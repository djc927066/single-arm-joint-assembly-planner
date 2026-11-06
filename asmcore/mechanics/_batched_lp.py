"""Small related LPs: common feasible basis, batched simplex, certificates.

GPU arithmetic only proposes solutions. Original FP64 primal/dual residuals
decide acceptance; unresolved problems are sent to the reference HiGHS solver.
"""
from time import perf_counter
import numpy as np
from scipy.optimize import linprog


def starting_basis(matrix, rhs, tolerance=1e-9):
    """Find one feasible basis of the shared alpha=0 system, or return None."""
    m,n=matrix.shape
    if not m or n<m or np.linalg.matrix_rank(matrix)<m: return None
    result=linprog(np.ones(n),A_eq=matrix,b_eq=rhs,bounds=(0,None),method='highs')
    if not result.success: return None
    positive=np.flatnonzero(result.x>tolerance)
    order=np.r_[positive,np.setdiff1d(np.arange(n),positive)]
    basis=[]; orthogonal=[]
    for index in order:
        vector=matrix[:,index].copy()
        for _ in range(2):
            for q in orthogonal: vector-=q@vector*q
        length=np.linalg.norm(vector)
        if length>1e-11*max(1.,np.linalg.norm(matrix[:,index])):
            basis.append(int(index)); orthogonal.append(vector/length)
        if len(basis)==m: break
    if len(basis)!=m: return None
    inverse=np.linalg.solve(matrix[:,basis],np.eye(m)); basic=inverse@rhs
    if np.min(basic)<-tolerance or np.max(np.abs(matrix[:,basis]@basic-rhs))>tolerance: return None
    return np.asarray(basis),inverse,np.maximum(basic,0.)


def simplex(tableau,basis,*,xp,max_iterations=256,pivot_tolerance=1e-11,optimality_tolerance=1e-10):
    """All LPs pivot together; inactive/converged rows remain unchanged."""
    gpu=xp is not np
    kw={'device':tableau.device} if gpu else {}
    amin=(lambda x,axis:xp.amin(x,dim=axis)) if gpu else np.min
    amax=(lambda x,axis:xp.amax(x,dim=axis)) if gpu else np.max
    batch,m1,n1=tableau.shape; m,n=m1-1,n1-1
    ids=xp.arange(batch,**kw); columns=xp.arange(n,**kw)
    active=xp.ones(batch,dtype=xp.bool if gpu else bool,**kw)
    iterations=xp.zeros(batch,dtype=xp.int64,**kw)
    for iteration in range(max_iterations):
        reduced=tableau[:,-1,:n]
        # Largest reduced cost avoids hundreds of tiny Bland pivots. Switch to
        # Bland after 128 iterations; any unresolved LP needs a certified fallback.
        eligible=reduced>optimality_tolerance
        if iteration<128:
            eligible=eligible & (reduced>=amax(reduced,axis=1)[:,None]-optimality_tolerance)
        entering=amin(xp.where(eligible,columns[None,:],n),axis=1)
        has_enter=entering<n
        entering=xp.minimum(entering,xp.full_like(entering,n-1))
        pivot_column=tableau[ids,:m,entering]
        positive=pivot_column>pivot_tolerance
        ratios=xp.where(positive,tableau[:,:m,-1]/xp.where(positive,pivot_column,1.),float('inf'))
        minimum=amin(ratios,axis=1)
        tied=positive & (ratios<=minimum[:,None]+pivot_tolerance)
        # Bland's leaving-variable tie break prevents exact-arithmetic cycling.
        leaving_variable=amin(xp.where(tied,basis,n),axis=1)
        leaving=amin(xp.where(tied & (basis==leaving_variable[:,None]),xp.arange(m,**kw)[None,:],m),axis=1)
        possible=(leaving<m)&has_enter
        active=active&possible
        leaving=xp.minimum(leaving,xp.full_like(leaving,m-1))
        pivot=tableau[ids,leaving,entering]
        row=tableau[ids,leaving,:]/xp.where(active,pivot,1.)[:,None]
        factors=tableau[ids,:,entering]
        factors*=active[:,None]
        tableau-=factors[:,:,None]*row[:,None,:]
        tableau[ids,leaving,:]=xp.where(active[:,None],row,tableau[ids,leaving,:])
        basis[ids,leaving]=xp.where(active,entering,basis[ids,leaving])
        iterations+=active
        # CUDA synchronization is amortized across eight batched pivots.
        if iteration%8==7 and not bool(active.any()): break
    optimal=amax(tableau[:,-1,:n],axis=1)<=optimality_tolerance
    return tableau,basis,optimal,iterations


class BatchedLP:
    """Related max c*x LPs, [common, varying columns] x = rhs, x>=0."""
    def __init__(self,common,rhs,objective_count,*,backend='numpy',max_iterations=256):
        self.common=np.asarray(common,float); self.rhs=np.asarray(rhs,float)
        self.objective_count=objective_count; self.backend=backend; self.max_iterations=max_iterations
        self.start=starting_basis(self.common,self.rhs)
        self.canonical=self.start[1]@self.common if self.start is not None else None
        self.cuda_setup_s=0.
        self.device_name=None
        if backend=='cuda':
            start=perf_counter()
            import torch
            if not torch.cuda.is_available(): raise RuntimeError('CUDA backend requested but torch.cuda is unavailable')
            self.xp=torch; self.device_name=torch.cuda.get_device_name(0)
            torch.cuda.synchronize(); self.cuda_setup_s=perf_counter()-start
        elif backend=='numpy': self.xp=np
        else: raise ValueError('BatchedLP backend must be numpy or cuda')

    def solve(self,columns):
        count,m,k=columns.shape; n=self.common.shape[1]+k
        if self.start is None: return None
        initial_basis,inverse,basic=self.start
        tableau=np.zeros((count,m+1,n+1))
        tableau[:,:m,:n-k]=self.canonical
        tableau[:,:m,n-k:n]=np.einsum('ij,bjk->bik',inverse,columns)
        tableau[:,:m,-1]=basic
        tableau[:,-1,n-k:n]=1.
        basis=np.broadcast_to(initial_basis,(count,m)).copy()
        if self.backend=='cuda':
            torch=self.xp
            tableau=torch.as_tensor(tableau,device='cuda',dtype=torch.float64)
            basis=torch.as_tensor(basis,device='cuda',dtype=torch.int64)
        tableau,basis,optimal,iterations=simplex(tableau,basis,xp=self.xp,max_iterations=self.max_iterations)
        if self.backend=='cuda':
            tableau,basis,optimal,iterations=[x.detach().cpu().numpy() for x in (tableau,basis,optimal,iterations)]
        weights=np.zeros((count,n)); weights[np.arange(count)[:,None],basis]=tableau[:,:m,-1]
        basic_cost=(basis>=n-k).astype(float)
        dual=np.einsum('bi,bij,jk->bk',basic_cost,tableau[:,:m,initial_basis],inverse)
        return weights,dual,optimal,iterations
