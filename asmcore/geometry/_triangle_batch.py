"""Vectorized closed-triangle distance witnesses, including edge/face crossing.

Operates on small BVH leaf batches; never allocates the full mesh-pair product.
The scalar mesh_bvh primitives remain the independent reference and overlap test.
"""
import numpy as np


def _dot(a,b): return np.sum(a*b,axis=-1)


def _divide(a,b):
    return np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b!=0)


def closest_points(points,triangles):
    a,b,c=np.moveaxis(triangles,1,0)
    normal=np.cross(b-a,c-a); n2=_dot(normal,normal)
    projected=points-_divide(_dot(points-a,normal),n2)[:,None]*normal
    starts=triangles; delta=np.roll(triangles,-1,axis=1)-starts
    margins=_dot(np.cross(delta,projected[:,None]-starts),normal[:,None])
    inside=(n2>np.finfo(float).tiny)&np.all(margins>=-n2[:,None]*1e-14,axis=1)
    t=np.clip(_divide(_dot(points[:,None]-starts,delta),_dot(delta,delta)),0,1)
    candidates=starts+t[:,:,None]*delta
    choice=np.argmin(_dot(candidates-points[:,None],candidates-points[:,None]),axis=1)
    edges=candidates[np.arange(len(points)),choice]
    return np.where(inside[:,None],projected,edges)


def segment_pairs(p,q,r,s):
    d1,d2,v=q-p,s-r,p-r
    a,e=_dot(d1,d1),_dot(d2,d2)
    b,c,f=_dot(d1,d2),_dot(d1,v),_dot(d2,v)
    denom=a*e-b*b
    u=np.where(denom>1e-14*a*e,np.clip(_divide(b*f-c*e,denom),0,1),0.)
    t=_divide(b*u+f,e)
    u=np.where(t<0,np.clip(_divide(-c,a),0,1),u)
    u=np.where(t>1,np.clip(_divide(b-c,a),0,1),u); t=np.clip(t,0,1)
    degenerate_a=a<=np.finfo(float).tiny; degenerate_b=e<=np.finfo(float).tiny
    t=np.where(degenerate_a,np.clip(_divide(f,e),0,1),t)
    u=np.where(degenerate_a,0.,u)
    u=np.where(degenerate_b & ~degenerate_a,np.clip(_divide(-c,a),0,1),u)
    t=np.where(degenerate_b,0.,t)
    return p+u[:,None]*d1,r+t[:,None]*d2


def triangle_pairs(a,b,tol=1e-10):
    """N pairs -> two N×3 nearest witnesses; closed surface distance only."""
    count=len(a); ids=np.arange(count); left=[]; right=[]
    for source,target,reverse in ((a,b,False),(b,a,True)):
        p=source.reshape(-1,3)
        q=closest_points(p,np.repeat(target,3,axis=0)).reshape(count,3,3)
        left.append(q if reverse else source); right.append(source if reverse else q)
    p=np.repeat(a,3,axis=1).reshape(-1,3)
    q=np.repeat(np.roll(a,-1,axis=1),3,axis=1).reshape(-1,3)
    r=np.tile(b,(1,3,1)).reshape(-1,3)
    s=np.tile(np.roll(b,-1,axis=1),(1,3,1)).reshape(-1,3)
    pa,pb=segment_pairs(p,q,r,s)
    left.append(pa.reshape(count,9,3)); right.append(pb.reshape(count,9,3))
    left,right=np.concatenate(left,axis=1),np.concatenate(right,axis=1)
    choice=np.argmin(_dot(left-right,left-right),axis=1)
    pa,pb=left[ids,choice].copy(),right[ids,choice].copy()
    # Vertex/face and edge/edge distances do not detect a piercing edge.
    for source,target in ((a,b),(b,a)):
        normal=np.cross(target[:,1]-target[:,0],target[:,2]-target[:,0])
        normal=_divide(normal,np.linalg.norm(normal,axis=1)[:,None])
        d=_dot(source-target[:,None,0],normal[:,None]); following=np.roll(d,-1,axis=1)
        p=source+(np.roll(source,-1,axis=1)-source)*_divide(d,d-following)[:,:,None]
        q=closest_points(p.reshape(-1,3),np.repeat(target,3,axis=0)).reshape(count,3,3)
        crossing=(d*following<0)&(np.linalg.norm(p-q,axis=2)<=tol)
        any_hit=np.any(crossing,axis=1); hit=p[ids,np.argmax(crossing,axis=1)]
        pa=np.where(any_hit[:,None],hit,pa); pb=np.where(any_hit[:,None],hit,pb)
    return pa,pb
