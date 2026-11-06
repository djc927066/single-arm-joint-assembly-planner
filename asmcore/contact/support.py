"""Certified point/edge contact in the interior of a solid's supporting face.

This deliberately excludes corner/corner and edge/edge contact: those can
require a union of motion cones rather than one set of linear inequalities.
"""
import numpy as np
from ..model import ContactPatch, Region
from ..geometry.planar import convex_intersection
from .planar import _segment_union


def support_face_contacts(part_a, tf_a, prep_a, surfaces_a,
                          part_b, tf_b, prep_b, *, config, budget):
    """Return bounded zero-area features plus an exhaustion flag.

    Both complete solids must lie in opposite supporting half-spaces. The
    supporting feature of B must have affine dimension at most one, and its
    clipped witnesses must lie in the relative interior of A's actual face.
    B's normal is an element of its edge/vertex normal cone, not a face normal.
    """
    eps = config.numerical_tol_m
    if not (prep_a.is_closed and prep_b.is_closed and prep_a.orientation_reliable and prep_b.orientation_reliable):
        return (), False
    error = prep_a.mesh.geometry_error_m+prep_b.mesh.geometry_error_m+config.pose_error_m
    if error > eps:
        return (), False
    va = prep_a.mesh.vertices@tf_a[:3,:3].T+tf_a[:3,3]
    vb = prep_b.mesh.vertices@tf_b[:3,:3].T+tf_b[:3,3]
    result = []
    for surface in surfaces_a:
        if surface.kind!='plane' or surface.residual_m+error>eps: continue
        normal = tf_a[:3,:3]@surface.normal
        origin = tf_a[:3,:3]@surface.origin+tf_a[:3,3]
        da,db = (va-origin)@normal,(vb-origin)@normal
        if da.max()>eps or db.min() < -eps or db.min()>eps: continue
        on = abs(db)<=eps-error-surface.residual_m
        support = vb[on]
        if not len(support): continue
        singular = np.linalg.svd(support-support.mean(0),compute_uv=False)
        if np.count_nonzero(singular>eps*max(1,len(support)))>1: continue
        basis = tf_a[:3,:3]@surface.basis
        chart = (va-origin)@basis
        triangles = chart[prep_a.mesh.faces[surface.face_ids]]
        bchart = (vb-origin)@basis
        edges = {}
        for face in prep_a.mesh.faces[surface.face_ids]:
            for p,q in zip(face,np.roll(face,-1)):
                key=tuple(sorted((int(p),int(q)))); edges[key]=edges.get(key,0)+1
        boundary = np.array([chart[list(edge)] for edge,count in edges.items() if count==1])
        if not len(boundary): continue

        def interior(point):
            start=boundary[:,0]; delta=boundary[:,1]-start
            length2=np.einsum('ij,ij->i',delta,delta)
            t=np.clip(np.einsum('ij,ij->i',point-start,delta)/length2,0,1)
            return np.min(np.linalg.norm(point-(start+t[:,None]*delta),axis=1))>4*eps

        features=[]; seen=set()
        for face in prep_b.mesh.faces:
            ids=tuple(sorted(int(i) for i in face if on[i]))
            if ids and ids not in seen:
                seen.add(ids); features.append(bchart[list(ids)])
        segments,points=[],[]
        for feature in features:
            for tri in triangles:
                if np.any(feature.max(0)+eps<tri.min(0)) or np.any(tri.max(0)+eps<feature.min(0)): continue
                if budget.used>=budget.max_tests: return tuple(result), True
                budget.consume()
                clipped=convex_intersection(feature,tri,eps)
                if not len(clipped): continue
                if len(clipped)>1 and np.linalg.norm(clipped[-1]-clipped[0])>config.min_length_m:
                    segments.append(clipped[[0,-1]])
                else: points.append(clipped.mean(0))
        segments=[s for s in _segment_union(segments,eps) if interior(s.mean(0))]
        unique=[]
        for p in points:
            if not interior(p) or any(np.linalg.norm(p-q)<=eps for q in unique): continue
            if any(np.linalg.norm(p-(s[0]+np.clip(np.dot(p-s[0],s[1]-s[0])/np.dot(s[1]-s[0],s[1]-s[0]),0,1)*(s[1]-s[0])))<=eps for s in segments): continue
            unique.append(p)
        for dimension,cells in ((1,segments),(0,[np.array([p]) for p in unique])):
            if not cells: continue
            world=[cell@basis.T+origin for cell in cells]
            witnesses=np.array([cell.mean(0) for cell in world])
            normals=np.tile(normal,(len(witnesses),1))
            result.append(ContactPatch(part_a.part_id,part_b.part_id,surface.patch_id,'support_feature',
                dimension,'active','bounded',witnesses,witnesses,normals,-normals,(Region(tuple(world)),),
                (-eps,eps),length_m=sum(np.linalg.norm(c[-1]-c[0]) for c in world) if dimension==1 else 0.,
                provenance=dict(backend='support_face/1',normal_kind='opposite_supporting_planes',
                    source_face_ids_a=list(map(int,surface.face_ids)),
                    source_vertex_ids_b=np.flatnonzero(on).tolist(),
                    error_bound_m=eps,relative_interior=True)))
    return tuple(result),False
