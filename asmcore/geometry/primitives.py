"""Deterministic metre-scale meshes for examples and analytical regression."""
import numpy as np
from ..model import MeshData, checked_tf


def pose(translation=(0, 0, 0), rotation=None):
    """Construct a local-to-world rigid pose, translation in metres."""
    tf = np.eye(4)
    tf[:3, 3] = translation
    if rotation is not None:
        tf[:3, :3] = rotation
    return checked_tf(tf)


def box(size=(0.1, 0.1, 0.1)):
    """Centered solid box with outward winding; side lengths in metres."""
    s = np.asarray(size, dtype=float)
    if s.shape != (3,) or not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError('size must contain three positive lengths')
    vs = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                   [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]]) * s / 2
    fs = [[0,2,1], [0,3,2], [4,5,6], [4,6,7], [0,1,5], [0,5,4],
          [1,2,6], [1,6,5], [2,3,7], [2,7,6], [3,0,4], [3,4,7]]
    return MeshData(vs, fs)


def rectangle(size=(0.1, 0.1), *, upward=True, alternate_diagonal=False):
    """A finite open support in z=0 with explicitly trusted winding."""
    x, y = np.asarray(size) / 2
    if min(x, y) <= 0:
        raise ValueError('Positive rectangle sizes required')
    vs = np.array([[-x,-y,0], [x,-y,0], [x,y,0], [-x,y,0]])
    fs = np.array([[0,1,2], [0,2,3]] if not alternate_diagonal else [[0,1,3], [1,2,3]])
    return MeshData(vs, fs if upward else fs[:, ::-1], 'trusted')


def rectangular_ring(outer=0.1, inner=0.04, *, upward=True):
    """Square planar ring, analytic area outer²-inner², with a real hole."""
    if not 0 < inner < outer:
        raise ValueError('Require 0 < inner < outer')
    vs = []
    for width in (outer, inner):
        vs.extend([[-width/2,-width/2,0], [width/2,-width/2,0],
                   [width/2,width/2,0], [-width/2,width/2,0]])
    fs = []
    for i in range(4):
        j = (i+1) % 4
        fs.extend([[i,j,j+4], [i,j+4,i+4]])
    fs = np.asarray(fs)
    return MeshData(vs, fs if upward else fs[:, ::-1], 'trusted')


def cylinder(radius=0.02, height=0.08, sections=48, *, inner_radius=0.0):
    """Closed Z-axis cylinder or annular tube, centered at the origin."""
    if radius <= 0 or height <= 0 or not 0 <= inner_radius < radius or sections < 3:
        raise ValueError('Invalid cylinder dimensions')
    angles = np.arange(sections) * 2*np.pi / sections
    vs, fs = [], []
    for r in ([radius, inner_radius] if inner_radius else [radius]):
        for z in (-height/2, height/2):
            vs.extend(np.column_stack((r*np.cos(angles), r*np.sin(angles), np.full(sections, z))))
    for i in range(sections):
        j, n = (i+1) % sections, sections
        fs.extend([[i,j,j+n], [i,j+n,i+n]])
        if inner_radius:
            fs.extend([[i+2*n,j+3*n,j+2*n], [i+2*n,i+3*n,j+3*n],
                       [i+n,j+n,j+3*n], [i+n,j+3*n,i+3*n],
                       [i,j+2*n,j], [i,i+2*n,j+2*n]])
    if not inner_radius:
        vs.extend([[0,0,-height/2], [0,0,height/2]])
        for i in range(sections):
            j = (i+1) % sections
            fs.extend([[2*sections,j,i], [2*sections+1,i+sections,j+sections]])
    return MeshData(vs, fs)


def sphere(radius=0.025, sections=24, rings=12):
    """Closed latitude mesh; exact poles allow point-contact fixtures."""
    if radius <= 0 or sections < 3 or rings < 3:
        raise ValueError('Invalid sphere parameters')
    vs = [[0,0,-radius], [0,0,radius]]
    for k in range(1, rings):
        lat = -np.pi/2 + np.pi*k/rings
        for j in range(sections):
            a = 2*np.pi*j/sections
            vs.append([radius*np.cos(lat)*np.cos(a), radius*np.cos(lat)*np.sin(a), radius*np.sin(lat)])
    fs = []
    for j in range(sections):
        nxt = (j+1) % sections
        fs.extend([[0,2+nxt,2+j], [1,2+(rings-2)*sections+j,2+(rings-2)*sections+nxt]])
    for k in range(rings-2):
        for j in range(sections):
            a, b = 2+k*sections+j, 2+k*sections+(j+1) % sections
            fs.extend([[a,b,b+sections], [a,b+sections,a+sections]])
    return MeshData(vs, fs)


def combine(meshes):
    """Combine disjoint meshes into one, preserving winding and components."""
    vs, fs, offset = [], [], 0
    for mesh in meshes:
        vs.append(mesh.vertices)
        fs.append(mesh.faces + offset)
        offset += len(mesh.vertices)
    return MeshData(np.concatenate(vs), np.concatenate(fs),
                    'trusted' if all(m.orientation == 'trusted' for m in meshes) else 'auto')
