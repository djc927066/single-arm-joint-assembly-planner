"""Topology-aware preprocessing with explicit precision and source face maps."""
from collections import defaultdict, deque
import numpy as np
from scipy.spatial import cKDTree
from ..model import GeometryConfig, MeshData, PreparedMesh


def prepare_mesh(mesh, *, config=None):
    """Clean a local mesh without mutating it; retain every original face ID.

    Exact duplicate vertices are always joined. Optional tolerance welding is
    representative-based (not transitive), so each displacement is <= weld_m.
    Auto orientation assumes non-self-intersecting closed manifold shells;
    nested shell ambiguity is reported instead of filling cavities silently.
    """
    config = config or GeometryConfig()
    vs, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
    if config.weld_m:
        tree = cKDTree(vs)
        representatives = np.full(len(vs), -1, dtype=int)
        for i in range(len(vs)):
            if representatives[i] < 0:
                neighbors = np.asarray(sorted(tree.query_ball_point(vs[i], config.weld_m)))
                neighbors = neighbors[representatives[neighbors] < 0]
                representatives[neighbors] = i
        chosen, merged = np.unique(representatives, return_inverse=True)
        vs, inverse = vs[chosen], merged[inverse]
    displacement = float(np.linalg.norm(vs[inverse] - mesh.vertices, axis=1).max())
    fs = inverse[mesh.faces]
    diagnostics = []
    raw = np.cross(vs[fs[:, 1]] - vs[fs[:, 0]], vs[fs[:, 2]] - vs[fs[:, 0]])
    areas = np.linalg.norm(raw, axis=1) / 2
    kept, sources, unique = [], [], {}
    removed_area = 0.0
    for i, face in enumerate(fs):
        if areas[i] <= config.degenerate_area_m2:
            removed_area += areas[i]
            continue
        key = tuple(sorted(face))
        if key in unique:
            sources[unique[key]].append(i)
        else:
            unique[key] = len(kept)
            kept.append(face)
            sources.append([i])
    if not kept:
        raise ValueError('All triangles are degenerate at the requested tolerance')
    if len(kept) != len(fs):
        diagnostics.append(f'cleaned_faces:{len(fs)-len(kept)};removed_degenerate_area_m2:{removed_area:.9g}')
    fs = np.asarray(kept, dtype=np.int64)
    used, compact = np.unique(fs, return_inverse=True)
    if len(used) != len(vs):
        diagnostics.append(f'removed_unreferenced_vertices:{len(vs)-len(used)}')
        vs, fs = vs[used], compact.reshape(-1, 3)
    edge_map = defaultdict(list)
    for i, face in enumerate(fs):
        for a, b in zip(face, np.roll(face, -1)):
            edge_map[tuple(sorted((a, b)))].append((i, 1 if a < b else -1))
    adjacent = [set() for _ in fs]
    orientation_links = [[] for _ in fs]
    for owners in edge_map.values():
        for i, si in owners:
            for j, sj in owners:
                if i != j:
                    adjacent[i].add(j)
                    orientation_links[i].append((j, -si * sj))
    nonmanifold = any(len(v) > 2 for v in edge_map.values())
    is_closed = all(len(v) == 2 for v in edge_map.values())
    if not is_closed:
        diagnostics.append('open_boundary')
    if nonmanifold:
        diagnostics.append('nonmanifold_edges')
    signs = np.zeros(len(fs), dtype=int)
    components, conflict = [], False
    for seed in range(len(fs)):
        if signs[seed]:
            continue
        signs[seed] = 1
        queue, comp = deque([seed]), []
        while queue:
            i = queue.popleft()
            comp.append(i)
            for j, relative in orientation_links[i]:
                expected = signs[i] * relative
                if signs[j] and signs[j] != expected:
                    conflict = True
                elif not signs[j]:
                    signs[j] = expected
                    queue.append(j)
        components.append(tuple(sorted(comp)))
    if conflict:
        diagnostics.append('nonorientable_or_nonmanifold')
    inconsistent = np.any(signs < 0)
    if inconsistent:
        diagnostics.append('inconsistent_winding')
        if config.repair_orientation and not conflict:
            fs[signs < 0] = fs[signs < 0][:, [0, 2, 1]]
            diagnostics.append('winding_repaired')
    bounds = [(vs[np.unique(fs[list(c)])].min(axis=0), vs[np.unique(fs[list(c)])].max(axis=0)) for c in components]
    nested = any(i != j and np.all(a >= c) and np.all(b <= d)
                 for i, (a, b) in enumerate(bounds) for j, (c, d) in enumerate(bounds))
    if nested and mesh.orientation == 'auto':
        diagnostics.append('nested_shells_require_trusted_winding')
    reliable = (not conflict and not nonmanifold and (not inconsistent or config.repair_orientation)
                and (mesh.orientation == 'trusted' or (is_closed and not nested)))
    if is_closed and mesh.orientation == 'auto' and not nested and not conflict:
        for comp in components:
            tri = vs[fs[list(comp)]]
            tri = tri - tri.mean(axis=(0, 1))
            volume = np.einsum('ij,ij->i', tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
            if abs(volume) <= np.finfo(float).eps * max(np.ptp(tri, axis=(0, 1))) ** 3:
                reliable = False
                diagnostics.append('zero_volume_shell')
            elif volume < 0:
                if config.repair_orientation:
                    fs[list(comp)] = fs[list(comp)][:, [0, 2, 1]]
                    diagnostics.append('outward_winding_repaired')
                else:
                    reliable = False
    raw = np.cross(vs[fs[:, 1]] - vs[fs[:, 0]], vs[fs[:, 2]] - vs[fs[:, 0]])
    lengths = np.linalg.norm(raw, axis=1)
    result = MeshData(vs, fs, mesh.orientation, mesh.geometry_error_m + displacement)
    return PreparedMesh(result, tuple(tuple(s) for s in sources), raw / lengths[:, None],
                        lengths / 2, tuple(tuple(sorted(a)) for a in adjacent),
                        tuple(components), is_closed, reliable, tuple(diagnostics), displacement)
