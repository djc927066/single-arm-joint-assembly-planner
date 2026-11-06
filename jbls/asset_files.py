"""Bridge STL inputs and reusable WRS-compatible nominal grasp catalogues.

All poses and contacts in a grasp file are in the part's local frame, in metres.
Loading a file does not certify force closure, IK or collision-free motion.
The existing analyzer still calibrates and checks the loaded candidates.
"""
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import pickle
import tempfile

import numpy as np

from asmcore._stl import read_stl_arrays
from asmcore.model import digest
from asmcore.robotics.grasp_model import GraspRecord, gripper_signature


DEFAULT_ASSET_DIR = Path(__file__).resolve().parent / 'assets' / 'bridge'
SCHEMA = 'jbls.nominal_grasps.v1'
ROLL_DEGREES = (0, 30, -30, 60, -60, 90, -90)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _stl_bytes(part):
    """ASCII STL retains double precision for exact nominal mating contacts.

Binary STL truncates vertices to float32. A nanometre-scale overlap can change
the conservative contact classifier, even though the drawing looks identical.
Do not loosen physical contact tests to compensate for a lossy file export.
"""
    rows = ['solid ' + part.part_id]
    for triangle in part.geometry.vertices[part.geometry.faces]:
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        length = np.linalg.norm(normal)
        if length:
            normal /= length
        rows.append('  facet normal ' + ' '.join(format(float(x), '.17g') for x in normal))
        rows.append('    outer loop')
        for vertex in triangle:
            rows.append('      vertex ' + ' '.join(format(float(x), '.17g') for x in vertex))
        rows.extend(('    endloop', '  endfacet'))
    rows.append('endsolid ' + part.part_id)
    return ('\n'.join(rows) + '\n').encode('ascii')


def read_part_stl(path):
    """Read generated ASCII STL without rounding away its contact precision."""
    raw = Path(path).read_bytes()
    if raw[:5].lower() != b'solid' or b'facet' not in raw[:2048]:
        return read_stl_arrays(path)
    points = [list(map(float, line.split()[1:4]))
              for line in raw.decode('ascii').splitlines()
              if line.strip().startswith('vertex')]
    triangle_vertices = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    vertices, indices = np.unique(triangle_vertices, axis=0, return_inverse=True)
    return vertices, indices.reshape(-1, 3).astype(np.int64)


def prepare_part_stls(parts, asset_dir, *, rebuild=False):
    """Materialize each assembly part, then use the STL-loaded geometry everywhere.

The task generator supplies the current Bridge shapes. Existing different STL
files are rejected unless rebuilding is explicitly requested; no stale scale is
silently used. The table remains the workspace's configured solid box.
"""
    mesh_dir = Path(asset_dir).resolve() / 'meshes'
    loaded, files = [], {}
    for part in parts:
        if part.part_id == 'table':
            loaded.append(part)
            continue
        path = mesh_dir / (part.part_id + '.stl')
        data = _stl_bytes(part)
        expected = hashlib.sha256(data).hexdigest()
        existed = path.exists()
        if existed and not rebuild and file_sha256(path) != expected:
            raise ValueError(f'STL differs from the current Bridge geometry: {path}. '
                             'Use --rebuild-assets or a different --asset-dir.')
        if not existed or rebuild:
            _atomic_write(path, data)
        vertices, faces = read_part_stl(path)
        geometry = replace(part.geometry, vertices=vertices, faces=faces)
        loaded.append(replace(part, geometry=geometry))
        files[part.part_id] = dict(stl=str(path), stl_sha256=file_sha256(path),
                                  geometry_id=geometry.geometry_id,
                                  stl_encoding='ascii_float64',
                                  stl_status='generated' if not existed or rebuild else 'reused')
    return loaded, files


def generate_nominal_grasps(part, gripper):
    """Same finite box/roll candidates as before, generated from the loaded STL."""
    from asmcore.robotics.grasp_geometry import box_grasps
    from scipy.spatial.transform import Rotation

    candidates = []
    for grasp in box_grasps(part, gripper):
        for degrees in ROLL_DEGREES:
            pose = np.array(grasp.pose)
            pose[:3, :3] = pose[:3, :3] @ Rotation.from_euler(
                'y', degrees, degrees=True).as_matrix()
            pre = pose.copy()
            pre[:3, 3] -= .06 * pose[:3, 2]
            candidates.append(grasp.replace(pose=pose, pre_pose=pre, provenance={
                'approach_roll_deg': degrees, 'file_candidate_id': len(candidates)}))
    return tuple(candidates)


def grasp_binding(part, stl_path, gripper):
    """Bind a local nominal catalogue to geometry, gripper and generator code.

Goal poses and layout are deliberately absent: IK/environment filtering happens
after loading, so moving the assembly site does not invalidate a local catalogue.
"""
    from asmcore.robotics import grasp_geometry

    return dict(schema=SCHEMA, part_id=part.part_id, units='metre', frame='part_local',
                stage='nominal_candidates', stl_sha256=file_sha256(stl_path),
                geometry_id=part.geometry.geometry_id,
                gripper_signature=digest(gripper_signature(gripper)),
                gripper_type=type(gripper).__module__ + '.' + type(gripper).__qualname__,
                jaw_range_m=list(gripper.jaw_range), roll_degrees=list(ROLL_DEGREES),
                generator_sha256=digest((file_sha256(__file__),
                                         file_sha256(grasp_geometry.__file__))),
                goal_ik_prefiltered=False, calibration_and_force_closure_pending=True)


def save_grasp_file(path, records, binding):
    """Write a WRS GraspCollection, retaining asmcore contact/pre-pose metadata."""
    from wrs.grasping.grasp import Grasp, GraspCollection

    collection = GraspCollection()
    payloads = []
    for candidate_id, record in enumerate(records):
        record = record.replace(provenance={'file_candidate_id': candidate_id})
        payload = record.to_dict()
        entry = Grasp(ee_values=record.jaw_width, ac_pos=np.array(record.pose[:3, 3]),
                      ac_rotmat=np.array(record.pose[:3, :3]))
        entry.asmcore_record = payload
        collection.append(entry)
        payloads.append(payload)
    collection.jbls_metadata = dict(binding=binding, count=len(payloads),
                                    records_sha256=digest(payloads))
    # WRS save_to_disk recreates the collection and drops custom metadata.
    _atomic_write(path, pickle.dumps(collection, protocol=pickle.HIGHEST_PROTOCOL))


def load_grasp_file(path, binding):
    """Read the on-disk catalogue; explicit empty files stay empty, never fallback."""
    from wrs.grasping.grasp import GraspCollection

    path = Path(path).resolve()
    collection = GraspCollection.load_from_disk(str(path))
    metadata = getattr(collection, 'jbls_metadata', {})
    if metadata.get('binding') != binding:
        raise ValueError(f'Grasp file binding mismatch: {path}. '
                         'Use --rebuild-assets or a different --asset-dir.')
    records = []
    for candidate_id, entry in enumerate(collection):
        if not hasattr(entry, 'asmcore_record'):
            raise ValueError(f'Missing contact/pre-pose data in {path}, entry {candidate_id}')
        record = GraspRecord.from_dict(entry.asmcore_record)
        if (record.provenance.get('file_candidate_id') != candidate_id
                or not np.array_equal(entry.ac_pos, record.pose[:3, 3])
                or not np.array_equal(entry.ac_rotmat, record.pose[:3, :3])
                or float(entry.ee_values) != record.jaw_width):
            raise ValueError(f'Inconsistent grasp data in {path}, entry {candidate_id}')
        records.append(record)
    if (metadata.get('count') != len(records)
            or metadata.get('records_sha256') != digest([g.to_dict() for g in records])):
        raise ValueError(f'Grasp file contents changed without updating metadata: {path}')
    return tuple(records)


def prepare_grasp_file(part, stl_path, gripper, asset_dir, *, rebuild=False):
    """Generate only if missing/rebuilding, then ALWAYS reload before screening."""
    path = Path(asset_dir).resolve() / 'grasps' / (part.part_id + '_grasps.pickle')
    binding = grasp_binding(part, stl_path, gripper)
    generated = rebuild or not path.exists()
    if generated:
        save_grasp_file(path, generate_nominal_grasps(part, gripper), binding)
    records = load_grasp_file(path, binding)
    return records, dict(grasp_file=str(path), grasp_file_sha256=file_sha256(path),
                         grasp_file_count=len(records), grasp_file_stage=binding['stage'],
                         grasp_file_status='generated' if generated else 'reused')
