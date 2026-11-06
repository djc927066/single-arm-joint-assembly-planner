"""File-backed Bridge grasp candidates preserve contact data and G behavior."""
from dataclasses import replace
import pickle

import numpy as np
import pytest

from asmcore.model import Assembly, AssemblyState
from asmcore.robotics import GraspabilityAnalyzer, GraspabilityConfig
from jbls import asset_files
from stub_gripper import StubParallelGripper
from tasks.bridge import build_bridge


@pytest.fixture
def assets(tmp_path):
    parts, files = asset_files.prepare_part_stls(build_bridge(scale=2 / 3).parts, tmp_path)
    return Assembly(tuple(parts)), files, tmp_path, StubParallelGripper()


def test_all_assembly_parts_have_reused_stls(assets):
    assembly, files, root, _ = assets
    original = {p.part_id: p for p in build_bridge(scale=2 / 3).parts}
    assert set(files) == {'base', 'post_l', 'post_r', 'beam', 'load'}
    for part in assembly.parts:
        if part.part_id != 'table':
            vertices, faces = asset_files.read_part_stl(files[part.part_id]['stl'])
            np.testing.assert_array_equal(vertices, part.geometry.vertices)
            np.testing.assert_array_equal(faces, part.geometry.faces)
            source = original[part.part_id].geometry
            np.testing.assert_array_equal(vertices[faces], source.vertices[source.faces])
    _, again = asset_files.prepare_part_stls(build_bridge(scale=2 / 3).parts, root)
    assert all(item['stl_status'] == 'reused' for item in again.values())
    with pytest.raises(ValueError, match='STL differs'):
        asset_files.prepare_part_stls(build_bridge(scale=1).parts, root)


@pytest.mark.parametrize('movable, expected', [
    (('post_l',), 'feasible'),
    (('post_l', 'post_r'), 'feasible'),
    (('post_l', 'post_r', 'beam'), 'feasible'),
    (('post_l', 'post_r', 'beam', 'load'), 'feasible'),
    (('post_l', 'beam', 'load'), 'infeasible'),
])
def test_stl_roundtrip_preserves_stable_and_unstable_bridge_states(assets, movable, expected):
    from asmcore.planning.sequence import SequenceConfig, SequenceEvaluator

    assembly, _, _, _ = assets
    evaluator = SequenceEvaluator(assembly, SequenceConfig(max_auxiliary_resources=0))
    state = AssemblyState({p.part_id: p.assembled_tf for p in assembly.parts
                           if p.fixed or p.part_id in movable})
    assert evaluator.equilibrium(state, ()).status == expected


def test_wrs_pickle_roundtrip_preserves_contacts_and_is_reused(assets, monkeypatch):
    from wrs.grasping.grasp import GraspCollection

    assembly, files, root, gripper = assets
    part = next(p for p in assembly.parts if p.part_id == 'post_l')
    expected = asset_files.generate_nominal_grasps(part, gripper)
    loaded, info = asset_files.prepare_grasp_file(part, files['post_l']['stl'], gripper, root)
    assert info['grasp_file_status'] == 'generated'
    assert [g.to_dict() for g in loaded] == [g.to_dict() for g in expected]
    wrs = GraspCollection.load_from_disk(info['grasp_file'])
    assert len(wrs) == len(loaded) > 0
    assert 'contacts_local_m' in loaded[0].provenance
    assert loaded[0].provenance['file_candidate_id'] == 0

    def unexpected_generation(*args):
        raise AssertionError('Existing file must be loaded, not regenerated')
    monkeypatch.setattr(asset_files, 'generate_nominal_grasps', unexpected_generation)
    second, again = asset_files.prepare_grasp_file(part, files['post_l']['stl'], gripper, root)
    assert again['grasp_file_status'] == 'reused'
    assert again['grasp_file_sha256'] == info['grasp_file_sha256']
    assert [g.to_dict() for g in second] == [g.to_dict() for g in expected]


def test_stale_gripper_and_corrupt_entry_are_rejected(assets):
    assembly, files, root, gripper = assets
    part = next(p for p in assembly.parts if p.part_id == 'beam')
    _, info = asset_files.prepare_grasp_file(part, files['beam']['stl'], gripper, root)
    with pytest.raises(ValueError, match='binding mismatch'):
        asset_files.prepare_grasp_file(part, files['beam']['stl'],
                                      StubParallelGripper(jaw_range=(0, .08)), root)
    with open(info['grasp_file'], 'rb') as stream:
        collection = pickle.load(stream)
    collection[0].ee_values += .001
    with open(info['grasp_file'], 'wb') as stream:
        pickle.dump(collection, stream)
    with pytest.raises(ValueError, match='Inconsistent grasp data'):
        asset_files.prepare_grasp_file(part, files['beam']['stl'], gripper, root)


def test_empty_file_never_generates_fallback_grasps(assets):
    assembly, files, root, gripper = assets
    part = next(p for p in assembly.parts if p.part_id == 'load')
    _, info = asset_files.prepare_grasp_file(part, files['load']['stl'], gripper, root)
    binding = asset_files.grasp_binding(part, files['load']['stl'], gripper)
    asset_files.save_grasp_file(info['grasp_file'], (), binding)
    loaded, again = asset_files.prepare_grasp_file(part, files['load']['stl'], gripper, root)
    assert loaded == ()
    assert again['grasp_file_status'] == 'reused'


def test_loaded_candidates_keep_calibration_force_closure_and_environment_filter(assets):
    assembly, files, root, gripper = assets
    part = next(p for p in assembly.parts if p.part_id == 'post_l')
    direct = asset_files.generate_nominal_grasps(part, gripper)
    loaded, _ = asset_files.prepare_grasp_file(part, files['post_l']['stl'], gripper, root)
    # Place a solid obstacle at one candidate's palm. This explicitly tests
    # environment filtering without assuming a particular scale blocks the stub.
    probe = gripper.clone()
    probe.grip_at(loaded[0].pose[:3, 3], loaded[0].pose[:3, :3], loaded[0].jaw_width)
    palm = probe.link_parts()[0]
    blocker = replace(palm, part_id='grasp_blocker', fixed=True,
                      assembled_tf=part.assembled_tf @ palm.assembled_tf)
    assembly = Assembly(assembly.parts + (blocker,))
    cfg = GraspabilityConfig(max_grasps=len(direct))
    a = GraspabilityAnalyzer(assembly, gripper, config=cfg, grasps={'post_l': direct})
    b = GraspabilityAnalyzer(assembly, gripper, config=cfg, grasps={'post_l': loaded})
    alone = AssemblyState({p.part_id: p.assembled_tf for p in assembly.parts
                           if (p.fixed and p.part_id != 'grasp_blocker') or p.part_id == 'post_l'})
    ra, rb = a.analyze(alone, 'post_l'), b.analyze(alone, 'post_l')
    assert rb.accepted_ids == ra.accepted_ids
    assert rb.accepted_ids
    assert [g.to_dict() for g in b.accepted_grasps(rb)] == [g.to_dict() for g in a.accepted_grasps(ra)]
    crowded = AssemblyState(dict(alone.poses, grasp_blocker=blocker.assembled_tf))
    full = b.analyze(crowded, 'post_l')
    assert len(full.accepted_ids) < len(rb.accepted_ids)
    assert 'environment_collision' in {c['status'] for c in full.diagnostics['candidates']}
