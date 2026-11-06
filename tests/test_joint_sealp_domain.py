"""SEALP XY selection retains its original semantics with Bridge's single arm."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jbls.sealp_domain import SealpCandidateDomain


class FakeCM:
    def __init__(self, initor=None, name=None, vertices=None):
        self.pos = np.zeros(3)
        self.rotmat = np.eye(3)
        self.vertices = vertices

    def is_mcdwith(self, other):
        a = self.vertices @ self.rotmat.T + self.pos
        b = other.vertices @ other.rotmat.T + other.pos
        return bool(np.all(a.max(0) >= b.min(0)) and np.all(b.max(0) >= a.min(0)))


class FakeArm:
    def __init__(self):
        self.q = np.ones(6)
        self.stack = []

    def backup_state(self):
        self.stack.append(self.q.copy())

    def restore_state(self):
        self.q = self.stack.pop()

    def goto_given_conf(self, jnt_values):
        self.q = np.array(jnt_values)

    def is_collided(self, obstacle_list):
        assert np.all(self.q == 0)  # the actual original routine must switch to HOME
        p = obstacle_list[0].pos
        return bool(abs(p[0] - .28) < .08 and abs(p[1] - .28) < .08)


def box_vertices(size):
    from itertools import product
    return np.array(list(product(*[(-s/2, s/2) for s in size])))


@pytest.fixture
def domain():
    # cuRobo is not instantiated. Load torch before SEALP's Panda3D imports on Windows.
    from curobo_bridge.runtime import prepare
    prepare()
    vertices = {'base': box_vertices((.16, .16, .04)),
                'post': box_vertices((.04, .04, .10)),
                'beam': box_vertices((.32, .04, .02))}
    parts = {p: SimpleNamespace(mesh=SimpleNamespace(vertices=v), mesh_path=Path(p+'.stl'))
             for p, v in vertices.items()}
    base = FakeCM(vertices=vertices['base'])
    base.pos = np.array([.35, 0, .02])
    problem = SimpleNamespace(ids=('post', 'beam'), parts=parts, arm=FakeArm(),
        table_center=np.array([.25, 0, -.01]), table_size=np.array([.8, .8, .02]),
        workspace=SimpleNamespace(top_z=0, robot_pos=np.zeros(3)), goal_cm={'base': base})

    def factory(initor, name):
        return FakeCM(vertices=vertices[Path(initor).stem])

    return SealpCandidateDomain(problem, collision_factory=factory)


def test_selected_centres_equal_original_bsfs_beam(domain):
    from sealp.examples.layout.bsfs.domain import continuous_domain
    selected, stats = domain.select_centres('post', .05, 12)
    original = continuous_domain(domain, 'post', .05, 12, 'base')
    np.testing.assert_array_equal(selected, original)
    assert len(selected) == 12
    assert stats['raw_centres'] > stats['geometry_centres'] >= 12
    # Goal positions do not influence the original centroid/farthest-point order.
    domain.problem.goals = {'post': np.array([100., 100., 100.])}
    repeated, _ = domain.select_centres('post', .05, 12)
    np.testing.assert_array_equal(selected, repeated)
    np.testing.assert_array_equal(domain.problem.arm.q, np.ones(6))
    assert not domain.problem.arm.stack


def test_cap_is_xy_count_not_pose_count_and_zero_means_all(domain):
    selected, _ = domain.select_centres('post', .05, 3)
    poses = [(xy, rot) for xy in selected for rot in domain.rot_cands['post']
             if domain.check_pose('post', xy, rot) is None]
    assert len(selected) == 3
    assert len(poses) == 6
    all_xy, stats = domain.select_centres('post', .05, 0)
    assert len(all_xy) == stats['geometry_centres']


def test_each_orientation_must_clear_table_base_and_home(domain):
    candidates = domain.rot_cands['beam']
    # Grid padding uses the smallest footprint, so one rotation may overhang.
    xy = np.array([domain.table_x_range[1] - .05, .15])
    assert domain.check_pose('beam', xy, candidates[0]) == 'oriented_footprint_outside_table_margin'
    assert domain.check_pose('beam', xy, candidates[1]) is None
    # A part inside the fixed base is excluded by the original mesh filter.
    assert domain.check_pose('post', [.35, 0], domain.rot_cands['post'][0]) is not None
    # HOME collision is excluded, even away from the arm-root keepout rectangle.
    assert 'robot_' in domain.check_pose('post', [.28, .28], domain.rot_cands['post'][0])
    np.testing.assert_array_equal(domain.problem.arm.q, np.ones(6))
    assert not domain.problem.arm.stack


@pytest.mark.parametrize('spacing,cap', [(0, 12), (float('nan'), 12), (.05, -1)])
def test_invalid_domain_budget(domain, spacing, cap):
    with pytest.raises(ValueError):
        domain.select_centres('post', spacing, cap)
