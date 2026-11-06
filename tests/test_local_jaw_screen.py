"""A collision only during finger closing must reject a local grasp witness."""
from types import SimpleNamespace
import numpy as np
from curobo_bridge.assembly_demo import screen_grasps, keyposes


def test_intermediate_jaw_collision_rejected_by_opt_in_screen():
    stage = np.eye(4); stage[2, 3] = .1
    goal = np.eye(4); goal[0, 3] = .2; goal[2, 3] = .2
    G = np.diag([1., -1., -1., 1.])
    part = SimpleNamespace(goal=goal, grasps=[(G, .04)])
    pick = keyposes(part, stage, G)['pick'][:3, 3]
    world = SimpleNamespace(layout={'p': stage}, table='table',
                            stage_cm={'p': 'pick_part'}, goal_cm={'p': 'goal_part'}, risers={})
    world.objects = lambda assembled, staged: ['table'] + list(assembled) + list(staged)

    class IK:
        def solve(self, poses):
            q = np.zeros((len(poses), 1, 6)); q[:, 0, :3] = poses[:, :3, 3]
            return q, np.ones((len(poses), 1), dtype=bool)

    class Arm:
        end_effector = SimpleNamespace(jaw_range=(0, .1))
        width = .04
        def change_jaw_width(self, width):
            self.width = width
        def goto_given_conf(self, q):
            pass

    class Oracle:
        arm = Arm()
        def tcp(self, q):
            T = np.eye(4); T[:3, 3] = q[:3]; return T
        def state_ok(self, q, obs, **kwargs):
            if np.allclose(q[:3], pick) and np.isclose(self.arm.width, .044):
                return 'finger:swept_obstacle'
            return None

    oracle = Oracle()
    args = (0, {'p': part}, ['p'], world, oracle, IK(), {'p': []})
    legacy, _ = screen_grasps(*args)
    assert legacy is not None
    checked, stats = screen_grasps(*args, check_jaw_motion=True)
    assert checked is None
    assert stats['rejections']['jaw_motion:finger:swept_obstacle'] == 1
