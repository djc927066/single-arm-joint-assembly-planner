"""End-to-end checks for the vendored WRS1/Panthera foundation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from adapters import PantheraArmProvider
from asmcore.mechanics.force_closure import check_force_closure


def test_vendored_panthera_fk_ik_and_collision() -> None:
    import wrs

    project_root = Path(__file__).resolve().parents[1]
    assert Path(wrs.__file__).resolve().is_relative_to(project_root / "wrs")

    provider = PantheraArmProvider(enable_cc=True)
    arm = provider.arm()
    home = provider.home_jnt_values()
    assert home.shape == (6,)
    assert arm.n_dof == 6

    tcp = provider.home_tcp_world()
    solution = arm.ik(
        tgt_pos=tcp[:3, 3],
        tgt_rotmat=tcp[:3, :3],
        seed_jnt_values=home,
    )
    assert solution is not None
    assert np.asarray(solution).shape == (6,)
    arm.goto_given_conf(solution)
    assert not arm.is_collided()


def test_force_closure_reference_cases() -> None:
    two_points = np.array([[-0.02, 0, 0], [0.02, 0, 0]])
    two_normals = np.array([[1, 0, 0], [-1, 0, 0]])
    assert check_force_closure(two_points, two_normals)["rank"] == 5

    points = np.array(
        [[x, y, z] for x in (-0.02, 0.02) for y in (-0.01, 0.01) for z in (-0.01, 0.01)]
    )
    normals = np.array([[-np.sign(x), 0, 0] for x, _, _ in points])
    assert check_force_closure(points, normals)["status"] == "force_closed"
    assert check_force_closure(points, normals, friction=0)["status"] == "not_force_closed"
