"""curobo_bridge 回归测试。只在 cuRobo venv（能 import curobo 且有 CUDA）中运行，其余环境跳过。

    & E:\\Projects\\wrs-sealp-assemble\\dual_sealp\\cuRobotest\\.venv\\Scripts\\python.exe `
        -X utf8 -m pytest tests/test_curobo_bridge.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
pytest.importorskip("curobo")

from curobo_bridge.runtime import ASSET_DIR, prepare  # noqa: E402

prepare()

MODEL = ASSET_DIR / "panthera_wrsext"
if not (MODEL / "panthera_d1.yml").exists():
    pytest.skip("run curobo_bridge.build_spheres first", allow_module_level=True)


@pytest.fixture(scope="module")
def arm():
    from curobo_bridge.export_model import make_wrs_arm
    return make_wrs_arm()


def test_export_matches_wrs_geometry(arm):
    from curobo_bridge.export_model import ExportedPanthera, load_chain, verify_export
    chain = load_chain(MODEL)
    ex = ExportedPanthera(out_dir=MODEL, urdf_path=MODEL / "panthera.urdf",
                          chain_path=MODEL / "chain.json", jaw_width=chain["jaw_width"],
                          collision_links=tuple(chain["collision_links"]),
                          base_homomat=np.asarray(chain["base"]), home=np.asarray(chain["home"]),
                          limits=np.asarray(chain["limits"]),
                          velocity_limits=np.asarray(chain["velocity_limits"]))
    r = verify_export(ex, arm, samples=8, seed=3)
    assert r["max_vertex_error_m"] < 1e-6
    assert r["max_tcp_error_m"] < 1e-9


def test_curobo_fk_matches_wrs(arm):
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import JointState
    from curobo_bridge.curobo_models import load_robot_dict
    kin = Kinematics(KinematicsCfg.from_data_dict(load_robot_dict(MODEL, 1.0)))
    rng = np.random.default_rng(0)
    lim = np.array([[-2.4, 2.4], [0, 3.2], [0, 4.0], [-1.6, 1.6], [-1.7, 1.7], [-2.5, 2.5]])
    qs = rng.uniform(lim[:, 0], lim[:, 1], size=(16, 6))
    js = JointState.from_position(torch.as_tensor(qs, dtype=torch.float32, device="cuda"),
                                  joint_names=kin.joint_names)
    pos = kin.compute_kinematics(js).tool_poses.position.reshape(-1, 3).cpu().numpy()
    for q, p in zip(qs, pos):
        assert np.linalg.norm(p - arm.fk(q)[0]) < 2e-5


def test_envelope_contains_original_surface():
    import trimesh
    for name in ("link3", "gripper_palm"):
        orig = trimesh.load(MODEL / f"meshes/{name}.stl")
        env = trimesh.load(MODEL / f"meshes_fit/{name}.stl")
        assert env.is_watertight
        pts = orig.sample(2000, seed=1)
        inside = env.contains(pts)
        _, d, _ = trimesh.proximity.closest_point(env, pts[~inside]) if (~inside).any() \
            else (None, np.zeros(1), None)
        # ray-parity misclassifies a few points hugging the surface; none may be outside by >0.5 mm
        assert inside.mean() > 0.95
        assert np.all(d < 0.0035)


def test_home_is_feasible_with_scene_buffer():
    """Guards the HOME_TOUCHING_PAIRS ignore and the upstream self-buffer workaround."""
    from curobo_bridge.curobo_models import load_robot_dict, make_collision_checker, with_buffer
    from curobo_bridge.scenes import random_scene
    robot = with_buffer(load_robot_dict(MODEL, 1.0), 0.005)
    chk = make_collision_checker(robot, random_scene(2000).curobo_scene())
    q = torch.zeros(1, 1, 6, device="cuda")
    assert bool(chk.validate(q).all())
