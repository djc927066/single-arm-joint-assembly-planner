"""Scaled geometry must preserve the assembly fit, not just its appearance."""
import numpy as np
import pytest

from tasks.bridge import build_bridge


def _parts(**kwargs):
    return {p.part_id: p for p in build_bridge(**kwargs).parts}


def _world_vertices(part):
    return part.geometry.vertices @ part.assembled_tf[:3, :3].T + part.assembled_tf[:3, 3]


def test_scaled_pockets_still_fit_posts_and_preserve_support_contacts():
    parts = _parts(scale=2 / 3)
    base = _world_vertices(parts["base"])
    # Six x breakpoints: outer sides and the four actual pocket walls in the STL.
    xs = np.unique(base[:, 0])
    ys = np.unique(base[:, 1])
    zs = np.unique(base[:, 2])
    assert len(xs) == 6 and len(ys) == 4 and len(zs) == 3
    beam = _world_vertices(parts["beam"])
    for pid, wall_pair in (("post_l", xs[1:3]), ("post_r", xs[3:5])):
        post = _world_vertices(parts[pid])
        low, high = post.min(axis=0), post.max(axis=0)
        expected_gap = 0.0002 * (2 / 3)
        assert low[0] - wall_pair[0] == pytest.approx(expected_gap, abs=1e-8)
        assert wall_pair[1] - high[0] == pytest.approx(expected_gap, abs=1e-8)
        assert low[1] - ys[1] == pytest.approx(expected_gap, abs=1e-8)
        assert ys[2] - high[1] == pytest.approx(expected_gap, abs=1e-8)
        assert low[2] == pytest.approx(zs[1], abs=1e-9)  # bottom on pocket floor
        assert high[2] == pytest.approx(beam[:, 2].min(), abs=1e-12)
        assert beam[:, 0].min() <= low[0] and high[0] <= beam[:, 0].max()
    load = _world_vertices(parts["load"])
    assert load[:, 2].min() == pytest.approx(beam[:, 2].max(), abs=1e-12)
    assert np.ptp(base, axis=0) == pytest.approx([0.34 * 2 / 3, 0.34 * 2 / 3, 0.02 * 2 / 3])


def test_scaling_preserves_density_and_scales_com_without_changing_table():
    original, scaled = _parts(), _parts(scale=2 / 3)
    for pid, before in original.items():
        after = scaled[pid]
        assert after.fixed == before.fixed
        assert after.friction == before.friction
        if pid == "table":
            np.testing.assert_array_equal(after.geometry.vertices, before.geometry.vertices)
            np.testing.assert_array_equal(after.assembled_tf, before.assembled_tf)
            assert after.mass_kg == before.mass_kg
            continue
        before_volume = np.prod(np.ptp(before.geometry.vertices, axis=0))
        after_volume = np.prod(np.ptp(after.geometry.vertices, axis=0))
        assert after.mass_kg / after_volume == pytest.approx(before.mass_kg / before_volume)
        np.testing.assert_allclose(after.com_local_m, before.com_local_m * (2 / 3))
        np.testing.assert_allclose(after.assembled_tf[:3, :3], np.eye(3))
        assert after.geometry.geometry_id != before.geometry.geometry_id
    assert scaled["base"].com_local_m[2] == pytest.approx(0.02 / 3)
    assert scaled["post_l"].mass_kg == pytest.approx(0.45 * (2 / 3) ** 3)


def test_scaled_baseless_posts_stand_on_unscaled_table():
    parts = _parts(scale=2 / 3, with_base=False)
    assert "base" not in parts
    tabletop = _world_vertices(parts["table"])[:, 2].max()
    for pid in ("post_l", "post_r"):
        assert _world_vertices(parts[pid])[:, 2].min() == pytest.approx(tabletop, abs=1e-12)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_reject_nonpositive_or_nonfinite_scale(scale):
    with pytest.raises(ValueError, match="scale must be finite and positive"):
        build_bridge(scale=scale)
