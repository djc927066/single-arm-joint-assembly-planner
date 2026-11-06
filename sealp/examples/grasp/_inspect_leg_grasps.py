"""
Quick inspection of the leg (yuanchair-part2) grasp library.

Goal: figure out whether the existing grasp set covers BOTH
upright-leg picks (leg long axis ≈ world z) and lying-leg picks
(leg long axis ≈ world x/y). We do this in object-local coordinates
since the saved grasps live there.

Reported metrics:
  * Total grasp count
  * Distribution of |approach · z_local|
      - approach axis = grasp.ac_rotmat[:, 2] (last column = gripper approach)
      - small value → side approach (good for upright-leg pick from the side)
      - large value → end-cap approach (good for lying-leg pick from one end)
  * Distribution of grasp position along leg long axis (z_local)
  * Yaw distribution around the long axis (atan2 of approach.x, approach.y)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from wrs.grasping.grasp import GraspCollection


def main():
    pickle_path = os.path.join(
        os.path.dirname(__file__), "_output", "demo_yuanchair-part2_grasps.pickle")
    gc = GraspCollection.load_from_disk(file_name=pickle_path)
    n = len(gc)
    print(f"Loaded {n} grasps from {pickle_path}")

    approaches = np.asarray(
        [g.ac_rotmat[:, 2] for g in gc])           # (N,3) approach in object frame
    closing = np.asarray(
        [g.ac_rotmat[:, 0] for g in gc])           # (N,3) jaw-closing in object frame
    positions = np.asarray([g.ac_pos for g in gc])

    cos_z = np.abs(approaches @ np.array([0, 0, 1.0]))   # |approach · z|
    cos_x = np.abs(approaches @ np.array([1.0, 0, 0]))
    cos_y = np.abs(approaches @ np.array([0, 1.0, 0]))

    bins = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.001]
    hist_z, _ = np.histogram(cos_z, bins=bins)
    hist_x, _ = np.histogram(cos_x, bins=bins)
    hist_y, _ = np.histogram(cos_y, bins=bins)

    def _fmt(h):
        return ", ".join(f"{c:>3d}" for c in h.tolist())

    print("\n|approach · axis| histogram, bins=[0,0.1,0.3,0.5,0.7,0.9,1.0]")
    print(f"  axis = z_local : {_fmt(hist_z)}")
    print(f"  axis = x_local : {_fmt(hist_x)}")
    print(f"  axis = y_local : {_fmt(hist_y)}")

    side = int(np.sum(cos_z < 0.3))
    angled = int(np.sum((cos_z >= 0.3) & (cos_z < 0.7)))
    end = int(np.sum(cos_z >= 0.7))
    print("\nSide vs end approach (relative to leg long axis = z_local):")
    print(f"  side   (|cos_z| < 0.3): {side:>4d}  ({100 * side / n:.1f}%)")
    print(f"  angled                : {angled:>4d}  ({100 * angled / n:.1f}%)")
    print(f"  end    (|cos_z| ≥ 0.7): {end:>4d}  ({100 * end / n:.1f}%)")

    print("\nGrasp position along z_local (leg long axis):")
    z = positions[:, 2]
    print(f"  z range: [{z.min():.3f}, {z.max():.3f}] m,  mean={z.mean():.3f}")
    z_bins = np.linspace(z.min(), z.max(), 6)
    z_hist, _ = np.histogram(z, bins=z_bins)
    print(f"  histogram (5 bins from min..max): {_fmt(z_hist)}")

    yaws = np.degrees(np.arctan2(approaches[:, 1], approaches[:, 0]))
    yaw_bins = np.linspace(-180, 180, 13)
    yaw_hist, _ = np.histogram(yaws, bins=yaw_bins)
    print("\nYaw of approach around z_local (deg, bins of 30°):")
    print(f"  bins  : {[f'{int(b)}' for b in yaw_bins[:-1]]}")
    print(f"  counts: {yaw_hist.tolist()}")

    print("\nClosing axis vs z_local |closing · z|:")
    cos_close_z = np.abs(closing @ np.array([0, 0, 1.0]))
    cz_hist, _ = np.histogram(cos_close_z, bins=bins)
    print(f"  hist : {_fmt(cz_hist)}")


if __name__ == "__main__":
    main()
