"""One-shot L2 check for shelf_unit seed-anchor staging layout."""
from __future__ import annotations

import numpy as np

import sealp.examples.layout.find_optimal_layout as fol
from sealp.examples.layout._tasks import shelf_unit


def _debug_traj_step(searcher, pid: str, placed: set) -> None:
    sl = fol._ScoredLayout(
        xy={p: searcher.task.staging_seeds[p][:2].copy() for p in searcher.task.part_ids}
    )
    searcher._apply_pose(sl)
    s = next(x for x in searcher.asm.steps if x.part_id == pid)
    sp_xy = sl.xy[pid]
    gp, gr = searcher.world_poses[pid]
    gc = searcher.grasp_cache[searcher.model_alias_fn(pid)]
    rot_cands = fol.STAGING_ROTMAT_CANDIDATES.get(pid, [(np.eye(3), 0.0)])
    rot, z_off = rot_cands[0]
    extra_lift = 0.03 if pid.startswith("shelf") else 0.0
    sp = np.array([float(sp_xy[0]), float(sp_xy[1]), float(z_off + extra_lift)])
    obs = searcher._step_aware_obs(pid, placed)
    arm = searcher.robot.lft_arm
    ok, n, valid_gids, msg = fol._reason_common_ok(
        arm, gc, sp, rot, gp, gr, obs, return_reason=True,
    )
    print(f"  reason_common ok={ok} n={n} msg={msg}")
    if not ok or not valid_gids:
        return
    gid = valid_gids[0]
    pick = fol.check_pose_reachability(arm, sp, rot, gc, obs, max_grasps=5)
    jv = pick.best_jnt_values
    ok_t, tmin, tmean, curve = fol._trajectory_manipulability(
        arm, gc[gid], sp, rot, gp, gr, obstacle_list=obs,
        seed_jnt_values=jv, n_waypoints=fol.N_TRAJ_WAYPOINTS,
    )
    print(f"  traj ok={ok_t} min={tmin:.4f} curve_len={len(curve)}")
    if ok_t:
        return
    # manual per-waypoint probe
    import wrs.basis.robot_math as rm
    sr, grm = np.asarray(rot), np.asarray(gr)
    alphas = np.linspace(0.0, 1.0, fol.N_TRAJ_WAYPOINTS)
    _, rot_ang = rm.axangle_between_rotmat(sr, grm)
    lift_peak = 0.0
    if rot_ang > fol.TRAJ_LIFT_ANGLE_RAD:
        lift_peak = min(fol.TRAJ_LIFT_PEAK_M, 0.04 + 0.10 * float(rot_ang / np.pi))
    obj_rot_seq = list(rm.rotmat_slerp(sr, grm, fol.N_TRAJ_WAYPOINTS))
    grasp = gc[gid]
    seed = jv
    for i, a in enumerate(alphas):
        p = (1.0 - a) * sp + a * gp
        if lift_peak > 0:
            p = np.asarray(p, dtype=float).copy()
            p[2] += lift_peak * fol._traj_lift_scale(a)
        R = obj_rot_seq[i]
        tcp_p = R @ grasp.ac_pos + p
        tcp_R = R @ grasp.ac_rotmat
        jv_i = arm.ik(tgt_pos=tcp_p, tgt_rotmat=tcp_R, seed_jnt_values=seed) if i else seed
        if jv_i is None:
            jv_i = arm.ik(tgt_pos=tcp_p, tgt_rotmat=tcp_R, seed_jnt_values=None)
        if jv_i is None:
            print(f"    wp {i}: IK fail  z={p[2]:.3f}")
            break
        arm.backup_state()
        try:
            arm.goto_given_conf(jv_i)
            coll = arm.is_collided(obstacle_list=obs)
            print(f"    wp {i}: IK ok coll={coll} z={p[2]:.3f}")
            if coll:
                break
        finally:
            arm.restore_state()
        seed = jv_i


def main() -> None:
    fol.TRAJ_GRASP_TRY_LIMIT = 50
    task, rotmat_dict = shelf_unit.register_task(verbose=False)
    fol.STAGING_ROTMAT_CANDIDATES.update(rotmat_dict)

    searcher = fol.FastLayoutSearcher(task, enable_l3=False, ik_retry_n=3)
    sl = fol._ScoredLayout(
        xy={pid: task.staging_seeds[pid][:2].copy() for pid in task.part_ids}
    )
    print("seed xy:")
    for pid in task.part_ids:
        print(f"  {pid}: {np.round(sl.xy[pid], 4).tolist()}")

    if not searcher.l1(sl):
        print("L1 FAIL:", sl.fail_reason)
        return
    print("L1 OK")

    if not searcher.l2(sl):
        print("L2 FAIL:", sl.fail_reason)
        print("[traj debug step=1 shelf_m, placed={side_l}]")
        _debug_traj_step(searcher, "shelf_m", {"side_l"})
        return
    print("L2 OK")
    print("grasp_counts:", sl.grasp_counts)
    print("arms:", sl.arm_choice)


if __name__ == "__main__":
    main()
