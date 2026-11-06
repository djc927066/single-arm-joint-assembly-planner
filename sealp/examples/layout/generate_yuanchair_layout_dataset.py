"""[DEPRECATED] 旧 YuanChair 双臂专用采集脚本。

请改用与 Tower / Totem 相同的通用采集入口 ``generate_layout_dataset``::

    python -m sealp.examples.layout.generate_layout_dataset ^
      --output-jsonl sealp/examples/layout/_output/yuanchair_layout_v1.jsonl ^
      --overwrite --gen-samples 200 --gen-seeds 0,1 ^
      --gen-station-mode center_continuous --gen-assembly-type chair ^
      --asmdef sealp/assembly_sequence/_demo_output/yuanchair.asmdef ^
      --grasp-dir sealp/examples/grasp/yuanchair_grasp ^
      --part-order seat,leg_bl,leg_br,leg_fl,leg_fr

抓取目录 ``yuanchair_grasp/`` 需包含 ``seat_grasps.pickle`` 与
``leg_model_grasps.pickle``（四腿共用 leg_model）。

本文件保留仅为历史参考；新数据请勿再使用双臂 FastLayoutSearcher 路径。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, Iterable, List, Tuple


def _configure_thread_env_from_argv() -> None:
    """Set BLAS/OpenMP limits before importing numpy and WRS."""
    value = None
    for i, arg in enumerate(sys.argv):
        if arg == "--gen-threads" and i + 1 < len(sys.argv):
            value = sys.argv[i + 1]
            break
        if arg.startswith("--gen-threads="):
            value = arg.split("=", 1)[1]
            break
    if value is None:
        return
    try:
        n_threads = max(1, int(value))
    except ValueError:
        return
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(n_threads)


_configure_thread_env_from_argv()

import numpy as np

from sealp.examples.layout import find_optimal_layout as yopt, find_optimal_initial_layout_tower_strict_pycharm as tower
from sealp.examples.layout import generate_layout_dataset as durable


SCORE_STANDARD = "tower_v9_5"


def _parse_int_list(spec: str) -> List[int]:
    values = [int(s) for s in str(spec).replace(",", " ").split()]
    if not values:
        raise argparse.ArgumentTypeError("至少提供一个 seed")
    return values


def _parse_args() -> argparse.Namespace:
    default_out = os.path.join(
        os.path.dirname(__file__), "_output", "layout_dataset_yuanchair_v1.jsonl")
    p = argparse.ArgumentParser(
        description="YuanChair dual-arm dataset generation with Tower scoring")
    p.add_argument("--dataset-out", default=default_out)
    p.add_argument("--gen-samples", type=int, default=100,
                   help="每个 seed 的样本数")
    p.add_argument("--gen-seeds", type=_parse_int_list, default=[0])
    p.add_argument("--gen-resume", action="store_true",
                   help="扫描已有 JSONL，并从每个 seed 的下一 sample_index 续采")
    p.add_argument("--gen-threads", type=int, default=1,
                   help="BLAS/OpenMP 线程上限；必须在进程启动参数中提供才影响导入")
    p.add_argument("--gen-fsync-every", type=int, default=1)
    p.add_argument("--gen-max-errors", type=int, default=20)
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--sleep-between-samples", type=float, default=0.0,
                   help="每条样本后暂停秒数；CPU 温度受限时可设 1~5")
    p.add_argument("--ik-retry-n", type=int, default=0)
    p.add_argument("--include-seed-anchor", action="store_true",
                   help="每个 seed 的 sample_index=0 使用已知 staging seed；通常不建议")
    p.add_argument("--w-grasp", type=float, default=tower.DEFAULT_W_GRASP)
    p.add_argument("--w-manip", type=float, default=tower.DEFAULT_W_MANIP)
    p.add_argument("--w-dist", type=float, default=tower.DEFAULT_W_DIST)
    p.add_argument("--w-rot", type=float, default=tower.DEFAULT_W_ROT)
    args = p.parse_args()
    if args.gen_samples <= 0:
        p.error("--gen-samples 必须 > 0")
    if args.gen_threads <= 0:
        p.error("--gen-threads 必须 > 0")
    if args.gen_fsync_every <= 0:
        p.error("--gen-fsync-every 必须 > 0")
    if args.gen_max_errors <= 0:
        p.error("--gen-max-errors 必须 > 0")
    if args.progress_every <= 0:
        p.error("--progress-every 必须 > 0")
    if min(args.w_grasp, args.w_manip, args.w_dist, args.w_rot) < 0:
        p.error("评分权重不能为负")
    if args.w_grasp + args.w_manip + args.w_dist + args.w_rot <= 0:
        p.error("评分权重之和必须 > 0")
    return args


def _normalized_weights(args: argparse.Namespace) -> Dict[str, float]:
    raw = {
        "grasp": float(args.w_grasp),
        "manip": float(args.w_manip),
        "dist": float(args.w_dist),
        "rot": float(args.w_rot),
    }
    total = sum(raw.values())
    return {key: value / total for key, value in raw.items()}


def _table_geometry(searcher: yopt.FastLayoutSearcher
                    ) -> Tuple[List[float], List[float], float]:
    table = searcher.table_def
    if table is not None and table.get("type") == "box":
        pos = np.asarray(table["pos"], dtype=float)
        ext = np.asarray(table["extent"], dtype=float)
        return (
            [float(pos[0] - ext[0] / 2), float(pos[0] + ext[0] / 2)],
            [float(pos[1] - ext[1] / 2), float(pos[1] + ext[1] / 2)],
            float(pos[2] + ext[2] / 2),
        )
    all_bounds = list(searcher.bounds.values())
    return (
        [min(b[0][0] for b in all_bounds), max(b[0][1] for b in all_bounds)],
        [min(b[1][0] for b in all_bounds), max(b[1][1] for b in all_bounds)],
        0.0,
    )


def _geometry_cache(searcher: yopt.FastLayoutSearcher) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for pid in searcher.search_part_ids:
        mesh_path = searcher.asm.model_path(pid)
        vertices = tower._load_mesh_vertices(mesh_path)
        _, _, extent_arr = tower._bounds_after_rotation(vertices, np.eye(3))
        extent = np.asarray(extent_arr, dtype=float)
        long_side = float(np.max(extent)) if extent.size else 0.0
        short_side = float(np.min(extent)) if extent.size else 0.0
        out[pid] = {
            "extent": extent.tolist(),
            "footprint": extent[:2].tolist(),
            "volume": float(np.prod(extent)) if extent.size >= 3 else 0.0,
            "aspect_ratio": (
                float(long_side / short_side) if short_side > 1e-9 else 0.0
            ),
            "thinness": (
                float(short_side / long_side) if long_side > 1e-9 else 0.0
            ),
        }
    return out


def _parent_map(searcher: yopt.FastLayoutSearcher) -> Dict[str, str]:
    return {
        str(step.part_id): str(step.parent_id)
        for step in searcher.asm.steps
    }


def _topdown_counts(searcher: yopt.FastLayoutSearcher) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for pid in searcher.search_part_ids:
        alias = searcher.model_alias_fn(pid)
        gc = searcher.grasp_cache.get(alias)
        out[pid] = int(tower._count_downward_grasps(
            gc, np.eye(3), align_cos=tower.DEFAULT_TOPDOWN_ALIGN_COS
        )) if gc is not None else 0
    return out


def _signature(searcher: yopt.FastLayoutSearcher,
               args: argparse.Namespace,
               weights: Dict[str, float]) -> str:
    payload = {
        "schema": 2,
        "collector": "yuanchair_dual",
        "score_standard": SCORE_STANDARD,
        "asmdef": os.path.abspath(searcher.task.asmdef_path),
        "parts": list(searcher.search_part_ids),
        "bounds": searcher.bounds,
        "fixture_pos": np.asarray(searcher.task.fixture_pos).tolist(),
        "weights": weights,
        "include_seed_anchor": bool(args.include_seed_anchor),
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _sample_xy(searcher: yopt.FastLayoutSearcher,
               seed: int,
               sample_index: int,
               attempt: int,
               include_seed_anchor: bool) -> Dict[str, np.ndarray]:
    if include_seed_anchor and sample_index == 0:
        return {
            pid: np.asarray(searcher.task.staging_seeds[pid], dtype=float)[:2].copy()
            for pid in searcher.search_part_ids
        }
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(sample_index), int(attempt)]))
    xy: Dict[str, np.ndarray] = {}
    for pid in searcher.search_part_ids:
        (xlo, xhi), (ylo, yhi) = searcher.bounds[pid]
        xy[pid] = np.array([
            float(rng.uniform(xlo, xhi)),
            float(rng.uniform(ylo, yhi)),
        ])
    return xy


def _spatial_score(sl: yopt._ScoredLayout,
                   fixture_y: float,
                   part_ids: Iterable[str]) -> float:
    ys = np.asarray([
        float(sl.xy[pid][1]) - float(fixture_y)
        for pid in part_ids if pid in sl.xy
    ], dtype=float)
    if len(ys) <= 1:
        return 1.0
    spread = float(np.clip(np.std(ys) / 0.28, 0.0, 1.0))
    n_pos = int(np.sum(ys > 0.03))
    n_neg = int(np.sum(ys < -0.03))
    balance = float(min(n_pos, n_neg) / max(max(n_pos, n_neg), 1))
    return float(np.clip(0.65 * spread + 0.35 * balance, 0.0, 1.0))


def _apply_tower_score(searcher: yopt.FastLayoutSearcher,
                       sl: yopt._ScoredLayout,
                       weights: Dict[str, float]) -> Dict[str, object]:
    """Overwrite the native YuanChair score with the exact Tower formula."""
    native_score = float(sl.layout_score)
    rot_angles: Dict[str, float] = {}
    for pid in searcher.search_part_ids:
        _, goal_rot = searcher.world_poses[pid]
        staging_rot = sl.chosen_rotmat.get(pid, np.eye(3))
        rot_angles[pid] = tower._rot_angle(staging_rot, goal_rot)

    counts = list(sl.grasp_counts.values())
    n_min = float(min(counts)) if counts else 0.0
    n_mean = float(np.mean(counts)) if counts else 0.0
    grasp_score = (
        0.70 * tower._hill(
            n_min, tower.NORM_GRASP_MIN_TARGET, tower.NORM_GRASP_HILL_K)
        + 0.30 * tower._hill(
            n_mean, tower.NORM_GRASP_MEAN_TARGET, tower.NORM_GRASP_HILL_K)
    )
    avg_manip = (
        float(np.mean(list(sl.per_part_manip.values())))
        if sl.per_part_manip else 0.0
    )
    manip_score = float(
        1.0 - np.exp(-max(avg_manip, 0.0) / tower.NORM_MANIP_TARGET))
    avg_dist = (
        float(np.mean(list(sl.per_part_dist.values())))
        if sl.per_part_dist else 0.0
    )
    dist_score = float(
        np.exp(-max(avg_dist, 0.0) / tower.NORM_DIST_DECAY))
    avg_rot = float(np.mean(list(rot_angles.values()))) if rot_angles else 0.0
    rot_score = float(
        np.exp(-max(avg_rot, 0.0) / tower.NORM_ROT_DECAY))
    spatial_score = _spatial_score(
        sl, float(searcher.task.fixture_pos[1]), searcher.search_part_ids)
    base = (
        weights["grasp"] * grasp_score
        + weights["manip"] * manip_score
        + weights["dist"] * dist_score
        + weights["rot"] * rot_score
    )
    tower_score = float(base * (0.90 + 0.10 * spatial_score))

    sl.layout_score = tower_score
    sl.grasp_score_norm = grasp_score
    sl.manip_score_norm = manip_score
    sl.dist_score_norm = dist_score
    sl.per_part_rot_angle = rot_angles
    sl.rot_score_norm = rot_score
    sl.spatial_score_norm = spatial_score
    return {
        "native_yuanchair_score": native_score,
        "native_traj_manip_score_norm": float(sl.traj_manip_score_norm),
        "tower_score": tower_score,
        "avg_rot": avg_rot,
    }


def _rotation_name(pid: str, rotmat: np.ndarray) -> str:
    candidates = yopt.STAGING_ROTMAT_CANDIDATES.get(
        pid, [(np.eye(3), 0.0)])
    for index, (candidate, _) in enumerate(candidates):
        if np.allclose(candidate, rotmat, atol=1e-7):
            return f"yuanchair_rot_{index:02d}"
    return "yuanchair_rot_unknown"


def _record(searcher: yopt.FastLayoutSearcher,
            sl: yopt._ScoredLayout,
            *,
            seed: int,
            sample_index: int,
            signature: str,
            geometry: Dict[str, Dict],
            parents: Dict[str, str],
            topdown: Dict[str, int],
            table_x_range: List[float],
            table_y_range: List[float],
            table_top_z: float,
            l1_pass: bool,
            score_diag: Dict[str, object] | None,
            eval_time: float) -> Dict:
    fixture = np.asarray(searcher.task.fixture_pos, dtype=float)
    tab_cx = 0.5 * (table_x_range[0] + table_x_range[1])
    tab_cy = 0.5 * (table_y_range[0] + table_y_range[1])
    parts: List[Dict] = []
    first_pid = searcher.search_part_ids[0] if searcher.search_part_ids else None

    for index, pid in enumerate(searcher.search_part_ids):
        goal_pos, goal_rot = searcher.world_poses[pid]
        alias = searcher.model_alias_fn(pid)
        gc = searcher.grasp_cache.get(alias)
        chosen_rot = sl.chosen_rotmat.get(pid, np.eye(3))
        g = geometry[pid]
        parts.append({
            "part_id": pid,
            "order_index": index,
            "is_first": bool(pid == first_pid),
            "extent": list(g["extent"]),
            "footprint": list(g["footprint"]),
            "volume": float(g["volume"]),
            "aspect_ratio": float(g["aspect_ratio"]),
            "thinness": float(g["thinness"]),
            "goal_pos": np.asarray(goal_pos, dtype=float).reshape(-1).tolist(),
            "goal_rotmat": np.asarray(goal_rot, dtype=float).reshape(-1).tolist(),
            "parent": parents.get(pid),
            "topdown_count": int(topdown.get(pid, 0)),
            "grasp_total": int(len(gc)) if gc is not None else 0,
            "staging_xy": np.asarray(sl.xy[pid], dtype=float).reshape(-1).tolist(),
            "pose_tag": sl.pose_tag.get(pid),
            "rot_name": (
                _rotation_name(pid, chosen_rot) if sl.l2_pass else None
            ),
            "grasp_count": int(sl.grasp_counts.get(pid, 0)),
            "arm_choice": sl.arm_choice.get(pid),
            "per_part_dist": float(sl.per_part_dist.get(pid, 0.0)),
            "per_part_manip": float(sl.per_part_manip.get(pid, 0.0)),
            "per_part_rot_angle": float(
                getattr(sl, "per_part_rot_angle", {}).get(pid, 0.0)),
        })

    fail_detail = {
        "l1_pass": bool(l1_pass),
        "fail_step_id": int(getattr(sl, "fail_step_id", -1)),
        "l2_fail_breakdown": dict(
            getattr(sl, "l2_fail_breakdown", {}) or {}),
    }
    return {
        "sample_id": durable._stable_sample_id(signature, seed, sample_index),
        "sample_index": int(sample_index),
        "generation_signature": signature,
        "sampler_version": 2,
        "collector": "yuanchair_dual",
        "score_standard": SCORE_STANDARD,
        "seed": int(seed),
        "task_id": "yuanchair",
        "assembly_id": "yuanchair",
        "assembly_type": "chair",
        "num_parts": len(parts),
        "assembly_region_id": "fixed",
        "assembly_region_rc": [-1, -1],
        "assembly_grid": 1,
        "assembly_station_pos": fixture.tolist(),
        "station_distance_to_center": float(np.hypot(
            fixture[0] - tab_cx, fixture[1] - tab_cy)),
        "table_x_range": table_x_range,
        "table_y_range": table_y_range,
        "table_top_z": float(table_top_z),
        "part_order": list(searcher.search_part_ids),
        "parts": parts,
        "l1_pass": bool(l1_pass),
        "l2_pass": bool(sl.l2_pass),
        "l3_pass": bool(sl.l3_pass),
        "layout_score": float(sl.layout_score) if sl.l2_pass else 0.0,
        "grasp_score_norm": float(sl.grasp_score_norm) if sl.l2_pass else 0.0,
        "manip_score_norm": float(sl.manip_score_norm) if sl.l2_pass else 0.0,
        "dist_score_norm": float(sl.dist_score_norm) if sl.l2_pass else 0.0,
        "rot_score_norm": float(
            getattr(sl, "rot_score_norm", 0.0)) if sl.l2_pass else 0.0,
        "spatial_score_norm": float(
            getattr(sl, "spatial_score_norm", 0.0)) if sl.l2_pass else 0.0,
        "native_yuanchair_score": (
            float(score_diag["native_yuanchair_score"])
            if score_diag is not None else 0.0
        ),
        "native_traj_manip_score_norm": (
            float(score_diag["native_traj_manip_score_norm"])
            if score_diag is not None else 0.0
        ),
        "fail_reason": str(sl.fail_reason),
        "fail_part": str(getattr(sl, "fail_part_id", "") or "") or None,
        "fail_detail": fail_detail,
        "eval_time": float(eval_time),
        "station_mode": "yuanchair_fixed_fixture",
    }


def _evaluate(searcher: yopt.FastLayoutSearcher,
              xy: Dict[str, np.ndarray],
              weights: Dict[str, float]
              ) -> Tuple[yopt._ScoredLayout, bool, Dict[str, object] | None]:
    sl = yopt._ScoredLayout(xy=xy)
    searcher.funnel["sampled"] += 1
    if not searcher.l1(sl):
        return sl, False, None
    searcher.funnel["l1_pass"] += 1
    if not searcher.l2(sl):
        return sl, True, None
    searcher.funnel["l2_pass"] += 1
    return sl, True, _apply_tower_score(searcher, sl, weights)


def collect(args: argparse.Namespace) -> None:
    task = yopt.YUANCHAIR_FAST_TASK
    for path in [task.asmdef_path, task.config_yaml_path, *task.grasp_pickles.values()]:
        if path and not os.path.isfile(path):
            raise FileNotFoundError(f"YuanChair resource missing: {path}")

    searcher = yopt.FastLayoutSearcher(
        task, enable_l3=False, ik_retry_n=int(args.ik_retry_n))
    weights = _normalized_weights(args)
    signature = _signature(searcher, args, weights)
    geometry = _geometry_cache(searcher)
    parents = _parent_map(searcher)
    topdown = _topdown_counts(searcher)
    table_x_range, table_y_range, table_top_z = _table_geometry(searcher)

    out_path = os.path.abspath(args.dataset_out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    resume_counts: Dict[int, int] = {}
    existing_feasible = 0
    existing_total = 0
    if args.gen_resume:
        resume_counts, existing_feasible, existing_total = (
            durable._scan_resume_state(out_path, signature))
    elif os.path.exists(out_path):
        raise FileExistsError(
            f"{out_path} 已存在。为防止覆盖，请换文件名或使用 --gen-resume。")

    print("\n========== YuanChair Dual-Arm Dataset Generation ==========")
    print(f"dataset_out    = {out_path}")
    print(f"samples/seed   = {args.gen_samples}")
    print(f"seeds          = {args.gen_seeds}")
    print(f"resume         = {args.gen_resume} {resume_counts or ''}")
    print(f"threads        = {args.gen_threads}")
    print(f"score_standard = {SCORE_STANDARD}")
    print(f"weights        = {weights}")
    print(f"signature      = {signature}")
    print("feasibility    = YuanChair FastLayoutSearcher L1+L2 (dual arm)")
    print(f"durability     = flush each sample, fsync every {args.gen_fsync_every}")

    open_mode = "a" if args.gen_resume else "w"
    errors_path = out_path + ".errors.jsonl"
    error_open_mode = "a" if args.gen_resume else "w"
    session_written = 0
    session_feasible = 0
    n_errors = 0
    started = time.time()

    with open(out_path, open_mode, encoding="utf-8") as fout, \
            open(errors_path, error_open_mode, encoding="utf-8") as ferr:
        for seed in args.gen_seeds:
            sample_index = int(resume_counts.get(seed, 0))
            if sample_index >= args.gen_samples:
                print(f"[yuanchair] seed={seed} complete "
                      f"({sample_index}/{args.gen_samples}), skip")
                continue
            print(f"[yuanchair] seed={seed} start at sample_index={sample_index}")
            while sample_index < args.gen_samples:
                attempt = 0
                while True:
                    attempt += 1
                    xy = _sample_xy(
                        searcher, seed, sample_index, attempt,
                        bool(args.include_seed_anchor))
                    eval_started = time.time()
                    try:
                        sl, l1_pass, score_diag = _evaluate(
                            searcher, xy, weights)
                        break
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        n_errors += 1
                        error_record = {
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "seed": int(seed),
                            "sample_index": int(sample_index),
                            "attempt": int(attempt),
                            "generation_signature": signature,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                        durable._durable_write(
                            ferr, error_record, n_errors, 1)
                        print(f"[yuanchair] WARN eval error "
                              f"{n_errors}/{args.gen_max_errors}: "
                              f"{type(exc).__name__}: {exc}")
                        try:
                            yopt._reset_robot_for_l3(searcher.robot)
                        except Exception:
                            pass
                        if n_errors >= args.gen_max_errors:
                            raise RuntimeError(
                                f"评估异常达到 {n_errors}; 详情见 {errors_path}"
                            ) from exc

                rec = _record(
                    searcher,
                    sl,
                    seed=seed,
                    sample_index=sample_index,
                    signature=signature,
                    geometry=geometry,
                    parents=parents,
                    topdown=topdown,
                    table_x_range=table_x_range,
                    table_y_range=table_y_range,
                    table_top_z=table_top_z,
                    l1_pass=l1_pass,
                    score_diag=score_diag,
                    eval_time=time.time() - eval_started,
                )
                session_written += 1
                durable._durable_write(
                    fout, rec, session_written, int(args.gen_fsync_every))
                session_feasible += int(bool(sl.l2_pass))
                sample_index += 1

                if session_written % args.progress_every == 0:
                    total_done = sum(resume_counts.values()) + session_written
                    total_feasible = existing_feasible + session_feasible
                    score = float(sl.layout_score) if sl.l2_pass else 0.0
                    print(
                        f"[yuanchair] new={session_written} total={total_done} "
                        f"feasible={total_feasible}/{total_done} "
                        f"({100.0 * total_feasible / max(total_done, 1):.1f}%) "
                        f"last_score={score:.4f} "
                        f"last_eval={rec['eval_time']:.1f}s"
                    )
                if args.sleep_between_samples > 0:
                    time.sleep(float(args.sleep_between_samples))

    total_relevant = sum(resume_counts.values()) + session_written
    total_feasible = existing_feasible + session_feasible
    print("\n========== YuanChair Dataset Summary ==========")
    print(f"new samples      = {session_written}")
    print(f"existing records = {existing_total if args.gen_resume else 0}")
    print(f"relevant total   = {total_relevant}")
    print(f"feasible samples = {total_feasible} "
          f"({100.0 * total_feasible / max(total_relevant, 1):.1f}%)")
    print(f"eval errors      = {n_errors}")
    print(f"wall time        = {time.time() - started:.1f}s")
    print(f"[OK] dataset -> {out_path}")


def main() -> None:
    args = _parse_args()
    try:
        collect(args)
    except KeyboardInterrupt:
        print("\n[yuanchair] 用户中断；已完成样本均已安全落盘。"
              "使用相同命令并保留 --gen-resume 即可续跑。")


if __name__ == "__main__":
    main()
