"""One backward expansion: decide part p (the LAST one installed among the undecided set R),
its initial pose x and its assembly grasp g, for every (x, g) at once.

Forward view of p's step: installed = R \\ {p} (at their goals), on the table = the decided
suffix (at their chosen poses) + p at x. Everything is known except where the arm comes from
(the part installed before p is still undecided).

Screening (IK from screen.py, collision here). The gripper is closed to the grasp's jaw width
while it holds p (if screen.CLOSE_FINGERS) and fully open otherwise. A failing check prunes the
(x, g) pair (or the grasp g for every x):
  per grasp, world A = installed R\\{p} + parts on the table (p in the hand)
    ins(g) arm + fingers; preins(g) holding p; preins -> ins arm + fingers (the part's own straight
    insertion is the mesh sweep of scene.step_ok; its sphere cover would touch the seat);
    ins -> retreat open, except the first SKIP, against everything but p
  per grasp, world B = installed R (p in its seat) + parts on the table, gripper open (released)
    retreat(g)
  per (x, g), world A
    pick(x, g) arm + fingers closed; hover(x, g) open, p at x as an obstacle (the hover is above the
    top of p, screen.py); descent hover -> pick open, except the last SKIP, without p; lift(x, g)
    holding p; pick -> lift holding p except the first SKIP (p still on the table)
  The open fingers straddle p on the descent and right after the release; the antipodal grasp
  itself is collision-free with p (WRS grasp planner), and the sphere cover of the open fingers is
  fatter than the fingers (it hit p on most posts), so p is left out there.
The free motions are cost only (lower bound of the straight joint line; blocked -> + penalty):
  transfer   lift(x,g) -> preins(g)            holding p, world installed R\\{p} + table suffix
  connection retreat(g) -> hover(head)          empty hand, world installed R + table suffix minus
             (or -> HOME at the root)          head, head part as extra obstacle. This RESOLVES the
                                               head's transit, unknown when the head was decided.
  fixed      hover->prepick->pick->lift + preins->ins->retreat (straight-line bounds)
  first part HOME -> hover(x, g)               (start known)
  pending    p's own transit, start unknown -> lower bound min over possible predecessors q and
             their grasps of T(retreat(q) -> hover(x, g)); resolved at the next expansion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lazy_cost.scene import key_xy

from .problem import HOME, t_acc

SKIP = 0.25     # share of a short motion next to a contact (part on the table / in its seat) not checked


@dataclass
class Child:
    p: str
    x: str
    g: int
    add_lb: float          # newly resolved segments: lower bound
    add_est: float         # newly resolved segments: estimate (bound + blocked-ness correction)
    pend_lb: float         # p's own transit (start unknown): lower bound
    segs: list = field(default_factory=list)   # records of the resolved segments


def _preds_retreats(pb, R1, p):
    """Retreat configurations of every part that can be installed right before p, all grasps."""
    cat = pb.cat
    P = [v["retreat"] for q in pb.preds(R1, p) for v in cat["goal"][q].values()] or \
        [v["retreat"] for q in R1 for v in cat["goal"][q].values()]
    return np.stack(P)


def _bound_order(pb, node, p, R1, cand, first):
    """{x: grasps of x sorted by the lower bound of fixed + transfer + connection + p's own transit}."""
    cat, goal = pb.cat, pb.cat["goal"][p]
    tgt = HOME if node.head is None else cat["pick"][node.head[0]][node.head[1]][node.head[2]]["hover"]
    E = lambda key: np.stack([cat["pick"][p][x][g][key] for x, g in cand])
    LF, HV = E("lift"), E("hover")
    PRE = np.stack([goal[g]["preins"] for _, g in cand])
    RET = np.stack([goal[g]["retreat"] for _, g in cand])
    b = t_acc(LF, PRE) + t_acc(RET, tgt) + np.array([pb.fix_pick[p, x, g] + pb.fix_goal[p, g] for x, g in cand])
    if first:
        b = b + t_acc(np.broadcast_to(HOME, HV.shape), HV)
    else:
        P = _preds_retreats(pb, R1, p)
        c = max(1, 2_000_000 // len(P))                     # chunk: pairs x retreats x 6 joints in memory
        b = b + np.concatenate([t_acc(P[None], HV[i:i + c, None]).min(1) for i in range(0, len(HV), c)])
    out = {}
    for (x, g), v in sorted(zip(cand, b), key=lambda t: t[1]):
        out.setdefault(x, []).append(g)
    return out


def expand(pb, node, p, max_per_x=None, trace=None):
    """All feasible children of ``node`` that decide part p; list[Child] (unsorted) and stats."""
    sc, cat, chk = pb.scene, pb.cat, pb.chk
    R1 = node.R - {p}
    stats = dict(pairs=0, pruned=0)
    if not sc.step_ok(R1, p):
        return [], dict(stats, reason="unstable_or_blocked_insertion")
    polys = [sc.polygons[q][k] for q, k in node.table.items()]
    xs = [x for x, _ in sc.candidates[p] if x in cat["pick"][p]
          and all(sc.polygons[p][x].distance(o) >= pb.gap for o in polys)]
    goal, part = cat["goal"][p], pb.held[p]
    jaws = (lambda gs: np.array([pb.jaw[p][g] for g in gs])) if pb.close_fingers else (lambda gs: None)
    first = len(R1) == 0
    rows_all = [(x, g) for x in xs for g in cat["pick"][p][x]]
    stats["pairs"] = len(rows_all)
    gl_all = sorted({g for _, g in rows_all})
    if not gl_all:
        return [], dict(stats, reason="no_pose_left")

    hb_all = [(part, sc.grasps()[p][g][0]) for g in gl_all]
    # ---- 1. per grasp: insertion side ---------------------------------------------------------------
    chk.set_world(chk.world_dict(R1, pb.placed(node.table)))
    INS, PRE, RET = (np.stack([goal[g][k] for g in gl_all]) for k in ("ins", "preins", "retreat"))
    Jg, Hg = jaws(gl_all), np.arange(len(gl_all))
    ok_a = chk.batch_free([(INS, None, -1, -1, Jg), (PRE, None, Hg, -1, Jg), (PRE, INS, -1, -1, Jg),
                           (INS + SKIP * (RET - INS), RET, -1, -1)], held_bank=hb_all, step=0.02)
    chk.set_world(chk.world_dict(node.R, pb.placed(node.table)))
    ok_b = chk.batch_free([(RET, None, -1, -1)], step=0.02)
    gok = {g for g, k in zip(gl_all, np.logical_and.reduce(ok_a + ok_b)) if k}
    stats["fail_goal"] = {name: int((~o).sum()) for name, o in zip(
        ("ins", "preins_held", "insert_arm", "release_line", "retreat"), ok_a + ok_b)}
    stats["grasps"], stats["grasps_ok"] = len(gl_all), len(gok)
    # ---- 2. per (x, g): pick side, only grasps that passed 1 ----------------------------------------
    # Only the k grasps per pose with the smallest bound of the newly resolved segments + p's own
    # transit are screened (the search keeps at most ``per_x`` grasps per pose anyway); a pose whose
    # k grasps all fail gets the next k, so no pose that has a feasible grasp is lost.
    # A position (x, y) may carry several orientations (Problem.orient). "first" (SEALP): the position
    # uses its first orientation in SEALP order that has a feasible grasp in THIS state, the later ones
    # are dropped; an orientation that overlaps a part on the table or has no grasp left counts as
    # failed. Each round screens, for every undecided position, only its current orientation (all
    # positions in one GPU batch); a position whose orientation fails moves on to the next one in the
    # next round. The grasp order (bound) is computed only for the orientations that get screened.
    # "best": every orientation is screened and kept, and the cost ranks them.
    chk.set_world(chk.world_dict(R1, pb.placed(node.table)))
    cand = [(x, g) for x, g in rows_all if g in gok]
    rows0, ok_pick, keep_x = [], np.zeros(0, bool), None
    if cand:
        gix, xix = {g: i for i, g in enumerate(gl_all)}, {x: i for i, x in enumerate(xs)}
        gx, prio = {}, {}
        for x, g in cand:
            gx.setdefault(x, []).append(g)

        def need(xs_):          # grasp order by bound, computed when an orientation is first screened
            new = [x for x in xs_ if x not in prio]
            if new:
                prio.update(_bound_order(pb, node, p, R1, [(x, g) for x in new for g in gx[x]], first))
        k = pb.grasps_per_pose or max(len(v) for v in gx.values())
        ex_all = [chk.obstacle_spheres(part, sc.pose[p][x]) for x in xs]
        at = {}                                             # position -> its orientations, SEALP order
        for x in xs:
            if x in gx:
                at.setdefault(key_xy(x), []).append(x)
        state, pos, oks = dict.fromkeys(gx, "open"), dict.fromkeys(gx, 0), []

        def chosen(xy):         # "first": the orientation used at xy; None = undecided, False = none
            for x in at[xy]:
                if state[x] != "dead":
                    return x if state[x] == "ok" else None
            return False

        def open_poses():
            if pb.orient == "first":           # the current orientation of each undecided position
                return [next(x for x in v if state[x] == "open") for xy, v in at.items() if chosen(xy) is None]
            return [x for x in gx if state[x] == "open"]
        todo = open_poses()
        while todo:
            need(todo)
            batch = [(x, g) for x in todo for g in prio[x][pos[x]:pos[x] + k]]
            E = lambda key: np.stack([cat["pick"][p][x][g][key] for x, g in batch])
            HV, PK, LF = E("hover"), E("pick"), E("lift")
            Hr, Xr, Jr = np.array([gix[g] for _, g in batch]), np.array([xix[x] for x, _ in batch]), jaws([g for _, g in batch])
            oks5 = chk.batch_free(
                [(PK, None, -1, -1, Jr), (HV, None, -1, Xr), (HV, PK + SKIP * (HV - PK), -1, -1),
                 (LF, None, Hr, -1, Jr), (PK + SKIP * (LF - PK), LF, Hr, -1, Jr)],
                held_bank=hb_all, extra_bank=ex_all, step=0.02)
            fp = stats.setdefault("fail_pick", dict.fromkeys(("pick", "hover", "descent", "lift_held", "lift_line"), 0))
            for name, o in zip(fp, oks5):
                fp[name] += int((~o).sum())
            ok = np.logical_and.reduce(oks5)
            rows0 += batch
            oks.append(ok)
            alive = {x for (x, _), o in zip(batch, ok) if o}
            for x in todo:
                pos[x] += k
                state[x] = "ok" if x in alive else "open" if pos[x] < len(gx[x]) else "dead"
            todo = open_poses()
        ok_pick = np.concatenate(oks)
        if pb.orient == "first":
            keep_x = {chosen(xy) for xy in at} - {None, False}
            lead = {}
            for x in cat["pick"][p]:                        # SEALP-first orientation of each position
                lead.setdefault(key_xy(x), x)
            stats["positions"] = len(keep_x)
            stats["orient_fallback"] = sum(x != lead[key_xy(x)] for x in keep_x)
    rows = [(x, g, cat["pick"][p][x][g]) for (x, g), k in zip(rows0, ok_pick)
            if k and (keep_x is None or x in keep_x)]
    stats["pruned"] = len(rows0) - int(ok_pick.sum()) + (len(rows_all) - len(cand))
    stats["screened"] = len(rows0)
    if trace is not None:
        trace.update(rows_all=len(rows_all), poses=len(xs), grasps_all=len(gl_all), grasps_goal_ok=len(gok),
                     pairs_goal_ok=len(cand), pairs_screened=len(rows0), rows=len(rows))
    if not rows:
        return [], dict(stats, reason="no_grasp")
    gl = sorted({g for _, g, _ in rows})
    gi = {g: i for i, g in enumerate(gl)}
    xi = {x: i for i, x in enumerate(xs)}
    n = len(rows)
    col = lambda k: np.stack([e[k] for _, _, e in rows])
    X = np.array([xi[x] for x, _, _ in rows])
    Gx = np.array([gi[g] for _, g, _ in rows])
    PRE = np.stack([goal[g]["preins"] for g in gl])
    RET = np.stack([goal[g]["retreat"] for g in gl])

    # ---- cost: transfer line (holding p) and, for the first part, HOME -> hover --------------------
    groups = [(col("lift"), PRE[Gx], Gx, -1, jaws([g for _, g, _ in rows]))]
    ex_p = []
    if first:
        ex_p = [chk.obstacle_spheres(part, sc.pose[p][x]) for x in xs]
        groups.append((np.broadcast_to(HOME, (n, 6)), col("hover"), -1, X))
    eA, fA = chk.batch_free(groups, held_bank=[(part, sc.grasps()[p][g][0]) for g in gl], extra_bank=ex_p,
                            step=0.02, features=True)
    line_tf, ft_tf = eA[0], fA[0]

    # ---- cost: connection retreat(g) -> head hover / HOME, world after releasing p ------------------
    if node.head is None:
        chk.set_world(chk.world_dict(node.R, pb.placed(node.table)))
        tgt, ex_h, ehx = HOME, [], -1
    else:
        h, hx, hg = node.head
        chk.set_world(chk.world_dict(node.R, pb.placed(node.table, skip=(h,))))
        tgt = cat["pick"][h][hx][hg]["hover"]
        ex_h, ehx = [chk.obstacle_spheres(pb.held[h], sc.pose[h][hx])], 0
    (line_conn,), (ft_conn,) = chk.batch_free([(RET, np.broadcast_to(tgt, RET.shape), -1, ehx)], extra_bank=ex_h,
                                              step=0.02, features=True)
    conn_lb = t_acc(RET, tgt)
    conn_est = np.array([pb.penalty.estimate(conn_lb[j], not line_conn[j], {k: v[j] for k, v in ft_conn.items()})
                         for j in range(len(gl))])

    # ---- pending transit of p (start unknown): lower bound over possible predecessors ------------------
    if first:
        tr_lb = t_acc(np.broadcast_to(HOME, (n, 6)), col("hover"))
        tr_est = np.array([pb.penalty.estimate(tr_lb[i], not eA[1][i], {k: v[i] for k, v in fA[1].items()}) for i in range(n)])
        pend = np.zeros(n)
    else:
        P, HV = _preds_retreats(pb, R1, p), col("hover")
        c = max(1, 2_000_000 // len(P))
        pend = np.concatenate([t_acc(P[None], HV[i:i + c, None]).min(1) for i in range(0, n, c)])

    tf_lb = t_acc(col("lift"), PRE[Gx])
    out = []
    for i, (x, g, e) in enumerate(rows):
        j = gi[g]
        fixed = pb.fix_pick[p, x, g] + pb.fix_goal[p, g]
        f_tf = {k: float(v[i]) for k, v in ft_tf.items()}
        tf_est = pb.penalty.estimate(float(tf_lb[i]), not line_tf[i], f_tf)
        segs = [dict(kind="transfer", part=p, lb=float(tf_lb[i]), est=tf_est, blocked=bool(not line_tf[i]),
                     depth=f_tf["depth"]),
                dict(kind="fixed", part=p, lb=fixed, est=fixed, blocked=False, depth=0.0),
                dict(kind="return_home" if node.head is None else "transit", part=None if node.head is None else node.head[0],
                     after=p, lb=float(conn_lb[j]), est=float(conn_est[j]), blocked=bool(not line_conn[j]),
                     depth=float(ft_conn["depth"][j]))]
        add_lb = float(tf_lb[i]) + fixed + float(conn_lb[j])
        add_est = tf_est + fixed + float(conn_est[j])
        if first:
            segs.append(dict(kind="transit", part=p, after="HOME", lb=float(tr_lb[i]), est=float(tr_est[i]),
                             blocked=bool(not eA[1][i]), depth=float(fA[1]["depth"][i])))
            add_lb += float(tr_lb[i])
            add_est += float(tr_est[i])
        out.append(Child(p, x, int(g), add_lb, add_est, float(pend[i]), segs))
    return out, stats
