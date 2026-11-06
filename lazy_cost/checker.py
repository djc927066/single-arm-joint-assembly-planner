"""Batched GPU collision checks (cuRobo V2 spheres vs scene) for configurations and joint lines.

Robot: MorphIt spheres of ``panthera_wrsext`` (links 2-5, palm, fingers) + ``buffer``.
World: table cuboid + assembly base + any parts (Warp mesh BVH) + boxes (cuboids).
Held part: AABB sphere cover placed through the TCP pose of the same FK pass.
Extra obstacles that differ per query (the part being picked, at its candidate pose) are
sphere sets checked sphere-to-sphere, so the world is not reloaded per candidate.
"""
from __future__ import annotations

import numpy as np


def pose7(h) -> list:
    """4x4 -> [x, y, z, qw, qx, qy, qz] (cuRobo convention)."""
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(np.asarray(h)[:3, :3]).as_quat()
    return [float(h[0, 3]), float(h[1, 3]), float(h[2, 3]), float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def _quat_wxyz_to_mat(q):
    import torch
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1
    ).reshape(q.shape[:-1] + (3, 3))


class GpuChecker:
    # density 2.0 (67 spheres, tighter than 1.0) and a 1 mm buffer: against the WRS mesh check on 960
    # search segments 95% agreement, 0 missed collisions, ~6% of our 'blocked' are free for WRS
    # (density 1.0 / 2 mm: 93%, ~9%); joint_search/diag_collision.py
    # self_relief 3 mm on link2 / link4 (self-collision only): the folded HOME is free again (WRS: free;
    # without it the density-2 spheres of link2 / link4 overlap 4.5 mm there); vs WRS mesh self-collision
    # on 2500 configurations: 95% agreement (81% without), 8 of 129 real self-collisions missed.
    def __init__(self, scene, *, buffer=0.001, density=2.0, chunk=32768, n_meshes=16, n_cuboids=16, self_relief=(('link2', 0.003), ('link4', 0.003))):
        import torch
        from curobo.scene import Scene
        from curobo_bridge.curobo_models import load_robot_dict, make_collision_checker, with_buffer
        from curobo_bridge.runtime import ASSET_DIR

        self.scene, self.buffer, self.chunk = scene, float(buffer), int(chunk)
        self.density = float(density)
        self.robot = with_buffer(load_robot_dict(ASSET_DIR / "panthera_wrsext", density), buffer)
        # self_relief {link: m}: shrink that link's spheres for SELF-collision only (not against the
        # scene). The folded HOME has link2/link4 meshes ~10 mm apart, but their spheres overlap.
        self.self_relief = dict(self_relief or {})
        pad = self.robot["kinematics"].setdefault("self_collision_buffer", {})
        for link, r in self.self_relief.items():
            pad[link] = float(pad.get(link, 0.0)) - float(r)
        self._mesh_lists = {}
        self.chk = make_collision_checker(self.robot, Scene.create(self.world_dict()),
                                          n_meshes=n_meshes, n_cuboids=n_cuboids)
        self.device = torch.device("cuda")
        # robot spheres of the ARM links (link2-link5). The held part touches the gripper (link6,
        # palm, fingers), so, as in cuRobo's attached-object setup (curobo_models.with_attach_link),
        # only the arm links are checked against it.
        kc = self.chk.kinematics.kinematics_config
        arm = [kc.link_name_to_idx_map[n] for n in ("link2", "link3", "link4", "link5")]
        self.arm_spheres = torch.nonzero(torch.isin(kc.link_sphere_idx_map,
                                                    torch.tensor(arm, device=kc.link_sphere_idx_map.device))).flatten().to(self.device)
        # finger spheres: the model's gripper is fully open (JAW_OPEN); a query may carry its own jaw
        # width, then both fingers are moved towards each other along the TCP y axis (closing axis).
        self.finger = {s: torch.nonzero(kc.link_sphere_idx_map == kc.link_name_to_idx_map[n]).flatten().to(self.device)
                       for s, n in ((1.0, "finger_left"), (-1.0, "finger_right"))}

    JAW_OPEN = 0.085

    def _close_fingers(self, sph, R, jaw):
        """sph [b,N,4] (modified in place), R [b,3,3] TCP rotations, jaw [b] widths (nan = open)."""
        import torch
        rows = torch.nonzero(~torch.isnan(jaw)).flatten()
        if not len(rows):
            return
        delta = ((self.JAW_OPEN - jaw[rows]).clamp(min=0) / 2)[:, None, None]
        y = R[rows, :, 1][:, None, :]
        for side, idx in self.finger.items():
            sph[rows[:, None], idx[None, :], :3] -= side * delta * y

    # ---- world ------------------------------------------------------------------------------
    def mesh_entry(self, pid, pose):
        if pid not in self._mesh_lists:
            m = self.scene.meshes[pid]
            self._mesh_lists[pid] = (np.asarray(m.vertices).tolist(), np.asarray(m.faces).reshape(-1).tolist())
        v, f = self._mesh_lists[pid]
        return dict(vertices=v, faces=f, pose=pose7(pose))

    def world_dict(self, installed=(), placed=(), boxes=()) -> dict:
        """installed: part ids at their goals; placed: (name, pid, 4x4) on the table;
        boxes: (name, dims[3], 4x4). Table and assembly base are always present."""
        T = np.eye(4)
        T[:3, 3] = self.scene.table_center
        d = {"cuboid": {"table": dict(dims=self.scene.table_size.tolist(), pose=pose7(T))},
             "mesh": {"base": self.mesh_entry("base", self.scene.goal["base"])}}
        for p in installed:
            d["mesh"][f"{p}@goal"] = self.mesh_entry(p, self.scene.goal[p])
        for name, pid, pose in placed:
            d["mesh"][name] = self.mesh_entry(pid, pose)
        for name, dims, pose in boxes:
            d["cuboid"][name] = dict(dims=[float(x) for x in dims], pose=pose7(pose))
        return d

    def set_world(self, d: dict) -> None:
        from curobo.scene import Scene
        self.chk.update_world(Scene.create(d))

    # ---- primitives -------------------------------------------------------------------------
    def kinematics(self, q):
        """q [B,6] torch -> robot spheres [B,N,4], TCP rotation [B,3,3], TCP position [B,3]."""
        st = self.chk.get_kinematics(q[:, None, :])
        tp = st.tool_poses
        return st.robot_spheres[:, 0], _quat_wxyz_to_mat(tp.quaternion[:, 0, 0]), tp.position[:, 0, 0]

    def held_world(self, R, p, spheres_obj, G):
        """Held-part spheres in world (object pose = T_tcp @ inv(G)), radii + buffer."""
        import torch
        Gi = np.linalg.inv(G)
        c = torch.as_tensor(spheres_obj[:, :3] @ Gi[:3, :3].T + Gi[:3, 3], dtype=torch.float32, device=self.device)
        r = torch.as_tensor(spheres_obj[:, 3] + self.buffer, dtype=torch.float32, device=self.device)
        centers = torch.einsum("bij,mj->bmi", R, c) + p[:, None, :]
        return torch.cat([centers, r.expand(centers.shape[0], -1)[..., None]], -1)

    def obstacle_spheres(self, spheres_obj, pose):
        """Sphere cover of a part at a world pose (radii + buffer), as a cuda tensor."""
        import torch
        c = spheres_obj[:, :3] @ pose[:3, :3].T + pose[:3, 3]
        return torch.as_tensor(np.c_[c, spheres_obj[:, 3] + self.buffer], dtype=torch.float32, device=self.device)

    def _hits(self, q, held, extras, owner):
        """Collision flag per state. owner[i]: index into ``extras`` or -1."""
        import torch
        out = np.zeros(len(q), bool)
        qt = torch.as_tensor(q, dtype=torch.float32, device=self.device)
        for s in range(0, len(q), self.chunk):
            sph, R, p = self.kinematics(qt[s:s + self.chunk])
            allsph = sph if held is None else torch.cat([sph, self.held_world(R, p, *held)], 1)
            b = allsph.shape[0]
            self.chk.setup_batch_tensors(b, 1)
            h = (self.chk.get_collision_constraint(allsph[:, None].contiguous())[:, 0] > 0).any(-1)
            h |= self.chk.get_self_collision(sph[:, None].contiguous()).reshape(b, -1).sum(-1) > 0
            if extras:
                o = torch.as_tensor(owner[s:s + b], device=self.device)
                for k, ex in enumerate(extras):
                    m = o == k
                    if m.any():
                        d = torch.cdist(allsph[m, :, :3], ex[:, :3]) - allsph[m, :, 3:4] - ex[None, :, 3]
                        h[m] |= (d < 0).flatten(1).any(-1)
            out[s:s + b] = h.cpu().numpy()
        return out

    # ---- queries ----------------------------------------------------------------------------
    def configs_free(self, q, held=None, extras=None, owner=None) -> np.ndarray:
        q = np.atleast_2d(np.asarray(q, float))
        owner = np.full(len(q), -1) if owner is None else np.asarray(owner)
        return ~self._hits(q, held, extras or [], owner)

    def lines_free(self, polys, held=None, extras=None, owner=None, step=0.02) -> np.ndarray:
        """polys: list of [K,6] joint polylines (K>=2), interpolated every ``step`` rad and
        checked in one batch. Returns a free flag per polyline."""
        states, idx = [], []
        for i, P in enumerate(polys):
            P = np.asarray(P, float)
            for a, b in zip(P[:-1], P[1:]):
                n = max(1, int(np.ceil(np.max(np.abs(b - a)) / step)))
                states.append(a + np.linspace(0, 1, n + 1)[:, None] * (b - a))
                idx.append(np.full(n + 1, i))
        if not states:
            return np.zeros(0, bool)
        S, I = np.concatenate(states), np.concatenate(idx)
        own = np.full(len(polys), -1) if owner is None else np.asarray(owner)
        hit = self._hits(S, held, extras or [], own[I])
        bad = np.zeros(len(polys), bool)
        np.logical_or.at(bad, I, hit)
        return ~bad

    # ---- one-call batch over heterogeneous queries ------------------------------------------------
    def batch_free(self, groups, held_bank=(), extra_bank=(), step=0.02, extra_chunk_elems=2.5e7, features=False):
        """Every query of a ranking step in ONE batched pass (chunked only for GPU memory).

        groups: list of (A [n,6], B [n,6] or None, held_idx [n], extra_idx [n]):
            B None  -> check the configurations A;
            else    -> check the joint lines A->B, interpolated every ``step`` rad.
        held_idx indexes ``held_bank`` = [(spheres_obj, G)] (-1: empty hand); extra_idx indexes
        ``extra_bank`` = [world sphere tensors] (-1: none). Different held parts / grasps and
        different extra obstacles are padded into one tensor and gathered per state, so the
        number of GPU passes no longer grows with the number of grasps or candidates.
        Returns a list of bool arrays (free per query), one per group. With features=True
        also returns, per group, a dict of per-query "how blocked" measures read from the SAME
        collision pass (no extra GPU pass): see _blocked_features."""
        import torch
        dev = self.device
        S, owner, H, E, J, sizes, T = [], [], [], [], [], [], []
        base = 0
        for grp in groups:
            # optional 5th element: jaw width per query (gripper closed to it; nan / absent = open)
            A, B, hi, ei = grp[:4]
            A = np.asarray(A, float).reshape(-1, 6)
            n = len(A)
            hi, ei = np.broadcast_to(np.asarray(hi), n), np.broadcast_to(np.asarray(ei), n)
            jw = np.broadcast_to(np.asarray(grp[4] if len(grp) > 4 and grp[4] is not None else np.nan, float), n)
            if B is None:
                S.append(A)
                owner.append(base + np.arange(n))
                H.append(hi), E.append(ei), J.append(jw)
                T.append(np.zeros(n))
            else:
                B = np.asarray(B, float).reshape(-1, 6)
                k = np.maximum(1, np.ceil(np.abs(B - A).max(1) / step).astype(int)) + 1   # points per line
                rep = np.repeat(np.arange(n), k)
                t = (np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)) / np.repeat(k - 1, k)
                S.append(A[rep] + t[:, None] * (B - A)[rep])
                owner.append(base + rep)
                H.append(hi[rep]), E.append(ei[rep]), J.append(jw[rep])
                T.append(t)
            sizes.append(n)
            base += n
        if base == 0:
            empty = [np.zeros(0, bool) for _ in groups]
            return (empty, [{} for _ in groups]) if features else empty
        S, owner, H, E, J, T = (np.concatenate(S), np.concatenate(owner), np.concatenate(H), np.concatenate(E),
                                np.concatenate(J), np.concatenate(T))
        Jt = torch.as_tensor(J, dtype=torch.float32, device=dev)
        far = 1e3
        HB = None
        if len(held_bank):                                    # held spheres in the TCP frame, padded
            loc = []
            for sph, G in held_bank:
                Gi = np.linalg.inv(G)
                loc.append(np.c_[sph[:, :3] @ Gi[:3, :3].T + Gi[:3, 3], sph[:, 3] + self.buffer])
            m = max(len(x) for x in loc)
            pad = lambda x: np.r_[x, np.tile([far, far, far, 0.0], (m - len(x), 1))]
            HB = torch.as_tensor(np.stack([pad(x) for x in loc] + [pad(np.zeros((0, 4)))]), dtype=torch.float32, device=dev)
        EB = None
        if len(extra_bank):
            m = max(len(x) for x in extra_bank)
            padt = torch.tensor([far, far, far, 0.0], device=dev)
            EB = torch.stack([torch.cat([x, padt.expand(m - len(x), 4)]) for x in extra_bank] + [padt.expand(m, 4)])
        hit = np.zeros(len(S), bool)
        depth = np.zeros((len(S), 3), np.float32)   # penetration of arm / held part / vs the picked part
        zdeep = np.zeros(len(S), np.float32)          # height of the deepest colliding sphere
        selfh = np.zeros(len(S), bool)
        qt = torch.as_tensor(S, dtype=torch.float32, device=dev)
        Ht = torch.as_tensor(np.where(H < 0, len(held_bank), H), device=dev)
        Et = torch.as_tensor(np.where(E < 0, len(extra_bank), E), device=dev)
        for s in range(0, len(S), self.chunk):
            sph, R, p = self.kinematics(qt[s:s + self.chunk])
            b = sph.shape[0]
            sph = sph.clone()
            self._close_fingers(sph, R, Jt[s:s + b])
            allsph = sph
            if HB is not None:
                hb = HB[Ht[s:s + b]]
                allsph = torch.cat([sph, torch.cat([torch.einsum("bij,bmj->bmi", R, hb[..., :3]) + p[:, None], hb[..., 3:]], -1)], 1)
            self.chk.setup_batch_tensors(b, 1)
            cv = self.chk.get_collision_constraint(allsph[:, None].contiguous())[:, 0]   # [b, spheres], >0 = penetration
            nr = sph.shape[1]
            h = (cv > 0).any(-1)
            sh = self.chk.get_self_collision(sph[:, None].contiguous()).reshape(b, -1).sum(-1) > 0
            if HB is not None and len(self.arm_spheres):      # held part vs the arm links
                rows = torch.nonzero(Ht[s:s + b] < len(held_bank)).flatten()
                arm = sph[:, self.arm_spheres]
                cs = max(16, int(extra_chunk_elems / (arm.shape[1] * HB.shape[1])))
                for r0 in range(0, len(rows), cs):
                    r = rows[r0:r0 + cs]
                    d = torch.cdist(allsph[r, nr:, :3], arm[r, :, :3]) - allsph[r, nr:, 3:4] - arm[r, None, :, 3]
                    sh[r] |= (d < 0).flatten(1).any(-1)
            h |= sh
            dep = torch.zeros(b, 3, device=dev)
            if features:
                dep[:, 0] = cv[:, :nr].clamp(min=0).max(-1).values
                if cv.shape[1] > nr:
                    dep[:, 1] = cv[:, nr:].clamp(min=0).max(-1).values
                zd = allsph[torch.arange(b, device=dev), cv.argmax(-1), 2]
            if EB is not None:
                rows = torch.nonzero(Et[s:s + b] < len(extra_bank)).flatten()
                cs = max(16, int(extra_chunk_elems / (allsph.shape[1] * EB.shape[1])))
                for r0 in range(0, len(rows), cs):
                    r = rows[r0:r0 + cs]
                    e = EB[Et[s:s + b][r]]
                    d = torch.cdist(allsph[r, :, :3], e[..., :3]) - allsph[r, :, 3:4] - e[:, None, :, 3]
                    dm = d.flatten(1).min(-1).values
                    h[r] |= dm < 0
                    if features:
                        dep[r, 2] = (-dm).clamp(min=0)
            hit[s:s + b] = h.cpu().numpy()
            if features:
                depth[s:s + b] = dep.cpu().numpy()
                zdeep[s:s + b] = zd.cpu().numpy()
                selfh[s:s + b] = sh.cpu().numpy()
        bad = np.zeros(base, bool)
        np.logical_or.at(bad, owner, hit)
        out, k = [], 0
        for n in sizes:
            out.append(~bad[k:k + n])
            k += n
        if not features:
            return out
        F = _blocked_features(owner, T, hit, depth, zdeep, selfh, base)
        feats, k = [], 0
        for n in sizes:
            feats.append({key: v[k:k + n] for key, v in F.items()})
            k += n
        return out, feats


def _blocked_features(owner, t, hit, depth, zdeep, selfh, n):
    """Per query (configuration or joint line) "how blocked" measures, aggregated over its
    interpolated states (owner[i] = query of state i, t[i] = position along the line in [0, 1]).

    frac        share of states in collision        s_in, s_out  first / last colliding t
    depth       max penetration (m) of any sphere   depth_mean   mean over states of the per-state max
    by_arm, by_held, by_part   max penetration of arm spheres / held-part spheres / spheres vs
                the part being picked (m);  by_self  share of self-colliding states
    z_low       lowest height (m) of the deepest colliding sphere over the colliding states
    """
    cnt = np.maximum(np.bincount(owner, minlength=n).astype(float), 1)
    dmax = depth.max(1)
    out = dict(frac=np.bincount(owner, hit.astype(float), minlength=n) / cnt,
               depth_mean=np.bincount(owner, dmax, minlength=n) / cnt,
               by_self=np.bincount(owner, selfh.astype(float), minlength=n) / cnt)
    big = 1e9
    s_in, s_out = np.full(n, big), np.full(n, -big)
    np.minimum.at(s_in, owner[hit], t[hit])
    np.maximum.at(s_out, owner[hit], t[hit])
    out["s_in"] = np.where(s_in < big, s_in, np.nan)
    out["s_out"] = np.where(s_out > -big, s_out, np.nan)
    for name, col in (("depth", dmax), ("by_arm", depth[:, 0]), ("by_held", depth[:, 1]), ("by_part", depth[:, 2])):
        m = np.zeros(n)
        np.maximum.at(m, owner, col)
        out[name] = m
    zl = np.full(n, big)
    np.minimum.at(zl, owner[hit], zdeep[hit])
    out["z_low"] = np.where(zl < big, zl, np.nan)
    return out
