"""摆放—抓取—两段运动的联合优化（只借用 cuRobo 的可微球体运动学与球-场景距离）。

一次反向扩展要回答：第 k 件放在桌上哪里、怎么抓，才能让
"前一件撤离位 → 空手 → 预抓取；抬升 → 持件 → 预插入" 整段最快。

为什么不能直接用 cuRobo TrajOpt：终端位姿代价权重 1e6、时间只在外层按 dt 缩放，放开
x/y/偏航后终端几乎不动（实测 45 个种子仅 1 个移动 5 mm）。

为什么不用"关节角 + 罚函数"表示摆放：约束梯度远大于时间梯度，Adam 归一化后沿约束面
几乎不走（实测摆放中位仅移动 3.7 mm）。

本实现直接优化**摆放 (x, y, 偏航) 与抓取 (滚转 theta, 滑移 s)**，外加两段路径的中间点：

* 三个端点位形（空手终点 = 预抓取、持件终点 = 抬升、持件起点 = 预插入）由摆放和抓取
  决定。每步先用阻尼最小二乘 IK（无梯度）把它们跟到当前目标，再用一步牛顿式
  ``q = q* - J^+ e(target(place, grasp))`` 把它们写成摆放/抓取的可微函数（在收敛点上
  数值等于 q*，导数为 J^+ d target）。约束因此是精确满足的，不靠罚函数。
* 目标 = 两段执行时间代理（每段"最慢关节时间"的光滑近似之和）
  + 沿路径（含插值点）的碰撞（cuRobo 机器人球 + 持件球的球-场景距离）
  + 摆放区域（桌面范围、禁放矩形）+ IK 残差（目标不可达时）。

梯度来源说明：cuRobo 融合运动学的反向对**位置与碰撞球**正确（与有限差分一致），对
**工具四元数不正确**（R22 自动求导 -0.65 vs 有限差分 -0.81，R01 符号相反），所以 TCP
位姿用本文件的 :class:`TorchFK`（导出链，与 numpy FK 误差 3e-7）计算。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

VMAX = torch.tensor([4.2, 5.0, 5.0, 4.2, 3.7, 3.7])
LIMITS = torch.tensor([[-2.4, 2.4], [0.0, 3.2], [0.0, 4.0], [-1.6, 1.6], [-1.7, 1.7], [-2.5, 2.5]])


def _rot_y(t):
    c, s = torch.cos(t), torch.sin(t)
    z, o = torch.zeros_like(t), torch.ones_like(t)
    return torch.stack([torch.stack([c, z, s], -1), torch.stack([z, o, z], -1),
                        torch.stack([-s, z, c], -1)], -2)


def _rot_z(t):
    c, s = torch.cos(t), torch.sin(t)
    z, o = torch.zeros_like(t), torch.ones_like(t)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1),
                        torch.stack([z, z, o], -1)], -2)


def _axis_rot(k, q):
    K = torch.stack([torch.stack([0 * k[0], -k[2], k[1]]), torch.stack([k[2], 0 * k[0], -k[0]]),
                     torch.stack([-k[1], k[0], 0 * k[0]])])
    s, c = torch.sin(q)[..., None, None], torch.cos(q)[..., None, None]
    return torch.eye(3, device=q.device) + s * K + (1 - c) * (K @ K)


class TorchFK:
    """Differentiable TCP pose from the exported Panthera chain (chain.json)."""

    def __init__(self, chain: dict, device="cuda"):
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
        self.base, self.tcp = f(chain["base"]), f(chain["tcp"])
        self.origins = [f(o) for o in chain["origins"]]
        self.axes = [f(a) / np.linalg.norm(a) for a in chain["axes"]]

    def __call__(self, q, with_jac=False):
        """q [...,6] -> R [...,3,3], p [...,3] (and the geometric Jacobian [...,6,6] if asked)"""
        R = self.base[:3, :3].expand(*q.shape[:-1], 3, 3)
        p = self.base[:3, 3].expand(*q.shape[:-1], 3)
        ws, ps = [], []
        for i, (o, a) in enumerate(zip(self.origins, self.axes)):
            p = p + (R @ o[:3, 3, None]).squeeze(-1)
            R = R @ o[:3, :3]
            ws.append((R @ a[:, None]).squeeze(-1))
            ps.append(p)
            R = R @ _axis_rot(a, q[..., i])
        p = p + (R @ self.tcp[:3, 3, None]).squeeze(-1)
        if not with_jac:
            return R @ self.tcp[:3, :3], p
        w = torch.stack(ws, -1)                                   # [...,3,6]
        o = torch.stack(ps, -1)
        jv = torch.linalg.cross(w, p[..., None] - o, dim=-2)
        return R @ self.tcp[:3, :3], p, torch.cat([jv, w], -2)


def _pose_err(R, p, Rt, pt):
    """6D error of (R, p) w.r.t. target (Rt, pt): [dp, 0.5*vee(Rt^T R - R^T Rt)]"""
    M = Rt.transpose(-1, -2) @ R
    er = 0.5 * torch.stack([M[..., 2, 1] - M[..., 1, 2], M[..., 0, 2] - M[..., 2, 0],
                            M[..., 1, 0] - M[..., 0, 1]], -1)
    return torch.cat([p - pt, er], -1)


@dataclass
class ChainProblem:
    q_retreat: np.ndarray          # [6]  start of the empty leg (previous part's retreat)
    goal: np.ndarray               # [4, 4] part pose at the assembly site
    G0: np.ndarray                 # [N, 4, 4] certified contact pair with zero roll / slide
    slide_axis: np.ndarray         # [N] part-frame axis for the slide
    slide_max: float
    obj_spheres: np.ndarray        # [M, 4] held-part spheres in the part frame
    obj_half_height: float         # part rests with its centre at this height
    obj_line: np.ndarray           # [P, 3] footprint points (part frame)
    obj_radius: float              # footprint half-width
    forbidden: list                # [(cx, cy, hx, hy)] no-staging rectangles
    table_lo: np.ndarray
    table_hi: np.ndarray
    lift: float
    approach: float
    preins: float


class ChainOptimizer:
    def __init__(self, checker, chain: dict, K: int = 6, interp: int = 3, iters: int = 300,
                 w_col: float = 50.0, w_reg: float = 1e3, w_ik: float = 1e3, lr_place: float = 0.01,
                 lr_path: float = 0.02, ik_steps: int = 3, log_every: int = 50):
        self.chk, self.K, self.m, self.iters = checker, K, interp, iters
        self.fk = TorchFK(chain)
        self.w = dict(col=w_col, reg=w_reg, ik=w_ik)
        self.lr_place, self.lr_path, self.ik_steps, self.log_every = lr_place, lr_path, ik_steps, log_every

        def err_jac(q, Rt, pt):
            R, p, J = self.fk(q, with_jac=True)
            J = torch.cat([J[..., :3, :], Rt.transpose(-1, -2) @ J[..., 3:, :]], -2)
            return _pose_err(R, p, Rt, pt), J
        self._err_jac = err_jac

    # --- helpers -------------------------------------------------------------------------
    def _spheres(self, q):
        st = self.chk.get_kinematics(q.contiguous())
        sph = st.robot_spheres
        if sph.requires_grad:   # cuRobo's CUDA backward needs contiguous incoming grads
            sph.register_hook(lambda g: None if g is None else g.contiguous())
        return sph

    def _dense(self, path):
        a, b = path[:, :-1], path[:, 1:]
        t = torch.linspace(0, 1, self.m + 1, device=path.device)[:-1].view(1, 1, -1, 1)
        dense = (a[:, :, None] + t * (b - a)[:, :, None]).reshape(path.shape[0], -1, 6)
        return torch.cat([dense, path[:, -1:]], 1)

    @staticmethod
    def _time(path, v):
        d = torch.sqrt((path[:, 1:] - path[:, :-1]) ** 2 + 1e-8) / v
        return (torch.logsumexp(40.0 * d, -1) / 40.0).sum(-1)

    def _track(self, q_star, Rt, pt, lim):
        """DLS IK (no grad) toward the current targets, then a differentiable Newton step."""
        eye = torch.eye(6, device=q_star.device)
        with torch.no_grad():
            Rt_d, pt_d = Rt.detach(), pt.detach()
            for _ in range(self.ik_steps):
                e, J = self._err_jac(q_star, Rt_d, pt_d)
                dq = J.transpose(-1, -2) @ torch.linalg.solve(J @ J.transpose(-1, -2) + 1e-4 * eye, e[..., None])
                q_star = torch.maximum(torch.minimum(q_star - dq.squeeze(-1), lim[:, 1]), lim[:, 0])
            _, J = self._err_jac(q_star, Rt_d, pt_d)
            Jp = J.transpose(-1, -2) @ torch.linalg.inv(J @ J.transpose(-1, -2) + 1e-4 * eye)
            R0, p0 = self.fk(q_star)
        e = _pose_err(R0, p0, Rt, pt)                       # grad flows through the target only
        q = q_star - (Jp @ e[..., None]).squeeze(-1)
        return q, q_star, e

    # --- main -----------------------------------------------------------------------------
    def solve(self, pb: ChainProblem, place0, theta0, slide0, q_pp0, q_lift0, q_pre0):
        dev = "cuda"
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)
        B = len(place0)
        v, lim = VMAX.to(dev), LIMITS.to(dev)
        place = f(place0).clone().requires_grad_(True)      # [B, 3] x, y, yaw
        th = f(theta0).clone().requires_grad_(True)
        sl = f(slide0).clone().requires_grad_(True)
        qs = dict(pp=f(q_pp0), lift=f(q_lift0), pre=f(q_pre0))
        qr = f(pb.q_retreat)[None].repeat(B, 1)
        s = torch.linspace(0, 1, self.K + 1, device=dev)[1:-1].view(1, -1, 1)
        qe = (qr[:, None] + s * (qs["pp"] - qr)[:, None]).clone().requires_grad_(True)
        qc = (qs["pre"][:, None] + s * (qs["lift"] - qs["pre"])[:, None]).clone().requires_grad_(True)
        G0, axis = f(pb.G0), torch.as_tensor(pb.slide_axis, device=dev).long()
        goal, osph, line, rect = f(pb.goal), f(pb.obj_spheres), f(pb.obj_line), f(pb.forbidden)
        lo, hi = f(pb.table_lo), f(pb.table_hi)
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev)
        opt = torch.optim.Adam([{"params": [place, th, sl], "lr": self.lr_place},
                                {"params": [qe, qc], "lr": self.lr_path}])
        log, t0 = [], time.perf_counter()
        for it in range(self.iters):
            # grasp G(theta, s) and the three targets implied by (place, grasp)
            Rg = G0[:, :3, :3] @ _rot_y(th)
            pg = G0[:, :3, 3] + torch.nn.functional.one_hot(axis, 3).float() * sl[:, None]
            Ro = _rot_z(place[:, 2])
            po = torch.cat([place[:, :2], torch.full((B, 1), pb.obj_half_height, device=dev)], -1)
            R_pick = Ro @ Rg
            p_pick = po + (Ro @ pg[..., None]).squeeze(-1)
            tgt = dict(pp=(R_pick, p_pick - pb.approach * R_pick[..., 2]),
                       lift=(R_pick, p_pick + pb.lift * ez),
                       pre=(goal[:3, :3] @ Rg, goal[:3, 3] + (goal[:3, :3] @ pg[..., None]).squeeze(-1)
                            + pb.preins * ez))
            q_end, ik_res = {}, 0.0
            for k in ("pp", "lift", "pre"):
                q_end[k], qs[k], e = self._track(qs[k], *tgt[k], lim)
                ik_res = ik_res + (e ** 2).sum(-1)
            pe = self._dense(torch.cat([qr[:, None], qe, q_end["pp"][:, None]], 1))
            pc = self._dense(torch.cat([q_end["pre"][:, None], qc, q_end["lift"][:, None]], 1))
            # held part along the carry path: T_obj = T_tcp @ G^-1
            Rc, pcp = self.fk(pc)
            Rgi = Rg.transpose(-1, -2)
            pgi = -(Rgi @ pg[..., None]).squeeze(-1)
            Rh = Rc @ Rgi[:, None]
            ph = (Rc @ pgi[:, None, :, None]).squeeze(-1) + pcp
            osp = (Rh[:, :, None] @ osph[None, None, :, :3, None]).squeeze(-1) + ph[:, :, None]
            held = torch.cat([osp, osph[None, None, :, 3:].expand(B, pc.shape[1], -1, 1)], -1)
            col = (self.chk.get_collision_distance(self._spheres(pe)).sum((1, 2))
                   + self.chk.get_collision_distance(torch.cat([self._spheres(pc), held], 2)).sum((1, 2)))
            # staging region: footprint inside the table, outside the no-staging rectangles
            foot = (Ro[:, None] @ line[None, :, :, None]).squeeze(-1)[..., :2] + po[:, None, :2]
            r = pb.obj_radius + 0.01
            reg = (torch.relu(lo + r - foot) + torch.relu(foot - hi + r)).pow(2).sum((1, 2))
            dx = (rect[None, None, :, 2] + r) - (foot[:, :, None, 0] - rect[None, None, :, 0]).abs()
            dy = (rect[None, None, :, 3] + r) - (foot[:, :, None, 1] - rect[None, None, :, 1]).abs()
            reg = reg + torch.minimum(torch.relu(dx), torch.relu(dy)).pow(2).sum((1, 2))
            T = self._time(pe, v) + self._time(pc, v)
            loss = T + self.w["col"] * col + self.w["reg"] * reg + self.w["ik"] * ik_res
            opt.zero_grad()
            loss.sum().backward()
            opt.step()
            with torch.no_grad():
                qe.copy_(torch.maximum(torch.minimum(qe, lim[:, 1]), lim[:, 0]))
                qc.copy_(torch.maximum(torch.minimum(qc, lim[:, 1]), lim[:, 0]))
                sl.clamp_(-pb.slide_max, pb.slide_max)
            if it % self.log_every == 0 or it == self.iters - 1:
                log.append(dict(it=it, T=float(T.detach().mean()), col=float(col.mean()), reg=float(reg.mean()),
                                ik=float(ik_res.mean()),
                                moved=float((place[:, :2] - f(place0)[:, :2]).norm(dim=-1).median())))
        torch.cuda.synchronize()
        with torch.no_grad():
            G = torch.zeros(B, 4, 4, device=dev)
            G[:, :3, :3], G[:, :3, 3], G[:, 3, 3] = Rg, pg, 1.0
            out = dict(place=place.detach().cpu().numpy(), theta=th.detach().cpu().numpy(),
                       slide=sl.detach().cpu().numpy(), G=G.cpu().numpy(),
                       T=T.detach().cpu().numpy(), col=col.detach().cpu().numpy(),
                       reg=reg.detach().cpu().numpy(), ik=ik_res.detach().cpu().numpy(),
                       q_pp=qs["pp"].cpu().numpy(), q_lift=qs["lift"].cpu().numpy(),
                       q_pre=qs["pre"].cpu().numpy(), seconds=time.perf_counter() - t0, log=log)
        return out
