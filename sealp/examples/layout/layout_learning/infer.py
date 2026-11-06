"""统一推理接口。

把训练好的 checkpoint 包装成 ``LayoutModelRunner``, 提供两条推理路径:

    - scorer 模型 : ``score_layouts(candidates)`` 对一批候选 layout 打分/排序,
                    供搜索器"采样 -> NN 预筛 -> evaluate_layout"。
    - generator   : ``propose_layouts(cond, k)`` 直接生成 k 个候选 layout xy,
                    供搜索器"NN 生成 -> evaluate_layout"。

关键: 本模块只输出**候选 xy / 排序**, 绝不判定最终可行性。最终仍由调用方的
evaluate_layout + motion-level validation 决定。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from . import features as F
from .dataset import sample_to_item, collate_items, move_batch
from .generator_dataset import rot_names_per_part
from .models import build_model, is_generator


class LayoutModelRunner:
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model_name = ckpt["model_name"]
        self.max_parts = int(ckpt.get("max_parts", F.MAX_PARTS_DEFAULT))
        self.feature_version = str(ckpt.get("feature_version", "v1"))
        self.is_generator = bool(ckpt.get("is_generator", is_generator(self.model_name)))
        self.model = build_model(self.model_name, flat_dim=ckpt["flat_dim"],
                                 **ckpt.get("model_kwargs", {})).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    # ---------- 单样本 -> batch ----------
    def _batch_from_sample(self, sample: Dict) -> Dict[str, torch.Tensor]:
        item = sample_to_item(sample, self.max_parts, self.feature_version)
        return move_batch(collate_items([item]), self.device)

    # ---------- scorer 路径 ----------
    @torch.no_grad()
    def score_layouts(
        self, samples: List[Dict], batch_size: int = 256
    ) -> Dict[str, np.ndarray]:
        """对含 ``staging_xy`` 的候选分批打分。

        分批推理避免 2k--10k 候选池一次性构图造成显存峰值。候选顺序严格
        保持不变，因此共享池的排序比较仍然可复现。
        """
        if not samples:
            return {
                "feas_prob": np.zeros(0, dtype=np.float32),
                "score": np.zeros(0, dtype=np.float32),
            }
        batch_size = max(1, int(batch_size))
        probs: List[np.ndarray] = []
        scores: List[np.ndarray] = []
        for start in range(0, len(samples), batch_size):
            chunk = samples[start:start + batch_size]
            items = [
                sample_to_item(s, self.max_parts, self.feature_version)
                for s in chunk
            ]
            batch = move_batch(collate_items(items), self.device)
            out = self.model(batch)
            probs.append(torch.sigmoid(out["feas_logit"]).cpu().numpy())
            scores.append(out["score_pred"].cpu().numpy())
        return {
            "feas_prob": np.concatenate(probs, axis=0),
            "score": np.concatenate(scores, axis=0),
        }

    # ---------- generator 路径 ----------
    @torch.no_grad()
    def propose_layouts(self, cond: Dict, k: int,
                        part_ids: Optional[List[str]] = None,
                        station_xy: Optional[np.ndarray] = None
                        ) -> List[Dict[str, np.ndarray]]:
        """给定条件 sample, 生成 k 个候选 layout。

        模型输出的是"相对装配站的偏移", 绝对 staging = 装配站 + 偏移:
            - station_xy 显式给定时直接使用 (scorer 采样站位 / CVAE / Diffusion);
            - 否则若模型能回归站位 (SAGPN) 则用 predict_station;
            - 再否则退回 cond 里的 assembly_station_pos。

        返回 [{pid: np.array([x,y]) (table 绝对坐标), ...}, ...]
        """
        batch = self._batch_from_sample(cond)
        off_norm = self.model.propose(batch, k).cpu().numpy()   # [k, N, 2] 偏移(归一化)
        bounds = F._table_bounds(cond)

        if station_xy is None:
            station_xy = self.predict_station(cond)
        if station_xy is None:
            station_xy = np.asarray(cond.get("assembly_station_pos", [0.0, 0.0, 0.0]),
                                    dtype=np.float32)[:2]
        station_xy = np.asarray(station_xy, dtype=np.float32)[:2]

        parts = F._ordered_parts(cond)
        pids = part_ids or [p["part_id"] for p in parts]
        out: List[Dict[str, np.ndarray]] = []
        for kk in range(off_norm.shape[0]):
            layout: Dict[str, np.ndarray] = {}
            for i, pid in enumerate(pids):
                if i >= off_norm.shape[1]:
                    break
                layout[pid] = station_xy + F.denormalize_offset(off_norm[kk, i], bounds)
            out.append(layout)
        return out

    @torch.no_grad()
    def propose_structured_layouts(
        self,
        cond: Dict,
        k: int,
        part_ids: Optional[List[str]] = None,
        seed: Optional[int] = None,
        temperature: float = 1.0,
    ) -> List[Dict]:
        """Structured generator output with station / pose / rotation / xy."""
        batch = self._batch_from_sample(cond)
        parts = F._ordered_parts(cond)
        pids = part_ids or [p["part_id"] for p in parts]
        bounds = F._table_bounds(cond)
        rot_name_maps = rot_names_per_part(cond)

        if hasattr(self.model, "propose_structured"):
            structured = self.model.propose_structured(
                batch, k=k, seed=seed, temperature=temperature)
        else:
            layouts = self.propose_layouts(cond, k=k, part_ids=pids)
            structured = []
            for layout_xy in layouts:
                st = self.predict_station(cond)
                if st is None:
                    st = np.asarray(cond.get("assembly_station_pos", [0, 0, 0]), float)[:2]
                st_norm = F.normalize_xy(st, bounds)
                parts_out = []
                for i, pid in enumerate(pids):
                    off = layout_xy.get(pid, np.zeros(2)) - st
                    parts_out.append({
                        "part_index": i,
                        "pose_index": 0,
                        "rotation_index": 0,
                        "offset_xy_norm": F.normalize_offset(off, bounds).tolist(),
                        "confidence": 1.0,
                    })
                structured.append({
                    "station_xy_norm": st_norm,
                    "parts": parts_out,
                    "proposal_logprob": 0.0,
                })

        from .projection import proposal_from_structured
        out: List[Dict] = []
        for prop in structured:
            base = proposal_from_structured(
                {"station_xy_norm": prop["station_xy_norm"],
                 "parts": prop["parts"],
                 "proposal_logprob": prop.get("proposal_logprob", 0.0)},
                cond, pids)
            pose_choice = {}
            rotation_choice = {}
            for i, pid in enumerate(pids):
                if i < len(prop["parts"]):
                    pi = int(prop["parts"][i].get("pose_index", 0))
                    ri = int(prop["parts"][i].get("rotation_index", 0))
                    pose_choice[pid] = pi
                    names = rot_name_maps[i] if i < len(rot_name_maps) else ["unknown"]
                    rotation_choice[pid] = names[ri] if ri < len(names) else str(ri)
            out.append({
                "assembly_station_xy": base["assembly_station_xy"],
                "xy": base["xy"],
                "pose_choice": pose_choice,
                "rotation_choice": rotation_choice,
                "proposal_logprob": base["proposal_logprob"],
            })
        return out

    @torch.no_grad()
    def predict_station(self, cond: Dict) -> Optional[np.ndarray]:
        """SAGPN: 回归连续装配站 xy (桌面绝对坐标 [x, y]); 其它模型返回 None。"""
        if not hasattr(self.model, "predict_station"):
            return None
        batch = self._batch_from_sample(cond)
        st_norm = self.model.predict_station(batch).cpu().numpy()[0]
        return F.denormalize_xy(st_norm, F._table_bounds(cond))
