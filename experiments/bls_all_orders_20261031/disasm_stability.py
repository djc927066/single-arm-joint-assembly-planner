"""Disassembly of the triangular frame by stability only (no IK catalogue, no layout) (2026-11-01).

Start from the assembled frame. A part p can be removed from the present set R if
  stable     the remaining set R - {p} (on the fixed base and table) is in static equilibrium:
             asmcore SequenceEvaluator.equilibrium = LP over contact forces at the contact-polygon
             vertices, Coulomb friction pyramids (mu 0.5, 16 sides), every part's net wrench zero
             under gravity, action = reaction between parts, no auxiliary support
  free path  p moved straight along its removal axis (+z) for 15 cm hits neither the remaining
             parts nor the base (scene.insertable, mesh sweep)
and for the record the stability margin S of R - {p}: smallest sampled disturbance (300 wrench
directions, each the largest load factor the contacts can still balance) = Chen T-ASE S.
Every reachable set is expanded (all removal sequences = reversed assembly orders).

    python -X utf8 -m experiments.bls_all_orders_20261031.disasm_stability
"""
from __future__ import annotations

import sys
from collections import deque

from curobo_bridge.runtime import prepare

prepare()  # torch before trimesh/Panda3D on Windows

import numpy as np  # noqa: E402

from .run import log  # noqa: E402


def main():
    import scenes.triangular_frame6.build_model as bm
    from asmcore.model import AssemblyState
    from asmcore.mechanics.stability_sweep import DirectionalStabilityAnalyzer, StabilitySweepConfig
    from asmcore.planning.sequence import SequenceConfig, SequenceEvaluator, _balanced
    from lazy_cost.scene import load_scene

    sc = load_scene("triangular_frame6")
    asm, _, _ = bm.build_assembly()
    seq = SequenceEvaluator(asm, SequenceConfig(max_auxiliary_resources=0))
    tf = {p.part_id: p.assembled_tf for p in asm.parts}
    fixed = {p.part_id: p.assembled_tf for p in asm.parts if p.fixed}
    S_cache = {}

    def margin(parts):
        key = frozenset(parts)
        if not key:
            return True, float("nan")          # only the fixed base is left
        if key not in S_cache:
            state = AssemblyState(dict(fixed, **{q: tf[q] for q in key}))
            nominal = seq.equilibrium(state, ())
            if _balanced(nominal):
                sw = DirectionalStabilityAnalyzer(asm, state, seq.graph(state), config=StabilitySweepConfig()).analyze()
                S_cache[key] = (True, sw.sampled_minimum_load_factor)
            else:
                S_cache[key] = (False, 0.0)
        return S_cache[key]

    short = lambda ps: "+".join(sorted(p.split("_")[1] for p in ps)) or "(只剩底座)"
    full = frozenset(sc.ids)
    st, S0 = margin(full)
    log(f"装好的框架: 稳定 {st}, S = {S0:.4f}")
    seen, queue, n_seq = {full: 1}, deque([full]), {}
    levels = {}
    while queue:
        R = queue.popleft()
        for p in sorted(R):
            rest = R - {p}
            stable, s = margin(rest)
            free = sc.insertable(p, rest)
            levels.setdefault(len(full) - len(R) + 1, []).append((R, p, stable, free, s))
            if stable and free:
                if rest not in seen:
                    seen[rest] = 0
                    queue.append(rest)
                seen[rest] += seen[R]
    for k in sorted(levels):
        rows = levels[k]
        log(f"\n== 第 {k} 步拆卸（= 倒数第 {k} 个装）：{len({r[0] for r in rows})} 种当前状态")
        for R, p, stable, free, s in rows:
            if k <= 2 or (stable and free):
                ok = "可以拆" if stable and free else ("剩下的会倒" if not stable else "拔出路径被挡")
                log(f"  现有 {short(R):14s} 拆 {p:8s} -> {ok:6s}  剩余稳定 {str(stable):5s}  竖直拔出 {str(free):5s}"
                    f"  剩余 S = {s:.4f}")
        bad = [r for r in rows if not (r[2] and r[3])]
        if k > 2 and bad:
            log(f"  （另有 {len(bad)} 个候选不能拆：剩下会倒 {sum(not r[2] for r in bad)}，路径被挡 {sum(r[2] and not r[3] for r in bad)}）")
    log(f"\n完整拆卸序列数 = {seen[frozenset()]}（= 可行装配顺序数）；场景里记录的装配顺序数 = {len(sc.orders)}")


if __name__ == "__main__":
    sys.exit(main())
