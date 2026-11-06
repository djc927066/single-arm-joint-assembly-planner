# Detour indicators for blocked segments — experiments (EXPERIMENTER)

Design: `lazy_cost/_design/detour_features.md` (thinker). Data: `lazy_cost/_output/exp_{rack10,triangular_frame6}_feat3_parts_*.jsonl`
(parts only, `--boxes 0,0 --crowd 0 --gap 0.03 --contexts 30 --cap 12`, seed 11 → the SAME 30 contexts as feat2;
lazy/LB/x1.25 summaries reproduce feat2 exactly). Evaluation: `python -m lazy_cost._design.eval_detour --tag _feat3_parts`
(full numbers in `lazy_cost/_design/eval_feat3_parts.json`).

## 1. What was implemented (no existing API changed)

* `lazy_cost/detour.py` — `DetourProbe(chk).run(segs, held_bank, extra_bank)`; one call per ranking step over all blocked
  segments (endpoints qa, qb, colliding stretch s_in..s_out, held part / picked part as in the screening).
  * **(A) escape**: 5 states on the colliding stretch are pushed out of collision by a projected Newton iteration on the total
    penetration P(q) (cuRobo collision constraint of robot + held spheres, picked-part overlap, held-vs-arm, self-collision).
    dP/dq = Σ J_sᵀ g_s is obtained directly by forward finite differences in joint space (7 states per point in one batched
    FK + collision pass per iteration, ≤10 iterations); this covers the held/picked-part terms that are outside cuRobo, so I did
    not use `get_gradient_buffer` + per-sphere Jacobians (same quantity, more plumbing). The step is orthogonal to the line;
    two metrics: plain and slack-weighted (W⁻¹ = displacement slack of each joint w.r.t. LB, design §3A). Indicators: freed share,
    max |dq|, decoupled bound T_dec incl. reversals and TOPP of the corner-blended escape path.
  * **(B) upper bound**: 19 smooth candidate curves per blocked segment, all checked in ONE `batch_free` call:
    raise_h (Bezier via mid-stretch with TCP +h, h∈{5,10,15,25} cm, TCP-position Jacobian pseudo-inverse), retract_h (towards the base),
    lift_h (blended polyline through the raised stretch), escB/escP (Bezier / blended polyline through escape-displaced states, ×1/1.5/2.5).
    Every free curve gets `lazy_cost.timing.topp` → UB (a real collision-free path under our sphere model).
* `lazy_cost/exp_detour.py` — exp_rack10 + the probe (timed separately, cuda-synchronised) + per blocked segment: `qa`, `qb`, `desc`,
  `detour{...}`; plus the **de-biased reference** `ref_sc` for every segment: min(TOPP(raw cuRobo path), TOPP(greedy shortcut of that
  path with `batch_free`), TOPP(corner-blended shortcut) if free). ~4 % of (cand, grasp) pairs are re-planned (`ref2`) to measure noise.
* `lazy_cost/_design/eval_detour.py` — segment errors, ranking metrics (`exp_rack10.metrics`), Wilcoxon vs the shipped model, and a
  simulation of the lazy certification (`LazyStepCost.rank` logic; refining a pair reveals its reference time) ordered by each
  estimator: segments refined to certify top-3, flips of the estimated best, refinements until the estimated best is within 2 % of
  the true best ("1st ok"), wasted refinements (not on the final top-3), anytime regret after 0/2/4 refined segments.
  All parameters fitted on the OTHER assembly.

## 2. Reference / noise (the thinker's point, confirmed)

| | rack10 | frame6 |
|---|---|---|
| free segments, raw cuRobo / LB (p10, p50, p90) | 1.07, 1.19, 1.46 | 1.08, 1.20, 1.45 |
| free segments, de-biased / LB | 1.00, 1.00, 1.00 | 1.00, 1.00, 1.00 |
| blocked, raw cuRobo / LB | 1.12, 1.25, 1.49 | 1.09, 1.22, 1.45 |
| **blocked, de-biased / LB** | **1.00, 1.02, 1.17** | **1.00, 1.06, 1.21** |
| blocked, min(de-biased, UB) / LB | 1.00, 1.01, 1.13 | 1.00, 1.04, 1.15 |
| cuRobo re-plan of the same segment differs | 0 / 68 | 0 / 54 |

* cuRobo MotionPlanner is **deterministic** here (identical times on re-plan), so the 0.11-0.14 scatter is not random noise but
  systematic path quality: ~20 % "tax" on any segment handed to the planner, free or blocked.
* After shortcut + blending (the free straight line is found again → exactly LB) the blocked excess is median **+2 % / +6 %**, p90 +17 / +21 %.
  **Parts-only detours are genuinely small.** The +24 % constant (and x1.25) is planner overhead, not detour cost.
* Our cheap curves beat cuRobo: UB found for **100 % / 99.1 %** of blocked segments; UB < raw cuRobo in 95 % / 86 %, UB < de-biased
  cuRobo in 45 % / 49 %; median UB/LB 1.07 / 1.08.

## 3. Results (test assembly; fitted on the other one)

Methods for blocked segments (free → LB): `lb`; `x1.25` (baseline only); `shipped` = penalty_model.json; `nnls` = same 7 features
refitted; `A` = (1+τ)·TOPP(escape path); `B` = (1+τ)·max(LB, UB) (A if no UB) = design E1; `Blite` = B with 4 curves only
(lift 5/10 cm, escBw ×1/×1.5, greedy cover chosen on rack10); `E2` = (1+τ)(LB + λ(UB−LB)); `E2aff` = LB·(c0 + c1(UB/LB−1)); `gbt` =
depth-2 boosted trees on all indicators. τ = median planner overhead on FREE segments (0.20/0.19 raw; 0 de-biased).
seg = mean |log(pred/true)| on blocked segments; ρ = Spearman of pred/LB vs true/LB; p = Wilcoxon on per-context regret vs shipped.

### 3.1 Truth R0 = raw cuRobo time on blocked segments (the spec's reference)

| test | method | seg | ρ | top1≤2% | top3 | mean regret | max regret | p | refined | flips | 1st ok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rack10 | lb | 0.241 | – | 0.83 | 0.89 | 1.66 % | 18.3 % | 0.11 | 5.30 | 0.90 | 0.40 |
| | x1.25 | 0.089 | −0.11 | 0.93 | 0.92 | 0.47 % | 9.6 % | – | 5.30 | 0.10 | 0.07 |
| | shipped | 0.089 | 0.03 | 0.93 | 0.92 | 0.47 % | 9.6 % | – | 5.30 | 0.10 | 0.07 |
| | A | 0.161 | 0.16 | 0.87 | 0.84 | 0.78 % | 9.6 % | 0.18 | 5.30 | 0.20 | 0.20 |
| | B | 0.101 | 0.13 | 0.87 | 0.94 | 0.97 % | 9.6 % | 0.18 | 5.33 | 0.17 | 0.17 |
| | E2 | 0.089 | 0.13 | 0.97 | 0.94 | **0.15 %** | **3.5 %** | 0.32 | 5.30 | 0.07 | 0.03 |
| | gbt | 0.082 | 0.28 | 0.90 | 0.92 | 0.57 % | 9.6 % | 0.65 | 5.30 | 0.10 | 0.10 |
| frame6 | lb | 0.217 | – | 0.70 | 0.78 | 3.69 % | 21.8 % | 0.03 | 11.03 | 2.57 | 1.90 |
| | x1.25 | 0.093 | 0.02 | 0.87 | 0.91 | 1.26 % | 18.1 % | – | 11.03 | 0.33 | 0.47 |
| | shipped | 0.094 | 0.14 | 0.87 | 0.90 | 1.26 % | 18.1 % | – | 11.03 | 0.23 | 0.33 |
| | A | 0.153 | 0.18 | 0.90 | 0.84 | 2.15 % | 33.3 % | 0.61 | 11.13 | 0.13 | 0.57 |
| | B | 0.109 | 0.10 | 0.90 | 0.94 | 0.88 % | 18.1 % | 0.18 | 11.03 | 0.23 | 0.53 |
| | E2 | 0.093 | 0.10 | 0.87 | 0.96 | 1.26 % | 18.1 % | 1.00 | 11.03 | 0.27 | 0.53 |
| | gbt | 0.078 | 0.46 | 0.87 | 0.94 | 1.31 % | 18.1 % | 0.32 | 11.03 | 0.23 | 0.43 |

Against the raw reference every estimator sits at the same noise floor (seg error 0.08-0.10 ≈ scatter of the planner tax);
ρ ≤ 0.28 except trees. No new indicator is significantly better than the shipped model (all p ≥ 0.18); E2's rack10 gain
(0.47 → 0.15 %, max 9.6 → 3.5 %) does not transfer to frame6. The frame6 18 % max regret (ctx 12) is common to all methods:
the true best candidate has a blocked segment that cuRobo solves *better* than our UB (UB not tight there).

### 3.2 Truth R1 = de-biased cuRobo (no probe input → no leakage)

| test | method | seg | ρ | top1≤2% | top3 | mean regret | max regret | p | refined | flips | 1st ok | wasted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rack10 | lb | 0.054 | – | 0.97 | 0.96 | 0.34 % | 8.6 % | 0.05 | 3.73 | 0.27 | 0.07 | 0.40 |
| | shipped | 0.176 | 0.70 | 0.70 | 0.88 | 1.60 % | 14.5 % | – | 3.87 | 0.33 | 0.30 | 0.43 |
| | nnls (refit, b≈0) | 0.039 | 0.70 | 1.00 | 0.99 | 0.06 % | 1.7 % | 0.01 | 3.73 | 0.10 | 0.00 | 0.40 |
| | A | 0.141 | 0.55 | 0.70 | 0.88 | 1.75 % | 14.5 % | 0.47 | 4.00 | 0.33 | 0.43 | 0.50 |
| | B | 0.054 | 0.42 | 1.00 | 0.97 | **0.04 %** | **1.2 %** | 0.01 | 4.03 | 0.03 | 0.00 | 0.53 |
| | Blite | 0.064 | 0.42 | 1.00 | 0.96 | 0.04 % | 1.2 % | 0.01 | 4.07 | 0.03 | 0.00 | 0.57 |
| | E2 | 0.043 | 0.42 | 1.00 | 0.97 | 0.04 % | 1.2 % | 0.01 | 3.93 | 0.03 | 0.00 | 0.47 |
| | gbt | 0.033 | 0.78 | 1.00 | 0.99 | 0.06 % | 1.7 % | 0.01 | 3.80 | 0.17 | 0.00 | 0.40 |
| frame6 | lb | 0.077 | – | 0.80 | 0.86 | 1.83 % | 11.6 % | 0.58 | 8.30 | 1.10 | 1.43 | 3.50 |
| | shipped | 0.159 | 0.71 | 0.73 | 0.86 | 1.62 % | 11.6 % | – | 8.30 | 0.30 | 0.97 | 3.50 |
| | nnls (refit) | 0.049 | 0.63 | 0.90 | 0.90 | **0.44 %** | **6.9 %** | 0.07 | 8.30 | 0.30 | 0.30 | 3.50 |
| | A | 0.109 | 0.41 | 0.73 | 0.86 | 1.81 % | 23.8 % | 0.75 | 8.47 | 0.40 | 1.03 | 3.60 |
| | B | 0.057 | 0.51 | 0.83 | 0.88 | 1.03 % | 11.6 % | 0.17 | 8.30 | 0.40 | 0.60 | 3.50 |
| | Blite | 0.070 | 0.44 | 0.87 | 0.84 | 0.63 % | 11.6 % | 0.04 | 8.43 | 0.30 | 0.50 | 3.63 |
| | E2aff | 0.049 | 0.51 | 0.93 | 0.87 | 0.50 % | 11.6 % | 0.03 | 8.30 | 0.27 | 0.43 | 3.50 |
| | gbt | 0.043 | 0.69 | 0.83 | 0.90 | 1.14 % | 11.0 % | 0.51 | 8.30 | 0.37 | 0.43 | 3.50 |

* Once the planner tax is removed, the shipped model and x1.25 are **worse than LB alone** (rack10 1.60 % vs 0.34 %): they add 24 % to
  segments that really cost 2-6 %.
* The best methods are (i) the old features refitted on the de-biased truth (NNLS → intercept ≈ 0, weight on depth / depth_mean;
  depth's ρ rises from 0.03-0.14 to 0.63-0.70 once the tax is gone), and (ii) UB-based B/E2. On rack10 B/E2 are marginally best
  (0.04 % / 1.2 % vs 0.06 % / 1.7 %), on frame6 refitted NNLS is best (0.44 % / 6.9 % vs E2aff 0.50 % / 11.6 %, B 1.03 %).
  The UB adds no consistent gain over a de-biased depth model.
* Escape-based A is poor everywhere (bias +0.09…+0.15): the escape path is jagged; its decoupled time T_dec equals LB in > 50 % (p90
  1.00/1.05) — the slowest joint is almost never the one that has to move, the time is lost to curvature coupling, which T_dec ignores.
* (R2 = min(R1, UB), leaks UB: E2/gbt/nnls all ~0-0.3 % mean regret; same conclusions, see json.)

### 3.3 Lazy procedure ("wrong turns")

* **Segments refined to certify the top-3 do not depend on the estimator** (R0: 5.30 / 11.03 for every method incl. LB order;
  R1: 3.7-4.1 / 8.3-8.5). Certification is driven by the lower bounds and τ, so ordering by a better estimate cannot remove
  refinements; B/A orders even cost +0.1-0.3 segments on average.
* What the estimator changes is the anytime behaviour: flips of the estimated best and refinements until it is right.
  LB order: 0.90/2.57 flips, 0.40/1.90 refinements to first-correct (rack10/frame6, R0); any blocked-penalty (x1.25, shipped, E2, B)
  cuts this to 0.1-0.3 flips and 0.03-0.5 refinements — but the constant already achieves it; the new indicators do not improve on it
  (R0) and only match a refitted depth model (R1).

## 4. Cost

| per context (mean) | rack10 | frame6 |
|---|---|---|
| bounds-only ranking (screen + LB) | 57 ms | 101 ms |
| blocked segments probed | 16.4 | 28.3 |
| escape (A), GPU | 67 ms (4 ms/seg) | 199 ms (7 ms/seg) |
| curve check (B, 19 curves/seg), GPU | 334 ms (20 ms/seg) | 1894 ms (67 ms/seg) |
| T_dec of all curves, CPU | 42 ms | 139 ms |
| TOPP of free curves (UB), CPU | 1886 ms (115 ms/seg) | 4346 ms (153 ms/seg) |
| one cuRobo refinement (light TrajOpt) | 89 ms/segment, 5.2 per ctx | 107 ms/segment, 11.2 per ctx |

The probe (2.6 s / 7.2 s per context) costs **5-6× more than the lazy refinements it could save** (0.46 s / 1.2 s), and the refinement
count does not drop anyway (3.3). Blite (4 curves) would still be ~70/400 ms GPU + ~25-30 ms TOPP per segment, i.e. about one cuRobo
refinement per blocked segment. TOPP-free proxies do not work: T_dec of the curves is ≈ LB (median 1.00).

## 4b. Robustness: dense scenes (`--boxes 6,10 --crowd 0.08`, 20 contexts, 18 usable; `eval_feat3_dense.json`)

Detours are real here: blocked de-biased / LB median **1.18 / 1.19**, p90 1.41 / 1.41 (rack10 / frame6); UB found for 93 % / 80 %;
free segments are rare (128 / 42). Mean / max regret (test assembly, fitted on the other):

| truth | test | lb | shipped | nnls refit | A | B | E2 | gbt |
|---|---|---|---|---|---|---|---|---|
| R0 | rack10 | 5.77 / 41.5 | 0.91 / 7.5 | 0.91 / 7.5 | **0.20 / 2.9** | 0.32 / 4.8 | 0.91 / 7.5 | 0.91 / 7.5 |
| R0 | frame6 | 2.43 / 20.7 | 1.05 / 11.9 | **0.52 / 4.7** | 3.65 / 17.0 | 2.06 / 14.4 | 1.61 / 14.4 | 1.44 / 11.9 |
| R1 | rack10 | 4.35 / 29.7 | 1.55 / 11.4 | 1.55 / 11.4 | **0.98 / 10.8** | 2.11 / 12.0 | 1.56 / 12.0 | 1.55 / 11.4 |
| R1 | frame6 | 1.65 / 11.7 | 0.87 / 6.1 | 1.27 / 15.0 | 1.89 / 14.2 | 0.84 / **3.7** | 0.84 / **3.7** | **0.53** / 3.8 |

(%). Certification refinements again identical across orders (11.1 / 28.7 R0; 8.7 / 25.5 R1). With 15 contexts with metrics per
assembly no difference is significant (all Wilcoxon p ≥ 0.11), and the winner changes from cell to cell. Segment-level, trees on all
indicators have the lowest error everywhere (0.05-0.09) and the UB raises ρ to 0.6-0.76 under R1, so the indicators DO carry signal
when detours are large — but it does not translate into a reliable ranking gain at this sample size.

## 5. Verdict

1. With the current (raw cuRobo) reference, richer physical indicators do **not** rank better than the shipped model or a constant:
   all are at the planner-noise floor; no paired test is significant; the lazy certification needs the same number of refinements.
2. The real finding is the reference: ~20 % of every planned segment is cuRobo path tax (deterministic, also on free lines); after
   shortcut + blending, parts-only detours cost +2 % / +6 % (median). The 0.243 constant should not be presented as a detour penalty.
3. Parts only, against the de-biased truth, a penalty that **scales with penetration depth and has no constant** (NNLS refit: intercept 0) is as good
   as the UB methods and far cheaper (it is computed in the existing screening pass). Escape-based A is not useful. UB-based B is the most
   physically explicit and marginally best on rack10, but 25-70× the screening cost.
3b. Dense scenes (real detours, +18 % median): the indicators carry segment-level signal (UB ρ 0.6-0.76, trees lowest error) but
   with 15 contexts per assembly no ranking gain is significant and the best method changes between cells; A is unreliable.
4. Recommended next steps: (a) de-tax the refiner itself (shortcut + blend each `CuroboRefiner` result — measured 22-56 ms per path with `batch_free` + TOPP, CPU-bound
   greedy loop; could be made cheaper) so lazy refinements report real times; then refit the depth-only penalty on de-biased data and drop the constant;
   (b) the B curves are a strong cheap *refiner* candidate (feasible curve for 99-100 % of blocked segments, better than raw cuRobo in
   86-95 %, ~20-70 ms GPU per segment) — worth testing as a first-try refiner before cuRobo, after validating feasibility under cuRobo's
   own collision model (ours uses the 2 mm-buffer sphere model only); (c) 30 contexts per assembly are too few for max-regret claims.
