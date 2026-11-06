# Detour-time indicators for blocked joint-straight segments (design, THINKER)

Scope: what makes a blocked segment slower than its lower bound LB = max_j t_trap(|b_j-a_j|),
which cheap cuRobo quantities can measure it, which estimators to try, and how to evaluate.
All file:line references are to the vendored cuRobo at
`E:\Projects\wrs-sealp-assemble\dual_sealp\cuRobotest\_vendor\curobo\curobo` (abbrev. `cu/`)
unless they start with `lazy_cost/`.

---------------------------------------------------------------------------------------------
## 0. Finding that changes the question (CPU analysis, `lazy_cost/_design/seg_stats.py`)

`seg_data` stores the cuRobo reference time `ref` for **free (straight) segments too**. On those
segments the straight line is collision-free, so the true optimum is LB, yet cuRobo + TOPP gives:

| run (feat2_parts) | segments | ref/LB p10 | p50 | p75 | p90 | max | mean(ref/LB-1) | std |
|---|---|---|---|---|---|---|---|---|
| rack10, FREE transit   | 637 | 1.029 | 1.197 | 1.334 | 1.449 | 1.57 | | |
| rack10, FREE transfer  | 716 | 1.092 | 1.179 | 1.309 | 1.461 | 1.75 | | |
| rack10, BLOCKED transit | 285 | 1.112 | 1.291 | 1.421 | 1.489 | 1.67 | | |
| rack10, BLOCKED transfer| 207 | 1.123 | 1.208 | 1.322 | 1.490 | 1.76 | | |
| frame6, FREE transit   | 278 | 1.043 | 1.179 | 1.280 | 1.333 | 1.48 | | |
| frame6, FREE transfer  | 438 | 1.106 | 1.220 | 1.374 | 1.497 | 1.74 | | |
| frame6, BLOCKED transit | 505 | 1.090 | 1.227 | 1.357 | 1.430 | 1.64 | | |
| frame6, BLOCKED transfer| 345 | 1.095 | 1.194 | 1.329 | 1.503 | 1.71 | | |
| rack10 all FREE / all BLOCKED | 1353 / 492 | | | | | | 0.229 / 0.281 | 0.143 / 0.141 |
| frame6 all FREE / all BLOCKED | 716 / 850 | | | | | | 0.237 / 0.250 | 0.143 / 0.140 |

Mean absolute deviation around the mean: FREE 0.114-0.116, BLOCKED 0.117 (both assemblies).
Spearman(ratio, depth) on blocked: +0.04 (rack10), +0.14 (frame6); (ratio, frac): +0.01 / +0.10.

Interpretation:
1. The excess of a blocked segment over LB under the current reference is **mostly the cuRobo
   path "tax"** (TrajOpt path is not straight / has curvature that the joint-coupled TOPP of
   `lazy_cost/timing.py` must pay for). The same +18..+22 % (median) appears on segments that need
   NO detour at all. The obstacle-attributable part is only about +1..+5 % on average
   (0.281-0.229, 0.250-0.237).
2. The fitted model's mean |error| 0.117 equals the scatter of the tax on FREE segments
   (0.114-0.116). That is the **noise floor of this reference**: no feature, however physical,
   can do much better against it. This is exactly why "x1.25 if blocked" ranks as well as the
   fitted model, and why `b = 0.243` ~= mean tax 0.23.
3. Consequently the constant is not a detour constant; it is "the executor's inefficiency when a
   segment is handed to cuRobo". It is physically meaningful only in that sense (and it can be
   measured on FREE segments, without fitting on blocked ones).
4. To study geometric detours at all, the reference for blocked segments must first be de-taxed
   (Sec. 3.1). Otherwise any geometric indicator is evaluated against noise of std ~0.14.

---------------------------------------------------------------------------------------------
## 1. What cuRobo V2 can provide cheaply (verified in source)

Our instance: `GpuChecker.chk` = `RobotCollisionChecker` built by
`curobo_bridge/curobo_models.py:144-153` (`collision_activation_distance=activation` default 0.0,
`self_collision_activation_distance=0`, `max_collision_distance=0.25`). Its class is
`RobotSceneCollision` (`cu/_src/collision/collision_robot_scene.py`). Config built in
`cu/_src/collision/collision_robot_scene_cfg.py:146-174`: two `SceneCollisionCost` objects sharing
one `SceneCollision`: `collision_cost` (eta = activation, `use_grad_input=False`) and
`collision_constraint` (eta = 0, `use_grad_input=True`); plus `self_collision_cost`.

| # | Quantity | API / path | In -> out | Extra cost | Works with `chk`? |
|---|---|---|---|---|---|
| Q1 | Per-sphere penetration (what we use) | `chk.get_collision_constraint(spheres)` collision_robot_scene.py:139-171 -> `SceneCollisionCost._discrete_fn` cost_scene_collision.py:160-198 -> `SphereObstacleCollision` geom/collision/wp_autograd.py:37-100 -> kernel wp_collision_kernel.py:70-166 | [b,h,n,4] -> [b,h,n]; value = sum over obstacles of penetration (eta=0: `apply_collision_activation` wp_collision_common.py:11-38 is linear, cost = penetration). Zero when not penetrating. | already paid | yes |
| Q2 | **Per-sphere escape direction** (into-obstacle unit vector, summed over penetrated obstacles) | written by the SAME kernel into the buffer: `accumulate_collision` wp_collision_common.py:84-96; read with `chk.collision_constraint.get_gradient_buffer()` cost_scene_collision.py:200-206 -> [b,h,n,4] (xyz, 0). Convention: `-grad(sdf)`, points INTO the obstacle (mesh: data_mesh.py:700-706; cuboid: data_cuboid.py:600-631). Escape = `-gradient`. | free: buffer is filled by the call we already make; must be read before the next call (zeroed per call, wp_autograd.py:76) | yes |
| Q3 | Signed clearance for non-colliding spheres within eta | set `chk.collision_cost.config.activation_distance.fill_(eta)` (tensor read per call, wp_autograd.py:82) then `chk.get_collision_distance` (collision_robot_scene.py:105-137) or `get_collision_vector` (203-245, returns (cost, gradient)). Cost per sphere = 0.5 p^2/eta if p<=eta else p-eta/2, p = eta - signed_dist (wp_collision_common.py:26-38) -> invertible to clearance in [0, eta) and penetration beyond. Summed over obstacles (approximate when two obstacles are near). | one more collision kernel pass (same cost as Q1) | yes (do not change `collision_constraint`; it is our collision test) |
| Q4 | Swept-sphere distance between consecutive horizon states | `SceneCollision.get_swept_sphere_distance` collision_scene.py:226-262 -> `SweptSphereObstacleCollision` wp_autograd.py:124+, kernel wp_sweep_collision_kernel.py (SWEEP_STEPS=3, Cartesian linear interpolation of each sphere center toward prev/next state, adaptive jumps). Enabled in the cost by `use_sweep=True` + `trajectory_dt` (cost_scene_collision.py:81-86, 124-158). | needs states laid out as [lines, K, n, 4] with equal K (padding). Cost is accumulated over sub-samples, so it is NOT a penetration depth. | similar to Q1 with fewer states | possible but needs a cost object with `use_sweep=True`; **low value**: only a speed-up of the free test (coarser step), no new information |
| Q5 | TCP pose (+ link poses of `tool_frames`) | `chk.get_kinematics(q)` collision_robot_scene.py:70-91 -> `Kinematics._forward` robot/kinematics/kinematics.py:102-170; `KinematicsState.tool_poses`. Our robot has `tool_frames: [wrs_tcp]` only (assets/panthera_wrsext/panthera_d*.yml:378). | [b,h,6] -> poses [b,h,L,3/4] | already paid | yes |
| Q6 | Analytic Jacobian of the tool frames | `Kinematics(compute_jacobian=...)` kinematics.py:49-63; attribute read in `_forward` (kinematics.py:136-153) so `chk.kinematics.compute_jacobian = True` at runtime works; buffer `batch_jacobian` [b,h,num_pose_links,6,dof] (curobolib/cuda_ops/kinematics.py:48-52), returned as `KinematicsState.tool_jacobians` (kinematics.py:167). **Only for wrs_tcp**, not per sphere. | same FK pass, jacobian kernel `launch_kinematics_forward_spheres_jacobian` (cuda_core_backend/kinematics.py:180) | small | yes. Exact for held-part spheres (rigid w.r.t. TCP): J_s = J_v - [r_s]x J_w |
| Q7 | J^T f for robot spheres (joint-space generalized force of a sphere-force field) | autograd: `q.requires_grad_()`, FK, `(spheres[...,:3]*f).sum().backward()` -> `KinematicsFusedFunction.backward` cuda_ops/kinematics.py:270-355 (one fused kernel). Per state because each state's spheres depend only on its q. | [b,h,n,3] force -> [b,h,6] | ~1 FK-backward kernel | yes (call `chk.kinematics.compute_kinematics` directly with grad on) |
| Q8 | Full per-sphere Jacobian (arm spheres) | not exposed; use **finite differences**: 6 extra FK calls `chk.get_kinematics(q + eps*e_j)` on the colliding states only (eps ~1e-3 rad, float32 OK) -> J_s [m,n,3,6]; held spheres via R,p of the same FK. | 6 FK passes over the colliding subset (FK is the cheapest kernel of the pass) | yes |
| Q9 | Sphere -> link map | `chk.kinematics.kinematics_config.link_sphere_idx_map`, `link_name_to_idx_map` (used in lazy_cost/checker.py:47-50) | static | free | yes |
| Q10 | Self-collision | `chk.get_self_collision(sph)` collision_robot_scene.py:184-201 -> `SelfCollisionCost.forward` cost_self_collision.py:91-130: max penetration over sphere pairs per state; per-sphere gradient kept in `self_collision_cost._out_grad` (cost_self_collision.py:44, 113); pair distances if `store_pair_distance`. | already paid | free | yes (self-collision is rare in our data; `by_self` low) |
| Q11 | `max_collision_distance` | stored in `CollisionChecker._max_distance_t` (geom/collision/checker_collision.py:60-71) but **not passed to the discrete kernel** (inputs at wp_autograd.py:93). Mesh SDF query radius is `max(0.5*bbox diagonal, r+eta)` (data_mesh.py:677-690); cuboids are analytic. | — | — | penetration/escape of any sphere inside a mesh is exact up to the mesh's sign quality |
| Q12 | Voxel ESDF | `cu/_src/geom/data/data_voxel.py` (float16 ESDF grids) | — | build cost | not needed; mesh BVH already exact; skip |

Pitfalls of Q1/Q2 visible in the code:
- Values are SUMMED over obstacles (atomic add) -> depth can exceed the real penetration where
  table and part overlap; escape vectors of two obstacles add (normalize; keep the count).
- Mesh sign from `mesh_query_point` (winding). For thin parts a sphere centre past the mid-plane
  gets the far face as nearest -> escape points THROUGH the part and "depth" saturates at
  ~half-thickness + radius. This alone explains part of why `depth` is a weak feature.
- Spheres vs the picked part (`extra_bank`) and held-part vs arm (cdist in checker.py:225-250) are
  outside cuRobo: escape = (c_robot - c_obs)/|.| of the closest pair, penetration = r1+r2-d.
  Compute them analytically in the same cdist (argmin index already available).

---------------------------------------------------------------------------------------------
## 2. First principles: what sets the detour time

Rest-to-rest, per-joint limits. For ANY path, joint j must travel its total variation TV_j and
must stop at each reversal of q_j. So for a given path P:
    T(P) >= T_dec(P) := max_j sum_{monotone pieces k of q_j} t_trap(|piece_k|)       (decoupled bound)
with equality only if the joints never share acceleration (straight line: T = LB). The executed
time adds a coupling term from curvature (|q'_j s'' + q''_j s'^2| <= A in timing.py).

So a detour costs time only through:
 (i)  extra displacement or reversal of the BOTTLENECK joint j* = argmax_j t_trap(|dq_j|), or of a
      joint whose slack is exhausted. Joint slack: sigma_j = LB - t_trap(|dq_j|) (time), or in
      displacement D_j = t_trap^{-1}(LB) - |dq_j|. A detour that only uses joints with slack, and
      stays monotone in them, is free in the decoupled sense;
 (ii) curvature coupling: a bend in joint space forces joints to share acceleration (this is what
      the cuRobo tax is made of, even without obstacles);
 (iii) where on the line it happens: near an endpoint the arm is slow (little coupling cost) but
      the endpoint is fixed, so the escape must be produced over a short stretch -> large q''.

Which joints must move is decided by the GEOMETRY: the escape direction n_i of each colliding
sphere i, mapped through that sphere's Jacobian J_i (3x6). A collision of link2 spheres (near the
base) needs joint 1/2 motion (large, often the bottleneck); a held-part collision near the TCP
can be fixed by wrist joints 4-6 (low VMAX 3.7/4.2 but usually lots of slack). Penetration depth
alone ignores both the mapping and the slack, which is why it explains so little.

---------------------------------------------------------------------------------------------
## 3. Indicators and estimators

### Indicator A — Jacobian escape profile -> decoupled detour time (cheap, local)
Per colliding state k on the line (t_k in [s_in, s_out]):
1. Active rows: for each colliding sphere i (scene from Q1/Q2, part/held from cdist):
   g_i = n_i^T J_i (1x6), r_i = p_i + m (m = margin, e.g. 0.01 m).
   J_i from Q8 (FD, all spheres) or Q6 (held spheres).
2. Slack-weighted minimum-time escape (one Gauss-Newton step):
   dq_k = W^-1 G^T (G W^-1 G^T + lambda I)^-1 r,  W = diag(w_j), w_j = 1/(D_j + eps)
   (cheap to move joints with displacement slack; expensive to move the bottleneck).
   Also compute W = diag(1/VMAX^2) and keep the faster result.
   Batched 6x6 / kxk solves in torch; k = number of active spheres, cap at the 16 deepest.
3. Detour profile: delta(t) = upper envelope of the dq_k over the stretch, ramped to 0 at
   s_in - rho and s_out + rho (rho = 0.1 or the distance to the endpoints), choose ONE side
   consistently (average n over the stretch; if the per-state directions disagree, flag it).
4. Corrected path P_A(t) = a + t(b-a) + delta(t). Features:
   - T_dec(P_A) (decoupled bound above, CPU, microseconds), and dT_A = T_dec(P_A) - LB;
   - argmax joint of the extra time; whether it equals the LB bottleneck j*;
   - ||delta||_W, number of active spheres, link of the deepest sphere (Q9), direction
     consistency (min cosine between per-state n's), stretch location min(s_in, 1-s_out).
5. Optional: re-check P_A in the next batch; if free, TOPP(P_A) is a valid UPPER bound;
   if not, one more GN step from the new penetrations (a 2-iteration CHOMP projection).
Failure modes: linearization fails for deep penetrations / large obstacles (the true detour goes
around, not straight out); sign flips on thin meshes; inconsistent sides along the stretch (local
escape left at one state, right at the next); held-part collisions that need wrist re-orientation
rather than translation. These are exactly the cases where B helps.

### Indicator B — cheap feasible detours (upper bound), same batch
For each blocked segment, generate a few smooth candidate paths and check them in the SAME
`batch_free` call (they are just more lines/states):
 - B1: quadratic Bezier a -> v -> b with v = q(t_m) + alpha*delta_A(t_m), alpha in {1, 1.5, 2.5},
   t_m = middle of the blocked stretch (uses A's direction);
 - B2: "lift": v = q(t_m) + J_z^+ * h, J_z = TCP z-row of Q6, h in {obstacle top - z of deepest
   sphere + 0.03, +0.08 m} (obstacle top from the part's AABB / the colliding sphere's z);
 - B3: "retract": same with the radial (base-to-TCP horizontal) direction, negative;
 - B4 (two-via): a -> v1 -> v2 -> b with v1,v2 = A-corrections at s_in and s_out.
Check the CURVE (sampled every 0.02 rad), not the control polygon. 6-10 curves per blocked
segment; only blocked segments get them. Time each free curve with `timing.topp` (CPU, ~100
points) or, for speed, T_dec plus a fitted curvature tax. UB = min over free curves.
UB is a real feasible time (w.r.t. our sphere model), so LB <= T* <= UB.
Failure modes: no free curve found (deep / enclosed configurations) -> fall back to A;
our sphere model (buffer 0.002) differs from cuRobo's planning model (activation distance,
32 attach spheres) -> a curve free for us may be infeasible for the executor (report rate).

### Indicator C — context features (for diagnosis / a small model)
LB bottleneck joint j*, its slack ratio sigma_2/LB (second slowest joint), link of the deepest
collision, collider type (table/part/installed/held/picked), stretch location, number of
separate colliding stretches, clearance at endpoints (Q3, eta = 0.05) — an endpoint that is
already within 1-2 cm of an obstacle forces the detour to start immediately.

### Estimators (in order of preference)
E1 (zero fitted geometric parameters, physically explicit):
    T_hat = (1 + tau) * T_geo,   T_geo = UB_B if found else max(LB, T_dec(P_A)),
    tau = executor tax measured on FREE segments only (currently ~0.20 median / 0.23 mean).
  Free segments keep T_hat = LB (the system executes the straight line).
  This separates "what the obstacle costs" (T_geo) from "what handing the segment to cuRobo
  costs" (tau) — the latter is a property of the planner, not of the obstacle.
E2 (one parameter): T_hat = (1+tau) * (LB + lambda*(UB - LB)), lambda in [0,1] fitted on one
  assembly, tested on the other. If no UB: lambda' * (T_dec(P_A) - LB).
E3 (only if E1/E2 leave clear structure in the residuals): monotone gradient-boosted trees or a
  depth-2 tree on [UB/LB, found_UB, T_dec(P_A)/LB, j*-involved, stretch location, collider type].
  1342 blocked rows across 2 assemblies is small; expect overfitting; report cross-assembly only.

Expectation: against the CURRENT reference (Sec. 0) none of these can beat the ~0.11 noise
floor at segment level, and ranking gains will be within noise. Against a de-taxed reference
(3.1) I expect E1 with B to give the largest gain on the cases that matter (the few large
detours that produce the 14-22 % max regret), because B finds the actual cheap way around and A
explains which joint pays. The constant model cannot distinguish a 2 % detour from a 40 % one;
UB can.

### 3.1 De-taxing the reference (prerequisite)
For each blocked segment keep TOPP(cuRobo) AND compute
  R1 = min( TOPP(cuRobo path), TOPP(smooth(shortcut(cuRobo path))) )
where `shortcut` = existing `CuroboRefiner._shortcut` (lazy_cost/refine.py:176-192, greedy
farthest free joint line, same checker) and `smooth` = cubic B-spline / corner blending through the
kept waypoints, accepted only if the smoothed curve is free (checked in one batch). Also apply the
same post-processing to a sample of FREE segments: the residual tax there tells how much of the
0.20 is path noise (removable) vs genuine. Do NOT include our own B-curves in R1 (leakage);
report separately "share of blocked segments where UB_B < R1".
If the executor will really run raw cuRobo paths, keep R0 as the system-level truth and R1 as
the geometric truth; report both.

---------------------------------------------------------------------------------------------
## 4. Evaluation protocol

Data: per (context, candidate, grasp, segment) store a, b endpoints (needed for A/B offline!),
LB, blocked flag, all features, R0 = TOPP(cuRobo light), R1 (3.1), and for ~100 blocked + ~100
free segments a repeated cuRobo solve with different seeds and with the full planner
(4 seeds, 100 iters, finetune) -> reference noise and light-vs-full bias.

Segment level (blocked only, also split transit/transfer, by assembly):
 - mean |log(T_hat/T_ref)|, median and p90; Spearman within blocked; calibration plot (deciles).
 - Noise floor: the same metric between two cuRobo runs of the same segment, and the tax
   scatter on free segments. A method is "better" only if it beats the constant by more than
   the floor's bootstrap CI.
 - Decomposition: share of (T_ref - LB) explained by T_geo - LB.

Ranking level (per context, K=3; candidate value = min over grasps of transit + transfer):
 - top-1 within 2 % (ties inside cuRobo noise are not errors), top-3 recall, mean / max regret,
   Spearman. Paired comparison between methods per context (Wilcoxon on regret), bootstrap CI
   over contexts (cluster = context). 30 contexts per assembly is too few for max-regret claims;
   target >= 100 contexts per assembly, plus the dense scenes (boxes/crowd) where detours are
   larger.
 - Methods: LB only | LB x const | current NNLS | E1 | E2 | (E3) | lazy refinement.

Lazy-procedure efficiency (the real goal: fewer cuRobo calls):
 - order candidates by T_hat but keep LB for certification (pruning stays exact);
 - #refined segments until the certified top-K equals the full-refinement top-K;
 - #refinements until the current best equals the final best ("time to first correct");
 - flip rate: fraction of contexts where the estimated top-1 changes after refinement;
 - anytime curve: regret vs refinement budget (0, 1, 2, 4, 8 segments);
 - wasted refinements: refined segments not on any final top-K candidate.

Cost: extra screening ms per candidate (cuda-synchronized, median / p90) for Q2 read-out, FD
Jacobian, GN solves, B-curves checking, CPU timing; at several batch sizes. A pays off only if
it saves more than its cost in refinements (~73-77 ms per refined segment, README).

Generalization / leakage:
 - fit on triangular_frame6, test on rack10 and vice versa; any hyper-parameter (tau, lambda,
   margins, alpha set) chosen on the fit assembly only; tau may be fitted on FREE segments of
   the fit assembly;
 - never use the cuRobo path (or R1 artefacts) to build a feature;
 - segments of one context are correlated -> split by context, not by segment;
 - handle inf references (cuRobo failure) explicitly (count, do not drop silently);
 - candidates pruned in layer 1 are absent from both sides: report the survivor set size.

---------------------------------------------------------------------------------------------
## 5. Opinion on LB * (1 + 0.243 + 1.093*depth)

- 0.243 is the cuRobo/TOPP execution tax (mean 0.23 on FREE segments), not a detour effect. It is
  defensible only when stated as such: "a segment handed to the planner costs ~20 % more than
  its straight-line bound because the planner's path is not straight". Measure it on free
  segments (no fitting on blocked data), not via NNLS on blocked data.
- The depth term contributes ~1-3 % at typical depths, has Spearman 0.04-0.14 with the target,
  and depth is ill-defined on thin meshes (saturation, sign flip). Drop it, or replace it with a
  geometric term from A/B.
- Proposed replacement: E1, T_hat = (1 + tau) * T_geo with T_geo from B (feasible upper bound)
  or A (Jacobian/slack escape). Free segments: LB. Pruning: LB only.
- Before investing in geometry, de-tax the reference (3.1). If after de-taxing the blocked excess
  is still small and unstructured, the honest conclusion is that parts-only scenes have small
  detours and the remaining cost is the planner's path quality — then improve the refiner's
  path (shortcut + smoothing) rather than the estimator.
