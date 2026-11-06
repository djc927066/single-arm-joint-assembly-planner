"""Bridge adapter: existing S/G/A + Panthera calibrated grasps + GPU IK/WRS local checks.

Search results are locally screened candidates, NOT global transport witnesses.
The collision coverage/contact exemptions follow curobo_bridge.assembly_demo.
"""
from dataclasses import replace
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import numpy as np
from .search import Placement


class CachedIK:
    def __init__(self):
        from curobo_bridge.assembly_demo import BatchIK
        self.backend = BatchIK(batch=128, seeds=32, returns=4)
        self.cache = {}
        self.targets = 0

    def solve(self, poses):
        keys = [np.asarray(T, dtype=np.float64).tobytes() for T in poses]
        missing = dict((k, T) for k, T in zip(keys, poses) if k not in self.cache)
        if missing:
            q, ok = self.backend.solve(np.stack(list(missing.values())))
            self.cache.update((k, (a, b)) for k, a, b in zip(missing, q, ok))
            self.targets += len(missing)
        return np.stack([self.cache[k][0] for k in keys]), np.stack([self.cache[k][1] for k in keys])


class BridgeProblem:
    def __init__(self, output, *, site_x=None, directions=64, workspace_config=None,
                 scene_config=None, assembly_scale=None, asset_dir=None, rebuild_assets=False):
        import trimesh
        import wrs.modeling.collision_model as mcm
        from asmcore.model import Assembly
        from asmcore.robotics import GraspabilityAnalyzer, GraspabilityConfig
        from asmcore.planning.sequence import SequenceConfig, SequenceEvaluator
        from adapters.panthera_gripper import PantheraGripperPort
        from adapters.panthera import PantheraArmProvider
        from tasks.bridge import build_bridge, box_mesh
        from curobo_bridge import assembly_demo as demo
        from .workspace import load_workspace, bridge_site, DEFAULT_CONFIG
        from .asset_files import DEFAULT_ASSET_DIR, prepare_part_stls, prepare_grasp_file
        self.demo, self.output, self.directions = demo, Path(output), directions
        self.output.mkdir(parents=True, exist_ok=True)
        self.workspace = load_workspace(workspace_config or DEFAULT_CONFIG)
        self.table_size = self.workspace.table_size.copy()
        self.table_center = self.workspace.table_center.copy()
        scene_path = Path(scene_config or Path(__file__).with_name('bridge_scene.json')).resolve()
        scene_raw = scene_path.read_bytes()
        scene_input = json.loads(scene_raw)
        self.assembly_scale = float(scene_input['assembly_scale'] if assembly_scale is None else assembly_scale)
        friction = float(scene_input['friction'])
        if not np.isfinite(friction) or friction < 0:
            raise ValueError('Bridge friction must be finite and nonnegative')
        source = build_bridge(scale=self.assembly_scale, friction=friction)
        base_part = next(p for p in source.parts if p.part_id == 'base')
        requested_x = scene_input['fixed_base_x_m'] if site_x is None else site_x
        self.site = bridge_site(self.workspace, base_part.geometry.vertices, requested_x)
        self.assembly_input = dict(source_config=str(scene_path),
            source_sha256=hashlib.sha256(scene_raw).hexdigest(), linear_scale=self.assembly_scale,
            mass_scale=self.assembly_scale**3, mass_assumption='constant density',
            friction=friction, fixed_base_position=self.site.tolist(),
            scope='base, its pockets, four movable parts and their relative goal translations; table and robot unchanged')
        parts = []
        for p in source.parts:
            T = np.array(p.assembled_tf)
            if p.part_id == 'table':
                T[:3, 3] = self.table_center
                p = replace(p, geometry=box_mesh(*self.table_size))
            else:
                T[:3, 3] += self.site
            parts.append(replace(p, assembled_tf=T))
        self.asset_dir = Path(asset_dir or DEFAULT_ASSET_DIR).resolve()
        parts, self.asset_files = prepare_part_stls(parts, self.asset_dir, rebuild=rebuild_assets)
        self.assembly = Assembly(tuple(parts))
        self.goal_state = self.assembly.initial_state()
        self.fixed = frozenset(p.part_id for p in parts if p.fixed)
        self.ids = tuple(p.part_id for p in parts if not p.fixed)
        self.part_by_id = {p.part_id: p for p in parts}
        self.goals = {p.part_id: np.array(p.assembled_tf[:3, 3]) for p in parts if not p.fixed}
        self.config = SequenceConfig(max_auxiliary_resources=0)
        self.sequence = SequenceEvaluator(self.assembly, self.config)
        self.transitions, self.states, self.graphs, self.stability = {}, {}, {}, {}
        self.counts = Counter()
        provider = PantheraArmProvider(base_pos=self.workspace.robot_pos, base_rotmat=self.workspace.robot_rot)
        self.arm = provider.arm()
        self.home = provider.home_tcp_world()[:3, 3]
        self.ik = CachedIK()
        # The sequence-only demo's 2-4 vertical grasps are not a sufficient arm
        # catalogue. Sample rotations about the closing axis, prefilter by goal
        # IK, then run every survivor through the SAME calibration/FC checker.
        supplied = {}
        grip = PantheraGripperPort()
        for p in parts:
            if p.fixed:
                continue
            candidates, grasp_file = prepare_grasp_file(
                p, self.asset_files[p.part_id]['stl'], grip, self.asset_dir,
                rebuild=rebuild_assets)
            self.asset_files[p.part_id].update(grasp_file)
            print('grasp file', p.part_id, grasp_file['grasp_file_status'],
                  len(candidates), grasp_file['grasp_file'], flush=True)
            dummy = SimpleNamespace(goal=np.array(p.assembled_tf))
            if not candidates:
                supplied[p.part_id] = []
                continue
            poses = np.stack([demo.keyposes(dummy, dummy.goal, g.pose)[k]
                              for g in candidates for k in ('preins','ins','retreat')])
            _, ok = self.ik.solve(poses)
            valid = ok.reshape(len(candidates),3,-1).any(-1).all(-1)
            supplied[p.part_id] = [g for g,v in zip(candidates,valid) if v]
            print('goal grasp IK', p.part_id, len(supplied[p.part_id]), '/', len(candidates), flush=True)
        self.ga = GraspabilityAnalyzer(self.assembly, grip,
            config=GraspabilityConfig(max_grasps=max(8, max(map(len,supplied.values()), default=0))), grasps=supplied)
        self.parts, self.records = {}, {}
        self.goal_cm, self.held_cm = {}, {}
        for p in parts:
            if p.part_id == 'table':
                continue
            mesh = trimesh.Trimesh(p.geometry.vertices, p.geometry.faces, process=False)
            path = self.output / (p.part_id + '.stl')
            source_path = Path(self.asset_files[p.part_id]['stl'])
            if source_path.resolve() != path.resolve():
                shutil.copyfile(source_path, path)
            records = {}
            if not p.fixed:
                result = self.ga.analyze(self.state(frozenset((p.part_id,))), p.part_id)
                records = dict(zip(result.accepted_ids, self.ga.accepted_grasps(result)))
                print('catalogue', p.part_id, 'force-closed=', len(records), flush=True)
            self.records[p.part_id] = records
            self.parts[p.part_id] = demo.TowerPart(p.part_id, path, mesh, mesh,
                [(np.array(g.pose), g.jaw_width) for g in records.values()], np.array(p.assembled_tf))
            cm = mcm.CollisionModel(initor=str(path), name=p.part_id)
            cm.pose = (np.array(p.assembled_tf[:3, 3]), np.array(p.assembled_tf[:3, :3]))
            self.goal_cm[p.part_id] = cm
            self.held_cm[p.part_id] = mcm.CollisionModel(initor=str(path), name='held_'+p.part_id)
        self.table_cm = mcm.gen_box(xyz_lengths=self.table_size, pos=self.table_center)
        self.oracle = demo.MeshOracle(self.arm, None)
        self.stage_cm, self.polys = {}, {}
        self.local_cache = {}

    def state(self, remaining):
        from asmcore.model import AssemblyState
        if remaining not in self.states:
            self.states[remaining] = AssemblyState({p: T for p, T in self.goal_state.poses.items()
                                                    if p in self.fixed | remaining})
        return self.states[remaining]

    def structure(self, remaining, pid):
        from asmcore.mechanics.stability_sweep import DirectionalStabilityAnalyzer, StabilitySweepConfig
        from asmcore.motion.directions import assembly_directions, DirectionConfig
        from asmcore.planning.quality import score_assemblability
        key = (remaining, pid)
        if key in self.transitions:
            self.counts['structure_cache_hits'] += 1
            return self.transitions[key]
        self.counts['structure_evaluations'] += 1
        remainder = remaining - {pid}
        if remainder:
            eq = self.sequence.equilibrium(self.state(remainder), ())
            if eq.status != 'feasible':
                result = (None, 'remainder_'+eq.status)
                self.transitions[key] = result
                return result
        if remaining not in self.stability:
            state = self.state(remaining)
            eq = self.sequence.equilibrium(state, ())
            if eq.status != 'feasible':
                self.stability[remaining] = None
            else:
                graph = self.sequence.graph(state)
                self.graphs[remaining] = graph
                sweep = DirectionalStabilityAnalyzer(self.assembly, state, graph,
                    config=StabilitySweepConfig(backend='numpy', direction_count=self.directions,
                                               batch_size=min(64, self.directions), stability=self.config.stability)).analyze()
                self.stability[remaining] = sweep.sampled_minimum_load_factor
        S = self.stability[remaining]
        if S is None or S <= 1e-8:
            result = (None, 'current_unstable_or_unresolved')
        else:
            gr = self.ga.analyze(self.state(remaining), pid)
            if not gr.accepted_ids:
                result = (None, 'G_unknown_or_zero')
            else:
                directions = assembly_directions(self.graphs[remaining], (pid,), config=DirectionConfig(method='socp'))
                score = score_assemblability(directions)
                if score.score in (None, 0):
                    result = (None, 'A_unknown_or_zero')
                else:
                    result = (dict(S=float(S), G=len(gr.accepted_ids), G_status=gr.status,
                                   G_scope='known accepted count within goal-IK-prefiltered catalogue', A=float(score.score),
                                   grasp_ids=list(gr.accepted_ids)), '')
        self.transitions[key] = result
        print('structure', sorted(remaining), 'remove', pid, result[1] or 'passed', flush=True)
        return result

    def domains(self, spacing=.05, per_part=12):
        """SEALP BSFS beam centres first, then Bridge poses and file-grasp IK."""
        from curobo_bridge.assembly_demo import keyposes, KEYS
        from .sealp_domain import SealpCandidateDomain
        import wrs.modeling.collision_model as mcm
        result, self.domain_stats = {}, {}
        domain = SealpCandidateDomain(self)
        self.candidate_generation = domain.metadata(spacing, per_part)
        self.stage_cm.clear()
        self.polys.clear()
        for pid in self.ids:
            part = self.parts[pid]
            centres, stats = domain.select_centres(pid, spacing, per_part)
            candidates = []
            rejected = Counter()
            for xy in centres:
                for rot in domain.rot_cands[pid]:
                    reason = domain.check_pose(pid, xy, rot)
                    if reason is not None:
                        rejected[reason] += 1
                        continue
                    T = np.eye(4)
                    T[:3, :3] = rot.rotmat
                    T[:3, 3] = [*xy, rot.z_offset]
                    key = f'{float(xy[0]):.17g},{float(xy[1]):.17g},{rot.tag}'
                    candidates.append(Placement(key, T))
                    self.polys[pid, key] = domain._polygon(pid, xy, rot)
            survivors = []
            for choice in candidates:
                if not part.grasps:
                    continue
                poses = np.stack([keyposes(part, choice.pose, G)[k] for G, jaw in part.grasps for k in KEYS])
                _, ok = self.ik.solve(poses)
                if ok.reshape(len(part.grasps), len(KEYS), -1).any(-1).all(-1).any():
                    survivors.append(choice)
            # XY cap was applied by original SEALP FPS before IK. Do not rank or
            # truncate oriented poses again with the removed goal-distance rule.
            result[pid] = survivors
            for c in survivors:
                cm = mcm.CollisionModel(initor=str(part.mesh_path), name=pid+'_staging')
                cm.pose = (c.pose[:3,3].copy(), c.pose[:3,:3].copy())
                self.stage_cm[pid, c.key] = cm
            stats.update(geometry_candidates=len(candidates), endpoint_ik_candidates=len(survivors),
                retained=len(survivors), retained_centres=len({tuple(c.position[:2]) for c in survivors}),
                orientation_rejections=dict(rejected),
                retained_poses=[dict(key=c.key, pose=c.pose.tolist()) for c in survivors])
            self.domain_stats[pid] = stats
            print('domain', pid, {k:v for k,v in stats.items()
                  if k not in ('raw_xy', 'geometry_xy', 'selected_xy', 'retained_poses')}, flush=True)
        return result

    def placement(self, parent, remaining, pid, choice, structure):
        self.counts['placement_queries'] += 1
        poly = self.polys[pid, choice.key]
        if any(poly.distance(self.polys[q, c.key]) < .01 for q, c in parent.layout.items()):
            return None, 'initial_footprints_overlap'
        key = (remaining, pid, choice.key, tuple(sorted((p, c.key) for p,c in parent.layout.items())))
        if key in self.local_cache:
            self.counts['local_cache_hits'] += 1
            return self.local_cache[key]
        layout = dict(parent.layout, **{pid: choice})
        stage = {p: self.stage_cm[p, c.key] for p,c in layout.items()}
        world = SimpleNamespace(layout={p:c.pose for p,c in layout.items()}, table=self.table_cm,
            goal_cm=self.goal_cm, stage_cm=stage, held_cm=self.held_cm, risers={})
        world.objects = lambda assembled, staged: [self.table_cm]+[self.goal_cm[p] for p in assembled]+[stage[p] for p in staged]
        self.oracle.world = world
        # Only the set of earlier installed parts matters, not its unknown order.
        before = ['base'] + sorted(remaining - {pid})
        order = before + [pid] + list(parent.order)
        original = self.parts[pid].grasps
        ids = [g for g in structure['grasp_ids'] if g in self.records[pid]]
        self.parts[pid].grasps = [(np.array(self.records[pid][g].pose), self.records[pid][g].jaw_width) for g in ids]
        # Exact face-contact mates for the axis-aligned Bridge only.
        bounds = {}
        for p in before + [pid]:
            part = self.parts[p]
            v = part.mesh.vertices @ part.goal[:3,:3].T + part.goal[:3,3]
            bounds[p] = (v.min(0), v.max(0))
        a,b = bounds[pid]
        mates = [p for p in before if np.all(b >= bounds[p][0]-.002) and np.all(bounds[p][1] >= a-.002)]
        try:
            plan, stats = self.demo.screen_grasps(len(before), self.parts, order, world, self.oracle,
                                                self.ik, {pid: mates}, max_verify=60, check_jaw_motion=True)
        finally:
            self.parts[pid].grasps = original
        if plan is None:
            reason = 'no_local_motion_in_candidate_budget'
            self.counts.update({'local_fail:'+r: n for r,n in stats.get('rejections',{}).items()})
            result = (None, reason)
        else:
            selected_id = ids[plan.grasp_id]
            selected = self.records[pid][selected_id]
            result = (dict(**structure, common_grasp_id=selected_id,
                           grasp_file=self.asset_files[pid]['grasp_file'],
                           grasp_file_sha256=self.asset_files[pid]['grasp_file_sha256'],
                           grasp_file_candidate_id=int(selected.provenance['file_candidate_id']),
                           grasp_file_stage=self.asset_files[pid]['grasp_file_stage'],
                           jaw_width=plan.jaw, object_to_tcp=plan.G.tolist(),
                           key_configurations={k: np.asarray(v).tolist() for k,v in plan.q.items()},
                           screening=stats, full_transport_pending=True), '')
        self.local_cache[key] = result
        return result
