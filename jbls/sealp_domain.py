"""SEALP's default BSFS beam XY domain, adapted to the current Bridge assets.

The original grid, geometric enumeration and centroid-seeded farthest-point
selection are called directly. This adapter does not run SEALP's fixed-order
search or replace Bridge's allowed upright orientations / grasp pipeline.
"""
from types import SimpleNamespace

import numpy as np


class SealpCandidateDomain:
    def __init__(self, problem, *, collision_factory=None):
        from sealp.examples.layout import find_optimal_initial_layout_tower_strict_pycharm as legacy
        from sealp.examples.layout.infer_assembly_ga import (
            _part_anchor_grid, _enumerate_xy_candidates, _farthest_point_order,
        )
        from shapely.geometry import MultiPoint, box

        if collision_factory is None:
            import wrs.modeling.collision_model as mcm
            collision_factory = mcm.CollisionModel
        self.problem, self.legacy = problem, legacy
        self._grid = _part_anchor_grid
        self._enumerate = _enumerate_xy_candidates
        self._spread = _farthest_point_order
        self._multipoint = MultiPoint
        self.table_margin = legacy.DEFAULT_TABLE_MARGIN  # BSFS _build_searcher: 6 cm
        self.min_staging_mesh_clearance = legacy.DEFAULT_MIN_STAGING_MESH_CLEARANCE
        self.filter_staging_near_arms = True
        self.staging_arm_x_clearance = legacy.DEFAULT_STAGING_ARM_X_CLEARANCE
        self.staging_arm_y_clearance = legacy.DEFAULT_STAGING_ARM_Y_CLEARANCE
        self.check_robot_home_collision = True
        self.single_arm_mode = True
        # The original home checker expects a robot with a .lft_arm attribute.
        # This facade wraps the existing single Panthera, without creating a robot.
        self.robot = SimpleNamespace(lft_arm=problem.arm)
        self.part_order = list(problem.ids)
        lo = problem.table_center[:2] - problem.table_size[:2] / 2 + self.table_margin
        hi = problem.table_center[:2] + problem.table_size[:2] / 2 - self.table_margin
        if np.any(hi <= lo):
            raise ValueError('SEALP table margin leaves no staging area')
        self.table_x_range = (float(lo[0]), float(hi[0]))
        self.table_y_range = (float(lo[1]), float(hi[1]))
        self.table_polygon = box(*lo, *hi)
        self.staging_models = {'base': problem.goal_cm['base']}
        self.mesh_vertices = {p: part.mesh.vertices for p, part in problem.parts.items()}
        self.rot_cands, self.cache = {}, {}
        self.staging_clearance = .0002  # retain Bridge's existing table contact convention
        for pid in problem.ids:
            part = problem.parts[pid]
            self.staging_models[pid] = collision_factory(initor=str(part.mesh_path), name=pid+'_domain')
            self.rot_cands[pid] = []
            for yaw in (0., np.pi / 2):
                c, s = np.cos(yaw), np.sin(yaw)
                R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
                vertices = part.mesh.vertices @ R.T
                extent = np.ptp(vertices, axis=0)
                self.rot_cands[pid].append(legacy.RotCandidate(
                    rotmat=R, z_offset=problem.workspace.top_z-vertices[:, 2].min()+self.staging_clearance,
                    tag=f'yaw_{int(round(np.degrees(yaw)))}', extent=extent,
                    footprint=extent[:2], rot_name='Bridge upright yaw'))
        # Reuse the original WRS geometry/home routines, including backup/restore.
        for name in ('_apply_staging_pose', '_pairwise_collision', '_mesh_clearance_reason',
                     '_world_vertices_for_staging', '_aabb_distance_between_world_vertices',
                     '_vertex_distance_between_world_vertices', '_robot_home_collision_reason'):
            setattr(self, name, getattr(legacy.WeightedInitialLayoutSearcher, name).__get__(self))

    def _arm_base_xy_map(self):
        return {'arm_base': tuple(self.problem.workspace.robot_pos[:2])}

    def _polygon(self, pid, xy, cand):
        vertices = self.mesh_vertices[pid] @ cand.rotmat.T
        return self._multipoint(vertices[:, :2] + xy).convex_hull

    def _staging_arm_keepout_reason(self, pid, xy, cand):
        reason = self.legacy.WeightedInitialLayoutSearcher._staging_arm_keepout_reason(self, pid, xy, cand)
        if reason:
            return reason
        # SEALP pads the grid using the minimum footprint over rotations. A longer
        # alternate rotation can still overhang; do not emit that particular pose.
        if not self.table_polygon.covers(self._polygon(pid, xy, cand)):
            return 'oriented_footprint_outside_table_margin'
        return None

    def check_pose(self, pid, xy, cand):
        key = (pid, tuple(np.asarray(xy, float)), cand.tag)
        if key not in self.cache:
            reason = self._staging_arm_keepout_reason(pid, xy, cand)
            if reason is None:
                self._apply_staging_pose(pid, xy, cand)
                active = [pid, 'base']
                reason = (self._pairwise_collision(active_pids=active)
                          or self._mesh_clearance_reason(active_pids=active)
                          or self._robot_home_collision_reason(active_pids=[pid]))
            self.cache[key] = reason
        return self.cache[key]

    def select_centres(self, pid, spacing, cap):
        """Identical XY selection to BSFS continuous_domain; cap counts XYs."""
        if not np.isfinite(spacing) or spacing <= 0 or cap < 0:
            raise ValueError('positive finite spacing and nonnegative centre cap required')
        raw = self._grid(self, pid, spacing)
        feasible = self._enumerate(self, pid, spacing, first_pid='base')
        indices = self._spread(feasible, cap or len(feasible))
        selected = [feasible[i] for i in indices]
        stats = dict(raw_centres=len(raw), geometry_centres=len(feasible),
                     selected_centres=len(selected),
                     raw_xy=[xy.tolist() for xy in raw],
                     geometry_xy=[xy.tolist() for xy in feasible],
                     selected_xy=[xy.tolist() for xy in selected])
        return selected, stats

    def metadata(self, spacing, cap):
        return dict(method='sealp_bsfs_beam_per_part_xy',
            grid_source='sealp.examples.layout.infer_assembly_ga._part_anchor_grid',
            filter_source='sealp.examples.layout.infer_assembly_ga._enumerate_xy_candidates',
            selector_source='sealp.examples.layout.infer_assembly_ga._farthest_point_order',
            centre_cap=cap or None, centre_cap_unit='XY positions, not oriented poses',
            requested_spacing_m=spacing, minimum_spacing_m=.03, grid_placement='linspace within per-part padded bounds',
            table_margin_m=self.table_margin, extra_footprint_pad_m=.02,
            table_x_range=list(self.table_x_range), table_y_range=list(self.table_y_range),
            arm_keepout_halfsize_m=[self.staging_arm_x_clearance, self.staging_arm_y_clearance],
            mesh_clearance_m=self.min_staging_mesh_clearance, robot_home_collision=True,
            orientation_degrees=[0, 90],
            adaptation='Bridge upright rotations retained; each selected XY may supply both orientations; whole oriented footprint must fit inside table margin; file-backed grasps/GPU IK applied after XY selection',
            staging_bottom_clearance_m=self.staging_clearance,
            staging_height_note='Bridge contact convention retained; original BSFS _build_searcher uses 0.01 m')
