"""Run: python -m jbls.run_bridge --candidates 12 --beam-width 8 --compare-fixed"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
import numpy as np
from .search import joint_search, cost_breakdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=int, default=12,
                        help='SEALP per-part XY centre cap (each may have 0/90-degree poses); 0=all')
    parser.add_argument('--beam-width', type=int, default=8)
    parser.add_argument('--spacing', type=float, default=.05,
                        help='SEALP per-part grid spacing in metres (minimum effective 0.03)')
    parser.add_argument('--directions', type=int, default=64)
    parser.add_argument('--site-x', type=float, default=None,
                        help='Override fixed base X from the scene input (metres)')
    parser.add_argument('--scene-config', type=Path,
                        help='Bridge input JSON; default jbls/bridge_scene.json')
    parser.add_argument('--assembly-scale', type=float,
                        help='Override assembly linear scale; 2/3 means shrink by one third')
    parser.add_argument('--workspace-config', type=Path,
                        help='SEALP single-table YAML; default sealp/config/sample_config.yaml')
    parser.add_argument('--compare-fixed', action='store_true')
    parser.add_argument('--asset-dir', type=Path,
                        help='Persistent STL/grasp directory; default jbls/assets/bridge')
    parser.add_argument('--rebuild-assets', action='store_true',
                        help='Regenerate STL files and nominal grasp pickles for current inputs')
    parser.add_argument('--prepare-assets-only', action='store_true',
                        help='Prepare/load files and target grasp catalogue, without joint search')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.candidates < 0 or args.beam_width < 1 or not np.isfinite(args.spacing) or args.spacing <= 0 or args.directions < 1:
        parser.error('invalid candidate/beam/spacing/direction budget')
    if args.assembly_scale is not None and (not np.isfinite(args.assembly_scale) or args.assembly_scale <= 0):
        parser.error('--assembly-scale must be finite and positive')
    output = args.output or Path(__file__).parent / '_output' / datetime.now().strftime('bridge_%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=True)
    from curobo_bridge.runtime import prepare
    prepare()  # torch must load before Panda3D on Windows
    from .bridge import BridgeProblem
    start = perf_counter()
    problem = BridgeProblem(output / 'meshes', site_x=args.site_x, directions=args.directions,
                             workspace_config=args.workspace_config, scene_config=args.scene_config,
                             assembly_scale=args.assembly_scale, asset_dir=args.asset_dir,
                             rebuild_assets=args.rebuild_assets)
    if args.prepare_assets_only:
        (output/'assets.json').write_text(json.dumps(problem.asset_files, indent=2), encoding='utf-8')
        print('ASSETS', problem.asset_dir, flush=True)
        return
    domains = problem.domains(args.spacing, args.candidates)
    setup_s = perf_counter()-start
    def progress(stats):
        print('layer', json.dumps(stats), flush=True)
        with (output / 'latest_layer.json').open('w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
    nodes, report = joint_search(problem.ids, domains, problem.goals, problem.home,
                                 problem.structure, problem.placement, beam_width=args.beam_width,
                                 progress=progress)
    candidates = []
    for node in nodes:
        rows = cost_breakdown(node.order, node.layout, problem.goals, problem.home)
        assert np.isclose(node.cost, sum(r['total_m'] for r in rows))
        candidates.append(dict(order=node.order, cost_m=node.cost,
            layout={p:c.pose.tolist() for p,c in node.layout.items()}, steps=node.steps, segments=rows))
    report.update(created=datetime.now().isoformat(), method='joint_backward_euclidean_v1',
        objective='sum distance(previous_goal_or_home, pick) + distance(pick, goal); no final return',
        cost_unit='metre', cost_is_execution_time=False, safe_to_execute=False,
        validation_scope='S/G/A; calibrated force-closed grasps; endpoint GPU IK; sampled WRS local motions and jaw motion. No global transit/transfer trajectories.',
        collision_scope='Existing panthera_wrsext link2..5, palm, fingers and configured self pairs; base/link1 excluded.',
        fixed_insertion_axis='world -Z (Bridge input); A is not a swept insertion proof',
        setup_s=setup_s, domain_stats=problem.domain_stats,
        candidate_generation=problem.candidate_generation, candidates=candidates,
        parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        site=problem.site.tolist(), home_tcp=problem.home.tolist(),
        goals={p:q.tolist() for p,q in problem.goals.items()},
        goal_poses={p:problem.parts[p].goal.tolist() for p in problem.ids},
        workspace=problem.workspace.metadata(),
        assembly_input=problem.assembly_input,
        asset_directory=str(problem.asset_dir), asset_files=problem.asset_files,
        assembly_parts={p:dict(size_m=problem.parts[p].mesh.extents.tolist(),
            mass_kg=problem.part_by_id[p].mass_kg,
            goal_pose=problem.parts[p].goal.tolist()) for p in problem.parts},
        table_center=problem.table_center.tolist(), table_size=problem.table_size.tolist(),
        sizes={p:problem.parts[p].mesh.extents.tolist() for p in problem.ids},
        counts=dict(problem.counts), ik_unique_targets=problem.ik.targets)
    if args.compare_fixed:
        baseline, base_report = joint_search(problem.ids, domains, problem.goals, problem.home,
            problem.structure, problem.placement, beam_width=args.beam_width,
            fixed_order=('post_l','post_r','beam','load'), progress=progress)
        report['fixed_order_baseline'] = dict(order=['post_l','post_r','beam','load'],
            best_cost_m=baseline[0].cost if baseline else None, report=base_report,
            timing_note='Runs after joint search with warm caches; wall times are NOT a fair speed comparison.')
    report['total_wall_s'] = perf_counter()-start
    # Preserve the search outcome even if an optional rendering backend fails.
    (output/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    from .scene import build_scene_visual, render_previews
    visual_start = perf_counter()
    report['scene_visual'] = build_scene_visual(problem, candidates)
    report['scene_preview_files'] = render_previews(report, output)
    report['visualization_s'] = perf_counter()-visual_start
    (output/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    from .viewer import write_viewer
    write_viewer(report, output/'viewer.html')
    print('RESULT', output, flush=True)
    print('BEST', candidates[0]['order'] if candidates else None,
          candidates[0]['cost_m'] if candidates else None, flush=True)


if __name__ == '__main__':
    main()
