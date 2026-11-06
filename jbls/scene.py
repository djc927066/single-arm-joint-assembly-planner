"""Same SEALP display geometry for offline 3-D and native rendered previews.

Visual holes/legs follow show_initial_layout_from_file; the search collision
table remains the solid box in SEALP's workspace YAML, as in the original.
"""
import base64
import numpy as np

PART_COLORS = dict(post_l=[.13,.55,.45], post_r=[.13,.44,.70],
                   beam=[.84,.57,.16], load=[.66,.27,.51])


def _encode(array, dtype):
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode('ascii')


def _body(name, kind, mesh, color, transform=None):
    return dict(name=name, kind=kind, v=_encode(mesh.vertices, '<f4'),
                n=_encode(mesh.vertex_normals, '<f4'), f=_encode(mesh.faces, '<u4'),
                pose=(np.eye(4) if transform is None else np.asarray(transform)).tolist(),
                color=list(map(float, color[:3])))


def build_scene_visual(problem, candidates=()):
    import trimesh
    from curobo_bridge.export_model import wrs_collision_parts
    from sealp.examples.layout.show_initial_layout_from_file import _build_perforated_table_mesh
    from sealp.layout._viz_common import attach_table_corner_legs, DEFAULT_TABLE_FLOOR_Z

    table, info = _build_perforated_table_mesh(problem.table_size, problem.table_center)
    meshes = [_body('work_table', 'fixed', table, problem.workspace.table_rgba)]

    # Collect models from the exact original SEALP leg helper without opening a window.
    from wrs.modeling.geometric_model import StaticGeometricModel
    root = StaticGeometricModel(name='visual_table_legs')
    legs = attach_table_corner_legs(root, problem.table_size, problem.table_center,
                                   floor_z=DEFAULT_TABLE_FLOOR_Z,
                                   rgb=problem.workspace.table_rgba[:3], alpha=1.)
    for i, leg in enumerate(legs):
        m = trimesh.Trimesh(leg.trm_mesh.vertices, leg.trm_mesh.faces, process=False)
        # SEALP's mgm.gen_box bakes its pose into StaticGeometricModel vertices.
        meshes.append(_body(f'table_leg_{i}', 'fixed', m, problem.workspace.table_rgba))
    root.pdndp.removeNode()

    problem.arm.goto_given_conf(np.zeros(6))
    problem.arm.change_jaw_width(float(problem.arm.end_effector.jaw_range[1]))
    for name, _, link in wrs_collision_parts(problem.arm, include_base=True):
        src = link.cmodel.trm_mesh
        mesh = trimesh.Trimesh(src.vertices, src.faces, process=False)
        if len(mesh.faces) > 8000:
            import open3d as o3d
            om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                          o3d.utility.Vector3iVector(mesh.faces))
            om = om.simplify_quadric_decimation(5000)
            mesh = trimesh.Trimesh(np.asarray(om.vertices), np.asarray(om.triangles), process=False)
        meshes.append(_body(name, 'robot', mesh, np.asarray(link.cmodel.rgba), link.gl_homomat))
    for pid, part in problem.parts.items():
        meshes.append(_body(pid, 'part' if pid in problem.ids else 'fixed', part.mesh,
                            PART_COLORS.get(pid, [.60,.61,.63]), part.goal))
    # Each discrete scene uses its locally validated retreat configuration.
    # No interpolation is created between these unconnected states.
    def robot_poses():
        return {name:np.asarray(link.gl_homomat).tolist()
                for name,_,link in wrs_collision_parts(problem.arm, include_base=True)}
    for candidate in candidates:
        problem.arm.goto_given_conf(np.zeros(6))
        problem.arm.change_jaw_width(float(problem.arm.end_effector.jaw_range[1]))
        states = [robot_poses()]
        for pid in candidate['order']:
            step = candidate['steps'][pid]
            problem.arm.change_jaw_width(min(float(problem.arm.end_effector.jaw_range[1]), step['jaw_width']+.008))
            problem.arm.goto_given_conf(np.asarray(step['key_configurations']['retreat']))
            states.append(robot_poses())
        candidate['robot_poses_by_step'] = states
    problem.arm.goto_given_conf(np.zeros(6))
    return dict(meshes=meshes, table_holes=info,
                source_visual='sealp/examples/layout/show_initial_layout_from_file.py::_build_perforated_table_mesh',
                source_legs='sealp/layout/_viz_common.py::attach_table_corner_legs',
                note='Visual perforated table and legs follow SEALP; collision uses its configured solid table box. Step 0 uses HOME, subsequent steps use screened retreat configurations. These are discrete states, not connected motion trajectories.')


def render_previews(report, output):
    """Render the same mesh payload with Panda3D offscreen; no browser needed."""
    from panda3d.core import loadPrcFileData, AmbientLight, DirectionalLight, Vec4, Vec3, Point3, Filename, PNMImage
    loadPrcFileData('', 'window-type offscreen\nwin-size 1400 1050\naudio-library-name null\nframebuffer-multisample 1\nmultisamples 4')
    from direct.showbase.ShowBase import ShowBase
    from wrs.basis.data_adapter import pdgeomndp_from_vvnf, npmat4_to_pdmat4
    base = ShowBase(windowType='offscreen')
    base.disableMouse()
    base.setBackgroundColor(.94,.95,.96,1)
    base.camLens.setNearFar(.01,20.)
    base.camLens.setFov(40)
    base.cam.setPos(1.55,-1.85,1.10)
    base.cam.lookAt(Point3(.22,0.,-.20), Vec3(0,0,1))
    ambient = AmbientLight('ambient'); ambient.setColor(Vec4(.52,.52,.52,1))
    base.render.setLight(base.render.attachNewNode(ambient))
    light = DirectionalLight('sun'); light.setColor(Vec4(.7,.7,.7,1))
    light_np = base.render.attachNewNode(light); light_np.setHpr(-35,-55,0)
    base.render.setLight(light_np)
    nodes = []
    for item in report['scene_visual']['meshes']:
        v = np.frombuffer(base64.b64decode(item['v']), '<f4').reshape(-1,3)
        n = np.frombuffer(base64.b64decode(item['n']), '<f4').reshape(-1,3)
        f = np.frombuffer(base64.b64decode(item['f']), '<u4').reshape(-1,3)
        node = pdgeomndp_from_vvnf(v,n,f,item['name'])
        node.reparentTo(base.render); node.setColor(*item['color'],1.)
        node.setTwoSided(True)
        nodes.append((item,node))
    candidate = report['candidates'][0] if report['candidates'] else None
    files = []
    modes = ('initial','assembled') if candidate else ('assembled',)
    try:
        for mode in modes:
            for item,node in nodes:
                pose = item['pose']
                if item['kind'] == 'part':
                    pose = (candidate['layout'][item['name']] if mode == 'initial'
                            else report['goal_poses'][item['name']])
                elif item['kind'] == 'robot' and candidate:
                    step_index = 0 if mode == 'initial' else len(candidate['order'])
                    pose = candidate['robot_poses_by_step'][step_index][item['name']]
                node.setMat(npmat4_to_pdmat4(np.asarray(pose,float)))
            for _ in range(4):
                base.graphicsEngine.renderFrame()
            base.graphicsEngine.syncFrame()
            path = output/f'scene_{mode}.png'
            shot = PNMImage()
            if not base.win.getScreenshot(shot):
                raise RuntimeError('Panda3D offscreen framebuffer could not be read')
            if not shot.write(Filename.fromOsSpecific(str(path.resolve()))):
                raise RuntimeError(f'Could not render {path}')
            files.append(path.name)
    finally:
        base.destroy()
    return files


def main():
    """Rebuild scene displays from a saved search result, without running IK/search."""
    import argparse
    import json
    from pathlib import Path
    from time import perf_counter
    from types import SimpleNamespace
    from curobo_bridge.runtime import prepare
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    prepare()
    import trimesh
    from adapters.panthera import PantheraArmProvider
    from .workspace import load_workspace
    from .viewer import write_viewer
    output = args.run_dir
    report = json.loads((output/'result.json').read_text(encoding='utf-8'))
    workspace = load_workspace(report['workspace']['source_config'])
    if workspace.sha256 != report['workspace']['source_sha256']:
        raise ValueError('Workspace configuration changed since this result was generated')
    ids = tuple(report['goals'])
    parts = {}
    for pid in ('base',)+ids:
        goal = np.eye(4)
        if pid == 'base':
            goal[:3,3] = report['site']
        else:
            goal = np.asarray(report['goal_poses'][pid],float)
        parts[pid] = SimpleNamespace(mesh=trimesh.load(output/'meshes'/f'{pid}.stl',process=False), goal=goal)
    problem = SimpleNamespace(parts=parts, ids=ids, workspace=workspace,
                              table_size=workspace.table_size, table_center=workspace.table_center,
                              arm=PantheraArmProvider(base_pos=workspace.robot_pos,base_rotmat=workspace.robot_rot).arm())
    started = perf_counter()
    report['scene_visual'] = build_scene_visual(problem, report['candidates'])
    report['scene_preview_files'] = render_previews(report, output)
    report['visualization_s'] = perf_counter()-started
    (output/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),encoding='utf-8')
    write_viewer(report, output/'viewer.html')
    print(output/'viewer.html', flush=True)


if __name__ == '__main__':
    main()
