"""Six movable parts on a fixed base; isolated, metre-based assembly benchmark.

Builds real STL files, verifies all 64 nominal static subsets and 720 orders,
and exports an offline viewer. This does not certify robot execution.
"""
from __future__ import annotations
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys, json, itertools, hashlib, math
from pathlib import Path
from dataclasses import replace
from time import perf_counter

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from curobo_bridge.runtime import prepare
prepare()  # Torch must precede trimesh/Panda3D on Windows.
import numpy as np
from asmcore.model import Assembly, AssemblyState, MeshData, Part, MatingRelation, to_dict
from tasks.bridge import box_mesh, pose
from jbls.asset_files import prepare_part_stls, prepare_grasp_file
from jbls.workspace import load_workspace

VERTICES={'A':(-.080,-.060),'B':(.080,-.060),'C':(.015,.080)}
POST_SIZE={'A':(.060,.054,.100),'B':(.058,.062,.100),'C':(.062,.056,.100)}
EDGES=(('AB','A','B',.022),('BC','B','C',.020),('CA','C','A',.018))
BASE_SIZE=(.240,.240,.014)
POCKET_DEPTH=.008
POCKET_FLOOR=BASE_SIZE[2]-POCKET_DEPTH
CLEARANCE=.0002
END_TRIM=.023
BEAM_HEIGHT=.020
SITE=np.array([.350,0.,0.])
DENSITY=650.
COLORS={'post_A':[.10,.56,.48],'post_B':[.16,.43,.73],'post_C':[.53,.36,.73],
        'beam_AB':[.94,.62,.17],'beam_BC':[.87,.33,.27],'beam_CA':[.78,.47,.16],
        'base':[.65,.69,.73]}
NAMES={'post_A':'立柱 A','post_B':'立柱 B','post_C':'立柱 C',
       'beam_AB':'横梁 AB','beam_BC':'横梁 BC','beam_CA':'横梁 CA','base':'固定底座'}


def pocket_plate():
    """Boundary of solid rectilinear cells: no booleans, no T-junctions."""
    holes=[]
    for key,(x,y) in VERTICES.items():
        sx,sy,_=POST_SIZE[key]
        holes.append((x-sx/2-CLEARANCE,x+sx/2+CLEARANCE,
                      y-sy/2-CLEARANCE,y+sy/2+CLEARANCE))
    xs=sorted({-BASE_SIZE[0]/2,BASE_SIZE[0]/2,*[v for h in holes for v in h[:2]]})
    ys=sorted({-BASE_SIZE[1]/2,BASE_SIZE[1]/2,*[v for h in holes for v in h[2:]]})
    zs=[0.,POCKET_FLOOR,BASE_SIZE[2]]
    solid=set()
    for i,j,k in itertools.product(range(len(xs)-1),range(len(ys)-1),range(2)):
        x=(xs[i]+xs[i+1])/2;y=(ys[j]+ys[j+1])/2
        if k==0 or not any(a<x<b and c<y<d for a,b,c,d in holes):
            solid.add((i,j,k))
    vertices=[];faces=[];lookup={}
    def quad(points,normal):
        pts=[np.array(p,float) for p in points]
        if np.dot(np.cross(pts[1]-pts[0],pts[2]-pts[0]),normal)<0:pts.reverse()
        ids=[]
        for p in pts:
            key=tuple(p)
            if key not in lookup:lookup[key]=len(vertices);vertices.append(p)
            ids.append(lookup[key])
        faces.extend(((ids[0],ids[1],ids[2]),(ids[0],ids[2],ids[3])))
    for i,j,k in sorted(solid):
        x0,x1=xs[i:i+2];y0,y1=ys[j:j+2];z0,z1=zs[k:k+2]
        sides=[((-1,0,0),[(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)]),
               ((1,0,0),[(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)]),
               ((0,-1,0),[(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)]),
               ((0,1,0),[(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)]),
               ((0,0,-1),[(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)]),
               ((0,0,1),[(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)])]
        for delta,points in sides:
            if (i+delta[0],j+delta[1],k+delta[2]) not in solid:quad(points,delta)
    return MeshData(np.array(vertices),np.array(faces,dtype=np.int64))


def build_assembly():
    ws=load_workspace()
    parts=[Part('table',box_mesh(*ws.table_size),pose(*ws.table_center),
                mass_kg=40.,com_local_m=np.zeros(3),friction=.5,fixed=True),
           Part('base',pocket_plate(),pose(*SITE),mass_kg=2.,
                com_local_m=np.array([0,0,BASE_SIZE[2]/2]),friction=.5,fixed=True)]
    meta={}
    mating=[]
    for key,(x,y) in VERTICES.items():
        pid='post_'+key;size=POST_SIZE[key]
        goal=pose(*(SITE+np.array([x,y,POCKET_FLOOR+size[2]/2])))
        parts.append(Part(pid,box_mesh(*size),goal,mass_kg=DENSITY*np.prod(size),
                          com_local_m=np.zeros(3),friction=.5))
        meta[pid]=dict(name=NAMES[pid],size_m=list(size),required_supports=['base'],
            insertion_axis_world=[0,0,-1],removal_axis_world=[0,0,1])
        mating.append(MatingRelation('base',pid,'rectangular_blind_pocket',
            dict(side_clearance_m=CLEARANCE,depth_m=POCKET_DEPTH,insertion_axis_world=[0,0,-1])))
    for label,a,b,width in EDGES:
        xy_a=np.array(VERTICES[a]);xy_b=np.array(VERTICES[b]);delta=xy_b-xy_a
        length=np.linalg.norm(delta)-2*END_TRIM
        yaw=math.atan2(delta[1],delta[0]);c,s=math.cos(yaw),math.sin(yaw)
        size=(float(length),width,BEAM_HEIGHT)
        tf=pose(*(SITE+np.r_[(xy_a+xy_b)/2,POCKET_FLOOR+.100+BEAM_HEIGHT/2]))
        tf[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
        pid='beam_'+label
        parts.append(Part(pid,box_mesh(*size),tf,mass_kg=DENSITY*np.prod(size),
                          com_local_m=np.zeros(3),friction=.5))
        meta[pid]=dict(name=NAMES[pid],size_m=list(size),required_supports=['post_'+a,'post_'+b],
            insertion_axis_world=[0,0,-1],removal_axis_world=[0,0,1],goal_yaw_deg=math.degrees(yaw),
            endpoint_trim_m=END_TRIM,connection='gravity-supported planar contact; no latch/fastener')
        for endpoint in (a,b):mating.append(MatingRelation('post_'+endpoint,pid,'planar_support',dict(insertion_axis_world=[0,0,-1])))
    return Assembly(tuple(parts),mating_relations=tuple(mating),provenance=dict(
        task='triangular_frame6',units='metre',six_movable_parts=True,
        fixed_base_not_counted=True,robot_motion_verified=False)),meta,ws


def world_vertices(p):return p.geometry.vertices@p.assembled_tf[:3,:3].T+p.assembled_tf[:3,3]


def validate(assembly,meta):
    import trimesh
    from shapely.geometry import MultiPoint
    from asmcore.planning.sequence import SequenceEvaluator,SequenceConfig
    t0=perf_counter();ids=tuple(meta);by={p.part_id:p for p in assembly.parts}
    mesh_checks={}
    for p in assembly.parts:
        mesh=trimesh.Trimesh(p.geometry.vertices,p.geometry.faces,process=False)
        assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume>0,p.part_id
        mesh_checks[p.part_id]=dict(watertight=True,winding_consistent=True,volume_m3=float(mesh.volume),
            vertices=len(mesh.vertices),faces=len(mesh.faces))
    v=world_vertices(by['base']);table=by['table'];tv=world_vertices(table)
    assert np.all(v.min(0)[:2]>=tv.min(0)[:2]) and np.all(v.max(0)[:2]<=tv.max(0)[:2])
    beams=[p for p in assembly.parts if p.part_id.startswith('beam_')]
    beam_gap={a.part_id+'|'+b.part_id:float(MultiPoint(world_vertices(a)[:,:2]).convex_hull.distance(
        MultiPoint(world_vertices(b)[:,:2]).convex_hull)) for a,b in itertools.combinations(beams,2)}
    assert min(beam_gap.values())>.0005,beam_gap
    evaluator=SequenceEvaluator(assembly,SequenceConfig(max_auxiliary_resources=0))
    fixed={p.part_id:p.assembled_tf for p in assembly.parts if p.fixed}
    statuses={};support_mismatches=[]
    for bits in range(1<<len(ids)):
        present=tuple(p for i,p in enumerate(ids) if bits&(1<<i))
        state=AssemblyState(dict(fixed,**{p:by[p].assembled_tf for p in present}))
        result=evaluator.equilibrium(state,())
        feasible=result.status=='feasible'
        expected=all(all(q in present or q=='base' for q in meta[p]['required_supports']) for p in present)
        if feasible!=expected:support_mismatches.append(dict(parts=present,expected=expected,status=result.status))
        statuses[bits]=dict(parts=present,status=result.status,nominal_static_feasible=feasible)
        if bits%16==0:print('static subsets',bits,'/64',flush=True)
    sequences=[]
    for order in itertools.permutations(ids):
        bits=0
        for p in order:
            bits|=1<<ids.index(p)
            if not statuses[bits]['nominal_static_feasible']:break
        else:sequences.append(order)
    assert statuses[(1<<len(ids))-1]['nominal_static_feasible'],'Final assembly not balanced'
    assert len(sequences)>2,'No branching improvement over Bridge'
    # Check object-only +Z removal on a representative order for each possible last beam.
    from asmcore.motion.part_motion import MotionConfig,plan_removal
    motions={};example_orders=[]
    for last in [p.part_id for p in beams]:
        order=next(o for o in sequences if o[-1]==last)
        example_orders.append(order)
        present=[]
        for p in order:
            present.append(p);key='|'.join(sorted(present))+'>'+p
            if key in motions:continue
            state=AssemblyState(dict(fixed,**{q:by[q].assembled_tf for q in present}))
            outcome=plan_removal(assembly,state,p,directions=np.array([[0.,0.,1.]]),
                config=MotionConfig(search_se3=False,max_candidates=1,clearance_m=.0001,time_limit_s=12.))
            motions[key]=dict(status=outcome.status,scope='object-only removal, no gripper or arm',
                sampled_pose_count=len(outcome.poses),diagnostics=to_dict(outcome.diagnostics))
            print('object removal',key,outcome.status,flush=True)
    return dict(mesh_checks=mesh_checks,beam_pair_clearances_m=beam_gap,
        subset_count=64,subset_results=statuses,nominal_stable_subset_count=sum(x['nominal_static_feasible'] for x in statuses.values()),
        support_rule_mismatches=support_mismatches,nominal_static_order_count=len(sequences),
        nominal_static_orders=sequences,example_orders=example_orders,object_removal_checks=motions,
        validation_s=perf_counter()-t0,full_robot_motion_verified=False,
        G_force_closure_verified=False,A_direction_space_computed=False,
        S_directional_robustness_computed=False,safe_to_execute=False)


def example_layout(assembly,meta,ws):
    from shapely.geometry import MultiPoint
    ids=tuple(meta);by={p.part_id:p for p in assembly.parts}
    places={}
    for p,xy in zip(ids,[(-.005,-.22),(.095,-.22),(.195,-.22),
                         (.060,-.075),(.060,.055),(.060,.195)]):
        T=np.eye(4);T[:3,3]=[*xy,meta[p]['size_m'][2]/2]
        places[p]=T
    polys={}
    table_lo=ws.table_center[:2]-ws.table_size[:2]/2
    table_hi=ws.table_center[:2]+ws.table_size[:2]/2
    for p,T in places.items():
        v=by[p].geometry.vertices@T[:3,:3].T+T[:3,3]
        assert np.all(v.min(0)[:2]>=table_lo-1e-12) and np.all(v.max(0)[:2]<=table_hi+1e-12),(p,'outside table')
        polys[p]=MultiPoint(v[:,:2]).convex_hull
    polys['base']=MultiPoint(world_vertices(by['base'])[:,:2]).convex_hull
    gaps={a+'|'+b:float(polys[a].distance(polys[b])) for a,b in itertools.combinations(polys,2)}
    assert min(gaps.values())>=.010-1e-9,gaps
    return places,gaps


def write_preview(assembly,meta,validation,layout,output):
    import trimesh
    from jbls.scene import _body
    report=dict(candidates=[dict(order=o,layout={p:T.tolist() for p,T in layout.items()},cost_m=0.,segments=[])
                    for o in validation['example_orders']],
        table_center=[.2335,0,-.01],table_size=[.6,.6,.02],site=SITE.tolist(),
        goal_poses={p.part_id:p.assembled_tf.tolist() for p in assembly.parts if not p.fixed},
        sizes={p:meta[p]['size_m'] for p in meta},
        assembly_parts={'base':dict(size_m=BASE_SIZE)},parameters={},
        scene_visual=dict(meshes=[_body(p.part_id,'fixed' if p.fixed else 'part',
            trimesh.Trimesh(p.geometry.vertices,p.geometry.faces,process=False),
            COLORS.get(p.part_id,[.88,.89,.91]),p.assembled_tf) for p in assembly.parts],
            note='6个可动件＋固定底座。页面初始摆放为几何无重叠的手工示意，顺序通过名义静力检查；没有机器人IK、全臂避障或连续轨迹验证。'))
    template=(ROOT/'jbls/scene_viewer.html').read_text(encoding='utf-8')
    template=template.replace('单桌场景 · 联合装配搜索','六零件 · 三角支撑框架').replace('JBLS / BRIDGE ASSEMBLY','CASE / TRIANGULAR FRAME 6')
    template=template.replace('sealp 洞洞桌 / Panthera 单臂 / Bridge 初始布局与完成结构','3 根立柱＋3 根横梁 / 固定底座 / 60 cm 单桌')
    template=template.replace('场景与离散安装状态展示','模型与名义静力顺序示意').replace('搜索候选','静力合格顺序示例')
    template=template.replace('欧氏距离总代价 · 不代表执行时间','布局为手工几何示意，未运行联合优化')
    template=template.replace("['机械臂','Panthera · 单臂']","['目标机械臂','Panthera（本案例未验证）']").replace("['候选数量',candidates.length]","['顺序示例数量',candidates.length]")
    template=template.replace('</style>','.bottom{grid-template-columns:1fr}.bottom>.card:nth-child(2){display:none}.plan-legend{display:none}</style>')
    custom='''\nObject.assign(displayNames, __NAMES__);Object.assign(partColors,__COLORS__);
const updateOriginal=update;
update=function(){updateOriginal();$('metric').innerHTML='6 <small>可动零件</small>';
$('robot-state').textContent='机器人尚未接入本案例';
$('robot-explanation').textContent='STL模型与名义静力顺序；未验证抓取、IK和机器人连续运动。';
$('step-note').textContent='手工初始布局＋名义静力合格顺序；切换直接改变位姿，不是运动轨迹。';
$('summary').textContent='名义稳定子集 __STATES__/64；名义稳定顺序 __ORDERS__/720。固定底座不计入6个可动件。';};
for(let i=0;i<candidates.length;i++)$('candidate').options[i].text='顺序示例 '+(i+1)+' · 最后安装 '+nameOf(candidates[i].order.at(-1));
$('routes').checked=false;installedCount=6;update();if(renderer)renderer.preset('table');
'''.replace('__NAMES__',json.dumps(NAMES,ensure_ascii=False)).replace('__COLORS__',json.dumps({p:'#'+''.join(f'{round(c*255):02x}' for c in rgb) for p,rgb in COLORS.items()})).replace('__STATES__',str(validation['nominal_stable_subset_count'])).replace('__ORDERS__',str(validation['nominal_static_order_count']))
    index=template.rfind('</script>');template=template[:index]+custom+template[index:]
    data=json.dumps(report,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    (output/'viewer.html').write_text(template.replace('__JBLS_REPORT_JSON__',data),encoding='utf-8')
    # Standalone image for immediate inspection; metre coordinates displayed in mm.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    fig=plt.figure(figsize=(12,7),layout='constrained')
    for index,exploded in enumerate((False,True),1):
        ax=fig.add_subplot(1,2,index,projection='3d')
        for p in assembly.parts:
            if p.part_id=='table':continue
            v=world_vertices(p)-SITE
            if exploded and not p.fixed:v[:,2]+=.035 if p.part_id.startswith('post') else .095
            tri=v[p.geometry.faces]
            normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
            normals/=np.linalg.norm(normals,axis=1)[:,None]
            light=np.array([-.4,-.6,1.]);light/=np.linalg.norm(light)
            shade=.72+.28*np.maximum(normals@light,0)
            if p.part_id=='base':
                shade[np.abs(tri[:,:,2].mean(1)-POCKET_FLOOR)<1e-8]*=.78
            colors=np.array(COLORS[p.part_id])[None,:]*shade[:,None]
            eye=np.array([math.cos(math.radians(-65))*math.cos(math.radians(25)),math.sin(math.radians(-65))*math.cos(math.radians(25)),math.sin(math.radians(25))])
            visible=normals@eye>1e-9
            coll=Poly3DCollection(tri[visible]*1000,facecolor=colors[visible],edgecolor='none',alpha=1)
            ax.add_collection3d(coll)
        ax.set(xlim=(-135,135),ylim=(-135,135),zlim=(0,235 if exploded else 145),xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)')
        ax.set_box_aspect((1,1,.85));ax.view_init(elev=25,azim=-65)
        ax.set_title('Exploded geometry' if exploded else 'Assembled geometry',fontsize=15)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=COLORS[p],label=p.replace('post_','Post ').replace('beam_','Beam ')) for p in meta],
               loc='lower center',ncol=6,bbox_to_anchor=(.5,-.04),frameon=False)
    fig.savefig(output/'model_preview.png' ,dpi=170,bbox_inches='tight',pad_inches=.25)
    plt.close(fig)


def main():
    start=perf_counter();assembly,meta,ws=build_assembly()
    assets=HERE/'assets'
    loaded,files=prepare_part_stls(assembly.parts,assets)
    assembly=replace(assembly,parts=tuple(loaded))
    print('STLs exported',len(files),flush=True)
    validation=validate(assembly,meta)
    layout,gaps=example_layout(assembly,meta,ws)
    (HERE/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
    # Reuse the real Panthera-compatible nominal catalogue pipeline; no FC claim.
    from curobo_bridge.runtime import prepare
    prepare()
    from adapters.panthera_gripper import PantheraGripperPort
    gripper=PantheraGripperPort()
    for p in assembly.parts:
        if p.fixed:continue
        records,info=prepare_grasp_file(p,files[p.part_id]['stl'],gripper,assets)
        assert records,p.part_id
        files[p.part_id].update(info)
        print('nominal grasp file',p.part_id,len(records),flush=True)
    task=dict(schema='assembly_benchmark/1',name='triangular_frame6',units='metre',
        fixed_base_position_m=SITE.tolist(),fixed_base_not_counted_in_six=True,
        movable_ids=list(meta),friction=.5,table=ws.metadata(),
        parts={p.part_id:dict(fixed=p.fixed,mass_kg=p.mass_kg,com_local_m=p.com_local_m.tolist(),
            goal_tf=p.assembled_tf.tolist(),**meta.get(p.part_id,{}),**files.get(p.part_id,{})) for p in assembly.parts},
        mating_relations=[dict(part_a=m.part_a,part_b=m.part_b,kind=m.kind,parameters=dict(m.parameters)) for m in assembly.mating_relations],
        initial_layout_example={p:T.tolist() for p,T in layout.items()},
        initial_layout_source='hand-authored nonoverlapping geometry example, NOT optimized',
        initial_layout_pair_clearances_m=gaps,full_robot_motion_verified=False,safe_to_execute=False)
    (HERE/'task.json').write_text(json.dumps(task,ensure_ascii=False,indent=2),encoding='utf-8')
    (HERE/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
    write_preview(assembly,meta,validation,layout,HERE)
    print(json.dumps(dict(states=validation['nominal_stable_subset_count'],orders=validation['nominal_static_order_count'],
        support_mismatches=validation['support_rule_mismatches'],object_checks=validation['object_removal_checks'],wall_s=perf_counter()-start),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    if '--preview-only' in sys.argv:
        assembly,meta,ws=build_assembly()
        validation=json.loads((HERE/'validation.json').read_text(encoding='utf-8'))
        task=json.loads((HERE/'task.json').read_text(encoding='utf-8'))
        layout={p:np.array(T,float) for p,T in task['initial_layout_example'].items()}
        write_preview(assembly,meta,validation,layout,HERE)
    else:
        main()


