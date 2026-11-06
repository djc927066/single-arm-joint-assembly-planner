"""Isolated experiment: generic JBLS on triangular_frame6 with discrete parking grid."""
import json, math, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from curobo_bridge.runtime import prepare
prepare()  # Keep Torch before numpy/WRS imports on this Windows environment.
import numpy as np
from adapters.panthera import PantheraArmProvider
from jbls.search import Placement, joint_search

task=json.loads((HERE/'task.json').read_text(encoding='utf-8'))
val=json.loads((HERE/'validation.json').read_text(encoding='utf-8'))
ids=task['movable_ids']; pmeta=task['parts']; goals={p:np.array(pmeta[p]['goal_tf'])[:3,3] for p in ids}
# Match the current SEALP single-table workspace (table top z=0).
tab=task['table']; lo=np.array(tab['table_center'][:2])-np.array(tab['table_size'][:2])/2
hi=np.array(tab['table_center'][:2])+np.array(tab['table_size'][:2])/2
base_center=np.array(task['fixed_base_position_m'][:2]); base_half=np.array([.12,.12])
# A coarse 70-mm occupancy grid. Every part has the same sampled XY locations,
# filtered by its own footprint, table edge, and fixed-base bounding rectangle.
xs=np.arange(-.035,.491,.07); ys=np.arange(-.245,.246,.07)
domains={}; dimensions={}
for pid in ids:
    sx,sy,sz=pmeta[pid]['size_m']; dimensions[pid]=(sx,sy,sz)
    places=[]
    for x in xs:
      for y in ys:
        if x-sx/2<lo[0]+.002 or x+sx/2>hi[0]-.002 or y-sy/2<lo[1]+.002 or y+sy/2>hi[1]-.002: continue
        # Conservative AABB exclusion around the fixed base; polygonal pockets
        # are intentionally excluded from this staging-space test.
        if abs(x-base_center[0]) < (sx+.24)/2 and abs(y-base_center[1]) < (sy+.24)/2: continue
        tf=np.eye(4);tf[:3,3]=[x,y,sz/2]
        places.append(Placement(f'{x:.3f},{y:.3f},yaw0',tf))
    domains[pid]=places

stable={frozenset(r['parts']):bool(r['nominal_static_feasible']) for r in val['subset_results'].values()}
# Forward and reverse memoized state-graph traversal, counting candidate
# transition checks. This isolates what reversal alone buys under static support.
def walk(reverse):
    start=frozenset(ids) if reverse else frozenset()
    seen={start};stack=[start];tests=0;rejected=0;edges=0;layers={}
    while stack:
        state=stack.pop();depth=len(state) if reverse else len(ids)-len(state)
        if (reverse and not state) or (not reverse and len(state)==len(ids)): continue
        for p in ids:
            if reverse:
                if p not in state: continue
                nxt=state-{p}
            else:
                if p in state: continue
                nxt=state|{p}
            tests+=1
            if not stable.get(nxt,False): rejected+=1;continue
            edges+=1
            layers[depth]=layers.get(depth,0)+1
            if nxt not in seen:seen.add(nxt);stack.append(nxt)
    return dict(reachable_states=len(seen),transition_tests=tests,stable_transitions=edges,
                pruned_transitions=rejected,states_by_depth={str(k):v for k,v in sorted(layers.items())})

# Static state predicate is equivalent to requiring the candidate set and its
# post-removal remainder to be feasible. This is the existing reverse callback.
def structure(remaining,pid):
    if not stable.get(frozenset(remaining),False):return None,'current_unstable'
    if not stable.get(frozenset(remaining-{pid}),False):return None,'remainder_unstable'
    return {'static':'feasible'},''

def rect(place,pid):
    sx,sy,sz=dimensions[pid]; x,y=place.position[:2]
    return (x-sx/2,x+sx/2,y-sy/2,y+sy/2)
def overlaps(a,b,gap=.001):
    return min(a[1],b[1])-max(a[0],b[0])>gap and min(a[3],b[3])-max(a[2],b[2])>gap

def make_placement_check(use_collision=True,counters=None):
    def check(parent,remaining,pid,place,structure_record):
        if counters is not None:counters['calls']+=1
        r=rect(place,pid)
        if use_collision:
            if overlaps(r,(base_center[0]-.12,base_center[0]+.12,base_center[1]-.12,base_center[1]+.12)):
                if counters is not None:counters['base_collision']+=1
                return None,'fixed_base_collision'
            for other,old in parent.layout.items():
                if overlaps(r,rect(old,other)):
                    if counters is not None:counters['part_collision']+=1
                    return None,'staging_part_collision'
        return {'collision_free_coarse_aabb':bool(use_collision)},''
    return check

home=PantheraArmProvider(base_pos=np.array(tab['robot_base_pos']),base_rotmat=np.array(tab['robot_base_rotmat'])).home_tcp_world()[:3,3]
records=[]
for width in (1,4,8,16):
  for collision in (True,False):
    stats={'calls':0,'base_collision':0,'part_collision':0}
    found,meta=joint_search(ids,domains,goals,home,structure,make_placement_check(collision,stats),beam_width=width)
    records.append(dict(beam_width=width,coarse_staging_collision_filter=collision,
        status=meta['status'],candidate_domains={p:len(domains[p]) for p in ids},
        layers=meta['layers'],wall_s=meta['wall_s'],placement_checks=stats,
        solutions=[dict(order=list(n.order),cost_proxy_m=n.cost,
            layout={p:{'position_m':n.layout[p].position.tolist(),'key':n.layout[p].key}
                    for p in n.order},coarse_geometry_certified=all(n.steps[p]['collision_free_coarse_aabb'] for p in n.order))
          for n in found[:min(3,len(found))]],full_robot_motion_verified=False))

# Grid-resolution ablation with physical XY spacing/part-collision checks active.
grid_runs=[]
for step in (.10,.07,.05):
  xs0=np.arange(-.035,.491,step);ys0=np.arange(-.245,.246,step);domains0={}
  for pid in ids:
    sx,sy,sz=dimensions[pid];poses=[]
    for x in xs0:
      for y in ys0:
        if x-sx/2<lo[0]+.002 or x+sx/2>hi[0]-.002 or y-sy/2<lo[1]+.002 or y+sy/2>hi[1]-.002:continue
        if abs(x-base_center[0])<(sx+.24)/2 and abs(y-base_center[1])<(sy+.24)/2:continue
        tf=np.eye(4);tf[:3,3]=[x,y,sz/2];poses.append(Placement(f'{x:.3f},{y:.3f},yaw0',tf))
    domains0[pid]=poses
  domains_saved=domains;domains=domains0
  for width in (4,8,16):
    c={'calls':0,'base_collision':0,'part_collision':0}
    found,m=joint_search(ids,domains,goals,home,structure,make_placement_check(True,c),beam_width=width)
    best=found[0] if found else None
    grid_runs.append(dict(grid_spacing_m=step,beam_width=width,
       domain_counts={p:len(domains[p]) for p in ids},status=m['status'],
       wall_s=m['wall_s'],placement_checks=c,
       best_cost_proxy_m=None if best is None else best.cost,
       best_order=None if best is None else list(best.order),
       best_layout={} if best is None else {p:best.layout[p].position.tolist() for p in best.order},
       coarse_geometry_certified=best is not None and all(best.steps[p]['collision_free_coarse_aabb'] for p in best.order)))
  domains=domains_saved

# Crowded parking-region ablation: incremental domain filtering / Hall matching.
# Hall here is a domain-feasibility bound; it does not include motion corridors.
step=.05;xs0=np.arange(-.035,.251,step);ys0=np.arange(-.245,-.044,step);crowded={}
for pid in ids:
  sx,sy,sz=dimensions[pid];arr=[]
  for x in xs0:
    for y in ys0:
      if x-sx/2<lo[0]+.002 or x+sx/2>hi[0]-.002 or y-sy/2<lo[1]+.002 or y+sy/2>hi[1]-.002:continue
      if abs(x-base_center[0])<(sx+.24)/2 and abs(y-base_center[1])<(sy+.24)/2:continue
      tf=np.eye(4);tf[:3,3]=[x,y,sz/2];arr.append(Placement(f'{x:.3f},{y:.3f},yaw0',tf))
  crowded[pid]=arr
def make_propagating(mode,counter):
  def check(parent,remaining,pid,place,structure_record):
    counter['calls']+=1;fixed=[]
    if overlaps(rect(place,pid),(base_center[0]-.12,base_center[0]+.12,base_center[1]-.12,base_center[1]+.12)):
      counter['fixed_base']+=1;return None,'fixed_base_collision'
    fixed.append((pid,rect(place,pid)))
    for q,old in parent.layout.items():
      if overlaps(fixed[0][1],rect(old,q)):
        counter['selected_part']+=1;return None,'staging_part_collision'
      fixed.append((q,rect(old,q)))
    rem=(set(remaining)-{pid})-set(parent.layout)
    if mode in ('forward_check','hall') and rem:
      free={};counter['domain_scans']+=1
      for q in rem:
        free[q]=[c.key for c in crowded[q]
          if not any(overlaps(rect(c,q),obs) for _,obs in fixed)]
        counter['domain_candidates_examined']+=len(crowded[q])
      empty=[q for q,choices in free.items() if not choices]
      if empty:
        counter['domain_wipeout']+=1;return None,'future_domain_empty:'+','.join(sorted(empty))
      if mode=='hall':
        owner={}
        def augment(q,seen):
          for cell in free[q]:
            if cell in seen:continue
            seen.add(cell)
            if cell not in owner or augment(owner[cell],seen):owner[cell]=q;return True
          return False
        for q in sorted(rem,key=lambda x:len(free[x])):
          if not augment(q,set()):
            counter['hall_wipeout']+=1;return None,'future_hall_violation'
    return {'collision_free_coarse_aabb':True,'propagation':mode},''
  return check
crowded_runs=[]
domains_saved=domains;domains=crowded
for mode in ('none','forward_check','hall'):
  for width in (1,4,8):
    cnt={'calls':0,'fixed_base':0,'selected_part':0,'base_collision':0,'part_collision':0,'domain_scans':0,
         'domain_candidates_examined':0,'domain_wipeout':0,'hall_wipeout':0}
    check=make_placement_check(True,cnt) if mode=='none' else make_propagating(mode,cnt)
    found,m=joint_search(ids,domains,goals,home,structure,check,beam_width=width)
    n=found[0] if found else None
    crowded_runs.append(dict(mode=mode,beam_width=width,domain_counts={p:len(domains[p]) for p in ids},
      status=m['status'],wall_s=m['wall_s'],layers=m['layers'],counters=cnt,
      best_cost_proxy_m=None if n is None else n.cost,
      best_order=None if n is None else list(n.order),
      best_layout={} if n is None else {p:n.layout[p].position.tolist() for p in n.order}))
domains=domains_saved

report=dict(experiment='triangular_frame6_joint_search_coarse_grid_v1',units='metre',
  model='6 movable parts plus fixed pocketed base',
  limits=['70-mm XY grid, yaw fixed at 0 for staging','AABB checks only; no mesh collision or robot/gripper motion',
          'Euclidean objective is a proxy, not cuRobo/RRT duration','beam truncation does not prove global optimum'],
  candidate_counts={p:len(domains[p]) for p in ids},
  static_reversal_comparison={'forward':walk(False),'reverse':walk(True)},
  beam_ablation=records,grid_resolution_ablation=grid_runs,crowded_domain_propagation=crowded_runs)
(HERE/'experiment_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
for row in records:
 print('width',row['beam_width'],'collision',row['coarse_staging_collision_filter'],'status',row['status'],
       'calls',row['placement_checks']['calls'],'wall',round(row['wall_s'],3),
       'topcost', [round(x['cost_proxy_m'],3) for x in row['solutions']],flush=True)
print('saved',HERE/'experiment_report.json',flush=True)


