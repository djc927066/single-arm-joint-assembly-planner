# SEALP — Sequence-Aware Layout Planner

> **S**equence-**A**ware **L**ayout **P**lanner for dual-arm furniture assembly.

This package provides tools for:
1. **Pick-and-place planning** using configurable robot arms
2. **Assembly sequence definition** (YAML-based format with 3D model references)
3. **Project configuration** (robot type, collision environment, sequence file)
4. **Static collision environment** management
5. *(Forthcoming)* Layout optimization under sequential feasibility constraints

## Folder Structure

```
sealp/
├── __init__.py
├── README.md
├── config/
│   ├── __init__.py
│   ├── sealp_config.py       # SEALPConfig dataclass + YAML loader + robot registry
│   ├── setup.py               # setup_from_config() facade
│   ├── sample_config.yaml     # Sample configuration file
│   └── demo_config.py         # Runnable demo: load config → visualize
├── colliders/
│   ├── __init__.py
│   ├── obstacle_manager.py    # Named obstacle dictionary
│   ├── static_environment.py  # Config-driven static obstacles
│   └── collision_world.py     # Unified: static env + runtime obstacles
├── examples/
│   ├── __init__.py
│   ├── grasp/
│   │   ├── __init__.py
│   │   ├── planning.py        # Antipodal grasp planning with Piper gripper
│   │   ├── filtering.py       # Filter grasps by orientation/position/width
│   │   └── visualization.py   # Visualize & analyze saved grasps
│   └── motion/
│       ├── __init__.py
│       ├── piper_pnp.py       # PiperPickAndPlace / DualPiperPickAndPlace
│       ├── pnp_demo.py        # Single-arm pick-and-place demo
│       └── dual_arm_pnp.py    # Dual-arm concurrent pick-and-place demo
├── editor/
│   └── ...
└── assembly_sequence/
    ├── __init__.py
    ├── assembly_part.py       # AssemblyPart dataclass
    ├── assembly_step.py       # AssemblyStep dataclass
    ├── assembly_sequence.py   # AssemblySequence container + DAG validation
    ├── sequence_io.py         # YAML / CSV / text I/O
    ├── sequence_generator.py  # Builder-pattern sequence construction
    └── demo_sequence.py       # Runnable demo: build, save, load a sequence
```

## Quick Start

### 1. Project Configuration (Recommended)

The fastest way to set up a SEALP project is via a YAML config file:

```python
from sealp.config import setup_from_config

setup = setup_from_config("my_project/sealp_config.yaml")
robot    = setup.robot             # instantiated robot (e.g. PiperSglArm)
sequence = setup.sequence          # loaded & validated AssemblySequence
world    = setup.collision_world   # CollisionWorld (static env + runtime)
```

**Config YAML format:**

```yaml
project_name: "MyAssembly"

robot:
  type: "piper"                    # piper | cobotta | nova2_wg | xarmlite6_wg
  pos: [0, 0, 0]
  rotmat: [[1,0,0],[0,1,0],[0,0,1]]
  enable_cc: true

assembly_sequence:
  file: "sequences/my_table.yaml"  # relative to config dir
  validate_on_load: true

environment:
  obstacles:
    - name: "work_table"
      type: "box"
      extent: [0.8, 1.2, 0.02]
      pos: [0.4, 0, 0]
      rgba: [0.6, 0.5, 0.4, 0.8]
    - name: "fixture"
      type: "stl"
      file: "meshes/fixture.stl"
      pos: [0.3, 0, 0.02]
```

**Registering custom robot types:**

```python
from sealp.config import ROBOT_REGISTRY

def _make_my_robot(pos, rotmat, name, enable_cc):
    from my_package import MyRobot
    return MyRobot(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)

ROBOT_REGISTRY["my_robot"] = _make_my_robot
# Now you can use type: "my_robot" in config YAML
```

### 2. Grasp Planning

```python
from sealp.examples.grasp.planning import plan_grasps
import wrs.modeling.collision_model as mcm

# Plan grasps on an object
obj = mcm.CollisionModel("my_object.stl")
grasp_collection, gripper = plan_grasps(obj, max_samples=100)
grasp_collection.save_to_disk("my_object_grasps.pickle")

# Filter grasps
from sealp.examples.grasp.filtering import filter_by_orientation
filtered = filter_by_orientation(grasp_collection, direction="down")
```

### 3. Collision Environment

```python
from sealp.colliders import CollisionWorld

# From config-driven definitions
world = CollisionWorld(obstacle_defs=[
    {"name": "table", "type": "box", "extent": [0.8, 1.2, 0.02],
     "pos": [0.4, 0, 0]},
])

# Add runtime obstacles
world.user_obstacles.add_box("block", extent=[0.1, 0.1, 0.1],
                             pos=[0.5, 0, 0.05])

# Use with collision detection
robot.is_collided(obstacle_list=world.obstacle_list)

# Visualize
world.show(base, robot=robot, toggle_cdprim=False)
```

### 4. Assembly Sequence (YAML)

**Create and save:**

```python
from sealp.assembly_sequence import SequenceGenerator, save_sequence

seq = (
    SequenceGenerator("MyFurniture")
    .add_part("top", "Table Top", "top.stl", assembly_pos=[0, 0, 0.4])
    .add_part("leg", "Leg", "leg.stl", assembly_pos=[0.1, 0, 0])
    .add_step(0, "top", parent="fixture")
    .add_step(1, "leg", parent="top", deps=[0])
    .build()
)

save_sequence(seq, "my_assembly.yaml")
```

**Load and inspect:**

```python
from sealp.assembly_sequence import load_sequence

seq = load_sequence("my_assembly.yaml")
print(seq.summary())

for step in seq.get_execution_order():
    print(f"Step {step.step_id}: assemble {step.part_id}")
```

### 5. Run Demos

```bash
# Config demo (loads config → creates robot → visualizes environment)
python -m sealp.config.demo_config

# Grasp planning demo (plan + save + visualize grasps)
python -m sealp.examples.grasp.planning

# Grasp filtering demo (filter by orientation/position)
python -m sealp.examples.grasp.filtering

# Grasp visualization demo (statistics + 3D view)
python -m sealp.examples.grasp.visualization

# Single-arm pick-and-place demo
python -m sealp.examples.motion.pnp_demo

# Dual-arm pick-and-place demo
python -m sealp.examples.motion.dual_arm_pnp

# Assembly sequence demo (terminal output + generates YAML)
python -m sealp.assembly_sequence.demo_sequence
```

## Assembly Sequence Validation

`AssemblySequence.validate()` performs the following checks:

1. Every step references a known `part_id`
2. Every dependency `step_id` exists
3. Dependency graph is acyclic (DAG)
4. No `part_id` appears in more than one step
5. Every part's `model_path` points to an existing file

With `strict=True` (default), validation raises `ValueError` on the first error found.

## Dependencies

- [WRS](https://github.com/wanweiwei07/wrs) — Robot planning & control framework
- [PyYAML](https://pypi.org/project/PyYAML/) — `pip install pyyaml`
- Panda3D (bundled with WRS)
