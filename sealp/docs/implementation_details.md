# SEALP Implementation Details

> Internal documentation for the Sequence-Aware Layout Planner project.

## Project Overview

**SEALP** (Sequence-Aware Layout Planner) is a framework for sequence-aware
workspace layout optimization for dual-arm furniture assembly.

### Core Modules

| Module | Description |
|--------|------------|
| `config/` | YAML project config, robot registry, setup facade |
| `colliders/` | Obstacle manager, static environment, collision world |
| `assembly_sequence/` | Assembly data formats (`.asmdef`, `.tplan`) & I/O |
| `editor/` | Interactive Panda3D-based assembly editor |
| `examples/` | Grasp planning, pick-and-place, and demo scripts |
| `assets/` | 3D models (STL) for assemblies |

---

## 1. Assembly Data Formats (`assembly_sequence/`)

SEALP uses **two file formats** to cleanly separate product definition from task planning:

### 1.1 Assembly Definition — `.asmdef`

**Purpose:** Defines the *product* (what is being assembled).  Contains only
information inherent to the assembly — not how or where a robot executes it.

**File:** `asmdef.py` → `AssemblyDef`, `PartDef`, `StepDef`

```
AssemblyDef
├── format_version            # "1.0"
├── name, description
├── models: Dict[alias, path] # shared model library (deduplicated)
├── symmetry_groups           # interchangeable parts (e.g. 4 identical legs)
├── parts: Dict[id, PartDef]
│   └── part_id, name, model (alias), mass, metadata
└── steps: List[StepDef]      # assembly DAG
    └── step_id, part_id, parent_id, rel_pos, rel_rotmat, deps, notes
```

**Key design decisions:**
- **Relative poses** — `rel_pos`/`rel_rotmat` relative to parent part, not world
- **Model library** — STL paths declared once by alias, referenced by parts
- **Symmetry groups** — enables layout optimizer to permute interchangeable parts
- **No task data** — no staging positions, colors, grasp IDs, primitive types

**API:**
```python
from sealp.assembly_sequence import AssemblyDef, PartDef, StepDef

asm = AssemblyDef.load("chair.asmdef")
asm.parts              # dict[str, PartDef]
asm.models             # dict[str, str] — alias → abs path
asm.symmetry_groups    # dict[str, list[str]]
asm.steps              # list[StepDef]

# Compute absolute world poses from relative
poses = asm.compute_world_poses(fixture_pos=np.array([0, 0, 0]))
# → {"seat": (pos, rotmat), "leg_fl": (pos, rotmat), ...}

asm.save("output.asmdef")
```

### 1.2 Task Plan — `.tplan`

**Purpose:** Task-specific execution plan.  References an `.asmdef` and adds all
data that varies per execution: staging positions, motion primitives, grasp
selections, robot configuration, fixture position.

Typically produced by the **layout optimizer** or configured by hand.

**File:** `tplan.py` → `TaskPlan`, `StagingPose`, `StepParams`, `RobotConfig`

```
TaskPlan
├── format_version            # "1.0"
├── name, description
├── assembly_file             # path to referenced .asmdef
├── fixture_pos, fixture_rotmat  # assembly station world pose
├── robot: RobotConfig
│   └── robot_type, base_pos, base_rotmat, start_conf
├── staging: Dict[part_id, StagingPose]
│   └── part_id, pos, rotmat  # where robot picks each part
└── step_params: Dict[step_id, StepParams]
    └── step_id, primitive, grasp_id, approach_distance,
        depart_distance, approach_direction, speed_factor
```

**Key design decisions:**
- **References `.asmdef`** — assembly link, loaded lazily via `plan.assembly`
- **Fixture pose** — enables placing the same assembly at different workspace locations
- **Per-step params** — primitive type and grasp ID assigned per step (auto or explicit)
- **Robot config** — base position, type, and initial configuration
- **Defaults omitted** — only non-default values are serialized (compact output)

**API:**
```python
from sealp.assembly_sequence import TaskPlan, StepParams

plan = TaskPlan.load("chair_plan.tplan")
plan.assembly              # linked AssemblyDef (lazy-loaded)
plan.fixture_pos           # assembly station world position
plan.staging               # dict[part_id → StagingPose]
plan.step_params           # dict[step_id → StepParams]
plan.robot                 # RobotConfig

# Compute assembly world poses using fixture position
poses = plan.compute_assembly_world_poses()

plan.save("output.tplan")
```

### 1.3 Format Comparison

| Data | `.asmdef` | `.tplan` | Legacy `.yaml` |
|------|:---------:|:--------:|:--------------:|
| Part identity & model | ✅ | — (ref) | ✅ |
| Assembly structure (DAG) | ✅ | — (ref) | ✅ |
| Assembly poses (relative) | ✅ | — | — |
| Assembly poses (absolute) | computed | computed | ✅ |
| Model library (deduplicated) | ✅ | — | — |
| Symmetry groups | ✅ | — | — |
| Staging positions | — | ✅ | ✅ (`init_pos`) |
| Primitive type | — | ✅ | ✅ |
| Grasp ID | — | ✅ | ✅ |
| Robot config | — | ✅ | — |
| Fixture world pose | — | ✅ | — |
| Color (visualization) | — | — | ✅ |

### 1.4 Legacy Format (`.yaml`)

The original `AssemblySequence` / `AssemblyPart` / `AssemblyStep` classes are
retained for backward compatibility.  The editor supports loading both formats.

### 1.5 Validation

`AssemblyDef.validate()` checks:
1. Every step references a known `part_id`
2. Every dependency `step_id` exists
3. The dependency graph is acyclic (DAG — Kahn's algorithm)
4. No `part_id` appears in more than one step
5. Every part's model alias exists in the model library
6. Every symmetry group member exists as a part

---

## 2. Assembly Editor (`editor/`)

### Architecture

```
assembly_editor.py       ← Main application (extends WRS World)
    ├── editor_gui.py    ← DirectGUI widget factories (dark theme)
    ├── transform_handler.py ← Grab/Rotate state machine
    ├── part_manager.py  ← 3D scene part management
    └── run_editor.py    ← Entry point
```

### Supported Formats

- **`.asmdef`** (preferred) — loads via bridge, computes world poses,
  saves back with re-derived relative poses
- **`.yaml`** (legacy) — direct load/save via `sequence_io`
- Auto-detected by file extension

### Key Bindings

| Key | Action |
|-----|--------|
| `G` | Grab/position mode |
| `R` | Rotate mode |
| `X`/`Y`/`Z` | Constrain to axis |
| `Escape` | Cancel / deselect |
| `Delete` | Remove part |
| `Ctrl+S` | Save |
| `Ctrl+O` | Load |

### Transform System

`TransformHandler`: state machine with `NONE → GRAB/ROTATE → confirm/cancel`.
- Grab: ray-plane intersection for XY, mouse-Y delta for Z axis
- Rotate: Rodrigues' formula for arbitrary axis rotation

---

## 3. Examples Module (`examples/`)

### `examples/grasp/`
- `planning.py` — antipodal grasp planning with `plan_grasps()`
- `filtering.py` — filter by orientation/position/width
- `visualization.py` — statistics + 3D rendering

### `examples/motion/`
- `piper_pnp.py` — `PiperPickAndPlace` / `DualPiperPickAndPlace`
- `pnp_demo.py` — single-arm FK/IK test
- `dual_arm_pnp.py` — dual-arm concurrent demo

---

## 4. Config & Colliders

### Config (`config/`)
- YAML-based `SEALPConfig` with robot type, environment, and sequence file
- Robot registry (piper, cobotta, nova2_wg, xarmlite6_wg)
- `setup_from_config()` — unified initialization facade

### Colliders (`colliders/`)
- `ObstacleManager` — named obstacle dictionary
- `StaticEnvironment` — config-driven obstacle definitions (box, STL)
- `CollisionWorld` — combines static + runtime obstacles

---

## 5. Piper Robot Migration

Migrated from `wrs-main`:
- **Source:** `d:\code\HTW\wrs-main\wrs\robot_sim\manipulators\piper\`
- **Destination:** `d:\code\layout_sq\wrs\wrs\robot_sim\manipulators\piper\`
- 6 revolute joints, URDF-based, STL collision meshes

---

## 6. Dependencies

| Package | Purpose |
|---------|---------|
| WRS | Robot planning & control framework (Panda3D, numpy) |
| PyYAML | `.asmdef` / `.tplan` serialization |
| tkinter | File dialogs in the editor (stdlib) |

---

## 7. Entry Points

```bash
# Generate yuanchair assembly files
python -m sealp.assembly_sequence.gen_yuanchair_asmdef   # → yuanchair.asmdef
python -m sealp.assembly_sequence.gen_yuanchair_tplan     # → yuanchair_plan.tplan

# Assembly editor
python -m sealp.editor.run_editor
python -m sealp.editor.run_editor path/to/assembly.asmdef

# Grasp demos
python -m sealp.examples.grasp.planning
python -m sealp.examples.grasp.filtering
python -m sealp.examples.grasp.visualization

# Motion demos
python -m sealp.examples.motion.pnp_demo
python -m sealp.examples.motion.dual_arm_pnp

# Legacy YAML demo
python -m sealp.assembly_sequence.demo_sequence
```

---

## 8. Pipeline Overview (Future)

```
                    ┌────────────────────┐
                    │   chair.asmdef     │  ← product definition
                    └────────┬───────────┘
                             │
                    ┌────────▼───────────┐
                    │  Layout Optimizer   │  ← constrained optimization
                    │  (Phase 3)         │
                    └────────┬───────────┘
                             │
                    ┌────────▼───────────┐
                    │  chair_plan.tplan  │  ← optimizer output
                    └────────┬───────────┘
                             │
                    ┌────────▼───────────┐
                    │ Sequence Executor  │  ← motion planning + execution
                    │  (Phase 2)         │
                    └────────┬───────────┘
                             │
                    ┌────────▼───────────┐
                    │  Robot Execution   │
                    └────────────────────┘
```

The `.asmdef` defines *what* to build.  The layout optimizer produces a `.tplan`
defining *where* (staging, fixture, robot base) and *how* (primitives, grasps).
The executor reads both files and generates motion plans.
