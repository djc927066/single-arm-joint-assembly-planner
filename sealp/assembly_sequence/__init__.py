"""
Assembly sequence definition, serialization, and generation.

Provides data structures for parts and steps, YAML I/O,
a builder-pattern generator, and validation utilities.

Formats:
    .asmdef  — Product-level assembly definition (models, parts, relative poses)
    .tplan   — Task plan: staging positions, primitives, grasps (future)
"""

# ── Shared types ─────────────────────────────────────────────
from .primitives import Primitive

# ── .asmdef format (preferred) ───────────────────────────────
from .asmdef import AssemblyDef, PartDef, StepDef

# ── Task plan .tplan format ──────────────────────────────────
from .tplan import TaskPlan, StagingPose, StepParams, RobotConfig

# ── Legacy format (backward compat) ─────────────────────────
from .assembly_part import AssemblyPart
from .assembly_step import AssemblyStep
from .assembly_sequence import AssemblySequence
from .sequence_io import save_sequence, load_sequence, export_summary_csv, export_summary_text
from .sequence_generator import SequenceGenerator

