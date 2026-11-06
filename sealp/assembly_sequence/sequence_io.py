"""
Assembly Sequence I/O
=====================

Read and write ``AssemblySequence`` objects to/from YAML files.
Also provides a CSV summary exporter.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Union

import yaml  # PyYAML

from .assembly_sequence import AssemblySequence


# ======================================================================
# YAML helpers — custom representers for clean output
# ======================================================================
def _float_representer(dumper: yaml.Dumper, value: float):
    """Avoid scientific notation for small floats."""
    if value != value:  # NaN
        return dumper.represent_scalar("tag:yaml.org,2002:float", ".nan")
    if value == float("inf"):
        return dumper.represent_scalar("tag:yaml.org,2002:float", ".inf")
    if value == float("-inf"):
        return dumper.represent_scalar("tag:yaml.org,2002:float", "-.inf")
    return dumper.represent_scalar(
        "tag:yaml.org,2002:float", f"{value:.6g}"
    )


yaml.add_representer(float, _float_representer)


# ======================================================================
# Public API
# ======================================================================
def save_sequence(sequence: AssemblySequence,
                  filepath: Union[str, Path]) -> None:
    """Serialise an ``AssemblySequence`` to a YAML file.

    Parameters
    ----------
    sequence : AssemblySequence
        The assembly to save.
    filepath : str or Path
        Destination path (will be created / overwritten).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data = sequence.to_dict()
    with open(filepath, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False,
                  sort_keys=False, allow_unicode=True)


def load_sequence(filepath: Union[str, Path]) -> AssemblySequence:
    """Load an ``AssemblySequence`` from a YAML file.

    Parameters
    ----------
    filepath : str or Path
        Path to a YAML file previously created by :func:`save_sequence`.

    Returns
    -------
    AssemblySequence
    """
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return AssemblySequence.from_dict(data)


def export_summary_csv(sequence: AssemblySequence,
                       filepath: Union[str, Path]) -> None:
    """Write a flat CSV summary of the assembly sequence.

    Columns
    -------
    step_id, part_id, part_name, parent_part_id, primitive_type,
    dependencies, assembly_pos, model_path, notes

    Parameters
    ----------
    sequence : AssemblySequence
    filepath : str or Path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step_id", "part_id", "part_name", "parent_part_id",
        "primitive_type", "dependencies", "assembly_pos",
        "model_path", "notes",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for step in sequence.get_execution_order():
            part = sequence.get_part(step.part_id)
            writer.writerow({
                "step_id": step.step_id,
                "part_id": step.part_id,
                "part_name": part.name,
                "parent_part_id": step.parent_part_id,
                "primitive_type": step.primitive_type,
                "dependencies": ";".join(str(d) for d in step.dependencies),
                "assembly_pos": f"[{', '.join(f'{v:.4f}' for v in step.assembly_pos)}]",
                "model_path": part.model_path,
                "notes": step.notes,
            })


def export_summary_text(sequence: AssemblySequence,
                        filepath: Union[str, Path]) -> None:
    """Write a human-readable text summary of the assembly.

    Parameters
    ----------
    sequence : AssemblySequence
    filepath : str or Path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(sequence.summary())
        fh.write("\n")
