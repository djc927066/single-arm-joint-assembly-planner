"""
SEALP Configuration
====================
Dataclasses and loader for the SEALP YAML configuration file.

The config specifies:
- **Robot**: type, base pose, collision checking flag.
- **Assembly sequence**: path to a sequence YAML file, validation flags.
- **Environment**: list of static collision obstacles.

Robot Registry
--------------
The ``ROBOT_REGISTRY`` dict maps string type names to factory functions
that accept ``(pos, rotmat, name, enable_cc)`` and return a robot
instance.  To add a new robot type, register it before calling
``load_config`` / ``setup_from_config``::

    from sealp.config import ROBOT_REGISTRY

    def _make_my_robot(pos, rotmat, name, enable_cc):
        from my_package import MyRobot
        return MyRobot(pos=pos, rotmat=rotmat, name=name,
                       enable_cc=enable_cc)

    ROBOT_REGISTRY["my_robot"] = _make_my_robot

YAML Format
-----------
.. code-block:: yaml

    project_name: "FurnitureAssembly_v1"

    robot:
      type: "panthera_ht"   # piper / cobotta / nova2_wg / xarmlite6_wg / panthera_ht
      pos: [0, 0, 0]
      rotmat: [[1,0,0],[0,1,0],[0,0,1]]
      enable_cc: true

    assembly_sequence:
      file: "sequences/my_table.yaml"
      validate_on_load: true

    environment:
      obstacles:
        - name: "table_surface"
          type: "box"
          extent: [0.8, 1.2, 0.02]
          pos: [0.4, 0, 0]
          rgba: [0.6, 0.5, 0.4, 0.8]
        - name: "fixture"
          type: "stl"
          file: "meshes/fixture.stl"
          pos: [0.3, 0, 0.02]
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import yaml


# ======================================================================
# Robot registry — maps type names to factory callables
# ======================================================================
def _make_piper(pos, rotmat, name, enable_cc):
    """Factory for PiperSglArm."""
    from wrs.robot_sim.robots.piper.piper_single_arm import PiperSglArm
    return PiperSglArm(pos=pos, rotmat=rotmat, name=name,
                       enable_cc=enable_cc)


def _make_cobotta(pos, rotmat, name, enable_cc):
    """Factory for Cobotta."""
    from wrs.robot_sim.robots.cobotta.cobotta import Cobotta
    return Cobotta(pos=pos, rotmat=rotmat, name=name,
                   enable_cc=enable_cc)


def _make_nova2_wg(pos, rotmat, name, enable_cc):
    """Factory for Nova2WG."""
    from wrs.robot_sim.robots.nova2_wg.nova2wg3 import Nova2WG3
    return Nova2WG3(pos=pos, rotmat=rotmat, name=name,
                   enable_cc=enable_cc)


def _make_xarmlite6_wg(pos, rotmat, name, enable_cc):
    """Factory for XArmLite6WG."""
    from wrs.robot_sim.robots.xarmlite6_wg.xarmlite6_wg import XArmLite6WG
    return XArmLite6WG(pos=pos, rotmat=rotmat, name=name,
                       enable_cc=enable_cc)


def _make_panthera_ht(pos, rotmat, name, enable_cc):
    """Factory for PantheraHTSglArm (Panthera-HT 6-DoF + Panthera 双指夹爪)."""
    from wrs.robot_sim.robots.robot_panthera_ht.panthera_ht import PantheraHTSglArm
    return PantheraHTSglArm(pos=pos, rotmat=rotmat, name=name,
                            enable_cc=enable_cc)


ROBOT_REGISTRY: Dict[str, Callable] = {
    "piper": _make_piper,
    "cobotta": _make_cobotta,
    "nova2_wg": _make_nova2_wg,
    "xarmlite6_wg": _make_xarmlite6_wg,
    "panthera_ht": _make_panthera_ht,
}
"""Registry of known robot types.

Keys are the ``type`` strings used in YAML configs.  Values are
factory callables ``(pos, rotmat, name, enable_cc) -> robot``.

Users can add custom robots at runtime::

    ROBOT_REGISTRY["my_robot"] = my_factory_fn
"""


# ======================================================================
# Dataclasses
# ======================================================================
@dataclass
class RobotConfig:
    """Robot configuration parsed from YAML.

    Attributes
    ----------
    type : str
        Robot type key (must exist in ``ROBOT_REGISTRY``).
    pos : np.ndarray
        Base position ``[x, y, z]``.
    rotmat : np.ndarray
        Base orientation (3×3 rotation matrix).
    enable_cc : bool
        Enable self-collision checking.
    name : str
        Instance name for the robot.
    """
    type: str = "piper"
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotmat: np.ndarray = field(default_factory=lambda: np.eye(3))
    enable_cc: bool = True
    name: str = "sealp_robot"


@dataclass
class SequenceConfig:
    """Assembly sequence configuration.

    Attributes
    ----------
    file : str
        Path to the assembly sequence YAML file.
        Relative paths are resolved against the config file directory.
    validate_on_load : bool
        If ``True``, run ``AssemblySequence.validate(strict=True)``
        immediately after loading the sequence.
    """
    file: str = ""
    validate_on_load: bool = True


@dataclass
class SEALPConfig:
    """Top-level SEALP project configuration.

    Attributes
    ----------
    project_name : str
        Human-readable project name.
    robot : RobotConfig
        Robot configuration.
    assembly_sequence : SequenceConfig
        Assembly sequence configuration.
    obstacle_defs : list of dict
        Raw obstacle definitions for the static environment.
    config_dir : str
        Directory containing the config file (for resolving relative
        paths).
    """
    project_name: str = "unnamed_project"
    robot: RobotConfig = field(default_factory=RobotConfig)
    assembly_sequence: SequenceConfig = field(default_factory=SequenceConfig)
    obstacle_defs: List[dict] = field(default_factory=list)
    config_dir: str = ""


# ======================================================================
# Loader
# ======================================================================
def load_config(filepath: Union[str, Path]) -> SEALPConfig:
    """Load a SEALP configuration from a YAML file.

    Parameters
    ----------
    filepath : str or Path
        Path to the YAML config file.

    Returns
    -------
    SEALPConfig

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the robot type is not in ``ROBOT_REGISTRY``.
    """
    filepath = Path(filepath).resolve()
    if not filepath.is_file():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    config_dir = str(filepath.parent)

    # --- Robot ---
    rbt_data = data.get("robot", {})
    robot_cfg = RobotConfig(
        type=rbt_data.get("type", "piper"),
        pos=np.asarray(rbt_data.get("pos", [0, 0, 0]), dtype=float),
        rotmat=np.asarray(
            rbt_data.get("rotmat", np.eye(3).tolist()), dtype=float
        ),
        enable_cc=rbt_data.get("enable_cc", True),
        name=rbt_data.get("name", "sealp_robot"),
    )

    # Validate that robot type is known
    if robot_cfg.type not in ROBOT_REGISTRY:
        known = ", ".join(sorted(ROBOT_REGISTRY.keys()))
        raise ValueError(
            f"Unknown robot type '{robot_cfg.type}'. "
            f"Known types: {known}. "
            f"Register custom robots via ROBOT_REGISTRY."
        )

    # --- Assembly sequence ---
    seq_data = data.get("assembly_sequence", {})
    seq_file = seq_data.get("file", "")
    # Resolve relative path against config dir
    if seq_file and not os.path.isabs(seq_file):
        seq_file = os.path.normpath(os.path.join(config_dir, seq_file))
    seq_cfg = SequenceConfig(
        file=seq_file,
        validate_on_load=seq_data.get("validate_on_load", True),
    )

    # --- Environment obstacles ---
    env_data = data.get("environment", {})
    obstacle_defs = env_data.get("obstacles", [])

    return SEALPConfig(
        project_name=data.get("project_name", "unnamed_project"),
        robot=robot_cfg,
        assembly_sequence=seq_cfg,
        obstacle_defs=obstacle_defs,
        config_dir=config_dir,
    )
