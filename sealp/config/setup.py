"""
SEALP Setup Facade
===================
High-level convenience function that wires together the robot,
assembly sequence, and collision environment from a single config file.

Usage::

    from sealp.config import setup_from_config

    setup = setup_from_config("my_project/sealp_config.yaml")
    robot = setup.robot
    sequence = setup.sequence
    world = setup.collision_world

    # Visualize
    import wrs.visualization.panda.world as wd
    base = wd.World(cam_pos=[1.5, 1.5, 1.0], lookat_pos=[0, 0, 0.3])
    setup.collision_world.show(base, robot=robot, toggle_cdprim=True)
    base.run()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from .sealp_config import SEALPConfig, ROBOT_REGISTRY, load_config


@dataclass
class SEALPSetup:
    """Container for a fully-initialized SEALP project.

    Attributes
    ----------
    config : SEALPConfig
        The loaded configuration.
    robot : object
        Instantiated robot (e.g., ``PiperSglArm``, ``Cobotta``, etc.).
    sequence : AssemblySequence or None
        Loaded and validated assembly sequence, or ``None`` if no
        sequence file was specified in the config.
    collision_world : CollisionWorld
        Static environment + user obstacle manager.
    """
    config: SEALPConfig
    robot: object
    sequence: object  # AssemblySequence | None
    collision_world: object  # CollisionWorld


def setup_from_config(
        config_path: Union[str, Path],
        config: Optional[SEALPConfig] = None,
) -> SEALPSetup:
    """Load config → create robot → load & validate sequence → build env.

    Parameters
    ----------
    config_path : str or Path
        Path to the SEALP YAML config file.  Ignored if *config* is
        provided directly.
    config : SEALPConfig, optional
        A pre-loaded config object.  If provided, *config_path* is
        ignored.

    Returns
    -------
    SEALPSetup
        Fully-initialized project container.

    Raises
    ------
    FileNotFoundError
        If the config file or assembly sequence file does not exist.
    ValueError
        If the robot type is unknown, or sequence validation fails.
    """
    if config is None:
        config = load_config(config_path)

    # --- 1. Create robot -------------------------------------------------
    factory = ROBOT_REGISTRY[config.robot.type]
    robot = factory(
        pos=config.robot.pos,
        rotmat=config.robot.rotmat,
        name=config.robot.name,
        enable_cc=config.robot.enable_cc,
    )

    # --- 2. Load assembly sequence (if specified) ------------------------
    sequence = None
    seq_file = config.assembly_sequence.file
    if seq_file:
        from sealp.assembly_sequence import load_sequence
        seq_path = Path(seq_file)
        if not seq_path.is_file():
            raise FileNotFoundError(
                f"Assembly sequence file not found: {seq_path}"
            )
        sequence = load_sequence(str(seq_path))
        if config.assembly_sequence.validate_on_load:
            # strict=True → raises ValueError on validation error
            sequence.validate(strict=True)

    # --- 3. Build collision world ----------------------------------------
    from sealp.colliders import CollisionWorld
    collision_world = CollisionWorld(
        obstacle_defs=config.obstacle_defs,
        base_dir=config.config_dir,
    )

    return SEALPSetup(
        config=config,
        robot=robot,
        sequence=sequence,
        collision_world=collision_world,
    )
