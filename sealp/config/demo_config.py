"""
SEALP Configuration Demo
==========================

Demonstrates loading a SEALP config file, creating the robot,
and visualizing the static environment in Panda3D.

Run directly::

    python -m sealp.config.demo_config
"""

import os
import numpy as np


def main():
    from sealp.config import load_config, setup_from_config, ROBOT_REGISTRY

    # ------------------------------------------------------------------
    # 1. Show available robot types
    # ------------------------------------------------------------------
    print("=" * 60)
    print("SEALP Configuration Demo")
    print("=" * 60)
    print(f"\nRegistered robot types: {sorted(ROBOT_REGISTRY.keys())}")

    # ------------------------------------------------------------------
    # 2. Load the sample config
    # ------------------------------------------------------------------
    config_path = os.path.join(os.path.dirname(__file__),
                               "sample_config.yaml")
    print(f"\nLoading config from: {config_path}")

    config = load_config(config_path)
    print(f"  Project: {config.project_name}")
    print(f"  Robot:   {config.robot.type} "
          f"(pos={config.robot.pos.tolist()}, cc={config.robot.enable_cc})")
    print(f"  Obstacles: {len(config.obstacle_defs)}")
    for obs in config.obstacle_defs:
        print(f"    • {obs['name']} ({obs['type']})")

    # ------------------------------------------------------------------
    # 3. Setup (create robot + collision world)
    # ------------------------------------------------------------------
    print("\nRunning setup_from_config()...")
    setup = setup_from_config(config_path, config=config)
    print(f"  Robot created: {type(setup.robot).__name__}")
    print(f"  Sequence: {setup.sequence}")
    print(f"  Collision world obstacles: "
          f"{len(setup.collision_world.obstacle_list)}")

    # ------------------------------------------------------------------
    # 4. Visualize in Panda3D
    # ------------------------------------------------------------------
    import wrs.visualization.panda.world as wd
    import wrs.modeling.geometric_model as mgm

    base = wd.World(cam_pos=[1.5, 1.5, 1.0], lookat_pos=[0, 0, 0.3])
    mgm.gen_frame().attach_to(base)

    # Show robot
    setup.robot.gen_meshmodel(alpha=1).attach_to(base)

    # Show collision environment
    setup.collision_world.show(base, toggle_cdprim=False, alpha=0.6)

    print("\n✓ Visualization ready. Press ESC to close.")
    base.run()


if __name__ == "__main__":
    main()
