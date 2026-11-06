"""
SEALP configuration management.

Provides YAML-based project configuration for robot selection,
collision environment setup, and assembly sequence loading.
"""

from .sealp_config import (
    SEALPConfig,
    RobotConfig,
    load_config,
    ROBOT_REGISTRY,
)
from .setup import SEALPSetup, setup_from_config
