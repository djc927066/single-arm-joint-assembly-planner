"""
Collision environment management for SEALP.

Provides obstacle management, static environment setup (from config),
and a unified collision world combining static + runtime obstacles.
"""

from .obstacle_manager import ObstacleManager
from .static_environment import StaticEnvironment
from .collision_world import CollisionWorld
