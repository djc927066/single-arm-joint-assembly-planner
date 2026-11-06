"""Joint backward search over assembly order, initial part layout and assembly grasp.

Forward execution: HOME -> pick part 1 -> insert -> (no return home) -> pick part 2 -> ...
-> insert part n -> HOME. The search runs backward (last part first), so that when a part is
decided every obstacle of its step is already known (see README.md).
"""
