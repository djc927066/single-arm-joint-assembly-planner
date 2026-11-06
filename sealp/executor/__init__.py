"""
Sequence Executor
==================

Step-by-step assembly execution engine.

- ``SequenceExecutor``  — plans all steps in topological order
- ``PrimitiveSelector``  — maps primitive enum → concrete planner
- ``ExecutionResult`` / ``StepResult`` — structured results
"""

from .sequence_executor import SequenceExecutor, ExecutionResult, StepResult
from .primitive_selector import PrimitiveSelector
