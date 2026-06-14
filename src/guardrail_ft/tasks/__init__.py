"""Bias tasks. Importing this package registers all concrete tasks.

Add a new task by creating ``<task>.py`` with a ``@register_task`` class and
importing it here.
"""

from .base import (
    BiasItem,
    BiasTask,
    ContrastPair,
    Dataset,
    Prediction,
    SpecialLabel,
    TASK_REGISTRY,
    get_task,
    register_task,
)

# Import concrete tasks for their registration side-effects.
from . import resume, bbq, winobias  # noqa: E402,F401

__all__ = [
    "BiasItem", "BiasTask", "ContrastPair", "Dataset", "Prediction",
    "SpecialLabel", "TASK_REGISTRY", "get_task", "register_task",
]
