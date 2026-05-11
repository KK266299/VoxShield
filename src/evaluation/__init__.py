"""Evaluation strategy package."""

# Import strategies so that they register themselves with the global registry
from . import brats19_eval  # noqa: F401
from . import flare21_eval  # noqa: F401

__all__ = [
    'brats19_eval',
    'flare21_eval',
]