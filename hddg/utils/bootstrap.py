"""Project root path setup for HDDG."""

from __future__ import annotations

import os
import sys


def setup_project_paths(*, chdir: bool = True) -> str:
    """Add project root to ``sys.path`` so ``from models.*`` imports work."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    if chdir:
        os.chdir(root)
    return root
