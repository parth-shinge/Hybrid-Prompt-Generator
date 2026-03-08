"""
utils/git_info.py

Git commit hash and dirty-state detection for experiment tracking.

Usage:
    from utils.git_info import get_git_commit, is_dirty
    commit = get_git_commit()  # e.g. "a1b2c3d..." or "unknown"
    dirty  = is_dirty()        # True / False / None
"""

import subprocess
from typing import Optional


def get_git_commit() -> str:
    """Return the current HEAD commit hash, or ``"unknown"`` if not in a git repo.

    Returns:
        40-character hex commit hash, or ``"unknown"``.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def is_dirty() -> Optional[bool]:
    """Check whether the working tree has uncommitted changes.

    Returns:
        ``True`` if dirty, ``False`` if clean, ``None`` if detection failed.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return len(result.stdout.strip()) > 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None
