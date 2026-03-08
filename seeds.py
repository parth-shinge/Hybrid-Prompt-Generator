"""
seeds.py

Deterministic seeding for full reproducibility.
Sets seeds for: random, numpy, torch (CPU + CUDA), and Python hash seed.

Usage:
    from seeds import set_deterministic
    set_deterministic(42)
"""

import os
import random

import numpy as np


def set_deterministic(seed: int = 42) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value. Default 42.
    """
    # Python built-in
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # Python hash seed (affects set/dict ordering in some versions)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch (optional — graceful skip if not installed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Enforce deterministic algorithms where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


if __name__ == "__main__":
    set_deterministic(42)
    print(f"Seeds set to 42. Verification: np.random.rand() = {np.random.rand():.6f}")
