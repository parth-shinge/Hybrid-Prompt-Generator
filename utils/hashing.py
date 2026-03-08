"""
utils/hashing.py

Dataset SHA-256 hashing for reproducibility.
Allows pinning a training run to the exact dataset version used.

Usage:
    from utils.hashing import compute_file_sha256, compute_data_sha256
    h = compute_file_sha256("data/processed/dataset.csv")
    h2 = compute_data_sha256([("text1", "label1"), ("text2", "label2")])
"""

import hashlib
import json
from pathlib import Path
from typing import Any, List, Tuple, Union


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Compute SHA-256 hex digest of a file.

    Reads in 64 KB chunks so arbitrarily large files are handled
    without loading everything into memory.

    Args:
        filepath: Path to the file.

    Returns:
        64-character lowercase hex digest string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    sha = hashlib.sha256()
    with open(filepath, "rb") as fh:
        while True:
            chunk = fh.read(65536)  # 64 KB
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def compute_data_sha256(data: Union[List[Tuple[str, str]], List[Any]]) -> str:
    """Compute SHA-256 hex digest of an in-memory dataset.

    Serializes the data to a canonical JSON string (sorted keys,
    no extra whitespace) before hashing, so the result is deterministic
    regardless of dict ordering.

    Args:
        data: List of tuples/rows representing the dataset.

    Returns:
        64-character lowercase hex digest string.
    """
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
