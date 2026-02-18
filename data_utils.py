"""Data loading and preprocessing utilities."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from config import config


def load_cost_vector(csv_path: Path) -> np.ndarray:
    """Load cost vector from CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Cost vector as numpy array
    """
    df = pd.read_csv(csv_path)
    if "costs" in df.columns:
        x = df["costs"].values
    else:
        x = df.iloc[:, 0].values
    x = x.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def find_parallel_file(folder: Path, key: str) -> Path:
    """Find parallel file in folder based on key pattern.

    Searches for a file in the folder that contains the pattern
    (e.g., 'ABP_RR-CBFV_RR') and ends with .csv.

    Args:
        folder: Directory to search in
        key: Key from parallel_files config (e.g., 'RR', 'SPO', 'SPP')

    Returns:
        Path to the matching file

    Raises:
        FileNotFoundError: If no matching file is found
    """
    pattern = config.parallel_files[key].lower()
    candidates = []
    for p in folder.iterdir():
        if p.is_file() and p.name.lower().endswith(".csv") and pattern in p.name.lower():
            candidates.append(p)

    if len(candidates) == 0:
        raise FileNotFoundError(f"No file found for {key} in {folder}")
    candidates = sorted(candidates)
    return candidates[-1]


def preprocess_cost_vector(x: np.ndarray) -> np.ndarray:
    """Preprocess cost vector with optional log transform and normalization.

    Args:
        x: Input cost vector

    Returns:
        Preprocessed cost vector
    """
    x = x.copy()

    if config.use_log1p:
        x = np.log1p(np.clip(x, 0, None))

    if config.use_norm_by_median:
        med = np.median(x)
        x = x / (med + 1e-8)

    # Optional: clip extremes
    x = np.clip(x, 0, np.percentile(x, 99.5))

    return x.astype(np.float32)
