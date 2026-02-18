"""Data pipeline for loading, windowing, and splitting dataset."""

import numpy as np
import pandas as pd
from typing import Tuple, List
from pathlib import Path

from config import config
from data_utils import load_cost_vector, find_parallel_file, preprocess_cost_vector
from feature_engineering import build_window_features


def load_category_data(category: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load and process data for a single category.

    Args:
        category: Category name (e.g., 'BAS', 'B6')

    Returns:
        Tuple of (feature matrix, metadata dataframe)
    """
    cat_dir = config.data_root / category

    # Find and load parallel files
    rr_path = find_parallel_file(cat_dir, "RR")
    spo_path = find_parallel_file(cat_dir, "SPO")
    spp_path = find_parallel_file(cat_dir, "SPP")

    # Load and preprocess cost vectors
    rr = preprocess_cost_vector(load_cost_vector(rr_path))
    spo = preprocess_cost_vector(load_cost_vector(spo_path))
    spp = preprocess_cost_vector(load_cost_vector(spp_path))

    # Align lengths
    L = min(len(rr), len(spo), len(spp))
    rr, spo, spp = rr[:L], spo[:L], spp[:L]

    # Generate windows
    windows = []
    t0_list = []

    for start in range(0, L - config.window_len + 1, config.window_step):
        rr_win = rr[start:start + config.window_len]
        spo_win = spo[start:start + config.window_len]
        spp_win = spp[start:start + config.window_len]

        feats = build_window_features(rr_win, spo_win, spp_win)
        windows.append(feats)
        t0_list.append(start)

    if len(windows) == 0:
        raise ValueError(
            f"Signal too short in class {category} (L={L}) for WINDOW_LEN={config.window_len}"
        )

    X_cat = np.vstack(windows)
    y_cat = np.array([category] * len(windows))

    df_cat = pd.DataFrame({
        "label": y_cat,
        "t0": t0_list
    })

    return X_cat, df_cat


def load_all_data(categories: List[str] = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load and combine data for all categories.

    Args:
        categories: List of categories to load. If None, uses config.categories

    Returns:
        Tuple of (feature matrix X, labels y, metadata dataframe)
    """
    if categories is None:
        categories = config.categories

    all_rows = []

    for cat in categories:
        print(f"Loading category: {cat}")
        X_cat, df_cat = load_category_data(cat)
        all_rows.append((X_cat, df_cat))

    X = np.vstack([x for x, _ in all_rows])
    meta = pd.concat([m for _, m in all_rows], ignore_index=True)
    # Convert to numpy array explicitly to avoid pandas StringArray issues
    y = np.array(list(meta["label"]), dtype="U")

    print(f"X shape: {X.shape}")
    print(f"y counts:\n{meta['label'].value_counts()}")

    return X, y, meta, all_rows


def split_train_val_test(all_rows: List[Tuple[np.ndarray, pd.DataFrame]],
                         X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Split data into train, validation, and test sets per category with time-awareness.

    This function ensures no time leakage by:
    1. Splitting based on chronological order (t0 values)
    2. Adding gaps between splits to prevent overlapping windows
    3. Splitting each category independently to maintain balance

    Args:
        all_rows: List of (X_cat, meta_cat) tuples
        X: Full feature matrix
        y: Full label array

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    train_idx = []
    val_idx = []
    test_idx = []

    offset = 0
    for (X_cat, meta_cat) in all_rows:
        n = len(meta_cat)

        # Calculate split points based on configured ratios
        train_split = int(n * config.train_ratio_per_class)
        val_split = int(n * (config.train_ratio_per_class + config.val_ratio_per_class))

        # Add safety gap to prevent overlapping windows at boundaries
        # Gap = window_len / window_step (number of overlapping windows)
        safety_gap = max(1, config.window_len // config.window_step)

        # Adjust splits to account for safety gaps
        train_end = max(0, train_split - safety_gap)
        val_start = train_split + safety_gap
        val_end = max(val_start, val_split - safety_gap)
        test_start = val_split + safety_gap

        # Ensure we don't go out of bounds
        val_start = min(val_start, n)
        val_end = min(val_end, n)
        test_start = min(test_start, n)

        idx = np.arange(offset, offset + n)

        # Split indices
        train_idx.extend(idx[:train_end])
        if val_start < val_end:
            val_idx.extend(idx[val_start:val_end])
        if test_start < n:
            test_idx.extend(idx[test_start:])

        # Report split info per category
        category = meta_cat['label'].iloc[0]
        print(f"  {category}: Train={train_end}, Val={len(idx[val_start:val_end])}, "
              f"Test={n - test_start}, Total={n}")

        offset += n

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"\nFinal split sizes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def split_train_test(all_rows: List[Tuple[np.ndarray, pd.DataFrame]],
                     X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backward compatibility wrapper for old train/test split.

    DEPRECATED: Use split_train_val_test instead.

    Args:
        all_rows: List of (X_cat, meta_cat) tuples
        X: Full feature matrix
        y: Full label array

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    print("Warning: Using deprecated split_train_test. Consider using split_train_val_test.")

    train_idx = []
    test_idx = []

    offset = 0
    for (X_cat, meta_cat) in all_rows:
        n = len(meta_cat)
        split = int(n * (1 - config.test_ratio_per_class))

        idx = np.arange(offset, offset + n)
        train_idx.extend(idx[:split])
        test_idx.extend(idx[split:])

        offset += n

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, y_train, X_test, y_test
