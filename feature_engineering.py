"""Feature engineering utilities for window-based analysis."""

import numpy as np
from typing import List

from config import config


def get_feature_names() -> List[str]:
    """Generate feature names matching the feature vector structure.

    Returns:
        List of feature names in the same order as build_window_features output
    """
    feature_names = []

    # Statistical features per channel
    stat_names = ['mean', 'std', 'min', 'max', 'median',
                  'p10', 'p25', 'p75', 'p90', 'iqr', 'slope', 'energy']
    channels = ['rr', 'spo', 'spp']

    for channel in channels:
        for stat in stat_names:
            feature_names.append(f"{channel}_{stat}")

    # Relational features between channel pairs
    channel_pairs = [('rr', 'spo'), ('rr', 'spp'), ('spo', 'spp')]
    relational_features = ['mean_diff', 'mean_ratio', 'std_diff']

    for ch1, ch2 in channel_pairs:
        for rel_feat in relational_features:
            feature_names.append(f"{ch1}_{ch2}_{rel_feat}")

    # Correlations
    for ch1, ch2 in channel_pairs:
        feature_names.append(f"{ch1}_{ch2}_corr")

    return feature_names


def window_stats(x: np.ndarray) -> List[float]:
    """Calculate statistical features from a 1D window using lambda dictionary.

    Args:
        x: Input window array

    Returns:
        List of statistical features: [mean, std, min, max, median,
        p10, p25, p75, p90, iqr, slope, energy]
    """
    x = np.asarray(x, dtype=np.float32)

    # Define statistical features as dictionary of lambdas
    stat_funcs = {
        'mean': lambda arr: float(arr.mean()),
        'std': lambda arr: float(arr.std()),
        'min': lambda arr: float(arr.min()),
        'max': lambda arr: float(arr.max()),
        'median': lambda arr: float(np.median(arr)),
        'p10': lambda arr: float(np.percentile(arr, 10)),
        'p25': lambda arr: float(np.percentile(arr, 25)),
        'p75': lambda arr: float(np.percentile(arr, 75)),
        'p90': lambda arr: float(np.percentile(arr, 90)),
    }

    # Calculate basic stats
    stats = {name: func(x) for name, func in stat_funcs.items()}

    # IQR (depends on previously calculated percentiles)
    stats['iqr'] = stats['p75'] - stats['p25']

    # Trend (slope) from linear regression
    t = np.arange(len(x), dtype=np.float32)
    t_mean = t.mean()
    denom = float(np.sum((t - t_mean) ** 2) + 1e-8)
    stats['slope'] = float(np.sum((t - t_mean) * (x - stats['mean'])) / denom)

    # Energy
    stats['energy'] = float(np.mean(x * x))

    # Return in consistent order
    return [stats['mean'], stats['std'], stats['min'], stats['max'], stats['median'],
            stats['p10'], stats['p25'], stats['p75'], stats['p90'], stats['iqr'],
            stats['slope'], stats['energy']]


def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate correlation between two arrays with safety checks.

    Args:
        a: First array
        b: Second array

    Returns:
        Pearson correlation coefficient, or 0.0 if calculation is not possible
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sa = a.std()
    sb = b.std()
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def build_window_features(rr_win: np.ndarray, spo_win: np.ndarray,
                         spp_win: np.ndarray) -> np.ndarray:
    """Build feature vector from window data across three channels using lambda dictionaries.

    Args:
        rr_win: RR channel window
        spo_win: SPO channel window
        spp_win: SPP channel window

    Returns:
        Feature vector as numpy array
    """
    feats = []

    # Statistical features per channel
    rr_f = window_stats(rr_win)
    spo_f = window_stats(spo_win)
    spp_f = window_stats(spp_win)

    feats += rr_f + spo_f + spp_f

    # Extract mean and std for relational features
    rr_mean, rr_std = rr_f[0], rr_f[1]
    spo_mean, spo_std = spo_f[0], spo_f[1]
    spp_mean, spp_std = spp_f[0], spp_f[1]

    eps = 1e-8

    # Define channel pairs for relational features
    channel_pairs = [
        ('rr', 'spo', rr_mean, spo_mean, rr_std, spo_std, rr_win, spo_win),
        ('rr', 'spp', rr_mean, spp_mean, rr_std, spp_std, rr_win, spp_win),
        ('spo', 'spp', spo_mean, spp_mean, spo_std, spp_std, spo_win, spp_win),
    ]

    # Relational features defined as lambdas
    relational_funcs = {
        'mean_diff': lambda m1, m2, s1, s2: m1 - m2,
        'mean_ratio': lambda m1, m2, s1, s2: m1 / (m2 + eps),
        'std_diff': lambda m1, m2, s1, s2: s1 - s2,
    }

    # Calculate relational features for each channel pair
    for _, _, m1, m2, s1, s2, _, _ in channel_pairs:
        for func in relational_funcs.values():
            feats.append(func(m1, m2, s1, s2))

    # Correlations between channel pairs
    for _, _, _, _, _, _, w1, w2 in channel_pairs:
        feats.append(corr_safe(w1, w2))

    return np.array(feats, dtype=np.float32)
