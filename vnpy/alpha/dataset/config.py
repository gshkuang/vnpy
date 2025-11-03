"""
Centralized configuration for dataset-specific parameters.

This module collects hardcoded constants used across datasets and feature
pipelines so they can be managed in a single place.

Example usage:

from vnpy.alpha.dataset.datasets.config importDATASET_CONFIG

cfg =DATASET_CONFIG
windows = cfg["feature_windows"]
label_cfg = cfg["label"]
"""

from typing import Any, Dict

# Default configuration for Alpha158 dataset
DATASET_CONFIG: Dict[str, Any] = {
    # Relative paths for dataset artifacts under lab_dir
    "paths": {
        "feat_dir": "feat",
        "stats_dir": "stats",
        "feat_norm_dir": "feat_norm",
        "splits_dir": "splits",
    },
    # Time-series window sizes for features
    "feature_windows": [5, 10, 20, 30, 60],
    # Ranking/period granularity, default minute ('m')
    "period": "m",
    # Label construction parameters
    # label = price(t+forward_shift) / price(t+base_shift) - 1
    "label": {
        "base_field": "close",
        "forward_shift": -3,
        "base_shift": -1,
        "type": "return",
    },
    # Normalization methods used in feature pipeline
    "normalization": {
        # Column-wise (global) normalization method: 'zscore' or 'robust'
        "col_method": "robust",
        # Row-wise (cross-sectional) normalization method: 'zscore' or 'robust'
        "row_method": "zscore",
    },
    # Splitting configuration for train/valid/test parquet generation
    "splits": {
        "shuffle": True,
        "seed": 42,
    },
}
