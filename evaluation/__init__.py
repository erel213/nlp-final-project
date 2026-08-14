from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import compute_all_metrics, bootstrap_ci
from .error_analysis import classify_errors, error_summary
from .ablation import load_weights, run_ablation, weight_sensitivity, fuse
from .kfold import aggregate_folds, fold_scores, paired_fold_test
from .visualization import (
    plot_confusion_matrices,
    plot_per_label_bars,
    plot_roc_curves,
    plot_pr_curves,
    plot_model_comparison_table,
    plot_ablation_table,
)
from .io import save_results, load_results, load_all_results

LABEL_COLS: list[str] = ["benign", "PII", "financial", "confidential"]


def from_dicts(preds: list[dict[str, int | float]]) -> np.ndarray:
    return np.array([[p[label] for label in LABEL_COLS] for p in preds])


def from_dataframe(df: pd.DataFrame, label_cols: list[str] = LABEL_COLS) -> np.ndarray:
    return df[label_cols].to_numpy()


__all__ = [
    "LABEL_COLS",
    "from_dicts",
    "from_dataframe",
    "compute_all_metrics",
    "bootstrap_ci",
    "classify_errors",
    "error_summary",
    "load_weights",
    "run_ablation",
    "weight_sensitivity",
    "fuse",
    "aggregate_folds",
    "fold_scores",
    "paired_fold_test",
    "plot_confusion_matrices",
    "plot_per_label_bars",
    "plot_roc_curves",
    "plot_pr_curves",
    "plot_model_comparison_table",
    "plot_ablation_table",
    "save_results",
    "load_results",
    "load_all_results",
]
