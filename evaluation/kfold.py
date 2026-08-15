"""Aggregate per-fold cross-validation results and test ensemble vs best-single.

The k-fold orchestrator (``evaluation/kfold_cv.py``) produces one ``run_ablation``
DataFrame per fold. These helpers collapse those into a mean ± std table across folds
and run a paired significance test between two configurations (e.g. ``full_ensemble``
vs ``best_single``).

Caveat, stated honestly: with k=5 the paired t-test has low power and CV folds are not
fully independent (they share training rows across the k-1 pooled folds). Treat the
p-value as descriptive robustness evidence, not a strong significance claim — consistent
with the project's scoping style.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Numeric columns emitted by evaluation.ablation.run_ablation that are worth averaging.
# ci_lower/ci_upper are per-fold bootstrap bounds and are intentionally dropped — the
# cross-fold std is the honest dispersion estimate at this level.
_LABELS = ["benign", "PII", "financial", "confidential"]
_METRIC_COLS = (
    ["macro_f1"]
    + [f"{l}_f1" for l in _LABELS]
    + [f"{l}_precision" for l in _LABELS]
    + [f"{l}_recall" for l in _LABELS]
    + [f"{l}_support" for l in _LABELS]
)


def aggregate_folds(fold_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Collapse per-fold ablation tables into mean ± std per configuration.

    ``fold_dfs`` is a list of ``run_ablation`` outputs (one per fold). Returns a tidy
    DataFrame with one row per ``configuration`` and, for each metric in
    ``_METRIC_COLS``, a ``{metric}_mean`` and ``{metric}_std`` column, plus ``n_folds``.
    Configurations are only averaged over the folds in which they appear.
    """
    if not fold_dfs:
        raise ValueError("aggregate_folds requires at least one fold DataFrame.")

    combined = pd.concat(fold_dfs, ignore_index=True)
    present = [c for c in _METRIC_COLS if c in combined.columns]

    grouped = combined.groupby("configuration")
    out = grouped[present].agg(["mean", "std"])
    out.columns = [f"{metric}_{stat}" for metric, stat in out.columns]
    out["n_folds"] = grouped.size()
    return out.reset_index()


def fold_scores(fold_dfs: list[pd.DataFrame], configuration: str,
                metric: str = "macro_f1") -> np.ndarray:
    """Return the per-fold ``metric`` values for one ``configuration``, fold-ordered."""
    scores = []
    for df in fold_dfs:
        row = df.loc[df["configuration"] == configuration, metric]
        if not row.empty:
            scores.append(float(row.iloc[0]))
    return np.asarray(scores, dtype=float)


def paired_fold_test(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """Paired t-test of two configurations across folds (``scores_a`` − ``scores_b``).

    Returns ``{"stat", "p_value", "mean_delta", "n_folds"}``. ``mean_delta > 0`` means
    configuration A scored higher on average. Requires ≥ 2 paired folds; with fewer, or
    when the two are identical on every fold, ``stat``/``p_value`` are ``None`` and only
    ``mean_delta`` is meaningful.
    """
    from scipy.stats import ttest_rel

    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired_fold_test needs equal-length inputs, got {a.shape} vs {b.shape}")

    mean_delta = float(np.mean(a - b)) if a.size else 0.0
    if a.size < 2 or np.allclose(a, b):
        return {"stat": None, "p_value": None, "mean_delta": mean_delta, "n_folds": int(a.size)}

    result = ttest_rel(a, b)
    return {
        "stat": float(result.statistic),
        "p_value": float(result.pvalue),
        "mean_delta": mean_delta,
        "n_folds": int(a.size),
    }
