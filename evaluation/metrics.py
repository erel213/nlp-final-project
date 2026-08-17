from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

LABEL_COLS: list[str] = ["benign", "PII", "financial", "confidential"]

# Labels averaged into the headline macro-F1. ``benign`` is EXCLUDED because it is
# definitionally the complement of "any entity present" (benign = 1 - PII at the row
# level; see documentation/preprocessing.md), so its F1 mirrors PII's and macro-
# averaging over all four double-counts a redundant, high-support category. benign
# stays in ``per_label`` for transparency but is not part of the headline metric.
# See comment 011 (Decision A1). Order matches LABEL_COLS.
MACRO_LABELS: list[str] = ["PII", "financial", "confidential"]
_MACRO_IDX: list[int] = [LABEL_COLS.index(label) for label in MACRO_LABELS]


def _macro_f1_over(labels_idx: list[int], yt: np.ndarray, yp: np.ndarray) -> float:
    """Unweighted mean of per-label binary F1 over the given column indices."""
    return float(
        f1_score(
            yt[:, labels_idx], yp[:, labels_idx], average="macro", zero_division=0
        )
    )


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict:
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    scores = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        scores[i] = metric_fn(y_true[idx], y_pred[idx])
    alpha = 1.0 - confidence
    lower, upper = np.percentile(scores, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return {"lower": float(lower), "upper": float(upper), "level": confidence}


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    per_label: dict[str, dict] = {}
    for i, label in enumerate(LABEL_COLS):
        p, r, f, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average="binary", zero_division=0
        )
        per_label[label] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(y_true[:, i].sum()),
        }

    # Headline macro-F1: mean of binary F1 over the informative categories only
    # (MACRO_LABELS = PII/financial/confidential). benign is excluded — see the
    # MACRO_LABELS docstring and comment 011.
    macro_f1 = _macro_f1_over(_MACRO_IDX, y_true, y_pred)

    def _macro_f1(yt: np.ndarray, yp: np.ndarray) -> float:
        return _macro_f1_over(_MACRO_IDX, yt, yp)

    ci = bootstrap_ci(
        y_true, y_pred, _macro_f1, n_resamples=n_bootstrap, random_state=random_state
    )

    result: dict = {
        "per_label": per_label,
        "macro_f1": macro_f1,
        "macro_f1_ci": ci,
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float)
        auc_per_label: dict[str, float | None] = {}
        valid_aucs: list[float] = []
        for i, label in enumerate(LABEL_COLS):
            try:
                auc_raw = roc_auc_score(y_true[:, i], y_proba[:, i])
                auc = None if np.isnan(auc_raw) else float(auc_raw)
            except ValueError:
                auc = None
            auc_per_label[label] = auc
            if auc is not None:
                valid_aucs.append(auc)

        macro_auc: float | None = (
            float(np.mean(valid_aucs)) if valid_aucs else None
        )
        auc_per_label["macro"] = macro_auc
        result["auc_roc"] = auc_per_label

    return result
