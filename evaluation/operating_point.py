"""DLP operating-point analysis: precision at a fixed recall floor.

Macro-F1 at an F1-optimal threshold is the wrong lens for email DLP. In DLP a missed
sensitive item (false negative) is far costlier than a false alarm (false positive), so
the system is run at a high-recall operating point — "catch almost everything, tolerate
some over-flagging". The right comparison is therefore *precision at a recall floor*
(e.g. recall >= 0.99) per category, not headline F1.

This module reports, per label, the lowest threshold that meets a target recall and the
precision achieved there, for each single model and for the fused ensemble. It tests
whether the ensemble buys anything at the operating point the application actually needs,
even when macro-F1 is statistically tied (RQ1).

Rule-based members have degenerate probabilities (mostly 0/1); their PR curve is a step
function, so the recall floor may be unreachable — this is reported honestly as NaN
precision rather than silently skipped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from .ablation import fuse

LABEL_COLS: list[str] = ["benign", "PII", "financial", "confidential"]


def precision_at_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_recall: float = 0.99,
) -> dict[str, dict[str, float]]:
    """Per-label precision at the lowest threshold meeting ``target_recall``.

    For each label we sweep the full precision-recall curve and pick the operating
    point with the highest precision among those whose recall >= ``target_recall``
    (equivalently, the lowest threshold that still clears the recall floor gives the
    best precision on the achievable frontier). If no threshold reaches the floor (e.g.
    a degenerate step-function score), precision/threshold are NaN and
    ``recall_achieved`` is the maximum recall attainable.

    Returns ``{label: {threshold, precision, recall_achieved}}`` over LABEL_COLS.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    out: dict[str, dict[str, float]] = {}

    for col_i, label in enumerate(LABEL_COLS):
        yt = y_true[:, col_i]
        yp = y_proba[:, col_i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            # single-class label on this split — recall is undefined
            out[label] = {
                "threshold": float("nan"),
                "precision": float("nan"),
                "recall_achieved": float("nan"),
            }
            continue

        # precision_recall_curve returns precision/recall of length T+1 and
        # thresholds of length T (last point is recall=0, precision=1 with no threshold)
        precision, recall, thresholds = precision_recall_curve(yt, yp)
        # align: drop the final sentinel point that has no threshold
        precision, recall = precision[:-1], recall[:-1]

        feasible = recall >= target_recall
        if not feasible.any():
            best_r_idx = int(np.argmax(recall))
            out[label] = {
                "threshold": float(thresholds[best_r_idx]),
                "precision": float("nan"),
                "recall_achieved": float(recall[best_r_idx]),
            }
            continue

        # among thresholds meeting the recall floor, take the best precision
        feasible_idx = np.where(feasible)[0]
        best = feasible_idx[int(np.argmax(precision[feasible_idx]))]
        out[label] = {
            "threshold": float(thresholds[best]),
            "precision": float(precision[best]),
            "recall_achieved": float(recall[best]),
        }
    return out


def operating_point_table(
    model_probas: dict[str, np.ndarray],
    y_true: np.ndarray,
    weights: dict[str, dict[str, float]] | None = None,
    target_recall: float = 0.99,
) -> pd.DataFrame:
    """Precision-at-recall comparison: every single model + the fused ensemble.

    Rows are each model in ``model_probas`` plus, when ``weights`` is provided, a final
    ``ensemble`` row built from the fused per-label probabilities (``fuse``). Columns
    are per-label precision at ``target_recall`` (``{label}_prec``). This directly
    answers whether the ensemble wins at the DLP high-recall operating point even though
    the headline macro-F1 (RQ1) is tied.

    A NaN cell means the recall floor was unreachable for that (model, label).
    """
    rows = []
    for model, proba in model_probas.items():
        par = precision_at_recall(y_true, proba, target_recall=target_recall)
        rows.append(
            {"model": model, **{f"{lab}_prec": par[lab]["precision"] for lab in LABEL_COLS}}
        )

    if weights is not None:
        _, fused_proba = fuse(model_probas, weights)
        par = precision_at_recall(y_true, fused_proba, target_recall=target_recall)
        rows.append(
            {"model": "ensemble", **{f"{lab}_prec": par[lab]["precision"] for lab in LABEL_COLS}}
        )

    df = pd.DataFrame(rows).set_index("model")
    df.attrs["target_recall"] = target_recall
    return df
