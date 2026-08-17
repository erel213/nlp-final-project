"""Row-level error-complementarity analysis.

These utilities explain *why* the late-fusion ensemble does not beat the best single
model (RQ1 null result). An ensemble can only improve on its best member when the
members make *different* mistakes — one member covers examples another gets wrong. If
the members mostly succeed and fail on the same examples, their errors are correlated
and there is no diversity for any combiner to exploit.

All functions operate on cached per-model probabilities plus ``y_true`` and reduce to
per-example, per-label *correctness* under a decision threshold. Correctness is used
(rather than raw probabilities) because that is what a hard-label combiner could ever
recover.

Everything here is pure numpy/pandas so notebooks never import sklearn directly (see
.claude/rules/evaluation-api.md). Metrics are reported per label over LABEL_COLS;
``benign`` is included per-label for transparency but the headline story is told over
MACRO_LABELS (PII/financial/confidential), matching comment 011.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import MACRO_LABELS

LABEL_COLS: list[str] = ["benign", "PII", "financial", "confidential"]

# Reuse the threshold-normalizing helper from ablation so a scalar OR a per-label
# {label: float} mapping (e.g. the DEV-tuned thresholds) is accepted everywhere.
from .ablation import Threshold, _threshold_vector  # noqa: E402


def per_model_correct(
    model_probas: dict[str, np.ndarray],
    y_true: np.ndarray,
    threshold: Threshold = 0.5,
) -> dict[str, np.ndarray]:
    """Per-model, per-example, per-label correctness under a decision threshold.

    Returns ``{model_name: (N, 4) bool}`` where ``True`` means the model's hard
    prediction for that (example, label) matches ``y_true``. The SAME threshold rule
    is applied to every model so their correctness masks are directly comparable.
    """
    y_true = np.asarray(y_true, dtype=int)
    thr = _threshold_vector(threshold)
    correct: dict[str, np.ndarray] = {}
    for model, proba in model_probas.items():
        y_pred = (np.asarray(proba, dtype=float) >= thr).astype(int)
        correct[model] = y_pred == y_true
    return correct


def oracle_gap(
    correct_by_model: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> pd.DataFrame:
    """Per-label best-single vs. oracle-upper-bound accuracy.

    The *oracle* is the best a hard-label combiner could ever do: an example counts as
    recoverable if **any** member is correct on it. ``oracle_gap`` is the headroom left
    for combination — ``oracle - best_single``. A small gap means even a perfect
    combiner could barely improve on the best member, which is the quantitative reason
    an ensemble cannot help.

    Returns one row per label in LABEL_COLS with columns:
    ``label | best_single_model | best_single_acc | oracle_acc | oracle_gap``.
    """
    y_true = np.asarray(y_true, dtype=int)
    n = y_true.shape[0]
    models = list(correct_by_model)

    rows = []
    for col_i, label in enumerate(LABEL_COLS):
        per_model_acc = {
            m: float(correct_by_model[m][:, col_i].mean()) for m in models
        }
        best_model = max(per_model_acc, key=per_model_acc.get)
        best_acc = per_model_acc[best_model]
        # oracle: example recoverable if ANY model is correct on this label
        any_correct = np.zeros(n, dtype=bool)
        for m in models:
            any_correct |= correct_by_model[m][:, col_i]
        oracle_acc = float(any_correct.mean())
        rows.append(
            {
                "label": label,
                "best_single_model": best_model,
                "best_single_acc": best_acc,
                "oracle_acc": oracle_acc,
                "oracle_gap": oracle_acc - best_acc,
            }
        )
    return pd.DataFrame(rows)


def error_overlap_matrix(
    correct_by_model: dict[str, np.ndarray],
    label: str,
) -> pd.DataFrame:
    """Pairwise error-correlation matrix for one label (Jaccard of error sets).

    For each pair of models, the value is the Jaccard overlap of the examples they get
    **wrong**: ``|errors_A ∩ errors_B| / |errors_A ∪ errors_B|``. 1.0 = identical error
    sets (fully correlated, no diversity); 0.0 = disjoint errors (maximal diversity, the
    regime where fusion helps). The diagonal is 1.0 by definition.

    Returns a square DataFrame indexed and columned by model name.
    """
    if label not in LABEL_COLS:
        raise ValueError(f"Unknown label {label!r}; expected one of {LABEL_COLS}")
    col_i = LABEL_COLS.index(label)
    models = list(correct_by_model)
    errors = {m: ~correct_by_model[m][:, col_i] for m in models}

    mat = np.zeros((len(models), len(models)), dtype=float)
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            union = np.count_nonzero(errors[a] | errors[b])
            inter = np.count_nonzero(errors[a] & errors[b])
            mat[i, j] = (inter / union) if union > 0 else 1.0
    return pd.DataFrame(mat, index=models, columns=models)


def unique_contribution(
    correct_by_model: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> pd.DataFrame:
    """Per-label, per-model count/fraction of examples ONLY that model gets right.

    An example is a model's *exclusive* correct catch on a label if that model is
    correct there and every other model is wrong. Near-zero exclusive catches means the
    member is redundant — removing it loses nothing, which is what the leave-one-out
    ablation shows in aggregate.

    Returns one row per (label, model) with columns:
    ``label | model | exclusive_correct | exclusive_frac`` where ``exclusive_frac`` is
    over all N examples of that label.
    """
    y_true = np.asarray(y_true, dtype=int)
    n = y_true.shape[0]
    models = list(correct_by_model)

    rows = []
    for col_i, label in enumerate(LABEL_COLS):
        stacked = np.stack([correct_by_model[m][:, col_i] for m in models], axis=1)
        # example is exclusively caught by model m if exactly one model is correct AND
        # it is m
        exactly_one = stacked.sum(axis=1) == 1
        for j, m in enumerate(models):
            excl = int(np.count_nonzero(exactly_one & stacked[:, j]))
            rows.append(
                {
                    "label": label,
                    "model": m,
                    "exclusive_correct": excl,
                    "exclusive_frac": excl / n if n else 0.0,
                }
            )
    return pd.DataFrame(rows)


def complementarity_summary(
    model_probas: dict[str, np.ndarray],
    y_true: np.ndarray,
    threshold: Threshold = 0.5,
) -> dict:
    """One-call bundle: oracle gap + per-label error-overlap + unique contributions.

    Returns a JSON-serializable dict summarizing the mechanism behind the RQ1 null
    result, keyed by ``oracle_gap`` (list of records over LABEL_COLS),
    ``error_overlap`` ({label: {model: {model: jaccard}}} over MACRO_LABELS), and
    ``unique_contribution`` (list of records). Convenience for the notebook / artifact
    saving; the individual functions above give the DataFrames for display.
    """
    correct = per_model_correct(model_probas, y_true, threshold=threshold)
    gap_df = oracle_gap(correct, y_true)
    uniq_df = unique_contribution(correct, y_true)
    overlap = {
        label: error_overlap_matrix(correct, label).to_dict()
        for label in MACRO_LABELS
    }
    return {
        "oracle_gap": gap_df.to_dict(orient="records"),
        "error_overlap": overlap,
        "unique_contribution": uniq_df.to_dict(orient="records"),
    }
