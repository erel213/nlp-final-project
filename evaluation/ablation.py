from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .metrics import MACRO_LABELS, bootstrap_ci, compute_all_metrics

LABEL_COLS: list[str] = ["benign", "PII", "financial", "confidential"]

# Column indices of the informative categories averaged into the headline macro-F1
# (benign excluded — see evaluation.metrics.MACRO_LABELS and comment 011). Every
# macro-F1 computed in this module (ablation rows, RQ1 delta, weight sensitivity)
# uses this SAME subset so the headline, RQ1 verdict, and CV aggregates share one
# metric definition.
_MACRO_IDX: list[int] = [LABEL_COLS.index(label) for label in MACRO_LABELS]


def _macro_f1(yt: np.ndarray, yp: np.ndarray) -> float:
    """Headline macro-F1 over MACRO_LABELS (PII/financial/confidential) only."""
    yt = np.asarray(yt, dtype=int)
    yp = np.asarray(yp, dtype=int)
    return float(
        f1_score(
            yt[:, _MACRO_IDX], yp[:, _MACRO_IDX], average="macro", zero_division=0
        )
    )

# A threshold may be a single float applied to every label, or a per-label
# mapping ``{label: float}`` (e.g. the ensemble's DEV-tuned per-label
# thresholds). Both forms are accepted everywhere ``threshold`` appears.
Threshold = float | dict[str, float]


def _threshold_vector(threshold: Threshold) -> np.ndarray:
    """Normalize a scalar-or-per-label threshold into a ``(len(LABEL_COLS),)`` vector.

    - ``float`` → same value broadcast to every label (backward compatible).
    - ``dict`` → looked up per label in ``LABEL_COLS`` order; a missing label
      defaults to 0.5.
    """
    if isinstance(threshold, dict):
        return np.array(
            [float(threshold.get(label, 0.5)) for label in LABEL_COLS], dtype=float
        )
    return np.full(len(LABEL_COLS), float(threshold), dtype=float)


def load_weights(weights_path: str | Path) -> dict[str, dict[str, float]]:
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Ensemble weights not found at {path}. "
            "Run ensemble weight training first — see the ensemble/ directory."
        )
    with path.open() as f:
        return json.load(f)


def fuse(
    model_probas: dict[str, np.ndarray],
    weights: dict[str, dict[str, float]],
    threshold: Threshold = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Confidence-weighted late fusion.

    ``threshold`` accepts either a single float applied to every label, or a
    per-label mapping ``{label: float}`` (e.g. the DEV-tuned per-label
    thresholds); the per-label form is applied column-wise.
    """
    model_names = list(model_probas)
    n = next(iter(model_probas.values())).shape[0]
    y_proba_fused = np.zeros((n, len(LABEL_COLS)), dtype=float)

    for col_i, label in enumerate(LABEL_COLS):
        total_weight = 0.0
        for model in model_names:
            w = weights.get(model, {}).get(label, 0.0)
            y_proba_fused[:, col_i] += w * model_probas[model][:, col_i]
            total_weight += w
        if total_weight > 0:
            y_proba_fused[:, col_i] /= total_weight

    thr = _threshold_vector(threshold)
    y_pred = (y_proba_fused >= thr).astype(int)
    return y_pred, y_proba_fused


def _solo_weights(model_name: str) -> dict[str, dict[str, float]]:
    return {model_name: {label: 1.0 for label in LABEL_COLS}}


def _leave_one_out_weights(
    weights: dict[str, dict[str, float]], excluded: str
) -> dict[str, dict[str, float]]:
    return {m: v for m, v in weights.items() if m != excluded}


def run_ablation(
    model_probas: dict[str, np.ndarray],
    weights: dict[str, dict[str, float]],
    y_true: np.ndarray,
    threshold: Threshold = 0.5,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Leave-one-out / solo ablation table.

    ``threshold`` accepts a float or a per-label ``{label: float}`` mapping; the
    SAME threshold rule is applied to the full ensemble, the leave-one-out
    configs, and the solo/best-single configs, so all rows are judged under one
    common decision rule (keeps RQ1 apples-to-apples — see comment 010/017).
    """
    y_true = np.asarray(y_true, dtype=int)

    def _eval_config(config_name: str, w: dict, probas: dict) -> dict:
        y_pred, _ = fuse(probas, w, threshold=threshold)
        ci = bootstrap_ci(y_true, y_pred, _macro_f1, n_resamples=n_bootstrap, random_state=random_state)
        metrics = compute_all_metrics(y_true, y_pred, n_bootstrap=n_bootstrap, random_state=random_state)
        per = metrics["per_label"]
        return {
            "configuration": config_name,
            "macro_f1": metrics["macro_f1"],
            "ci_lower": ci["lower"],
            "ci_upper": ci["upper"],
            **{f"{label}_f1": per[label]["f1"] for label in LABEL_COLS},
            **{f"{label}_precision": per[label]["precision"] for label in LABEL_COLS},
            **{f"{label}_recall": per[label]["recall"] for label in LABEL_COLS},
            **{f"{label}_support": per[label]["support"] for label in LABEL_COLS},
        }

    rows = []

    # Full ensemble
    rows.append(_eval_config("full_ensemble", weights, model_probas))

    # Leave-one-out
    for model in list(model_probas):
        loo_weights = _leave_one_out_weights(weights, model)
        loo_probas = {m: p for m, p in model_probas.items() if m != model}
        if loo_probas:
            rows.append(_eval_config(f"minus_{model}", loo_weights, loo_probas))

    # Solo models
    best_solo_f1 = -1.0
    best_solo_row: dict | None = None
    for model in list(model_probas):
        solo_w = _solo_weights(model)
        solo_probas = {model: model_probas[model]}
        row = _eval_config(f"solo_{model}", solo_w, solo_probas)
        rows.append(row)
        if row["macro_f1"] > best_solo_f1:
            best_solo_f1 = row["macro_f1"]
            best_solo_row = {**row, "configuration": "best_single"}

    if best_solo_row:
        rows.append(best_solo_row)

    return pd.DataFrame(rows)


def paired_bootstrap_delta(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict:
    """Paired bootstrap on the per-row-resampled metric difference (A minus B).

    Mirrors the resampling style of ``metrics.bootstrap_ci`` but draws ONE index
    vector per resample and applies it to BOTH prediction arrays, so the delta is
    measured on the same rows for both configs. Returns the mean delta, a
    ``confidence``-level CI on the delta, a two-sided bootstrap p-value, and a
    ``significant`` flag that is True iff the delta CI excludes 0.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_a = np.asarray(y_pred_a, dtype=int)
    y_pred_b = np.asarray(y_pred_b, dtype=int)

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        deltas[i] = metric_fn(y_true[idx], y_pred_a[idx]) - metric_fn(
            y_true[idx], y_pred_b[idx]
        )

    alpha = 1.0 - confidence
    lower, upper = np.percentile(deltas, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    frac_le = float(np.mean(deltas <= 0.0))
    frac_ge = float(np.mean(deltas >= 0.0))
    p_value = min(1.0, 2.0 * min(frac_le, frac_ge))
    significant = bool(lower > 0.0 or upper < 0.0)

    return {
        "mean_delta": float(np.mean(deltas)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_value": float(p_value),
        "n_resamples": int(n_resamples),
        "level": float(confidence),
        "significant": significant,
    }


def ensemble_vs_best_single(
    model_probas: dict[str, np.ndarray],
    weights: dict[str, dict[str, float]],
    y_true: np.ndarray,
    threshold: Threshold = 0.5,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict:
    """Paired-bootstrap RQ1 test: full ensemble vs. the best single model.

    Fuses the full ensemble and each solo model on the SAME ``y_true`` rows,
    identifies the best single model by point macro-F1, then runs
    ``paired_bootstrap_delta`` on (ensemble - best_single) macro-F1 using one set
    of resample indices per draw. The RQ1 improvement claim holds only when the
    returned delta CI excludes 0 (``significant`` True).

    ``threshold`` accepts a float or a per-label ``{label: float}`` mapping and
    is applied identically to BOTH the ensemble and the solo models, so the RQ1
    delta is measured under one common decision rule.

    Returns the ``paired_bootstrap_delta`` dict augmented with
    ``best_single_model``, ``ensemble_macro_f1`` and ``best_single_macro_f1``.
    Leaves ``run_ablation``'s DataFrame contract untouched.
    """
    y_true = np.asarray(y_true, dtype=int)

    y_pred_ensemble, _ = fuse(model_probas, weights, threshold=threshold)
    ensemble_f1 = _macro_f1(y_true, y_pred_ensemble)

    best_model: str | None = None
    best_f1 = -1.0
    best_pred: np.ndarray | None = None
    for model in model_probas:
        solo_pred, _ = fuse(
            {model: model_probas[model]}, _solo_weights(model), threshold=threshold
        )
        solo_f1 = _macro_f1(y_true, solo_pred)
        if solo_f1 > best_f1:
            best_f1 = solo_f1
            best_model = model
            best_pred = solo_pred

    if best_pred is None:
        raise ValueError("ensemble_vs_best_single requires at least one model in model_probas")

    result = paired_bootstrap_delta(
        y_true,
        y_pred_ensemble,
        best_pred,
        _macro_f1,
        n_resamples=n_resamples,
        confidence=confidence,
        random_state=random_state,
    )
    result["best_single_model"] = best_model
    result["ensemble_macro_f1"] = ensemble_f1
    result["best_single_macro_f1"] = best_f1
    return result


def weight_sensitivity(
    model_probas: dict[str, np.ndarray],
    weights: dict[str, dict[str, float]],
    y_true: np.ndarray,
    model_name: str,
    label: str,
    weight_range: tuple[float, float] = (0.0, 1.0),
    n_steps: int = 21,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    label_idx = LABEL_COLS.index(label)
    w_values = np.linspace(weight_range[0], weight_range[1], n_steps)

    rows = []
    for w_val in w_values:
        varied = {m: dict(v) for m, v in weights.items()}
        if model_name not in varied:
            varied[model_name] = {lbl: 0.0 for lbl in LABEL_COLS}
        varied[model_name][label] = float(w_val)

        y_pred, _ = fuse(model_probas, varied)
        # Headline macro-F1 over MACRO_LABELS only (benign excluded), same definition
        # as run_ablation / the RQ1 verdict. The varied ``label``'s own F1 is still
        # reported separately below even if it is benign.
        macro_f1 = _macro_f1(y_true, y_pred)
        label_f1 = float(f1_score(y_true[:, label_idx], y_pred[:, label_idx], average="binary", zero_division=0))
        rows.append({"weight": float(w_val), "macro_f1": macro_f1, f"{label}_f1": label_f1})

    return pd.DataFrame(rows)
