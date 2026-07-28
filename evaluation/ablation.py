from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .metrics import bootstrap_ci, compute_all_metrics

LABEL_COLS: list[str] = ["benign", "PII", "financial", "confidential"]


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
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
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

    y_pred = (y_proba_fused >= threshold).astype(int)
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
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)

    def _macro_f1(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(f1_score(yt, yp, average="macro", zero_division=0))

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
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        label_f1 = float(f1_score(y_true[:, label_idx], y_pred[:, label_idx], average="binary", zero_division=0))
        rows.append({"weight": float(w_val), "macro_f1": macro_f1, f"{label}_f1": label_f1})

    return pd.DataFrame(rows)
