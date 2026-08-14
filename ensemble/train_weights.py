"""Learn per-category late-fusion weights for the SentinelMail ensemble.

The fusion formula in ``evaluation.ablation.fuse`` is separable per label:
``score_L = Σ_m w[m][L]·p[m][:,L] / Σ_m w[m][L]``. Label L's binary F1 depends
only on ``{w[m][L]}_m``, and macro-F1 is the mean of the four per-label F1s.
Therefore maximising macro-F1 over all 16 weights decomposes into four
independent per-label problems, each over its own model weights.

Because F1 is computed after a 0.5 threshold it is piecewise-constant, so the
gradient is zero almost everywhere and gradient-based L-BFGS-B stalls at its
start point (see ``fit_weights_lbfgsb`` for the surrogate variant that honours
the spec's letter). The default optimiser is therefore a gradient-free
simplex grid + Nelder-Mead refinement on the TRUE binary F1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score

from evaluation import LABEL_COLS, fuse

_DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.json"
_DEFAULT_THRESHOLDS_PATH = Path(__file__).resolve().parent / "thresholds.json"


# --------------------------------------------------------------------------- #
# per-label fusion helpers                                                     #
# --------------------------------------------------------------------------- #
def _stack_label(model_probas: dict[str, np.ndarray], label_idx: int,
                 model_order: list[str]) -> np.ndarray:
    """Return the per-label probabilities of every model as an (M, N) array."""
    return np.stack([model_probas[m][:, label_idx] for m in model_order], axis=0)


def _fused_label_score(w: np.ndarray, stacked: np.ndarray) -> np.ndarray | None:
    """Weighted-average fusion for one label. Returns (N,) or None if degenerate."""
    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total <= 0:
        return None
    return (w[:, None] * stacked).sum(axis=0) / total


def _binary_f1(score: np.ndarray, y_col: np.ndarray, threshold: float) -> float:
    return float(f1_score(y_col, (score >= threshold).astype(int), zero_division=0))


def _simplex_grid(n_models: int, n_steps: int):
    """Yield weight vectors of length n_models, non-negative, summing to 1,
    at resolution 1/n_steps (integer compositions of n_steps into n_models parts)."""
    def rec(remaining: int, parts_left: int):
        if parts_left == 1:
            yield (remaining,)
            return
        for i in range(remaining + 1):
            for rest in rec(remaining - i, parts_left - 1):
                yield (i, *rest)

    for combo in rec(n_steps, n_models):
        yield np.array(combo, dtype=float) / n_steps


def _label_threshold(threshold: float | dict[str, float], label: str) -> float:
    return float(threshold[label]) if isinstance(threshold, dict) else float(threshold)


# --------------------------------------------------------------------------- #
# default optimiser: grid + Nelder-Mead on TRUE F1                             #
# --------------------------------------------------------------------------- #
def fit_label_weights(
    model_probas: dict[str, np.ndarray],
    y_true: np.ndarray,
    label: str,
    grid_step: float = 0.1,
    threshold: float = 0.5,
    refine: bool = True,
    model_order: list[str] | None = None,
) -> tuple[dict[str, float], float]:
    """Fit one label's per-model weights to maximise its binary F1.

    Returns ``({model: weight}, best_f1)`` with weights normalised to sum to 1.
    """
    model_order = model_order or list(model_probas)
    label_idx = LABEL_COLS.index(label)
    stacked = _stack_label(model_probas, label_idx, model_order)
    y_col = np.asarray(y_true, dtype=int)[:, label_idx]
    n_steps = max(1, round(1.0 / grid_step))

    best_w: np.ndarray | None = None
    best_f1 = -1.0
    for w in _simplex_grid(len(model_order), n_steps):
        score = _fused_label_score(w, stacked)
        if score is None:
            continue
        f1 = _binary_f1(score, y_col, threshold)
        if f1 > best_f1:
            best_f1, best_w = f1, w

    if best_w is None:  # all-zero grid (shouldn't happen); fall back to uniform
        best_w = np.ones(len(model_order)) / len(model_order)
        best_f1 = _binary_f1(_fused_label_score(best_w, stacked), y_col, threshold)

    if refine:
        def neg_f1(w: np.ndarray) -> float:
            score = _fused_label_score(w, stacked)
            return 1.0 if score is None else -_binary_f1(score, y_col, threshold)

        res = minimize(
            neg_f1, best_w, method="Nelder-Mead",
            options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 500},
        )
        w_ref = np.clip(res.x, 0.0, None)
        if w_ref.sum() > 0:
            f1_ref = _binary_f1(_fused_label_score(w_ref, stacked), y_col, threshold)
            if f1_ref > best_f1:
                best_f1, best_w = f1_ref, w_ref / w_ref.sum()

    weights = {m: float(best_w[i]) for i, m in enumerate(model_order)}
    return weights, float(best_f1)


def fit_weights(
    model_probas: dict[str, np.ndarray],
    y_true: np.ndarray,
    threshold: float | dict[str, float] = 0.5,
    grid_step: float = 0.1,
    refine: bool = True,
) -> dict[str, dict[str, float]]:
    """Fit per-category weights for all labels. Returns ``{model: {label: w}}``."""
    model_order = list(model_probas)
    weights: dict[str, dict[str, float]] = {m: {} for m in model_order}
    for label in LABEL_COLS:
        label_w, _ = fit_label_weights(
            model_probas, y_true, label,
            grid_step=grid_step,
            threshold=_label_threshold(threshold, label),
            refine=refine,
            model_order=model_order,
        )
        for m in model_order:
            weights[m][label] = label_w[m]
    return weights


# --------------------------------------------------------------------------- #
# spec-letter alternative: L-BFGS-B on a smooth soft-F1 surrogate              #
# --------------------------------------------------------------------------- #
def fit_weights_lbfgsb(
    model_probas: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Fit weights with L-BFGS-B on a differentiable soft-F1 surrogate.

    Documented alternative that follows the ensemble rule's prescription of
    L-BFGS-B. Soft-F1 replaces the thresholded counts with the fused probability
    itself: ``tp=Σ p·y``, ``fp=Σ p·(1-y)``, ``fn=Σ (1-p)·y``, so the objective is
    differentiable. The surrogate optimum may differ from the true-F1 optimum,
    which is exactly why ``fit_weights`` (grid + Nelder-Mead on true F1) is the
    default.
    """
    model_order = list(model_probas)
    weights: dict[str, dict[str, float]] = {m: {} for m in model_order}

    for label in LABEL_COLS:
        label_idx = LABEL_COLS.index(label)
        stacked = _stack_label(model_probas, label_idx, model_order)
        y_col = np.asarray(y_true, dtype=float)[:, label_idx]

        def neg_soft_f1(w: np.ndarray) -> float:
            w = np.clip(w, 0.0, None)
            total = w.sum()
            if total <= 0:
                return 1.0
            p = (w[:, None] * stacked).sum(axis=0) / total
            tp = float((p * y_col).sum())
            fp = float((p * (1.0 - y_col)).sum())
            fn = float(((1.0 - p) * y_col).sum())
            denom = 2 * tp + fp + fn
            return 1.0 if denom == 0 else -(2 * tp) / denom

        x0 = np.ones(len(model_order)) / len(model_order)
        res = minimize(
            neg_soft_f1, x0, method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * len(model_order),
        )
        w = np.clip(res.x, 0.0, None)
        w = w / w.sum() if w.sum() > 0 else x0
        for i, m in enumerate(model_order):
            weights[m][label] = float(w[i])

    return weights


# --------------------------------------------------------------------------- #
# per-label threshold tuning                                                   #
# --------------------------------------------------------------------------- #
def tune_thresholds(
    model_probas: dict[str, np.ndarray],
    y_true: np.ndarray,
    weights: dict[str, dict[str, float]],
    grid: np.ndarray | None = None,
) -> dict[str, float]:
    """Scan per-label thresholds on the fused score to maximise each label's F1."""
    grid = np.linspace(0.05, 0.95, 19) if grid is None else grid
    _, fused = fuse(model_probas, weights, threshold=0.5)
    y_true = np.asarray(y_true, dtype=int)

    thresholds: dict[str, float] = {}
    for i, label in enumerate(LABEL_COLS):
        y_col = y_true[:, i]
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            f1 = float(f1_score(y_col, (fused[:, i] >= t).astype(int), zero_division=0))
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[label] = best_t
    return thresholds


# --------------------------------------------------------------------------- #
# persistence                                                                  #
# --------------------------------------------------------------------------- #
def save_weights(weights: dict[str, dict[str, float]],
                 path: str | Path = _DEFAULT_WEIGHTS_PATH) -> None:
    """Persist weights as ``{model: {label: float}}`` (schema read by load_weights)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights, indent=2))


def save_thresholds(thresholds: dict[str, float],
                    path: str | Path = _DEFAULT_THRESHOLDS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(thresholds, indent=2))
