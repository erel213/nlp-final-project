"""K-fold cross-validation orchestrator for the SentinelMail ensemble.

Runs 5-fold CV over the ai4privacy ``train`` split (the ``validation`` TEST split stays
untouched — see ``data.preprocessing.load_kfold_splits``). Per fold it retrains the three
neural models (rule-based is zero-trained, only re-scored), fits ensemble weights on the
fold's inner-dev partition, evaluates every configuration on the held-out fold via
``run_ablation``, then aggregates mean ± std across folds and runs a paired ensemble-vs-
best-single significance test.

Design guarantees:
- Leakage-safe: ensemble weights are fit on ``inner_dev`` and reported on ``test_fold``,
  which are disjoint (see ``load_kfold_splits``).
- Non-destructive: per-fold checkpoints go to ``models/<m>/checkpoint/kfold/fold{i}/`` —
  the deployed ``checkpoint/best_model.pt`` and the single-TEST report are never touched.
- Resumable: each model's dev/test probabilities are cached as ``.npy`` under the fold's
  results dir. A fold whose caches exist is loaded without retraining.

Usage:
    python -m evaluation.kfold_cv [--k 5] [--epochs N] [--device cpu]
                                  [--max-train N] [--max-eval N] [--force]
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from data.preprocessing import load_kfold_splits
from ensemble.train_weights import fit_weights
from evaluation import (
    LABEL_COLS,
    aggregate_folds,
    fold_scores,
    from_dataframe,
    paired_fold_test,
    run_ablation,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "evaluation" / "results" / "kfold"
SUMMARY_PATH = REPO_ROOT / "evaluation" / "results" / "kfold_cv.json"

MODEL_NAMES = ["bert", "roberta", "bilstm", "rule_based"]
NEURAL_MODELS = ["bert", "roberta", "bilstm"]

# Per-model training defaults, mirroring each train.py's CLI argparse defaults so the
# CV runs reproduce the deployed single-model training regime. --epochs overrides all.
_NEURAL_DEFAULTS = {
    "bert": {"batch_size": 16, "lr": 2e-5, "epochs": 5},
    "roberta": {"batch_size": 16, "lr": 2e-5, "epochs": 5},
    "bilstm": {"batch_size": 64, "lr": 1e-3, "epochs": 20},
}


def _neural_args(model: str, epochs_override: int | None) -> Namespace:
    cfg = dict(_NEURAL_DEFAULTS[model])
    if epochs_override is not None:
        cfg["epochs"] = epochs_override
    return Namespace(**cfg)


def _fold_ckpt_dir(model: str, fold_idx: int) -> Path:
    return REPO_ROOT / "models" / model / "checkpoint" / "kfold" / f"fold{fold_idx}"


def _train_neural(model: str, train_df, dev_df, args: Namespace, fold_idx: int, device: str) -> Path:
    """Train one neural model for a fold and return its checkpoint path."""
    import torch

    ckpt = _fold_ckpt_dir(model, fold_idx) / "best_model.pt"
    dev = torch.device(device)
    if model == "bert":
        from models.bert.train import fit
    elif model == "roberta":
        from models.roberta.train import fit
    elif model == "bilstm":
        from models.bilstm.train import fit
    else:  # pragma: no cover - guarded by caller
        raise ValueError(model)
    fit(train_df, dev_df, args, ckpt, device=dev)
    return ckpt


def _build_detector(model: str, fold_idx: int, device: str):
    """Construct a detector from a fold's checkpoint (imports are lazy)."""
    ckpt = _fold_ckpt_dir(model, fold_idx) / "best_model.pt"
    if model == "bert":
        from models.bert.predict import BertDLPDetector

        return BertDLPDetector(str(ckpt), device=device)
    if model == "roberta":
        from models.roberta.predict import RobertaDLPDetector

        return RobertaDLPDetector(str(ckpt), device=device)
    if model == "bilstm":
        from models.bilstm.predict import BiLSTMDLPDetector

        vocab = _fold_ckpt_dir(model, fold_idx) / "vocab.json"
        return BiLSTMDLPDetector(str(ckpt), str(vocab), device=device)
    if model == "rule_based":
        from models.rule_based.detector import RuleBasedDetector

        return RuleBasedDetector()
    raise ValueError(f"Unknown model name: {model!r}")


def _subsample(df, n: int | None, seed: int):
    """Return a seeded ``n``-row subsample of ``df`` (or ``df`` if ``n`` is None/≥len)."""
    if n is None or n >= len(df):
        return df
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))[:n]
    return df.iloc[idx].reset_index(drop=True)


def _model_probas_for_fold(
    fold_idx: int,
    train_fit_df,
    inner_dev_df,
    test_fold_df,
    epochs_override: int | None,
    device: str,
    force: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return ``(dev_probas, test_probas)`` for every model on one fold.

    Neural models are trained (writing a per-fold checkpoint) unless a cached ``.npy``
    already exists for the model on this fold; rule-based is scored directly. Every
    model's dev/test probabilities are cached so a re-run skips completed work.
    """
    fold_dir = RESULTS_DIR / f"fold{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    dev_texts = inner_dev_df["source_text"].tolist()
    test_texts = test_fold_df["source_text"].tolist()

    dev_probas: dict[str, np.ndarray] = {}
    test_probas: dict[str, np.ndarray] = {}

    for model in MODEL_NAMES:
        dev_npy = fold_dir / f"probas_{model}_dev.npy"
        test_npy = fold_dir / f"probas_{model}_test.npy"

        if not force and dev_npy.exists() and test_npy.exists():
            print(f"[fold {fold_idx}] {model}: cached — loading probabilities.")
            dev_probas[model] = np.load(dev_npy)
            test_probas[model] = np.load(test_npy)
            continue

        if model in NEURAL_MODELS:
            print(f"[fold {fold_idx}] {model}: training on {len(train_fit_df):,} rows ...")
            _train_neural(
                model, train_fit_df, inner_dev_df,
                _neural_args(model, epochs_override), fold_idx, device,
            )

        print(f"[fold {fold_idx}] {model}: scoring dev ({len(dev_texts):,}) + test ({len(test_texts):,}) ...")
        detector = _build_detector(model, fold_idx, device)
        dev_p = detector.predict_proba(dev_texts).astype(np.float32)
        test_p = detector.predict_proba(test_texts).astype(np.float32)
        np.save(dev_npy, dev_p)
        np.save(test_npy, test_p)
        dev_probas[model] = dev_p
        test_probas[model] = test_p

    return dev_probas, test_probas


def run_kfold_cv(
    k: int = 5,
    seed: int = 42,
    inner_dev_frac: float = 0.10,
    epochs_override: int | None = None,
    device: str = "cpu",
    max_train: int | None = None,
    max_eval: int | None = None,
    force: bool = False,
) -> dict:
    """Run k-fold CV end-to-end and write the aggregated summary to ``kfold_cv.json``."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_tables = []

    for fold_idx, train_fit_df, inner_dev_df, test_fold_df in load_kfold_splits(
        k=k, seed=seed, inner_dev_frac=inner_dev_frac
    ):
        # Optional subsampling keeps smoke runs fast; full runs pass None.
        train_fit_df = _subsample(train_fit_df, max_train, seed + fold_idx)
        inner_dev_df = _subsample(inner_dev_df, max_eval, seed + fold_idx)
        test_fold_df = _subsample(test_fold_df, max_eval, seed + fold_idx)
        print(
            f"\n=== Fold {fold_idx + 1}/{k}  "
            f"train_fit={len(train_fit_df):,}  inner_dev={len(inner_dev_df):,}  "
            f"test={len(test_fold_df):,} ==="
        )

        dev_probas, test_probas = _model_probas_for_fold(
            fold_idx, train_fit_df, inner_dev_df, test_fold_df,
            epochs_override, device, force,
        )

        y_true_dev = from_dataframe(inner_dev_df).astype(int)
        y_true_test = from_dataframe(test_fold_df).astype(int)

        # Fit fusion weights on the DEV partition only (disjoint from the TEST fold),
        # then evaluate every configuration on the held-out TEST fold at threshold 0.5.
        weights = fit_weights(dev_probas, y_true_dev)
        ablation_df = run_ablation(test_probas, weights, y_true_test)

        fold_dir = RESULTS_DIR / f"fold{fold_idx}"
        ablation_df.to_csv(fold_dir / "ablation.csv", index=False)
        (fold_dir / "weights.json").write_text(json.dumps(weights, indent=2))
        fold_tables.append(ablation_df)
        print(ablation_df.to_string(index=False))

    aggregated = aggregate_folds(fold_tables)
    sig = paired_fold_test(
        fold_scores(fold_tables, "full_ensemble"),
        fold_scores(fold_tables, "best_single"),
    )

    summary = {
        "k": k,
        "seed": seed,
        "inner_dev_frac": inner_dev_frac,
        "n_folds_completed": len(fold_tables),
        "label_cols": LABEL_COLS,
        "aggregated": aggregated.to_dict(orient="records"),
        "ensemble_vs_best_single": sig,
        "notes": (
            "CV over ai4privacy train split; validation TEST split untouched. "
            "Weights fit on inner-dev, evaluated on held-out fold. Paired t-test has "
            "low power at k=5 and folds are not fully independent — descriptive robustness."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print("\n=== Aggregated (mean ± std across folds) ===")
    print(aggregated.to_string(index=False))
    delta, p = sig["mean_delta"], sig["p_value"]
    p_str = f"{p:.4g}" if p is not None else "n/a"
    print(f"\nfull_ensemble − best_single macro-F1: Δ={delta:+.4f}  paired-t p={p_str}")
    print(f"Summary written to {SUMMARY_PATH}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run k-fold cross-validation for the ensemble.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inner-dev-frac", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override every neural model's epoch count (for fast smoke runs).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-train", type=int, default=None,
                        help="Subsample each fold's training rows (smoke runs).")
    parser.add_argument("--max-eval", type=int, default=None,
                        help="Subsample each fold's dev/test rows (smoke runs).")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cached per-fold probabilities and recompute.")
    args = parser.parse_args()

    run_kfold_cv(
        k=args.k,
        seed=args.seed,
        inner_dev_frac=args.inner_dev_frac,
        epochs_override=args.epochs,
        device=args.device,
        max_train=args.max_train,
        max_eval=args.max_eval,
        force=args.force,
    )


if __name__ == "__main__":
    main()
