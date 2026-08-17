"""Compute trivial baselines + a consolidated per-class comparison table.

Answers reviewer comment 004. All numbers are on the TEST split (ai4privacy
``validation``, English). Model probabilities are computed fresh via
``ensemble.probas.compute_probas`` (the on-disk proba cache was removed —
correctness by construction over cached-but-possibly-stale). No retraining;
expect BERT/RoBERTa inference wall-clock. Pass ``--device mps`` to speed it up.

Outputs:
  * evaluation/results/majority_metrics.json
  * evaluation/results/trivial_metrics.json          (regex text-only baseline)
  * evaluation/results/per_class_comparison.json
  * evaluation/results/per_class_comparison.csv

Run:  .venv/bin/python -m evaluation.run_baselines
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from data.preprocessing import load_split
from ensemble.probas import compute_probas
from evaluation import (
    LABEL_COLS,
    compute_all_metrics,
    from_dicts,
    save_results,
)
from evaluation.baselines import (
    MajorityBaseline,
    RegexTextBaseline,
    training_priors_english,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "evaluation" / "results"
WEIGHTS_PATH = REPO_ROOT / "ensemble" / "weights.json"
ENSEMBLE_METRICS = RESULTS_DIR / "ensemble_metrics.json"


def _fuse_with_per_label_thresholds(model_probas, weights, thresholds):
    """Late fusion with per-label weights AND per-label thresholds.

    Mirrors ensemble/ evaluation but applies the stored per-label thresholds so
    the Ensemble column of the comparison table matches the reported ensemble.
    """
    n = next(iter(model_probas.values())).shape[0]
    fused = np.zeros((n, len(LABEL_COLS)), dtype=float)
    for col_i, label in enumerate(LABEL_COLS):
        total = 0.0
        for m, p in model_probas.items():
            w = weights.get(m, {}).get(label, 0.0)
            fused[:, col_i] += w * p[:, col_i]
            total += w
        if total > 0:
            fused[:, col_i] /= total
    y_pred = np.zeros_like(fused, dtype=int)
    for col_i, label in enumerate(LABEL_COLS):
        y_pred[:, col_i] = (fused[:, col_i] >= thresholds[label]).astype(int)
    return y_pred, fused


def main(device: str = "cpu") -> None:
    # 1. Compute aligned model probabilities, y_true, and TEST texts (fresh inference).
    model_probas, y_true, texts = compute_probas("validation", device=device)
    n = len(texts)
    print(f"[baselines] TEST split loaded: n={n}")

    priors = training_priors_english()
    print("[baselines] training priors (English):",
          {k: round(v, 4) for k, v in priors.items()})

    per_class: dict[str, dict] = {}

    def _record(name, y_pred, y_proba=None):
        m = compute_all_metrics(y_true, y_pred, y_proba=y_proba)
        per_class[name] = {
            **{lbl: round(m["per_label"][lbl]["f1"], 4) for lbl in LABEL_COLS},
            "macro_f1": round(m["macro_f1"], 4),
        }
        return m

    # 2. Majority baseline.
    maj = MajorityBaseline(priors)
    y_pred_maj = from_dicts(maj.predict_batch(texts))
    m_maj = _record("Majority", y_pred_maj)
    save_results(
        m_maj, RESULTS_DIR / "majority_metrics.json",
        model_name="majority_prior", n_samples=n, threshold=0.5,
        extra_metadata={
            "baseline": "per-label training-prior constant predictor",
            "constant_prediction": maj.constant,
            "training_priors": {k: round(v, 6) for k, v in priors.items()},
        },
    )
    print("[baselines] wrote majority_metrics.json")

    # 3. Regex text-only baseline.
    regex = RegexTextBaseline()
    y_pred_regex = from_dicts(regex.predict_batch(texts))
    m_regex = _record("Regex", y_pred_regex)
    save_results(
        m_regex, RESULTS_DIR / "trivial_metrics.json",
        model_name="regex_text_only", n_samples=n, threshold=0.5,
        extra_metadata={
            "baseline": "text-only hard-pattern classifier (no privacy_mask access)",
            "note": "pure-patterns variant: NO TitleCase name-guess; harshest PII floor",
            "distinct_from": "Presidio rule_based model uses statistical NER; this does not",
        },
    )
    print("[baselines] wrote trivial_metrics.json")

    # 4. Existing models from cached probas @ 0.5 (per-model reporting threshold).
    name_map = {
        "rule_based": "Presidio rule_based",
        "bilstm": "Bi-LSTM",
        "bert": "BERT",
        "roberta": "RoBERTa",
    }
    for key, disp in name_map.items():
        proba = model_probas[key]
        y_pred = (proba >= 0.5).astype(int)
        _record(disp, y_pred, y_proba=proba)

    # 5. Ensemble on the SAME full TEST rows, using stored weights + per-label thresholds.
    weights = json.loads(WEIGHTS_PATH.read_text())
    ens_meta = json.loads(ENSEMBLE_METRICS.read_text())
    thresholds = ens_meta["metadata"]["thresholds"]
    y_pred_ens, y_proba_ens = _fuse_with_per_label_thresholds(
        model_probas, weights, thresholds
    )
    _record("Ensemble", y_pred_ens, y_proba=y_proba_ens)

    # 6. Persist the consolidated per-class comparison table.
    order = ["Majority", "Regex", "Presidio rule_based", "Bi-LSTM",
             "BERT", "RoBERTa", "Ensemble"]
    table = {name: per_class[name] for name in order if name in per_class}

    (RESULTS_DIR / "per_class_comparison.json").write_text(
        json.dumps(
            {
                "split": "validation",
                "n_samples": n,
                "note": "per-label F1 on full TEST split from cached probas; "
                        "models @0.5, Ensemble @per-label thresholds.",
                "table": table,
            },
            indent=2,
        )
    )
    with (RESULTS_DIR / "per_class_comparison.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", *LABEL_COLS, "macro_f1"])
        for name in order:
            if name in table:
                r = table[name]
                w.writerow([name, *[r[l] for l in LABEL_COLS], r["macro_f1"]])
    print("[baselines] wrote per_class_comparison.{json,csv}")

    # 7. Print the crux: regex PII recall vs transformer PII recall.
    def _pii_recall(y_pred):
        idx = LABEL_COLS.index("PII")
        tp = int(((y_pred[:, idx] == 1) & (y_true[:, idx] == 1)).sum())
        pos = int((y_true[:, idx] == 1).sum())
        return tp / pos if pos else float("nan")

    print("\n=== per-class F1 comparison (TEST, n={}) ===".format(n))
    header = f"{'model':<20} " + " ".join(f"{l:>13}" for l in LABEL_COLS) + f"{'macro':>10}"
    print(header)
    for name in order:
        if name in table:
            r = table[name]
            print(f"{name:<20} " + " ".join(f"{r[l]:>13.4f}" for l in LABEL_COLS)
                  + f"{r['macro_f1']:>10.4f}")

    print("\n=== PII recall (the crux) ===")
    print(f"Regex   PII recall: {_pii_recall(y_pred_regex):.4f}")
    for key, disp in name_map.items():
        yp = (model_probas[key] >= 0.5).astype(int)
        print(f"{disp:<8}PII recall: {_pii_recall(yp):.4f}")
    print(f"Majority PII recall: {_pii_recall(y_pred_maj):.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baselines + per-class comparison table.")
    parser.add_argument("--device", default="cpu")
    main(device=parser.parse_args().device)
