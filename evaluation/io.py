from __future__ import annotations

import json
from pathlib import Path

LABEL_COLS: list[str] = ["benign", "PII", "financial", "confidential"]

_AUC_DEFAULTS = {label: None for label in LABEL_COLS}
_AUC_DEFAULTS["macro"] = None


def save_results(
    metrics: dict,
    path: str | Path,
    model_name: str,
    dataset: str = "ai4privacy/pii-masking-300k",
    split: str = "validation",
    n_samples: int | None = None,
    threshold: float = 0.5,
    extra_metadata: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "model": model_name,
        "dataset": dataset,
        "split": split,
        "language_filter": "English",
        "n_samples": n_samples,
        "threshold": threshold,
        "per_label": metrics.get("per_label", {}),
        "macro_f1": metrics.get("macro_f1"),
        "macro_f1_ci": metrics.get("macro_f1_ci"),
        "auc_roc": metrics.get("auc_roc", {**_AUC_DEFAULTS}),
        "metadata": {
            "n_bootstrap": 1000,
            "random_state": 42,
            **(extra_metadata or {}),
        },
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def load_results(path: str | Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    if "auc_roc" not in data:
        data["auc_roc"] = {**_AUC_DEFAULTS}
    else:
        for key in list(_AUC_DEFAULTS):
            data["auc_roc"].setdefault(key, None)
    if "macro_f1_ci" not in data:
        data["macro_f1_ci"] = None
    return data


def load_all_results(results_dir: str | Path) -> dict[str, dict]:
    results_dir = Path(results_dir)
    results: dict[str, dict] = {}
    for json_file in sorted(results_dir.glob("*.json")):
        data = load_results(json_file)
        model_name = data.get("model", json_file.stem)
        results[model_name] = data
    return results
