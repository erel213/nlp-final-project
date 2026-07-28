from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
)

LABEL_COLS: list[str] = ["benign", "PII", "financial", "confidential"]


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_cols: list[str] = LABEL_COLS,
    normalize: Literal["true", "pred", "all"] | None = None,
    figsize: tuple[int, int] | None = None,
    return_fig: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], plt.Figure]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    n = len(label_cols)
    figsize = figsize or (4 * n, 4)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    cms: dict[str, np.ndarray] = {}
    for ax, label in zip(axes, label_cols):
        i = label_cols.index(label)
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], normalize=normalize)
        cms[label] = cm
        fmt = ".2f" if normalize else "d"
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            ax=ax,
            cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        ax.set_title(label)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.tight_layout()
    if return_fig:
        return cms, fig
    return cms


def plot_per_label_bars(
    metrics_results: dict,
    metric_keys: tuple[str, ...] = ("precision", "recall", "f1"),
    figsize: tuple[int, int] | None = None,
    return_fig: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, plt.Figure]:
    per_label = metrics_results["per_label"]
    rows = []
    for label, vals in per_label.items():
        row = {"label": label}
        row.update({k: vals.get(k, 0.0) for k in metric_keys})
        row["support"] = vals.get("support", 0)
        rows.append(row)
    df = pd.DataFrame(rows)

    figsize = figsize or (max(8, 2 * len(metric_keys) * len(rows)), 5)
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(rows))
    bar_width = 0.8 / len(metric_keys)
    for i, key in enumerate(metric_keys):
        offset = (i - len(metric_keys) / 2 + 0.5) * bar_width
        ax.bar(x + offset, df[key], width=bar_width, label=key)

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-label metrics")
    ax.legend()
    fig.tight_layout()

    if return_fig:
        return df, fig
    return df


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label_cols: list[str] = LABEL_COLS,
    figsize: tuple[int, int] | None = None,
    return_fig: bool = False,
) -> dict[str, dict] | tuple[dict[str, dict], plt.Figure]:
    if y_proba is None:
        raise ValueError("y_proba is required for ROC curves; rule-based models do not provide probabilities.")

    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    figsize = figsize or (7, 6)
    fig, ax = plt.subplots(figsize=figsize)

    curves: dict[str, dict] = {}
    mean_fpr = np.linspace(0, 1, 200)
    tprs_interp: list[np.ndarray] = []

    for i, label in enumerate(label_cols):
        try:
            auc = float(roc_auc_score(y_true[:, i], y_proba[:, i]))
            fpr, tpr, _ = roc_curve(y_true[:, i], y_proba[:, i])
            curves[label] = {"fpr": fpr, "tpr": tpr, "auc": auc}
            ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
            tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        except ValueError:
            curves[label] = {"fpr": np.array([]), "tpr": np.array([]), "auc": None}

    if tprs_interp:
        mean_tpr = np.mean(tprs_interp, axis=0)
        macro_auc = float(np.mean([c["auc"] for c in curves.values() if c["auc"] is not None]))
        ax.plot(mean_fpr, mean_tpr, "k--", label=f"macro avg (AUC={macro_auc:.3f})")
        curves["macro"] = {"fpr": mean_fpr, "tpr": mean_tpr, "auc": macro_auc}

    ax.plot([0, 1], [0, 1], "gray", linestyle=":")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if return_fig:
        return curves, fig
    return curves


def plot_pr_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label_cols: list[str] = LABEL_COLS,
    figsize: tuple[int, int] | None = None,
    return_fig: bool = False,
) -> dict[str, dict] | tuple[dict[str, dict], plt.Figure]:
    if y_proba is None:
        raise ValueError("y_proba is required for PR curves; rule-based models do not provide probabilities.")

    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    figsize = figsize or (7, 6)
    fig, ax = plt.subplots(figsize=figsize)

    # Iso-F1 contours
    f1_levels = [0.2, 0.4, 0.6, 0.8]
    recall_grid = np.linspace(0.01, 1.0, 200)
    for f1 in f1_levels:
        precision_grid = f1 * recall_grid / (2 * recall_grid - f1 + 1e-12)
        mask = (precision_grid >= 0) & (precision_grid <= 1)
        ax.plot(recall_grid[mask], precision_grid[mask], color="lightgray", linestyle=":", linewidth=0.8)
        ax.annotate(f"F1={f1}", xy=(recall_grid[mask][-1], precision_grid[mask][-1]), fontsize=7, color="gray")

    curves: dict[str, dict] = {}
    for i, label in enumerate(label_cols):
        try:
            ap = float(average_precision_score(y_true[:, i], y_proba[:, i]))
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_proba[:, i])
            curves[label] = {"precision": precision, "recall": recall, "ap": ap}
            ax.plot(recall, precision, label=f"{label} (AP={ap:.3f})")
        except ValueError:
            curves[label] = {"precision": np.array([]), "recall": np.array([]), "ap": None}

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left")
    fig.tight_layout()

    if return_fig:
        return curves, fig
    return curves


def plot_model_comparison_table(
    results: dict[str, dict],
    highlight_best: bool = True,
    return_fig: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, plt.Figure]:
    rows = []
    all_have_auc = all(
        r.get("auc_roc", {}).get("macro") is not None for r in results.values()
    )

    for model_name, r in results.items():
        ci = r.get("macro_f1_ci") or {}
        row: dict = {
            "model": model_name,
            "macro_f1": r.get("macro_f1"),
            "macro_f1_ci": f"[{ci.get('lower', 0):.3f}, {ci.get('upper', 0):.3f}]" if ci else "",
        }
        per = r.get("per_label", {})
        for label in LABEL_COLS:
            row[f"{label}_f1"] = per.get(label, {}).get("f1")
        if all_have_auc:
            row["macro_auc"] = r.get("auc_roc", {}).get("macro")
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")

    figsize = (max(10, len(df.columns) * 1.5), max(3, len(df) * 0.6 + 1))
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    col_labels = list(df.columns)
    cell_text = []
    for _, row in df.iterrows():
        cell_text.append([
            f"{v:.4f}" if isinstance(v, float) else (str(v) if v is not None else "—")
            for v in row
        ])

    table = ax.table(
        cellText=cell_text,
        rowLabels=list(df.index),
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(list(range(len(col_labels))))

    if highlight_best:
        numeric_col_indices = [
            i for i, c in enumerate(col_labels)
            if c not in ("macro_f1_ci",)
        ]
        for col_i in numeric_col_indices:
            col_vals = []
            for row_i, row_data in enumerate(cell_text):
                try:
                    col_vals.append((float(row_data[col_i]), row_i))
                except (ValueError, TypeError):
                    pass
            if col_vals:
                best_row = max(col_vals, key=lambda x: x[0])[1]
                cell = table[best_row + 1, col_i]
                cell.set_facecolor("#d4f0d4")

    fig.tight_layout()
    if return_fig:
        return df, fig
    return df


def plot_ablation_table(
    ablation_df: pd.DataFrame,
    return_fig: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, plt.Figure]:
    df = ablation_df.copy()
    baseline_mask = df["configuration"] == "full_ensemble"
    if baseline_mask.any():
        baseline_macro = df.loc[baseline_mask, "macro_f1"].values[0]
    else:
        baseline_macro = None

    figsize = (max(12, len(df.columns) * 1.4), max(3, len(df) * 0.7 + 1))
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    display_cols = [c for c in df.columns if c != "configuration"]
    cell_text = []
    cell_colors = []

    for _, row in df.iterrows():
        row_text = []
        row_colors = []
        for col in display_cols:
            val = row[col]
            if isinstance(val, float):
                row_text.append(f"{val:.4f}")
                if col == "macro_f1" and baseline_macro is not None:
                    delta = val - baseline_macro
                    if abs(delta) < 0.001:
                        row_colors.append("#ffffff")
                    elif delta < 0:
                        intensity = min(1.0, abs(delta) * 10)
                        r_val = 1.0
                        g_val = 1.0 - 0.5 * intensity
                        row_colors.append(f"#{int(r_val*255):02x}{int(g_val*255):02x}{int(g_val*255):02x}")
                    else:
                        intensity = min(1.0, delta * 10)
                        g_val = 1.0
                        r_val = 1.0 - 0.5 * intensity
                        row_colors.append(f"#{int(r_val*255):02x}{int(g_val*255):02x}{int(r_val*255):02x}")
                else:
                    row_colors.append("#ffffff")
            else:
                row_text.append(str(val) if val is not None else "—")
                row_colors.append("#ffffff")
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    table = ax.table(
        cellText=cell_text,
        rowLabels=list(df["configuration"]),
        colLabels=display_cols,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(list(range(len(display_cols))))
    ax.set_title("Ablation Study", pad=12)
    fig.tight_layout()

    if return_fig:
        return df, fig
    return df
