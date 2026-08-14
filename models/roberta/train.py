"""Fine-tuning entry point for RoBERTa DLP classifier.

Usage:
    python -m models.roberta.train [--epochs 5] [--batch_size 16] [--lr 2e-5]

Requires:
    pip install transformers torch
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data.preprocessing import load_train_holdout, LABEL_COLS
from models.roberta.dataset import DLPDataset, collate_fn, MODEL_NAME
from models.roberta.model import RobertaForDLPClassification

_ROOT = Path(__file__).parent
CHECKPOINT_DIR = _ROOT / "checkpoint"

_LABEL_SMOOTHING = 0.05
_WARMUP_FRACTION = 0.10


def _pos_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    pos = labels.sum(axis=0).clip(1)
    neg = len(labels) - pos
    return torch.tensor(neg / pos, dtype=torch.float32).to(device)


def _smooth_labels(labels: torch.Tensor, eps: float = _LABEL_SMOOTHING) -> torch.Tensor:
    """Binary label smoothing: 1 → (1-eps), 0 → eps."""
    return labels * (1.0 - eps) + (1.0 - labels) * eps


def _macro_f1(logits_all: list[torch.Tensor], labels_all: list[torch.Tensor]) -> float:
    proba = torch.sigmoid(torch.cat(logits_all)).cpu().numpy()
    y_pred = (proba >= 0.5).astype(int)
    y_true = torch.cat(labels_all).cpu().numpy().astype(int)
    f1s = []
    for i in range(4):
        tp = ((y_pred[:, i] == 1) & (y_true[:, i] == 1)).sum()
        fp = ((y_pred[:, i] == 1) & (y_true[:, i] == 0)).sum()
        fn = ((y_pred[:, i] == 0) & (y_true[:, i] == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s))


def fit(
    train_df,
    val_df,
    args: argparse.Namespace,
    ckpt_path: Path,
    device: torch.device | None = None,
) -> tuple[Path, float]:
    """Fine-tune RoBERTa on ``train_df``, selecting the checkpoint on ``val_df``.

    ``val_df`` is the selection/DEV partition (early stopping only) — never a TEST
    split. Writes the best checkpoint to ``ckpt_path`` and returns
    ``(ckpt_path, best_val_macro_f1)``. Shared by the CLI ``train`` and the k-fold
    orchestrator so their training regimes are identical.
    """
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Device: {device}")

    train_texts = train_df["source_text"].tolist()
    val_texts = val_df["source_text"].tolist()
    train_labels = train_df[LABEL_COLS].values.astype(np.float32)
    val_labels = val_df[LABEL_COLS].values.astype(np.float32)
    print(f"Train: {len(train_texts):,}  |  Selection holdout: {len(val_texts):,}")

    print("Encoding inputs...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = DLPDataset(train_texts, train_labels, tokenizer)
    val_ds = DLPDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
    )

    model = RobertaForDLPClassification().to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,} total, {trainable:,} trainable")

    criterion = nn.BCEWithLogitsLoss(pos_weight=_pos_weights(train_labels, device))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(_WARMUP_FRACTION * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, _smooth_labels(labels))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_logits_all, val_labels_all = [], []
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels_dev = labels.to(device)
                logits = model(input_ids, attention_mask)
                val_loss += criterion(logits, _smooth_labels(labels_dev)).item()
                val_logits_all.append(logits.detach())
                val_labels_all.append(labels)

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        val_f1 = _macro_f1(val_logits_all, val_labels_all)

        marker = "  ✓ saved" if val_f1 > best_val_f1 else ""
        print(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"train={avg_train:.4f}  val={avg_val:.4f}  macro-F1={val_f1:.4f}{marker}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)

    print(f"\nBest val macro-F1: {best_val_f1:.4f}  →  {ckpt_path}")
    return ckpt_path, best_val_f1


def train(args: argparse.Namespace) -> None:
    print("Loading data...")
    # Early stopping / checkpoint selection uses a seeded 10% held-out slice of
    # `train` (per .claude/rules/model-roberta.md ADR-005 / model-bert.md). The
    # `validation` split is reserved untouched as the TEST set for final reporting
    # only — never read here.
    train_df, val_df = load_train_holdout(frac=0.10, seed=42)
    fit(train_df, val_df, args, CHECKPOINT_DIR / "best_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    train(parser.parse_args())
