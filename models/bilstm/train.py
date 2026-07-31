"""Training entry point for the Bi-LSTM + char-CNN model.

Usage:
    python -m models.bilstm.train [--epochs 20] [--batch_size 64] [--lr 1e-3]

GloVe vectors must be downloaded separately and placed at:
    data/embeddings/glove.6B.100d.txt
Download: https://nlp.stanford.edu/data/glove.6B.zip  (822 MB)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.preprocessing import load_split
from models.bilstm.dataset import DLPTextDataset, collate_fn
from models.bilstm.model import BiLSTMCharCNN
from models.bilstm.vocab import Vocabulary

LABEL_COLS = ["benign", "PII", "financial", "confidential"]
_ROOT = Path(__file__).parent
CHECKPOINT_DIR = _ROOT / "checkpoint"
GLOVE_PATH = _ROOT.parent.parent / "data" / "embeddings" / "glove.6B.100d.txt"


def _load_glove(path: Path, vocab: Vocabulary, emb_dim: int = 100) -> torch.Tensor:
    emb = torch.zeros(vocab.n_words, emb_dim)
    nn.init.normal_(emb[2:], mean=0.0, std=0.01)  # PAD=0 and UNK=1 stay zero

    if not path.exists():
        print(f"[warn] GloVe not found at {path} — using random init for all words.")
        return emb

    found = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word in vocab.word2idx:
                idx = vocab.word2idx[word]
                emb[idx] = torch.tensor([float(v) for v in parts[1 : emb_dim + 1]])
                found += 1
    print(f"GloVe: {found}/{vocab.n_words - 2} vocab words initialised from GloVe vectors.")
    return emb


def _pos_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    pos = labels.sum(axis=0).clip(1)
    neg = len(labels) - pos
    return torch.tensor(neg / pos, dtype=torch.float32).to(device)


def train(args: argparse.Namespace) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    train_df = load_split("train")
    val_df = load_split("validation")
    train_texts = train_df["source_text"].tolist()
    val_texts = val_df["source_text"].tolist()
    train_labels = train_df[LABEL_COLS].values.astype(np.float32)
    val_labels = val_df[LABEL_COLS].values.astype(np.float32)

    print("Building vocabulary...")
    vocab = Vocabulary()
    vocab.build_from_texts(train_texts, min_freq=2)
    print(f"Vocab: {vocab.n_words} words, {vocab.n_chars} chars")

    pretrained = _load_glove(GLOVE_PATH, vocab)

    train_ds = DLPTextDataset(train_texts, train_labels, vocab)
    val_ds = DLPTextDataset(val_texts, val_labels, vocab)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    model = BiLSTMCharCNN(vocab.n_words, vocab.n_chars, pretrained_emb=pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=_pos_weights(train_labels, device))
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for word_ids, char_ids, mask, labels in train_loader:
            word_ids, char_ids, mask, labels = (
                word_ids.to(device), char_ids.to(device), mask.to(device), labels.to(device),
            )
            optimizer.zero_grad()
            loss = criterion(model(word_ids, char_ids, mask), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for word_ids, char_ids, mask, labels in val_loader:
                word_ids, char_ids, mask, labels = (
                    word_ids.to(device), char_ids.to(device), mask.to(device), labels.to(device),
                )
                val_loss += criterion(model(word_ids, char_ids, mask), labels).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch:02d}/{args.epochs}  train={avg_train:.4f}  val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pt")
            vocab.save(CHECKPOINT_DIR / "vocab.json")
            print(f"  → checkpoint saved (val_loss={best_val_loss:.4f})")

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    train(parser.parse_args())
