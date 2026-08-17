"""Training entry point for the Bi-LSTM + char-CNN model.

Usage:
    python -m models.bilstm.train [--epochs 20] [--batch_size 64] [--lr 1e-3]

GloVe vectors must be downloaded separately and placed at:
    data/embeddings/glove.6B.100d.txt
Download: https://nlp.stanford.edu/data/glove.6B.zip  (822 MB)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.preprocessing import load_train_holdout
from models.bilstm.dataset import DLPTextDataset, collate_fn
from models.bilstm.model import BiLSTMCharCNN
from models.bilstm.vocab import Vocabulary

LABEL_COLS = ["benign", "PII", "financial", "confidential"]
_ROOT = Path(__file__).parent
CHECKPOINT_DIR = _ROOT / "checkpoint"
GLOVE_PATH = _ROOT.parent.parent / "data" / "embeddings" / "glove.6B.100d.txt"


def _load_glove(
    path: Path,
    vocab: Vocabulary,
    emb_dim: int = 100,
    allow_random_init: bool = False,
) -> tuple[torch.Tensor, bool]:
    """Load GloVe vectors for ``vocab`` and report whether GloVe was actually used.

    Returns ``(emb, glove_loaded)``. ``glove_loaded`` is ``True`` only when the GloVe
    file was found and at least one vocab word was initialised from it. When the file
    is missing this HARD-FAILS by default (a random-init run must not be silently
    reported as the specified GloVe design, per comment 012). Pass
    ``allow_random_init=True`` to opt into a random-init run explicitly; the returned
    ``glove_loaded=False`` is then threaded into saved metadata so it cannot drift.
    """
    emb = torch.zeros(vocab.n_words, emb_dim)
    nn.init.normal_(emb[2:], mean=0.0, std=0.01)  # PAD=0 and UNK=1 stay zero

    if not path.exists():
        msg = (
            f"GloVe not found at {path}. The Bi-LSTM design specifies GloVe 6B 100d "
            f"(ADR-005). Download it (https://nlp.stanford.edu/data/glove.6B.zip, "
            f"extract glove.6B.100d.txt into {path.parent}/), or pass "
            f"allow_random_init=True (CLI: --allow-random-init) to run an explicit "
            f"random-init variant that will be recorded as glove_loaded=false."
        )
        if not allow_random_init:
            raise FileNotFoundError(msg)
        print(f"[warn] {msg}\n[warn] Proceeding with random init (glove_loaded=false).")
        return emb, False

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
    glove_loaded = found > 0
    if not glove_loaded and not allow_random_init:
        raise RuntimeError(
            f"GloVe file {path} was found but 0/{vocab.n_words - 2} vocab words matched "
            f"— refusing to report this as a GloVe run. Check the file/format, or pass "
            f"allow_random_init=True to accept a random-init variant."
        )
    return emb, glove_loaded


def _pos_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    pos = labels.sum(axis=0).clip(1)
    neg = len(labels) - pos
    return torch.tensor(neg / pos, dtype=torch.float32).to(device)


def fit(
    train_df,
    val_df,
    args: argparse.Namespace,
    ckpt_path: Path,
    device: torch.device | None = None,
) -> tuple[Path, float]:
    """Train the Bi-LSTM on ``train_df``, selecting the checkpoint on ``val_df``.

    A fresh ``Vocabulary`` is built from ``train_df`` and saved as ``vocab.json`` next
    to ``ckpt_path`` (its ``predict.py`` needs the matching vocab). ``val_df`` is the
    selection/DEV partition (early stopping only) — never a TEST split. Returns
    ``(ckpt_path, best_val_loss)``. Shared by the CLI entry point, the training
    notebook, and the k-fold orchestrator so their training regimes are identical.

    Recognised optional ``args`` attributes (default to the specified design when
    absent): ``allow_random_init`` (bool) opts into a random-init word channel when
    GloVe is missing; ``use_char_cnn`` (bool, default True) toggles the char-CNN for
    the with/without-char-CNN ablation. The *actual* GloVe-loaded status and the
    char-CNN setting are written to ``run_meta.json`` next to the checkpoint so the
    reported configuration cannot silently drift from what was trained.
    """
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path = ckpt_path.parent / "vocab.json"
    allow_random_init = bool(getattr(args, "allow_random_init", False))
    use_char_cnn = bool(getattr(args, "use_char_cnn", True))
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

    print("Building vocabulary...")
    vocab = Vocabulary()
    vocab.build_from_texts(train_texts, min_freq=2)
    print(f"Vocab: {vocab.n_words} words, {vocab.n_chars} chars")

    pretrained, glove_loaded = _load_glove(
        GLOVE_PATH, vocab, allow_random_init=allow_random_init
    )
    print(
        f"Config: glove_loaded={glove_loaded}  use_char_cnn={use_char_cnn}  "
        f"freeze_word_emb={glove_loaded}"
    )

    train_ds = DLPTextDataset(train_texts, train_labels, vocab)
    val_ds = DLPTextDataset(val_texts, val_labels, vocab)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    # Freeze the word channel only when real GloVe was loaded (freezing a random
    # matrix would train on frozen noise — comment 012).
    model = BiLSTMCharCNN(
        vocab.n_words,
        vocab.n_chars,
        pretrained_emb=pretrained,
        freeze_word_emb=glove_loaded,
        use_char_cnn=use_char_cnn,
    ).to(device)
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
            torch.save(model.state_dict(), ckpt_path)
            vocab.save(vocab_path)
            print(f"  → checkpoint saved (val_loss={best_val_loss:.4f})")

    # Persist the *actual* run configuration next to the checkpoint so downstream
    # eval/metadata reflects what was trained rather than a re-derived guess
    # (comment 012: glove_available was previously a path-existence check).
    meta_path = ckpt_path.parent / "run_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "glove_loaded": glove_loaded,
                "use_char_cnn": use_char_cnn,
                "freeze_word_emb": glove_loaded,
                "word_emb_dim": 100,
                "vocab_size": vocab.n_words,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "checkpoint": str(ckpt_path),
            },
            indent=2,
        )
    )

    print("Training complete.")
    return ckpt_path, best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--allow-random-init", dest="allow_random_init", action="store_true",
        help="Run with random word embeddings when GloVe is missing (recorded as "
             "glove_loaded=false); by default a missing GloVe file hard-fails.",
    )
    parser.add_argument(
        "--no-char-cnn", dest="use_char_cnn", action="store_false",
        help="Train the char-CNN ablation variant (word channel only) for the "
             "with/without-char-CNN comparison.",
    )
    parser.set_defaults(use_char_cnn=True)
    args = parser.parse_args()

    print("Loading data...")
    # Early stopping / checkpoint selection uses a seeded 10% held-out slice of
    # `train`. The `validation` split is reserved untouched as the TEST set for
    # final reporting only — never read here.
    train_df, val_df = load_train_holdout(frac=0.10, seed=42)
    # The no-char-CNN ablation writes to a separate checkpoint dir so it never
    # clobbers the deployed full model.
    ckpt = CHECKPOINT_DIR / ("best_model_no_charcnn.pt" if not args.use_char_cnn
                             else "best_model.pt")
    fit(train_df, val_df, args, ckpt)
