from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PAD_WORD = "<PAD>"
UNK_WORD = "<UNK>"
PAD_CHAR = "\x00"


class Vocabulary:
    """Word + character vocabulary serialized to vocab.pkl."""

    def __init__(self) -> None:
        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str] = []
        self.char2idx: dict[str, int] = {}
        self.idx2char: list[str] = []

    def build_from_texts(self, texts: list[str], min_freq: int = 2) -> None:
        word_counter: Counter = Counter()
        char_set: set[str] = set()
        for text in texts:
            tokens = text.lower().split()
            word_counter.update(tokens)
            for token in tokens:
                char_set.update(token)

        # Word vocab: 0=PAD, 1=UNK, 2..n=words with count >= min_freq
        self.idx2word = [PAD_WORD, UNK_WORD] + [
            w for w, c in word_counter.items() if c >= min_freq
        ]
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}

        # Char vocab: 0=PAD, 1..n=observed chars (sorted for reproducibility)
        self.idx2char = [PAD_CHAR] + sorted(char_set)
        self.char2idx = {c: i for i, c in enumerate(self.idx2char)}

    def encode_word(self, word: str) -> int:
        return self.word2idx.get(word.lower(), 1)  # 1 = UNK

    def encode_chars(self, word: str, max_word_len: int = 25) -> list[int]:
        chars = [self.char2idx.get(c, 0) for c in word.lower()[:max_word_len]]
        chars += [0] * (max_word_len - len(chars))
        return chars

    @property
    def n_words(self) -> int:
        return len(self.idx2word)

    @property
    def n_chars(self) -> int:
        return len(self.idx2char)

    def save(self, path: str | Path) -> None:
        data = {
            "idx2word": self.idx2word,
            "idx2char": self.idx2char,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        obj = cls()
        obj.idx2word = data["idx2word"]
        obj.word2idx = {w: i for i, w in enumerate(obj.idx2word)}
        obj.idx2char = data["idx2char"]
        obj.char2idx = {c: i for i, c in enumerate(obj.idx2char)}
        return obj
