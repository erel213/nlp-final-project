from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CHAR_EMB_DIM = 30
CHAR_KERNELS = [2, 3, 4]
CHAR_FILTERS = 64
CHAR_OUT_DIM = len(CHAR_KERNELS) * CHAR_FILTERS  # 192

WORD_EMB_DIM = 100
LSTM_INPUT_DIM = WORD_EMB_DIM + CHAR_OUT_DIM  # 292 (word + char-CNN)
LSTM_INPUT_DIM_NO_CHAR = WORD_EMB_DIM         # 100 (word only, char-CNN ablation)
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
LSTM_OUT_DIM = LSTM_HIDDEN * 2  # 512 (bidirectional)
N_LABELS = 4


class _CharCNN(nn.Module):
    def __init__(self, n_chars: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_chars, CHAR_EMB_DIM, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(CHAR_EMB_DIM, CHAR_FILTERS, k) for k in CHAR_KERNELS]
        )

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        # char_ids: (N, max_word_len)  N = batch * seq_len
        x = self.embedding(char_ids).transpose(1, 2)  # (N, emb_dim, max_word_len)
        pooled = [
            F.adaptive_max_pool1d(F.relu(conv(x)), 1).squeeze(-1) for conv in self.convs
        ]
        return torch.cat(pooled, dim=-1)  # (N, 192)


class BiLSTMCharCNN(nn.Module):
    def __init__(
        self,
        n_words: int,
        n_chars: int,
        pretrained_emb: torch.Tensor | None = None,
        freeze_word_emb: bool = True,
        use_char_cnn: bool = True,
    ) -> None:
        super().__init__()
        self.use_char_cnn = use_char_cnn
        self.word_embedding = nn.Embedding(n_words, WORD_EMB_DIM, padding_idx=0)
        if pretrained_emb is not None:
            self.word_embedding.weight.data.copy_(pretrained_emb)
        # Freeze only when real GloVe vectors were loaded (ADR-002). Freezing a random
        # N(0, 0.01) matrix would train the model on frozen noise (comment 012), so a
        # random-init variant keeps the word channel learnable.
        self.word_embedding.weight.requires_grad = not freeze_word_emb

        self.char_cnn = _CharCNN(n_chars) if use_char_cnn else None

        self.lstm = nn.LSTM(
            LSTM_INPUT_DIM if use_char_cnn else LSTM_INPUT_DIM_NO_CHAR,
            LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            bidirectional=True,
            dropout=0.3,  # inter-layer dropout (ADR-004)
            batch_first=True,
        )
        self.output_dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(LSTM_OUT_DIM, N_LABELS)

    def forward(
        self,
        word_ids: torch.Tensor,   # (B, L)
        char_ids: torch.Tensor,   # (B, L, max_word_len)
        mask: torch.Tensor,        # (B, L) bool
    ) -> torch.Tensor:
        B, L = word_ids.shape

        word_emb = self.word_embedding(word_ids)              # (B, L, 100)

        if self.use_char_cnn:
            char_emb = self.char_cnn(char_ids.view(B * L, -1))    # (B*L, 192)
            char_emb = char_emb.view(B, L, -1)                    # (B, L, 192)
            x = torch.cat([word_emb, char_emb], dim=-1)           # (B, L, 292)
        else:
            x = word_emb                                          # (B, L, 100)

        lengths = mask.sum(dim=1).cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        lstm_out = self.output_dropout(lstm_out)               # (B, L, 512)

        # Mean pooling over non-padding positions (ADR-003)
        mask_f = mask[:, :lstm_out.size(1)].unsqueeze(-1).float()
        pooled = (lstm_out * mask_f).sum(dim=1) / lengths.to(lstm_out.device).unsqueeze(1).float()

        return self.classifier(pooled)                         # (B, 4)
