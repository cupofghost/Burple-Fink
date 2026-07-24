"""The char-RNN itself: Embedding -> LSTM -> Linear.

This is the same family of model Janelle Shane used for paint colors. It reads a
sequence of characters and, at every position, outputs a score for each possible
next character.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Config


class CharRNN(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config, pad_id: int = 0, predict_value: bool = False):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.predict_value = predict_value

        # Each character id -> a small learned vector. padding_idx keeps PAD's
        # embedding fixed at zero and out of the gradient.
        self.embedding = nn.Embedding(vocab_size, cfg.embedding_dim, padding_idx=pad_id)

        # The recurrent core. batch_first => tensors are (batch, time, features).
        self.lstm = nn.LSTM(
            input_size=cfg.embedding_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Projects each LSTM output back to one score per vocabulary character.
        self.head = nn.Linear(cfg.hidden_dim, vocab_size)

        # Optional second head (WS-4): regresses one scalar attribute per name,
        # e.g. Shane's paint-color RGB trick generalized to any numeric attribute.
        # Kept separate from `forward` so the existing (logits, hidden) contract
        # every caller relies on never changes shape.
        if predict_value:
            self.value_head = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, x: torch.Tensor, hidden=None):
        """x: (batch, time) integer ids -> logits: (batch, time, vocab_size).

        ``hidden`` lets generation carry the LSTM state forward one step at a time.
        """
        emb = self.embedding(x)                 # (batch, time, embedding_dim)
        out, hidden = self.lstm(emb, hidden)    # (batch, time, hidden_dim)
        logits = self.head(out)                 # (batch, time, vocab_size)
        return logits, hidden

    def regress_value(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Predict the scalar attribute for each sequence in ``x``.

        ``x`` is a batch of encoded names (no END token, see ``src/train_dual.py``);
        ``lengths`` is each row's true length. Reads the value head off the LSTM's
        hidden state at each row's last real position, so padding never leaks in.
        """
        if not self.predict_value:
            raise RuntimeError("This model was not built with predict_value=True.")
        emb = self.embedding(x)
        out, _ = self.lstm(emb)                 # (batch, time, hidden_dim)
        idx = (lengths - 1).clamp(min=0)
        last = out[torch.arange(out.size(0), device=x.device), idx]  # (batch, hidden_dim)
        return self.value_head(last).squeeze(-1)
