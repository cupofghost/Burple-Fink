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
    def __init__(self, vocab_size: int, cfg: Config, pad_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size

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

        # WS-4 dual-output: an optional second head that regresses one scalar
        # attribute (e.g. a car brand's founding year) from the same LSTM encoder.
        # None for every ordinary (non-dual) config, so existing checkpoints and
        # callers are completely unaffected.
        self.value_head = nn.Linear(cfg.hidden_dim, 1) if cfg.dual_output else None

    def encode(self, x: torch.Tensor, hidden=None):
        """Run just the embedding + LSTM, exposing the per-timestep output.

        Factored out of :meth:`forward` so the dual-output value head (which needs
        each sequence's *last valid* timestep, not just the vocab logits) can read
        the same LSTM output without duplicating the embedding/LSTM call.
        """
        emb = self.embedding(x)                 # (batch, time, embedding_dim)
        out, hidden = self.lstm(emb, hidden)    # (batch, time, hidden_dim)
        return out, hidden

    def forward(self, x: torch.Tensor, hidden=None):
        """x: (batch, time) integer ids -> logits: (batch, time, vocab_size).

        ``hidden`` lets generation carry the LSTM state forward one step at a time.
        """
        out, hidden = self.encode(x, hidden)
        logits = self.head(out)                 # (batch, time, vocab_size)
        return logits, hidden

    def predict_value(self, state: torch.Tensor) -> torch.Tensor:
        """Regress the trained scalar attribute from an LSTM hidden vector.

        ``state`` is (batch, hidden_dim): either the top-layer hidden state after
        a single generation step, or ``encode()``'s output gathered at each
        sequence's last non-pad timestep during training. Only valid when this
        model was built with ``cfg.dual_output=True``.
        """
        if self.value_head is None:
            raise RuntimeError("predict_value() requires a model built with dual_output=True")
        return self.value_head(state).squeeze(-1)
