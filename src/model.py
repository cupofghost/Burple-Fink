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

    def forward(self, x: torch.Tensor, hidden=None):
        """x: (batch, time) integer ids -> logits: (batch, time, vocab_size).

        ``hidden`` lets generation carry the LSTM state forward one step at a time.
        """
        emb = self.embedding(x)                 # (batch, time, embedding_dim)
        out, hidden = self.lstm(emb, hidden)    # (batch, time, hidden_dim)
        logits = self.head(out)                 # (batch, time, vocab_size)
        return logits, hidden


class DualCharRNN(CharRNN):
    """CharRNN plus a second head that regresses a numeric attribute from a name.

    This is Shane's name+RGB trick, generalized to any scalar attribute (WS-4):
    the same LSTM encoding that learns to *spell* a name also learns to predict a
    *number* associated with it (e.g. a paint color's perceived brightness).
    ``forward`` is inherited unchanged, so this class is a drop-in replacement for
    ``CharRNN`` wherever only character generation is needed (e.g. ``src.sample``'s
    ``generate_one``/``generate_many``).
    """

    def __init__(self, vocab_size: int, cfg: Config, pad_id: int = 0):
        super().__init__(vocab_size, cfg, pad_id=pad_id)
        self.attr_head = nn.Linear(cfg.hidden_dim, 1)

    def forward_attr(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x: (batch, time) ids, lengths: (batch,) -> (batch,) predicted attribute.

        ``x`` is expected in the same ``[START, ...chars]`` shape ``data.make_pairs``
        produces as its input side. The output at position ``lengths - 1`` is the
        LSTM's state right after reading START and every character of the name, so
        the attribute head predicts from a representation of the whole name.
        """
        emb = self.embedding(x)
        out, _ = self.lstm(emb)                            # (batch, time, hidden_dim)
        last = out[torch.arange(out.size(0)), lengths - 1]  # (batch, hidden_dim)
        return self.attr_head(last).squeeze(-1)
