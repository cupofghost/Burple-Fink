"""The char generator itself: Embedding -> core -> Linear.

This is the same family of model Janelle Shane used for paint colors. It reads a
sequence of characters and, at every position, outputs a score for each possible
next character.

The *core* in the middle is selectable via `cfg.arch` ("lstm" | "gru" |
"transformer") and lives in `src/arch/`. Everything around it -- the embedding,
the vocabulary head, the optional dual-output value head, and the public API --
is identical for all three, which is why `train.py`, `sample.py`, `finetune.py`,
`pretrain.py` and `export_web.py` needed no changes when the choice appeared.

The class is still called `CharRNN` even though a transformer is not recurrent.
That is deliberate: five call sites and every checkpoint on disk name it, and a
rename would buy nothing but churn.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import arch as arch_pkg
from .config import Config


class CharRNN(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config, pad_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        # Plain str, not a submodule -- state_dict is unaffected.
        self.arch = arch_pkg.validate_arch(getattr(cfg, "arch", "lstm"))

        # Each character id -> a small learned vector. padding_idx keeps PAD's
        # embedding fixed at zero and out of the gradient.
        self.embedding = nn.Embedding(vocab_size, cfg.embedding_dim, padding_idx=pad_id)

        # The core. Registered under an architecture-specific attribute name
        # ("lstm" / "gru" / "transformer") so that a default-config model's
        # state_dict keys are exactly what they were before this lane existed --
        # old checkpoints load unchanged, and export_web.py still finds
        # `lstm.weight_ih_l0`.
        core_attr, core = arch_pkg.build_core(cfg)
        self._core_attr = core_attr
        setattr(self, core_attr, core)

        # Projects each core output back to one score per vocabulary character.
        self.head = nn.Linear(cfg.hidden_dim, vocab_size)

        # WS-4 dual-output: an optional second head that regresses one scalar
        # attribute (e.g. a car brand's founding year) from the same encoder.
        # None for every ordinary (non-dual) config, so existing checkpoints and
        # callers are completely unaffected.
        self.value_head = nn.Linear(cfg.hidden_dim, 1) if cfg.dual_output else None

    @property
    def core(self) -> nn.Module:
        """The selected architecture's core module, whatever it is called."""
        return getattr(self, self._core_attr)

    def encode(self, x: torch.Tensor, hidden=None):
        """Run just the embedding + core, exposing the per-timestep output.

        Factored out of :meth:`forward` so the dual-output value head (which needs
        each sequence's *last valid* timestep, not just the vocab logits) can read
        the same core output without duplicating the embedding/core call.
        """
        emb = self.embedding(x)                                    # (batch, time, emb)
        out, hidden = arch_pkg.run_core(self.arch, self.core, emb, hidden)
        return out, hidden                                         # (batch, time, hidden)

    def forward(self, x: torch.Tensor, hidden=None):
        """x: (batch, time) integer ids -> logits: (batch, time, vocab_size).

        ``hidden`` lets generation carry state forward one step at a time. Its
        contents depend on the architecture -- an LSTM's ``(h, c)``, a GRU's
        ``(h,)``, or the transformer's ``PrefixState`` -- but callers only ever
        pass it straight back in, and ``hidden[0][-1]`` is the last timestep's
        (batch, hidden_dim) vector for all three. See ``src/arch/__init__.py``.
        """
        out, hidden = self.encode(x, hidden)
        logits = self.head(out)                 # (batch, time, vocab_size)
        return logits, hidden

    def predict_value(self, state: torch.Tensor) -> torch.Tensor:
        """Regress the trained scalar attribute from an encoder hidden vector.

        ``state`` is (batch, hidden_dim): either the top-layer hidden state after
        a single generation step, or ``encode()``'s output gathered at each
        sequence's last non-pad timestep during training. Only valid when this
        model was built with ``cfg.dual_output=True``.
        """
        if self.value_head is None:
            raise RuntimeError("predict_value() requires a model built with dual_output=True")
        return self.value_head(state).squeeze(-1)
