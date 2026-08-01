"""Selectable model cores (WS-9).

`src/model.py` owns the parts every architecture shares -- the character embedding,
the vocabulary head, and the optional dual-output value head. This package owns the
*core*: the thing in the middle that turns a sequence of character embeddings into a
sequence of hidden vectors.

The core contract, which every architecture here honors:

    run_core(name, core, emb, state) -> (out, state)

      emb    (batch, time, embedding_dim) embedded input characters
      out    (batch, time, hidden_dim)    one hidden vector per input position
      state  opaque; pass the returned value back in to continue the sequence

`state` is what makes one-character-at-a-time generation work in `src/sample.py`
without the sampler knowing which architecture it is driving. Two properties are
required of it, because the rest of the repo relies on them:

  1. Feeding a sequence in chunks and carrying `state` forward must give the same
     `out` as feeding the whole sequence at once (`tests/test_arch.py` proves it).
  2. `state[0][-1]` must be the (batch, hidden_dim) representation of the most
     recent timestep -- `src/sample.py` reads exactly that to drive the dual-output
     value head. LSTM's native `(h, c)` already satisfies it; GRU's bare `h` does
     not, so it is wrapped in a 1-tuple, and the transformer's `PrefixState` puts
     its last hidden vector first for the same reason.

Adding a fourth architecture means: add a builder branch to `build_core`, a dispatch
branch to `run_core`, and its name to `ARCHITECTURES`. Nothing outside this package
and `src/model.py` should need to change.
"""

from __future__ import annotations

import torch.nn as nn

from .recurrent import build_gru, build_lstm, run_recurrent
from .transformer import PrefixState, TransformerCore

__all__ = [
    "ARCHITECTURES",
    "PrefixState",
    "TransformerCore",
    "build_core",
    "run_core",
    "validate_arch",
]

# The valid values of `cfg.arch`. "lstm" is the default and reproduces every
# pre-wave-3 checkpoint bit for bit.
ARCHITECTURES = ("lstm", "gru", "transformer")

# Which attribute of CharRNN each architecture's core module is stored under.
# "lstm" MUST stay "lstm": existing checkpoints have `lstm.weight_ih_l0` etc. in
# their state dict, and `src/export_web.py` reads those exact key names.
_ATTR = {"lstm": "lstm", "gru": "gru", "transformer": "transformer"}


def validate_arch(name: str) -> str:
    """Return `name` if it is a supported architecture, else raise ValueError."""
    if name not in ARCHITECTURES:
        raise ValueError(
            f"unknown arch {name!r}; expected one of {', '.join(ARCHITECTURES)}"
        )
    return name


def core_attr(name: str) -> str:
    """The CharRNN attribute name that holds this architecture's core module."""
    return _ATTR[validate_arch(name)]


def build_core(cfg) -> tuple[str, nn.Module]:
    """Build the core for `cfg.arch`.

    Returns `(attribute_name, module)` rather than just the module so `CharRNN` can
    register it under an architecture-specific name -- which is what keeps the
    default LSTM's state-dict keys unchanged.
    """
    name = validate_arch(cfg.arch)
    if name == "lstm":
        return _ATTR[name], build_lstm(cfg)
    if name == "gru":
        return _ATTR[name], build_gru(cfg)
    return _ATTR[name], TransformerCore(cfg)


def run_core(name: str, core: nn.Module, emb, state=None):
    """Run one architecture's core over `emb`, continuing from `state`."""
    if name == "transformer":
        return core(emb, state)
    return run_recurrent(core, emb, state)
