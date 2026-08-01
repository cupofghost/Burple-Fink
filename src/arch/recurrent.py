"""The two recurrent cores: LSTM (the default) and GRU.

Both are thin wrappers around a stock `torch.nn` module, because both already do
incremental decoding natively -- their whole point is that the sequence so far is
summarized in a fixed-size state.

The only wrinkle is the *shape* of that state. `src/sample.py` reads
`hidden[0][-1]` to get the last timestep's top-layer vector for the dual-output
value head:

    LSTM  hidden = (h, c)   h: (layers, batch, hidden)  ->  h[-1] is (batch, hidden) OK
    GRU   hidden = h        h: (layers, batch, hidden)  ->  h[0] is layer 0, not the
                                                            top layer, and h[0][-1]
                                                            is one batch row. Wrong.

So the GRU's state is wrapped in a 1-tuple `(h,)` on the way out and unwrapped on
the way back in. That costs nothing and makes `state[0][-1]` mean the same thing
for every architecture in this package.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _dropout(cfg) -> float:
    # PyTorch warns (and does nothing) if dropout is set on a 1-layer RNN.
    return cfg.dropout if cfg.num_layers > 1 else 0.0


def build_lstm(cfg) -> nn.LSTM:
    """The original core, unchanged: 2 stacked LSTM layers, batch-first."""
    return nn.LSTM(
        input_size=cfg.embedding_dim,
        hidden_size=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=_dropout(cfg),
        batch_first=True,
    )


def build_gru(cfg) -> nn.GRU:
    """Same shape as the LSTM, one gate fewer.

    A GRU has 3 gates to the LSTM's 4, so at equal width it is ~25% smaller and
    proportionally faster per step. Whether that costs accuracy on short names is
    an empirical question -- see the measurement table in the WS-9 report.
    """
    return nn.GRU(
        input_size=cfg.embedding_dim,
        hidden_size=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=_dropout(cfg),
        batch_first=True,
    )


def run_recurrent(core: nn.Module, emb: torch.Tensor, state=None):
    """Run an `nn.LSTM`/`nn.GRU`, normalizing the GRU's state to a 1-tuple."""
    if isinstance(core, nn.GRU):
        h = None if state is None else state[0]
        out, h = core(emb, h)
        return out, (h,)
    return core(emb, state)
