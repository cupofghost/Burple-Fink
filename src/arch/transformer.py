"""A causal (decoder-only) transformer core, plus the state trick that lets it
pretend to be recurrent.

Why this file is more interesting than `recurrent.py`
-----------------------------------------------------
`src/sample.py` generates one character at a time: it feeds a single token in,
gets one step of logits back, and carries `hidden` forward to the next step. An
LSTM or GRU does that natively -- the whole sequence so far *is* the state.

A transformer has no recurrent state. Every layer of self-attention looks at the
entire prefix. To honor the same `(logits, hidden)` contract, `hidden` has to carry
enough of the past to reconstruct that attention. So `PrefixState` carries the
accumulated *embedded prefix*, and each step re-runs attention over prefix + new
token.

Why recompute the prefix instead of caching keys and values
-----------------------------------------------------------
A KV cache is the fast answer: store each layer's K and V, append one column per
step, and the per-step cost drops from O(T^2) to O(T). It is also the answer that
is easy to get subtly wrong -- one stale position index or one off-by-one in the
mask and generation still *looks* fine while quietly conditioning on the wrong
thing. The bug shows up as slightly worse names, which is exactly the kind of
regression nobody notices.

Recomputing the prefix is correct by construction: the tensor fed to the stack at
step t is literally the same tensor a single full-sequence forward pass would have
built, so the equivalence test in `tests/test_arch.py` is checking arithmetic, not
bookkeeping. The cost it buys back is bounded and small here -- names are capped at
`cfg.max_length` (40) and positions at `cfg.max_position` (64), so a full name is a
few dozen 64-wide attention matrices. Generation is not this repo's bottleneck;
training is. If names ever get long enough for this to matter, swap this class for a
KV cache and keep the same test.

Padding
-------
No key-padding mask is needed and none is used. `src/data.py` right-pads every
batch, and attention here is causal, so a real position at time t only ever attends
to positions <= t -- all of which are real. PAD's own embedding is zeroed by the
shared `nn.Embedding(padding_idx=...)` in `src/model.py`, `in_proj` is bias-free so
it stays zero through the input projection, and the loss ignores PAD targets. PAD
therefore contributes nothing to any real position's output. `tests/test_arch.py`
pins that.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn


class PrefixState(NamedTuple):
    """The transformer's stand-in for a recurrent hidden state.

    `hidden` is (1, batch, hidden_dim): the last timestep's output, shaped like an
    RNN's top-layer state so `state[0][-1]` is (batch, hidden_dim) -- which is what
    `src/sample.py` feeds to the dual-output value head.

    `prefix` is (batch, time, embedding_dim): every embedded token seen so far,
    including the ones just consumed. It is the actual state; `hidden` is a
    convenience view for callers written against the RNN contract.
    """

    hidden: torch.Tensor
    prefix: torch.Tensor


class _Block(nn.Module):
    """One pre-norm decoder block: causal self-attention, then a feed-forward.

    Pre-norm (normalize *before* each sublayer, add the residual raw) rather than
    the original post-norm, because it trains without a warmup schedule -- and the
    warmup knob in `Config` belongs to another lane.
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg.hidden_dim
        self.ln_attn = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.ln_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, cfg.ff_dim),
            nn.GELU(),
            nn.Linear(cfg.ff_dim, d),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, h: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        normed = self.ln_attn(h)
        attended, _ = self.attn(
            normed, normed, normed, attn_mask=causal_mask, need_weights=False
        )
        h = h + self.drop(attended)
        h = h + self.drop(self.ff(self.ln_ff(h)))
        return h


class TransformerCore(nn.Module):
    """Embeddings in, one hidden vector per position out -- same as the RNN cores.

    Reuses the config fields that already exist for the RNNs (`hidden_dim` is the
    model width, `num_layers` is the block count, `dropout`) plus the three declared
    for this lane: `num_heads`, `ff_dim`, `max_position`.
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg.hidden_dim
        if cfg.num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {cfg.num_heads}")
        if d % cfg.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({d}) must be divisible by num_heads ({cfg.num_heads}); "
                f"each head gets hidden_dim / num_heads channels"
            )
        if cfg.max_position < 1:
            raise ValueError(f"max_position must be >= 1, got {cfg.max_position}")

        self.max_position = cfg.max_position

        # The shared character embedding is embedding_dim wide (32 by default) so
        # that every architecture reads the same table; the transformer works at
        # hidden_dim. bias=False keeps PAD's zero embedding at exactly zero.
        self.in_proj = nn.Linear(cfg.embedding_dim, d, bias=False)

        # Learned absolute positions. Learned rather than sinusoidal because names
        # are short and position 0 ("first letter") genuinely behaves differently
        # from position 7 in this domain -- there is nothing to extrapolate to.
        self.pos_embedding = nn.Embedding(cfg.max_position, d)

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(_Block(cfg) for _ in range(cfg.num_layers))
        self.ln_final = nn.LayerNorm(d)

    def _causal_mask(self, time: int, device) -> torch.Tensor:
        """(time, time) boolean mask, True where attention is forbidden.

        Row t is allowed columns 0..t. A name generator that could see the future
        would score its own answer, so this is load-bearing, not decoration.
        """
        return torch.ones(time, time, dtype=torch.bool, device=device).triu(1)

    def forward(self, emb: torch.Tensor, state: PrefixState | None = None):
        """emb: (batch, new_time, embedding_dim) -> (batch, new_time, hidden_dim).

        With `state`, `emb` is only the *new* tokens; the returned outputs still
        cover just those new positions, but they attend over the whole prefix.
        """
        prefix = None if state is None else state.prefix
        full = emb if prefix is None else torch.cat([prefix, emb], dim=1)

        time = full.size(1)
        if time > self.max_position:
            raise ValueError(
                f"sequence length {time} exceeds max_position={self.max_position}. "
                f"The transformer has no positional embedding for position {time - 1}. "
                f"Raise Config.max_position (and retrain) or shorten the input."
            )

        positions = torch.arange(time, device=full.device)
        h = self.drop(self.in_proj(full) + self.pos_embedding(positions))

        mask = self._causal_mask(time, full.device)
        for block in self.blocks:
            h = block(h, mask)
        h = self.ln_final(h)

        out = h[:, full.size(1) - emb.size(1):, :]
        new_state = PrefixState(
            hidden=out[:, -1:, :].transpose(0, 1).contiguous(),  # (1, batch, hidden)
            prefix=full,
        )
        return out, new_state
