"""Data loading for WS-4 (dual-output: name + numeric attribute).

Mirrors ``src/data.py``'s name-only pipeline but for ``name<TAB>value`` files under
``data/dual/``. Kept as its own module (rather than extending ``data.py``) so the
single-column dataset loader that every other stage depends on stays untouched.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from .data import Vocab


def load_name_value_pairs(path: str) -> List[Tuple[str, float]]:
    """Read ``name<TAB>value`` lines, stripped and de-duplicated by name."""
    seen = set()
    pairs: List[Tuple[str, float]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            name, _, value = line.partition("\t")
            if not _ or name in seen:
                continue
            seen.add(name)
            pairs.append((name, float(value)))
    if not pairs:
        raise ValueError(f"No usable name<TAB>value pairs found in {path!r}")
    return pairs


def normalize_values(values: List[float]) -> Tuple[List[float], float, float]:
    """Z-score normalize so the regression head trains on a well-scaled target.

    Returns (normalized values, mean, std) — the mean/std are saved in the
    checkpoint so sampling can map predictions back to the original scale.
    """
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = max(variance ** 0.5, 1e-8)
    return [(v - mean) / std for v in values], mean, std


def make_dual_pairs(
    pairs: List[Tuple[str, float]], vocab: Vocab
) -> List[Tuple[List[int], List[int], float]]:
    """Like ``data.make_pairs`` but each row also carries its (normalized) value."""
    out = []
    for name, value in pairs:
        ids = vocab.encode(name)
        out.append((ids[:-1], ids[1:], value))
    return out


def make_dual_batches(
    triples: List[Tuple[List[int], List[int], float]],
    batch_size: int,
    pad_id: int,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
):
    """Yield padded (inputs, targets, values, lengths) tensors, one batch at a time.

    ``lengths`` counts the input side, matching what ``DualCharRNN.forward_attr``
    expects: the input sequence is ``[START, ...chars]`` (mirrors
    ``data.make_pairs``'s input/target shift, which drops the trailing END), so the
    LSTM output at position ``lengths - 1`` is the state right after reading START
    and every character of the name — primed to predict END next.
    """
    order = list(range(len(triples)))
    if shuffle:
        order = torch.randperm(len(triples), generator=generator).tolist()

    for start in range(0, len(order), batch_size):
        idxs = order[start:start + batch_size]
        batch = [triples[i] for i in idxs]
        lengths = [len(inp) for inp, _, _ in batch]
        max_len = max(lengths)

        inputs = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        targets = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        values = torch.zeros(len(batch), dtype=torch.float)
        for row, (inp, tgt, value) in enumerate(batch):
            inputs[row, : len(inp)] = torch.tensor(inp, dtype=torch.long)
            targets[row, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)
            values[row] = value

        yield inputs, targets, values, torch.tensor(lengths, dtype=torch.long)
