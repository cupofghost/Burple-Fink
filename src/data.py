"""Data loading and the character <-> integer vocabulary.

Pipeline: read a newline-separated name list -> clean it -> build a vocabulary of
every character (plus the special START/END/PAD tokens) -> turn each name into an
(input, target) pair for next-character prediction.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from .config import PAD_TOKEN, START_TOKEN, END_TOKEN


def load_names(path: str) -> List[str]:
    """Read names from a file, one per line, stripped and de-duplicated.

    Order is preserved (first occurrence wins) so runs are reproducible.
    """
    seen = set()
    names: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            name = line.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
    if not names:
        raise ValueError(f"No usable names found in {path!r}")
    return names


class Vocab:
    """Bidirectional mapping between characters and integer ids.

    The three special tokens always occupy the first, fixed indices so a saved
    checkpoint's vocabulary can be reconstructed unambiguously.
    """

    def __init__(self, names: List[str]):
        # PAD must be index 0 so it doubles as the padding_idx of the embedding.
        specials = [PAD_TOKEN, START_TOKEN, END_TOKEN]
        chars = sorted({ch for name in names for ch in name})
        self.itos = specials + chars                       # id  -> char
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}  # char -> id

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def start_id(self) -> int:
        return self.stoi[START_TOKEN]

    @property
    def end_id(self) -> int:
        return self.stoi[END_TOKEN]

    def encode(self, name: str) -> List[int]:
        """Wrap a name in START/END and convert to ids."""
        return [self.start_id] + [self.stoi[ch] for ch in name] + [self.end_id]

    def decode(self, ids: List[int]) -> str:
        """Turn ids back into a string, dropping any special tokens."""
        specials = {self.pad_id, self.start_id, self.end_id}
        return "".join(self.itos[i] for i in ids if i not in specials)

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d: dict) -> "Vocab":
        obj = cls.__new__(cls)
        obj.itos = list(d["itos"])
        obj.stoi = {ch: i for i, ch in enumerate(obj.itos)}
        return obj


def make_pairs(names: List[str], vocab: Vocab) -> List[Tuple[List[int], List[int]]]:
    """For each name build (input, target) where target is input shifted by one.

    Example for "Go":  encoded = [START, G, o, END]
        input  = [START, G, o]
        target = [G,     o, END]
    i.e. at every position the model must predict the *following* character.
    """
    pairs = []
    for name in names:
        ids = vocab.encode(name)
        pairs.append((ids[:-1], ids[1:]))
    return pairs


def make_batches(
    pairs: List[Tuple[List[int], List[int]]],
    batch_size: int,
    pad_id: int,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
):
    """Yield padded (inputs, targets, lengths) tensors, one batch at a time.

    Names within a batch are padded to the longest name in that batch; ``lengths``
    lets the model and the loss ignore the padding.
    """
    order = list(range(len(pairs)))
    if shuffle:
        perm = torch.randperm(len(pairs), generator=generator).tolist()
        order = perm

    for start in range(0, len(order), batch_size):
        idxs = order[start:start + batch_size]
        batch = [pairs[i] for i in idxs]
        lengths = [len(inp) for inp, _ in batch]
        max_len = max(lengths)

        inputs = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        targets = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        for row, (inp, tgt) in enumerate(batch):
            inputs[row, : len(inp)] = torch.tensor(inp, dtype=torch.long)
            targets[row, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)

        yield inputs, targets, torch.tensor(lengths, dtype=torch.long)
