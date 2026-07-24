"""Data loading and the character <-> integer vocabulary.

Pipeline: read a newline-separated name list -> clean it -> build a vocabulary of
every character (plus the special START/END/PAD tokens) -> turn each name into an
(input, target) pair for next-character prediction.
"""

from __future__ import annotations

import glob
import json
import os
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


def load_name_value_pairs(path: str) -> List[Tuple[str, float]]:
    """Read ``name<TAB>value`` rows for dual-output training (WS-4, see src/train_dual.py).

    Same order-preserving, first-occurrence-wins de-dupe as :func:`load_names`.
    """
    seen = set()
    pairs: List[Tuple[str, float]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            name, _, value = line.partition("\t")
            name = name.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            pairs.append((name, float(value)))
    if not pairs:
        raise ValueError(f"No usable name/value pairs found in {path!r}")
    return pairs


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


# --- shared vocabulary (the precondition for fine-tuning) ------------------------
#
# Stage 0 builds a Vocab per dataset. That is fine when every model is trained from
# scratch, but it *breaks fine-tuning*: a base model's embedding and output layers
# are sized to its vocabulary, so re-using its weights against a differently-sized
# per-dataset vocab is impossible. WS-2 therefore fixes ONE vocabulary spanning every
# dataset (persisted to ``data/shared_vocab.json``) and builds ``Vocab`` from it for
# both pretraining and every fine-tune, so all checkpoints share identical
# embedding/head dimensions. See HANDOFF §6 and docs/PLAN.md.

DEFAULT_VOCAB_PATH = "data/shared_vocab.json"


def list_dataset_files(data_dir: str = "data") -> List[str]:
    """Return dataset paths (``*.txt``) under ``data_dir``, sorted.

    Sorting makes the shared vocabulary deterministic regardless of the order the
    filesystem happens to return files in.
    """
    return sorted(glob.glob(os.path.join(data_dir, "*.txt")))


def load_all_names(paths: List[str]) -> List[str]:
    """Concatenate the names from several files, de-duplicated across all of them.

    This is the corpus the base model pretrains on: every dataset at once, so it
    learns the spelling regularities common to all name domains before any one of
    them is specialized by fine-tuning.
    """
    seen = set()
    names: List[str] = []
    for path in paths:
        for name in load_names(path):
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
    return names


def build_shared_vocab(paths: List[str]) -> Vocab:
    """Build one vocabulary covering every character across every dataset.

    Because ``Vocab`` derives its character set from the names it is given, feeding
    it the union of all datasets yields a superset that every per-dataset vocab is
    contained in — exactly what fine-tuning needs.
    """
    return Vocab(load_all_names(paths))


def save_shared_vocab(vocab: Vocab, path: str = DEFAULT_VOCAB_PATH) -> None:
    """Persist a vocab as JSON. Control-character special tokens are ``\\uXXXX``-escaped."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(vocab.to_dict(), fh, indent=2)
        fh.write("\n")


def load_shared_vocab(path: str = DEFAULT_VOCAB_PATH) -> Vocab:
    """Reload a shared vocab saved by :func:`save_shared_vocab`."""
    with open(path, "r", encoding="utf-8") as fh:
        return Vocab.from_dict(json.load(fh))


def filter_to_vocab(names: List[str], vocab: Vocab) -> Tuple[List[str], List[str]]:
    """Split ``names`` into (representable, dropped) given a (shared) vocab.

    Fine-tuning a base model onto a new dataset only works for characters the base
    model was built with; a stray character would otherwise raise a ``KeyError`` deep
    inside encoding. Returning the dropped names lets callers report them honestly
    instead of crashing.
    """
    known = set(vocab.stoi)
    kept, dropped = [], []
    for name in names:
        (kept if all(ch in known for ch in name) else dropped).append(name)
    return kept, dropped


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
