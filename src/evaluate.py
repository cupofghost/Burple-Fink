"""Evaluation harness: does a checkpoint actually generate good, novel names? (WS-3)

Sampling *looks* fun but "looks fun" is not measurable, so we cannot tell whether
fine-tuning (WS-2) truly beats the from-scratch baseline without numbers. This module
computes three cheap, honest metrics for a checkpoint against the training set stored
inside it:

* **Novelty** — the fraction of generated names that are *not* memorized copies of
  training names. A generator that parrots its data is worthless; we want new words.
* **Plausibility** — a character-bigram log-likelihood under the training
  distribution. Real names score some average log-likelihood; if generated names score
  close to that, they are "spelled like" the domain. Far below = gibberish, far above =
  the model is playing it too safe and echoing common fragments. We report both so the
  ratio is interpretable. We also report the model's own next-char loss (NLL) on the
  training names as a fit sanity-check (labeled clearly as *not* held-out).
* **Diversity** — how many distinct names it produces (uniqueness) and how different
  they are from each other (mean pairwise edit distance). A model that emits the same
  five names forever is diverse-on-paper but useless.

Usage:
    python -m src.evaluate --checkpoint checkpoints/car_models_ft.pt
    python -m src.evaluate --checkpoint checkpoints/base.pt --num 300 --temperature 0.9
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from typing import Dict, List, Set

import torch
import torch.nn.functional as F

from .config import START_TOKEN, END_TOKEN
from .data import Vocab, make_batches, make_pairs
from .model import CharRNN
from .sample import generate_one, load_checkpoint


# --- plausibility: a smoothed character-bigram model of the training names -----------

def _bigram_logprobs(names: List[str]) -> Dict[str, Dict[str, float]]:
    """Add-1-smoothed P(next_char | char) table over START/END-bracketed names.

    This is a deliberately dumb reference model: it captures "which letters tend to
    follow which" in the domain without any of the RNN's machinery, so scoring the
    RNN's output against it is an independent check on spelling plausibility.
    """
    alphabet: Set[str] = set()
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for name in names:
        seq = START_TOKEN + name + END_TOKEN
        alphabet.update(seq)
        for a, b in zip(seq, seq[1:]):
            counts[a][b] += 1

    vocab_size = len(alphabet)
    logprobs: Dict[str, Dict[str, float]] = {}
    for a in alphabet:
        total = sum(counts[a].values()) + vocab_size  # +1 smoothing over the alphabet
        logprobs[a] = {
            b: math.log((counts[a].get(b, 0) + 1) / total) for b in alphabet
        }
    return logprobs


def _score_name(name: str, table: Dict[str, Dict[str, float]]) -> float:
    """Mean per-character bigram log-likelihood (nats/char) of one name."""
    seq = START_TOKEN + name + END_TOKEN
    total, n = 0.0, 0
    for a, b in zip(seq, seq[1:]):
        row = table.get(a)
        if row is None or b not in row:
            # A character combination unseen even in smoothing (out-of-alphabet char).
            total += math.log(1e-6)
        else:
            total += row[b]
        n += 1
    return total / max(n, 1)


def mean_bigram_ll(names: List[str], table: Dict[str, Dict[str, float]]) -> float:
    if not names:
        return float("nan")
    return sum(_score_name(n, table) for n in names) / len(names)


# --- diversity: Levenshtein edit distance --------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Classic Levenshtein distance (iterative, O(len(a)*len(b)) time, O(len(b)) space)."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,           # deletion
                cur[j - 1] + 1,        # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = cur
    return prev[-1]


def mean_pairwise_edit_distance(names: List[str], max_pairs: int = 2000,
                                rng: random.Random | None = None) -> float:
    """Average edit distance over a random sample of name pairs (full set is O(n^2))."""
    rng = rng or random.Random(0)
    if len(names) < 2:
        return float("nan")
    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
    if len(pairs) > max_pairs:
        pairs = rng.sample(pairs, max_pairs)
    return sum(_edit_distance(names[i], names[j]) for i, j in pairs) / len(pairs)


# --- fit sanity-check: the model's own next-char loss on the training names ----------

@torch.no_grad()
def training_nll(model: CharRNN, vocab: Vocab, names: List[str], cfg,
                 device: str = "cpu") -> float:
    """Mean cross-entropy (nats/char) of the model on its *training* names.

    Not held-out — the model has seen these — so it measures fit, not generalization.
    Reported only as a sanity check that training converged; lower is a better fit.
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_id, reduction="sum")
    pairs = make_pairs(names, vocab)
    total_loss, total_tokens = 0.0, 0
    for inputs, targets, _ in make_batches(pairs, cfg.batch_size, vocab.pad_id,
                                           shuffle=False):
        inputs, targets = inputs.to(device), targets.to(device)
        logits, _ = model(inputs)
        total_loss += criterion(logits.reshape(-1, logits.size(-1)),
                                targets.reshape(-1)).item()
        total_tokens += int((targets != vocab.pad_id).sum().item())
    return total_loss / max(total_tokens, 1)


# --- the harness ---------------------------------------------------------------------

def evaluate(checkpoint: str, num: int = 200, temperature: float = 0.8,
             min_length: int = 2, device: str = "cpu",
             seed: int | None = 0) -> Dict[str, float]:
    """Generate ``num`` names and return a dict of metrics (also printed as a table)."""
    if seed is not None:
        torch.manual_seed(seed)
    model, vocab, cfg, training_names = load_checkpoint(checkpoint, device)
    training_set = set(training_names)

    # Raw sample: keep duplicates and training-set collisions so we can *measure* them.
    raw: List[str] = []
    attempts, cap = 0, num * 50
    while len(raw) < num and attempts < cap:
        attempts += 1
        name = generate_one(model, vocab, temperature, cfg.max_length, device=device)
        if len(name) >= min_length:
            raw.append(name)

    distinct = sorted(set(raw))
    novel = [n for n in distinct if n not in training_set]
    table = _bigram_logprobs(list(training_names))

    metrics = {
        "generated": len(raw),
        "distinct": len(distinct),
        "uniqueness": len(distinct) / max(len(raw), 1),
        "novelty": len(novel) / max(len(distinct), 1),
        "avg_length": sum(len(n) for n in raw) / max(len(raw), 1),
        "avg_edit_distance": mean_pairwise_edit_distance(distinct),
        "bigram_ll_generated": mean_bigram_ll(distinct, table),
        "bigram_ll_training": mean_bigram_ll(list(training_names), table),
        "training_nll": training_nll(model, vocab, list(training_names), cfg, device),
    }

    _print_report(checkpoint, temperature, metrics)
    return metrics


def _print_report(checkpoint: str, temperature: float, m: Dict[str, float]) -> None:
    gen_ll, train_ll = m["bigram_ll_generated"], m["bigram_ll_training"]
    ratio = gen_ll / train_ll if train_ll else float("nan")
    print(f"\n=== Evaluation: {checkpoint} @ temperature {temperature} ===")
    print(f"  generated            : {m['generated']} names "
          f"({m['distinct']} distinct)")
    print(f"  novelty              : {m['novelty']*100:5.1f}%   "
          f"(distinct names not in training set)")
    print(f"  uniqueness           : {m['uniqueness']*100:5.1f}%   "
          f"(distinct / generated)")
    print(f"  avg edit distance    : {m['avg_edit_distance']:5.2f}    "
          f"(mean pairwise Levenshtein)")
    print(f"  avg length           : {m['avg_length']:5.2f} chars")
    print(f"  plausibility (bigram log-likelihood, nats/char — higher = more typical):")
    print(f"     generated         : {gen_ll:6.3f}")
    print(f"     training  (ref)   : {train_ll:6.3f}")
    print(f"     ratio             : {ratio:5.2f}    "
          f"(~1 = generated as typical as real names)")
    print(f"  model NLL on training: {m['training_nll']:6.3f} nats/char "
          f"(fit check, not held-out)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a char-RNN checkpoint (novelty / plausibility / diversity).")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--num", type=int, default=200, help="How many names to sample.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for reproducible metrics (use -1 for random).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate(args.checkpoint, num=args.num, temperature=args.temperature,
             device=device, seed=None if args.seed == -1 else args.seed)


if __name__ == "__main__":
    main()
