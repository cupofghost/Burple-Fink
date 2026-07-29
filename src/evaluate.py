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

WS-7 adds three more: a **near-duplicate rate** (names within edit distance 1, and
separately 2, of a training name — the memorization novelty alone can't see), an
**honest held-out NLL** when the checkpoint carries WS-6's ``val_names`` (degrades
to "n/a" otherwise, no waiting on WS-6), and ``--sweep``, a temperature x decoding
grid that recommends a setting from the numbers instead of eyeballing one run.

Usage:
    python -m src.evaluate --checkpoint checkpoints/car_models_ft.pt
    python -m src.evaluate --checkpoint checkpoints/base.pt --num 300 --temperature 0.9
    python -m src.evaluate --checkpoint checkpoints/base.pt --sweep --compare checkpoints/base_ft.pt
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


# --- memorization: near-duplicates of the training set, not just exact copies -------

def near_duplicate_rate(names: List[str], training_names: List[str], max_dist: int = 1) -> float:
    """Fraction of ``names`` within edit distance <= ``max_dist`` of some training name.

    ``novelty`` only catches exact copies (distance 0); a model that learned the
    *list* rather than the *style* will produce a lot of one-letter-off variants
    that novelty scores as "new". This is the number that tells them apart.
    Length-difference pruning (edit distance >= abs length diff) keeps this cheap
    without needing a smarter string index.
    """
    if not names:
        return float("nan")
    by_length: Dict[int, List[str]] = defaultdict(list)
    for t in training_names:
        by_length[len(t)].append(t)

    hits = 0
    for name in names:
        found = False
        for length in range(len(name) - max_dist, len(name) + max_dist + 1):
            for t in by_length.get(length, ()):
                if _edit_distance(name, t) <= max_dist:
                    found = True
                    break
            if found:
                break
        if found:
            hits += 1
    return hits / len(names)


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
             seed: int | None = 0, *,
             top_k: int = 0, top_p: float = 1.0, repetition_penalty: float = 1.0,
             verbose: bool = True) -> Dict[str, float]:
    """Generate ``num`` names and return a dict of metrics (also printed as a table).

    ``top_k``/``top_p``/``repetition_penalty`` (WS-7) default to off, matching
    ``src/sample.py``'s decoding controls exactly, so a call with no new
    arguments samples the same way evaluation always has. ``verbose=False``
    suppresses the printed table -- used by ``sweep()`` so it can print one
    combined table instead of one per grid point.
    """
    if seed is not None:
        torch.manual_seed(seed)
    model, vocab, cfg, training_names = load_checkpoint(checkpoint, device)
    training_set = set(training_names)
    val_names: List[str] = torch.load(checkpoint, map_location=device).get("val_names", [])

    # Raw sample: keep duplicates and training-set collisions so we can *measure* them.
    raw: List[str] = []
    attempts, cap = 0, num * 50
    while len(raw) < num and attempts < cap:
        attempts += 1
        name = generate_one(model, vocab, temperature, cfg.max_length, device=device,
                             top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                             min_length=min_length)
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
        "near_dup_rate_1": near_duplicate_rate(distinct, list(training_names), max_dist=1),
        "near_dup_rate_2": near_duplicate_rate(distinct, list(training_names), max_dist=2),
        "held_out_nll": (training_nll(model, vocab, list(val_names), cfg, device)
                          if val_names else None),
    }

    if verbose:
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
    print(f"  near-duplicate <=1   : {m['near_dup_rate_1']*100:5.1f}%   "
          f"(within 1 edit of a training name — the memorization novelty misses)")
    print(f"  near-duplicate <=2   : {m['near_dup_rate_2']*100:5.1f}%")
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
    if m["held_out_nll"] is not None:
        print(f"  model NLL held-out   : {m['held_out_nll']:6.3f} nats/char "
              f"(WS-6 val_names — honest generalization check)")
    else:
        print(f"  model NLL held-out   : n/a (checkpoint has no val_names)")


# --- the sweep: pick decoding settings with numbers, not vibes -----------------------

_DEFAULT_SWEEP_TEMPERATURES = [0.7, 0.9, 1.1, 1.3]
_DEFAULT_SWEEP_TOP_KS = [10, 5]
_DEFAULT_SWEEP_TOP_PS = [0.9, 0.8]


def _decoding_settings(top_ks: List[int], top_ps: List[float]) -> List[Dict]:
    """One "plain" baseline, plus one row per requested top-k, plus one per top-p.

    Deliberately not the full top_k x top_p cross product -- that grid grows
    fast and mixes two questions (does truncating help? does nucleus help?)
    into one number. Keeping them as separate axes against the same
    temperature list makes each comparison legible in one table.
    """
    settings = [{"top_k": 0, "top_p": 1.0, "label": "plain"}]
    for k in top_ks:
        settings.append({"top_k": k, "top_p": 1.0, "label": f"top_k={k}"})
    for p in top_ps:
        settings.append({"top_k": 0, "top_p": p, "label": f"top_p={p}"})
    return settings


def sweep(checkpoint: str, num: int = 150,
          temperatures: List[float] | None = None,
          top_ks: List[int] | None = None,
          top_ps: List[float] | None = None,
          min_length: int = 2, device: str = "cpu", seed: int | None = 0,
          compare: List[str] | None = None) -> List[Dict]:
    """Grid over temperature x decoding setting; prints one table, returns one row per point.

    ``compare`` runs the same grid over additional checkpoints and folds them
    into the same table (tagged by checkpoint) instead of printing separately.
    """
    temperatures = temperatures or _DEFAULT_SWEEP_TEMPERATURES
    settings = _decoding_settings(top_ks or _DEFAULT_SWEEP_TOP_KS, top_ps or _DEFAULT_SWEEP_TOP_PS)
    checkpoints = [checkpoint] + list(compare or [])

    rows: List[Dict] = []
    for ckpt_path in checkpoints:
        for temp in temperatures:
            for setting in settings:
                m = evaluate(ckpt_path, num=num, temperature=temp, min_length=min_length,
                             device=device, seed=seed, top_k=setting["top_k"],
                             top_p=setting["top_p"], verbose=False)
                rows.append({"checkpoint": ckpt_path, "temperature": temp,
                             "decoding": setting["label"], **m})

    _print_sweep_table(rows, multi=len(checkpoints) > 1)
    return rows


def _plausibility_ratio(row: Dict) -> float:
    train_ll = row["bigram_ll_training"]
    return row["bigram_ll_generated"] / train_ll if train_ll else float("nan")


def _recommend(rows: List[Dict]) -> Dict:
    """Highest novelty, penalized for memorization (near-dup rate) and for
    drifting away from ``ratio == 1`` (too gibberish or too safe)."""
    def score(r: Dict) -> float:
        return r["novelty"] - r["near_dup_rate_1"] - abs(1.0 - _plausibility_ratio(r))
    return max(rows, key=score)


def _print_sweep_table(rows: List[Dict], multi: bool = False) -> None:
    print(f"\n=== Decoding sweep ({len(rows)} settings) ===")
    cols = "{:>18} {:>5} {:>10} {:>8} {:>10} {:>10} {:>11} {:>10} {:>9}"
    header = ["checkpoint", "temp", "decoding", "novelty", "near_dup1", "near_dup2",
              "plaus.ratio", "unique", "avg_edit"]
    print(cols.format(*header) if multi else
          cols.format("", *header[1:]))
    for r in rows:
        ratio = _plausibility_ratio(r)
        values = [
            r["checkpoint"] if multi else "",
            f"{r['temperature']:.1f}", r["decoding"],
            f"{r['novelty']*100:.1f}%", f"{r['near_dup_rate_1']*100:.1f}%",
            f"{r['near_dup_rate_2']*100:.1f}%", f"{ratio:.2f}",
            f"{r['uniqueness']*100:.1f}%", f"{r['avg_edit_distance']:.2f}",
        ]
        print(cols.format(*values))

    best = _recommend(rows)
    print(f"\nRecommended: {(best['checkpoint'] + ' @ ') if multi else ''}"
          f"temperature={best['temperature']}, decoding={best['decoding']} -- "
          f"novelty {best['novelty']*100:.1f}%, near-duplicate<=1 "
          f"{best['near_dup_rate_1']*100:.1f}%, plausibility ratio "
          f"{_plausibility_ratio(best):.2f} (closest to the novelty/quality tradeoff "
          f"of everything tried).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a char-RNN checkpoint (novelty / plausibility / diversity).")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--num", type=int, default=200, help="How many names to sample.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--min-length", type=int, default=None,
                        help="Discard names shorter than this. Defaults to the checkpoint's config.")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Keep only the k likeliest next characters (0 = off). "
                             "Defaults to the checkpoint's config.")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Nucleus sampling threshold (1.0 = off). "
                             "Defaults to the checkpoint's config.")
    parser.add_argument("--repetition-penalty", type=float, default=None,
                        help="Penalize repeated characters (1.0 = off). "
                             "Defaults to the checkpoint's config.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for reproducible metrics (use -1 for random).")
    parser.add_argument("--sweep", action="store_true",
                        help="Grid-search temperature x decoding setting instead of one run.")
    parser.add_argument("--compare", nargs="+", default=None,
                        help="Extra checkpoints folded into the --sweep table.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = None if args.seed == -1 else args.seed

    if args.sweep:
        sweep(args.checkpoint, num=args.num, device=device, seed=seed, compare=args.compare)
        return

    _, _, cfg, _ = load_checkpoint(args.checkpoint, device)
    top_k = args.top_k if args.top_k is not None else cfg.top_k
    top_p = args.top_p if args.top_p is not None else cfg.top_p
    repetition_penalty = (args.repetition_penalty if args.repetition_penalty is not None
                           else cfg.repetition_penalty)
    min_length = args.min_length if args.min_length is not None else cfg.min_length

    evaluate(args.checkpoint, num=args.num, temperature=args.temperature, min_length=min_length,
             device=device, seed=seed, top_k=top_k, top_p=top_p,
             repetition_penalty=repetition_penalty)


if __name__ == "__main__":
    main()
