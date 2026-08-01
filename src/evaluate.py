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

WS-11 finishes the held-out path WS-7 could only stub. Checkpoints written by
``src/train.py --val-fraction F`` now really do carry ``val_names``, so the held-out
NLL is reported *beside* the training NLL together with the **gap** between them —
the single most informative number this project prints. A small gap means the model
learned the style; a large one means it memorized the list, and no amount of decoding
tuning fixes that. WS-11 also adds ``--report``, which writes the whole evaluation
(identity, config, metrics, real generated names, recommended decoding) to a
self-contained, deterministic markdown file.

Usage:
    python -m src.evaluate --checkpoint checkpoints/car_models_ft.pt
    python -m src.evaluate --checkpoint checkpoints/base.pt --num 300 --temperature 0.9
    python -m src.evaluate --checkpoint checkpoints/base.pt --sweep --compare checkpoints/base_ft.pt
    python -m src.evaluate --checkpoint checkpoints/base.pt --report reports/base.md
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set

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


# --- fit vs. generalization: the model's own next-char loss on a set of names --------

@torch.no_grad()
def mean_char_nll(model: CharRNN, vocab: Vocab, names: Sequence[str], cfg,
                  device: str = "cpu") -> float:
    """Mean cross-entropy in nats/char of the model over ``names``.

    Token-weighted, not name-weighted: the totals are summed over every non-pad
    target and divided once at the end, so a set of long names and a set of short
    ones are on the same scale. That is what makes the train number and the held-out
    number below directly subtractable.
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_id, reduction="sum")
    pairs = make_pairs(list(names), vocab)
    total_loss, total_tokens = 0.0, 0
    for inputs, targets, _ in make_batches(pairs, cfg.batch_size, vocab.pad_id,
                                           shuffle=False):
        inputs, targets = inputs.to(device), targets.to(device)
        logits, _ = model(inputs)
        total_loss += criterion(logits.reshape(-1, logits.size(-1)),
                                targets.reshape(-1)).item()
        total_tokens += int((targets != vocab.pad_id).sum().item())
    return total_loss / max(total_tokens, 1)


def training_nll(model: CharRNN, vocab: Vocab, names: Sequence[str], cfg,
                 device: str = "cpu") -> float:
    """Mean cross-entropy (nats/char) on the model's *training* names.

    Not held-out — the model has seen these — so it measures fit, not generalization.
    Kept as a named entry point (WS-7 callers use it) but it is now literally
    :func:`mean_char_nll`; the only difference between this and the held-out number
    is which list of names you hand it.
    """
    return mean_char_nll(model, vocab, names, cfg, device)


def held_out_nll(model: CharRNN, vocab: Vocab, val_names: Sequence[str], cfg,
                 device: str = "cpu") -> Optional[float]:
    """Mean cross-entropy (nats/char) on names the model never saw, or ``None``.

    ``None`` — not ``nan``, not ``0.0`` — when there are no held-out names, because
    "we cannot measure this" and "this measured zero" must never render the same way
    in a report. Every caller therefore has to handle the missing case explicitly.
    """
    if not val_names:
        return None
    return mean_char_nll(model, vocab, val_names, cfg, device)


def generalization_gap(train_nll: float, val_nll: Optional[float]) -> Optional[float]:
    """``val_nll - train_nll`` in nats/char, or ``None`` when there is no held-out set.

    Positive means the model does worse on names it never saw — the expected sign.
    The *size* is the point: wave 2 measured 6.15 nats/char on a 135-name dataset,
    which is not "slightly overfit", it is a model that memorized a list. Because
    both terms are per-character, the gap is comparable across datasets of very
    different name lengths, which is exactly what the dataset-size question needs.
    """
    if val_nll is None:
        return None
    return val_nll - train_nll


# --- one checkpoint, loaded and derived-from exactly once -----------------------------

_UNSET = object()


class _Checkpoint:
    """Everything evaluation needs from one checkpoint file, computed at most once.

    ``--sweep`` evaluates the same checkpoint at dozens of grid points, and the
    training NLL, the held-out NLL and the bigram reference table are identical at
    every one of them — they depend on the weights and the data, not on the decoding
    settings. Computing them once here rather than per grid point is what makes a
    full sweep (and the benchmark that runs one per dataset) affordable.
    """

    def __init__(self, path: str, device: str = "cpu") -> None:
        self.path = path
        self.device = device
        self.model, self.vocab, self.cfg, names = load_checkpoint(path, device)
        # load_checkpoint hands back a *set*. Sort it: Python randomizes string
        # hashing per process, so set iteration order differs between runs, and a
        # report that must be byte-identical across runs cannot be built on it.
        self.training_names: List[str] = sorted(names)
        raw = torch.load(path, map_location=device)
        # WS-6/WS-10 write this key unconditionally; anything older simply lacks it.
        self.val_names: List[str] = list(raw.get("val_names", []) or [])
        self._bigram: Optional[Dict[str, Dict[str, float]]] = None
        self._train_nll: object = _UNSET
        self._val_nll: object = _UNSET

    @property
    def bigram_table(self) -> Dict[str, Dict[str, float]]:
        if self._bigram is None:
            self._bigram = _bigram_logprobs(self.training_names)
        return self._bigram

    @property
    def train_nll(self) -> float:
        if self._train_nll is _UNSET:
            self._train_nll = mean_char_nll(self.model, self.vocab,
                                            self.training_names, self.cfg, self.device)
        return self._train_nll  # type: ignore[return-value]

    @property
    def val_nll(self) -> Optional[float]:
        if self._val_nll is _UNSET:
            self._val_nll = held_out_nll(self.model, self.vocab, self.val_names,
                                         self.cfg, self.device)
        return self._val_nll  # type: ignore[return-value]


_CHECKPOINT_CACHE: Dict[tuple, _Checkpoint] = {}


def _load(path: str, device: str = "cpu") -> _Checkpoint:
    """Cached :class:`_Checkpoint`, keyed on the file's identity *and* its mtime/size.

    Keying on mtime/size rather than the path alone means retraining a checkpoint
    under the same name inside one process (which tests do) is picked up instead of
    silently serving stale weights.
    """
    try:
        st = os.stat(path)
        stamp: object = (st.st_mtime_ns, st.st_size)
    except OSError:
        stamp = None
    key = (os.path.abspath(path), device, stamp)
    if key not in _CHECKPOINT_CACHE:
        _CHECKPOINT_CACHE[key] = _Checkpoint(path, device)
    return _CHECKPOINT_CACHE[key]


def clear_checkpoint_cache() -> None:
    """Drop every cached checkpoint. Used by tests that rewrite files in place."""
    _CHECKPOINT_CACHE.clear()


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
    ckpt = _load(checkpoint, device)
    model, vocab, cfg = ckpt.model, ckpt.vocab, ckpt.cfg
    training_names = ckpt.training_names
    training_set = set(training_names)

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
    table = ckpt.bigram_table

    metrics = {
        "generated": len(raw),
        "distinct": len(distinct),
        "uniqueness": len(distinct) / max(len(raw), 1),
        "novelty": len(novel) / max(len(distinct), 1),
        "avg_length": sum(len(n) for n in raw) / max(len(raw), 1),
        "avg_edit_distance": mean_pairwise_edit_distance(distinct),
        "bigram_ll_generated": mean_bigram_ll(distinct, table),
        "bigram_ll_training": mean_bigram_ll(training_names, table),
        "training_nll": ckpt.train_nll,
        "near_dup_rate_1": near_duplicate_rate(distinct, training_names, max_dist=1),
        "near_dup_rate_2": near_duplicate_rate(distinct, training_names, max_dist=2),
        "held_out_nll": ckpt.val_nll,
        "nll_gap": generalization_gap(ckpt.train_nll, ckpt.val_nll),
        "num_training_names": len(training_names),
        "num_val_names": len(ckpt.val_names),
        # The names themselves ride along so --report can show the qualitative
        # check no metric replaces, without generating a second, different sample.
        "samples": raw,
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
    print(f"  model NLL (nats/char — train vs. the names it never saw):")
    print(f"     train             : {m['training_nll']:6.3f}   "
          f"({m.get('num_training_names', '?')} names, fit only)")
    if m["held_out_nll"] is not None:
        print(f"     held-out          : {m['held_out_nll']:6.3f}   "
              f"({m.get('num_val_names', '?')} names the model never saw)")
        print(f"     gap               : {m['nll_gap']:+6.3f}   "
              f"{_gap_verdict(m['nll_gap'])}")
    else:
        print(f"     held-out          :    n/a   (checkpoint has no val_names; "
              f"retrain with --val-fraction 0.15 to get an honest number)")
        print(f"     gap               :    n/a")


# How far apart the two NLLs have to be before it stops being noise. These are
# calibration, not physics: 0.5 nats/char is roughly "each character is 1.6x less
# likely on unseen names", which is normal for a small char model, and 2.0 is
# "the held-out names are ~7x less likely per character", which is memorization.
GAP_OK = 0.5
GAP_BAD = 2.0


def _gap_verdict(gap: float) -> str:
    if gap <= GAP_OK:
        return "(generalizes — learned the style)"
    if gap <= GAP_BAD:
        return "(some overfitting)"
    return "(memorized the list — needs more data, not more tuning)"


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
          compare: List[str] | None = None, verbose: bool = True) -> List[Dict]:
    """Grid over temperature x decoding setting; prints one table, returns one row per point.

    ``compare`` runs the same grid over additional checkpoints and folds them
    into the same table (tagged by checkpoint) instead of printing separately.
    ``verbose=False`` returns the rows without printing — used by ``--report``,
    which renders the same rows as markdown instead.
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
                             "decoding": setting["label"],
                             "top_k": setting["top_k"], "top_p": setting["top_p"], **m})

    if verbose:
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


# --- the markdown report -------------------------------------------------------------

# The grid --report runs on its own to justify a recommendation. Deliberately smaller
# than the full --sweep grid: one representative top-k and one representative top-p
# against the standard temperature ladder is enough to *rank* settings, and a report
# is meant to be cheap enough to generate for every checkpoint in the library.
_REPORT_SWEEP_TOP_KS = [10]
_REPORT_SWEEP_TOP_PS = [0.9]

# Config fields worth printing. Anything not listed is either an implementation
# detail or a dual-output field that would be noise for a plain name model.
_REPORTED_CONFIG = [
    ("arch", "architecture"), ("embedding_dim", "embedding dim"),
    ("hidden_dim", "hidden dim"), ("num_layers", "layers"), ("dropout", "dropout"),
    ("epochs", "epoch budget"), ("batch_size", "batch size"),
    ("learning_rate", "learning rate"), ("val_fraction", "val fraction"),
    ("early_stop_patience", "early-stop patience"), ("lr_schedule", "lr schedule"),
    ("weight_decay", "weight decay"), ("label_smoothing", "label smoothing"),
    ("seed", "training seed"),
]


def _fmt(value, spec: str = ".3f", missing: str = "n/a") -> str:
    """Format a number, or render ``None``/``nan`` as an explicit "n/a".

    Reports are read by people deciding whether to trust a model. A blank cell or a
    stray ``nan`` invites them to guess; "n/a" says the measurement does not exist.
    """
    if value is None:
        return missing
    if isinstance(value, float) and math.isnan(value):
        return missing
    return format(value, spec)


def _pct(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value * 100:.1f}%"


def _checkpoint_fingerprint(path: str) -> str:
    """First 16 hex chars of the file's SHA-256, or "unavailable".

    Identity, not integrity: it is what lets a reader confirm two reports describe
    the same weights, and it keeps the report deterministic (a timestamp would not).
    """
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()[:16]
    except OSError:
        return "unavailable"


def _dedup_keep_order(names: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _decoding_label(top_k: int, top_p: float, repetition_penalty: float) -> str:
    parts = []
    if top_k:
        parts.append(f"top_k={top_k}")
    if top_p < 1.0:
        parts.append(f"top_p={top_p}")
    if repetition_penalty != 1.0:
        parts.append(f"repetition_penalty={repetition_penalty}")
    return ", ".join(parts) if parts else "plain (no truncation)"


def write_report(path: str, checkpoint: str, metrics: Dict, *,
                 temperature: float, num: int, seed: int | None,
                 top_k: int = 0, top_p: float = 1.0, repetition_penalty: float = 1.0,
                 min_length: int = 2, device: str = "cpu",
                 sweep_rows: List[Dict] | None = None,
                 num_samples: int = 30) -> str:
    """Write a self-contained markdown evaluation report and return its path.

    Self-contained: a reader who has only this file can tell which weights it
    describes, what data they came from, how the names were sampled, and what the
    numbers were — without the checkpoint, the dataset, or the terminal scrollback.

    Deterministic: given the same checkpoint and the same ``seed`` the bytes are
    identical, so two runs of the same command can be diffed and a report can be
    committed. That rules out timestamps, wall-clock timings and hostnames, which is
    why none appear below.
    """
    ckpt = _load(checkpoint, device)
    cfg = ckpt.cfg
    label = getattr(cfg, "dataset_label", "") or os.path.splitext(
        os.path.basename(checkpoint))[0]
    gap = metrics.get("nll_gap")
    ratio = (metrics["bigram_ll_generated"] / metrics["bigram_ll_training"]
             if metrics["bigram_ll_training"] else float("nan"))

    L: List[str] = []
    L.append(f"# Evaluation report — {label}")
    L.append("")
    L.append("Generated by `python -m src.evaluate --report` "
             "(deterministic given the seed below).")
    L.append("")

    L.append("## Checkpoint")
    L.append("")
    L.append("| field | value |")
    L.append("|---|---|")
    L.append(f"| path | `{checkpoint}` |")
    L.append(f"| sha256 (first 16) | `{_checkpoint_fingerprint(checkpoint)}` |")
    for field, pretty in _REPORTED_CONFIG:
        if hasattr(cfg, field):
            L.append(f"| {pretty} | {getattr(cfg, field)} |")
    L.append("")

    L.append("## Dataset")
    L.append("")
    L.append("| field | value |")
    L.append("|---|---|")
    L.append(f"| label | {getattr(cfg, 'dataset_label', '') or '(not recorded)'} |")
    L.append(f"| source path | "
             f"{('`' + cfg.dataset_path + '`') if getattr(cfg, 'dataset_path', '') else '(not recorded)'} |")
    L.append(f"| training names (in checkpoint) | {metrics.get('num_training_names', 0)} |")
    L.append(f"| held-out names (`val_names`) | "
             f"{metrics.get('num_val_names', 0) or 'none — checkpoint predates --val-fraction'} |")
    L.append("")

    L.append("## Generalization — the headline number")
    L.append("")
    if metrics.get("held_out_nll") is None:
        L.append("This checkpoint carries no `val_names`, so **no honest generalization "
                 "number exists for it**. Retrain with `--val-fraction 0.15` to get one; "
                 "the training NLL below measures fit only and a model that memorized "
                 "its data scores *better* on it, not worse.")
        L.append("")
        L.append("| measure | nats/char |")
        L.append("|---|---|")
        L.append(f"| model NLL on training names | {_fmt(metrics['training_nll'])} |")
        L.append("| model NLL on held-out names | n/a |")
        L.append("| **train → held-out gap** | **n/a** |")
    else:
        L.append("| measure | nats/char | over |")
        L.append("|---|---|---|")
        L.append(f"| model NLL on training names | {_fmt(metrics['training_nll'])} | "
                 f"{metrics.get('num_training_names', 0)} names it trained on |")
        L.append(f"| model NLL on held-out names | {_fmt(metrics['held_out_nll'])} | "
                 f"{metrics.get('num_val_names', 0)} names it never saw |")
        L.append(f"| **train → held-out gap** | **{_fmt(gap, '+.3f')}** | "
                 f"{_gap_verdict(gap).strip('()')} |")
        L.append("")
        L.append(f"Reference points: a gap under {GAP_OK} nats/char means the model "
                 f"learned the domain's spelling rather than its word list; a gap over "
                 f"{GAP_BAD} means it memorized, and decoding settings cannot repair that "
                 f"— only more data can.")
    L.append("")

    L.append("## Sampling")
    L.append("")
    L.append("| field | value |")
    L.append("|---|---|")
    L.append(f"| names requested | {num} |")
    L.append(f"| names generated | {metrics['generated']} ({metrics['distinct']} distinct) |")
    L.append(f"| temperature | {temperature} |")
    L.append(f"| decoding | {_decoding_label(top_k, top_p, repetition_penalty)} |")
    L.append(f"| min length | {min_length} |")
    L.append(f"| seed | {seed if seed is not None else 'random (not reproducible)'} |")
    L.append("")

    L.append("## Metrics")
    L.append("")
    L.append("| metric | value | reading |")
    L.append("|---|---|---|")
    L.append(f"| novelty | {_pct(metrics['novelty'])} | distinct names absent from the "
             f"training set; higher is better |")
    L.append(f"| near-duplicate ≤1 | {_pct(metrics['near_dup_rate_1'])} | within one edit "
             f"of a training name — memorization novelty misses; lower is better |")
    L.append(f"| near-duplicate ≤2 | {_pct(metrics['near_dup_rate_2'])} | within two edits; "
             f"lower is better |")
    L.append(f"| uniqueness | {_pct(metrics['uniqueness'])} | distinct / generated; low means "
             f"the model loops |")
    L.append(f"| mean pairwise edit distance | {_fmt(metrics['avg_edit_distance'], '.2f')} | "
             f"how different the outputs are from each other |")
    L.append(f"| mean length | {_fmt(metrics['avg_length'], '.2f')} chars | |")
    L.append(f"| bigram log-likelihood, generated | {_fmt(metrics['bigram_ll_generated'])} | "
             f"nats/char under a bigram model of the training names |")
    L.append(f"| bigram log-likelihood, real names | {_fmt(metrics['bigram_ll_training'])} | "
             f"the reference the line above is judged against |")
    L.append(f"| plausibility ratio | {_fmt(ratio, '.2f')} | ~1.00 = generated names are as "
             f"typical as real ones |")
    L.append("")

    shown = _dedup_keep_order(metrics.get("samples") or [])[:num_samples]
    L.append(f"## Generated names ({len(shown)})")
    L.append("")
    L.append("The check no metric replaces — read them.")
    L.append("")
    L.append("```")
    for i in range(0, len(shown), 5):
        L.append(", ".join(shown[i:i + 5]))
    if not shown:
        L.append("(no names were generated)")
    L.append("```")
    L.append("")

    L.append("## Recommended decoding")
    L.append("")
    if sweep_rows:
        best = _recommend(sweep_rows)
        L.append(f"**temperature {best['temperature']}, {best['decoding']}** — "
                 f"novelty {_pct(best['novelty'])}, near-duplicate ≤1 "
                 f"{_pct(best['near_dup_rate_1'])}, plausibility ratio "
                 f"{_fmt(_plausibility_ratio(best), '.2f')}.")
        L.append("")
        L.append(f"Chosen by `novelty − near_dup≤1 − |1 − plausibility ratio|` over the "
                 f"{len(sweep_rows)} settings below "
                 f"({sweep_rows[0]['generated']} names each, same seed).")
        L.append("")
        L.append("| temp | decoding | novelty | near-dup ≤1 | near-dup ≤2 | plaus. ratio | "
                 "uniqueness | avg edit |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in sweep_rows:
            marker = " **←**" if r is best else ""
            L.append(f"| {r['temperature']} | {r['decoding']}{marker} | "
                     f"{_pct(r['novelty'])} | {_pct(r['near_dup_rate_1'])} | "
                     f"{_pct(r['near_dup_rate_2'])} | "
                     f"{_fmt(_plausibility_ratio(r), '.2f')} | {_pct(r['uniqueness'])} | "
                     f"{_fmt(r['avg_edit_distance'], '.2f')} |")
    else:
        L.append("No sweep was run, so this report cannot recommend a setting from "
                 "evidence. The single setting it did measure is "
                 f"**temperature {temperature}, {_decoding_label(top_k, top_p, repetition_penalty)}**, "
                 f"scoring novelty {_pct(metrics['novelty'])} and near-duplicate ≤1 "
                 f"{_pct(metrics['near_dup_rate_1'])}. Re-run with `--sweep` for a ranked grid.")
    L.append("")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return path


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
    parser.add_argument("--report", default=None, metavar="PATH.md",
                        help="Also write a self-contained markdown report to PATH.md. "
                             "Deterministic given --seed.")
    parser.add_argument("--report-samples", type=int, default=30,
                        help="How many generated names to show in the report's "
                             "qualitative block (default 30).")
    parser.add_argument("--report-sweep-num", type=int, default=60,
                        help="Names per grid point in the compact sweep --report runs "
                             "to justify its recommendation (0 = skip the sweep).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = None if args.seed == -1 else args.seed

    cfg = _load(args.checkpoint, device).cfg
    top_k = args.top_k if args.top_k is not None else cfg.top_k
    top_p = args.top_p if args.top_p is not None else cfg.top_p
    repetition_penalty = (args.repetition_penalty if args.repetition_penalty is not None
                           else cfg.repetition_penalty)
    min_length = args.min_length if args.min_length is not None else cfg.min_length

    sweep_rows = None
    if args.sweep:
        sweep_rows = sweep(args.checkpoint, num=args.num, min_length=min_length,
                           device=device, seed=seed, compare=args.compare)
        if not args.report:
            return
    elif args.report and args.report_sweep_num > 0:
        # A recommendation with no grid behind it is an opinion. Run a compact one.
        sweep_rows = sweep(args.checkpoint, num=args.report_sweep_num,
                           top_ks=_REPORT_SWEEP_TOP_KS, top_ps=_REPORT_SWEEP_TOP_PS,
                           min_length=min_length, device=device, seed=seed, verbose=False)

    metrics = evaluate(args.checkpoint, num=args.num, temperature=args.temperature,
                       min_length=min_length, device=device, seed=seed, top_k=top_k,
                       top_p=top_p, repetition_penalty=repetition_penalty)

    if args.report:
        if sweep_rows:
            # --compare folds other checkpoints into the same sweep; a report is about
            # one checkpoint, so it may only recommend from that checkpoint's rows.
            sweep_rows = [r for r in sweep_rows if r["checkpoint"] == args.checkpoint]
        out = write_report(args.report, args.checkpoint, metrics,
                           temperature=args.temperature, num=args.num, seed=seed,
                           top_k=top_k, top_p=top_p,
                           repetition_penalty=repetition_penalty,
                           min_length=min_length, device=device,
                           sweep_rows=sweep_rows, num_samples=args.report_samples)
        print(f"\nWrote report -> {out}")


if __name__ == "__main__":
    main()
