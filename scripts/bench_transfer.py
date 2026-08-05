#!/usr/bin/env python3
"""WS-20 · does pretrain-then-finetune actually beat from-scratch?

The README has claimed since wave 1 that pretraining one base model on every dataset
and fine-tuning specialized copies from it beats training each dataset from scratch.
Nothing in the repo has ever measured that. This driver does, and writes one JSON per
run to ``reports/_transfer/`` so a lost session costs at most one run.

Two things decide whether the comparison means anything, and both are checked by
``check`` before any training happens:

1. **The validation split must be identical between arms.** ``src.train`` splits
   ``load_names(path)``; ``src.finetune`` splits ``filter_to_vocab(load_names(path),
   shared_vocab)``. Those are the same list only when nothing is dropped.
2. **Leakage.** ``src.pretrain`` trains the base on ``load_all_names`` over *every*
   dataset, which contains the target's validation names. A fine-tune from that base
   is scored on names the base already read, so its "held-out" loss is not held out.
   ``pretrain --variant clean`` builds a base with the targets' validation names
   removed from the corpus; ``--variant leaky`` reproduces the shipped
   ``python -m src.pretrain`` exactly, so the size of the artifact can be measured
   rather than assumed.

Arms per target, all on the identical split, identical epoch budget and identical
patience (both arms derive them from ``--auto-epochs`` on the same dataset size):

    scratch      python -m src.train  --data <t> --val-fraction 0.15 --auto-epochs
    scratch_sv   the same, but built against the *shared* vocab, so arm B's larger
                 softmax is not mistaken for a transfer effect
    ft_clean     fine-tune from the leakage-free base, at src.finetune's default 5e-4
    ft_clean_lr  the same at 3e-3, matching from-scratch's LR, to separate "transfer"
                 from "trained at a different learning rate"
    ft_leaky     fine-tune from the shipped, leaking base — the number the README's
                 claim would produce if nobody checked

Usage:
    python scripts/bench_transfer.py check
    python scripts/bench_transfer.py pretrain --variant clean
    python scripts/bench_transfer.py pretrain --variant leaky
    python scripts/bench_transfer.py arms                  # every arm, every target
    python scripts/bench_transfer.py arms --target typefaces --arm scratch
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

# Two other lanes share this 4-core box; keep every run to one thread.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import finetune as finetune_mod          # noqa: E402
from src import pretrain as pretrain_mod          # noqa: E402
from src import train as train_mod                # noqa: E402
from src.config import Config                     # noqa: E402
from src.data import (                            # noqa: E402
    DEFAULT_VOCAB_PATH,
    Vocab,
    filter_to_vocab,
    list_dataset_files,
    load_all_names,
    load_names,
    load_shared_vocab,
    split_names,
)
from src.model import CharRNN                     # noqa: E402

# Targets span the shipped size range; the transfer story should be strongest at the
# small end, which is where wave 3 found the model struggling most (BENCHMARK §1).
TARGETS = ["motorcycle_brands", "typefaces", "spacecraft", "pharma_drugs"]

VAL_FRACTION = 0.15
SEED = 1337
DATA_DIR = "data"
CKPT_DIR = "checkpoints"
OUT_DIR = "reports/_transfer"
PREFIX = "ws20_"

# The base sees 26,591 names, so an epoch costs ~43s and patience is the whole budget.
# WS-10 measured the held-out bottom at epoch 7 on 8,631 names and never later than 26
# across a 54x size range, and `fit` restores the best epoch's weights regardless of
# when the run ends -- so a shorter patience can only cost us a *later* bottom, not a
# worse checkpoint at the bottom we found. 20 is ~3x the largest-corpus bottom measured
# and saves ~15 minutes per base. Stated here because it is the one budget cut we make.
BASE_PATIENCE = 20


# --------------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------------

def _patched_fit(sink: dict):
    """Wrap ``src.train.fit`` so it fills ``sink`` with its per-epoch report.

    ``fit`` already supports ``report=``, but ``train()``, ``pretrain()`` and
    ``finetune()`` don't forward it. Rather than reimplement those three functions here
    -- which would risk measuring something subtly different from what the repo ships --
    we rebind the name each of them looks ``fit`` up under and let the real function run.
    """
    real = train_mod.fit

    def wrapper(*args, **kwargs):
        kwargs.setdefault("report", sink)
        return real(*args, **kwargs)

    return real, wrapper


class capture:
    """Context manager: run the shipped code paths, keep their training report."""

    def __init__(self):
        self.report: dict = {}

    def __enter__(self):
        self._real, wrapper = _patched_fit(self.report)
        for mod in (train_mod, pretrain_mod, finetune_mod):
            mod.fit = wrapper
        return self

    def __exit__(self, *exc):
        for mod in (train_mod, pretrain_mod, finetune_mod):
            mod.fit = self._real
        return False


def write_json(name: str, payload: dict) -> str:
    """Persist one run immediately. Never hold a batch of results in memory."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"  -> {path}")
    return path


def target_path(target: str) -> str:
    return os.path.join(DATA_DIR, f"{target}.txt")


def budget_for(n_names: int) -> tuple[int, int]:
    """The epoch budget and patience both arms derive from ``--auto-epochs``."""
    epochs = train_mod.derive_epochs(n_names)
    return epochs, train_mod.derive_patience(epochs)


def summarize(report: dict) -> dict:
    return {
        "best_val_loss": report.get("best_val_loss"),
        "best_epoch": report.get("best_epoch"),
        "epochs_run": report.get("epochs_run"),
        "stopped_early": report.get("stopped_early"),
        "train_loss_at_best": (
            report["train_losses"][report["best_epoch"] - 1]
            if report.get("best_epoch") else None
        ),
        "final_train_loss": (report.get("train_losses") or [None])[-1],
        "val_losses": report.get("val_losses"),
    }


# --------------------------------------------------------------------------------
# check: the two questions that decide whether any of this is worth running
# --------------------------------------------------------------------------------

def cmd_check(_args) -> None:
    paths = list_dataset_files(DATA_DIR)
    vocab = load_shared_vocab(DEFAULT_VOCAB_PATH)
    corpus = load_all_names(paths)
    corpus, corpus_dropped = filter_to_vocab(corpus, vocab)
    base_train, base_val = split_names(corpus, VAL_FRACTION, SEED)
    base_train_set, corpus_set = set(base_train), set(corpus)

    out = {
        "datasets": len(paths),
        "corpus_names_deduped": len(corpus),
        "corpus_dropped_by_shared_vocab": len(corpus_dropped),
        "shared_vocab_size": len(vocab),
        "base_split": {"train": len(base_train), "val": len(base_val)},
        "targets": {},
    }
    print(f"{len(paths)} datasets | {len(corpus)} names after cross-file dedup "
          f"| shared vocab {len(vocab)}")

    for target in TARGETS:
        names = load_names(target_path(target))
        own_vocab = Vocab(names)
        a_train, a_val = split_names(names, VAL_FRACTION, SEED)         # src.train
        ft_names, dropped = filter_to_vocab(names, vocab)
        b_train, b_val = split_names(ft_names, VAL_FRACTION, SEED)      # src.finetune
        epochs, patience = budget_for(len(names))
        rec = {
            "names": len(names),
            "own_vocab_size": len(own_vocab),
            "dropped_by_shared_vocab": len(dropped),
            "splits_identical": a_train == b_train and a_val == b_val,
            "val_names": len(a_val),
            "val_in_base_corpus": sum(1 for n in a_val if n in corpus_set),
            "val_in_base_train_half": sum(1 for n in a_val if n in base_train_set),
            "auto_epochs": epochs,
            "auto_patience": patience,
        }
        rec["leak_fraction"] = rec["val_in_base_corpus"] / max(1, len(a_val))
        out["targets"][target] = rec
        print(f"  {target:20s} n={rec['names']:5d} splits_identical="
              f"{rec['splits_identical']} val={rec['val_names']:4d} "
              f"leaked={rec['val_in_base_corpus']:4d} "
              f"({rec['leak_fraction']:.0%}) budget={epochs}/pat {patience}")

    write_json("preflight", out)


# --------------------------------------------------------------------------------
# stage 1: the two bases
# --------------------------------------------------------------------------------

def excluded_val_names() -> list[str]:
    """Every target's validation names -- what the clean base must never see.

    Excluded by *string*, not by file: ``load_all_names`` de-duplicates across files, so
    a name that lives in both the target and some other dataset would otherwise slip
    back into the corpus under the other file's flag.
    """
    out: set[str] = set()
    for target in TARGETS:
        _, val = split_names(load_names(target_path(target)), VAL_FRACTION, SEED)
        out.update(val)
    return sorted(out)


def cmd_pretrain(args) -> None:
    variant = args.variant
    name = f"{PREFIX}base_{variant}"
    paths = list_dataset_files(DATA_DIR)
    vocab = load_shared_vocab(DEFAULT_VOCAB_PATH)

    corpus, dropped = filter_to_vocab(load_all_names(paths), vocab)
    excluded = excluded_val_names() if variant == "clean" else []
    if excluded:
        blocked = set(excluded)
        kept = [n for n in corpus if n not in blocked]
        print(f"clean base: removed {len(corpus) - len(kept)} target validation names "
              f"from the {len(corpus)}-name corpus")
        corpus = kept

    cfg = Config()
    cfg.val_fraction = VAL_FRACTION
    cfg.early_stop_patience = BASE_PATIENCE
    train_names, val_names = split_names(corpus, cfg.val_fraction, cfg.seed)
    train_mod.apply_auto_epochs(cfg, len(corpus))

    print(f"pretraining {name}: {len(corpus)} names from {len(paths)} datasets "
          f"| {len(train_names)} train / {len(val_names)} val "
          f"| {cfg.epochs} epochs, patience {cfg.early_stop_patience}")

    train_mod.seed_for_init(cfg)
    model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id)
    t0 = time.time()
    with capture() as cap:
        train_mod.fit(model, vocab, train_names, cfg, device="cpu", val_names=val_names)
    wall = time.time() - t0

    ckpt = train_mod.save_checkpoint(
        os.path.join(CKPT_DIR, f"{name}.pt"), model, cfg, vocab, train_names, val_names)

    payload = {
        "run": name,
        "kind": "pretrain",
        "variant": variant,
        "datasets": len(paths),
        "corpus_names": len(corpus),
        "dropped_by_shared_vocab": len(dropped),
        "excluded_target_val_names": len(excluded),
        "train_names": len(train_names),
        "val_names": len(val_names),
        "vocab_size": len(vocab),
        "epoch_budget": cfg.epochs,
        "patience": cfg.early_stop_patience,
        "learning_rate": cfg.learning_rate,
        "checkpoint": ckpt,
        "wall_seconds": round(wall, 1),
        **summarize(cap.report),
    }
    write_json(name, payload)


# --------------------------------------------------------------------------------
# stage 2: the arms
# --------------------------------------------------------------------------------

def arm_scratch(target: str) -> dict:
    """`python -m src.train --data <t> --val-fraction 0.15 --auto-epochs --patience N`."""
    cfg = Config()
    cfg.val_fraction = VAL_FRACTION
    epochs, patience = budget_for(len(load_names(target_path(target))))
    cfg.early_stop_patience = patience
    name = f"{PREFIX}{target}_scratch"
    t0 = time.time()
    with capture() as cap:
        ckpt = train_mod.train(target_path(target), name, cfg,
                               checkpoint_dir=CKPT_DIR, device="cpu", auto_epochs=True)
    return {"checkpoint": ckpt, "wall_seconds": round(time.time() - t0, 1),
            "learning_rate": cfg.learning_rate, "epoch_budget": cfg.epochs,
            "patience": cfg.early_stop_patience, "vocab": "per-dataset",
            **summarize(cap.report)}


def arm_scratch_shared_vocab(target: str) -> dict:
    """From scratch, but against the shared vocab.

    ``src.train`` sizes the softmax to the dataset's own characters (55-67 here) while
    every fine-tune inherits the base's 67-token shared vocab. That difference is not
    transfer, but it lands in the same loss number, so this arm removes it: identical
    split, identical budget, identical LR, identical architecture -- the *only*
    difference from ``ft_*`` is that the weights start random instead of pretrained.
    """
    names = load_names(target_path(target))
    vocab = load_shared_vocab(DEFAULT_VOCAB_PATH)
    names, _ = filter_to_vocab(names, vocab)
    cfg = Config()
    cfg.val_fraction = VAL_FRACTION
    epochs, patience = budget_for(len(names))
    cfg.early_stop_patience = patience
    train_names, val_names = split_names(names, cfg.val_fraction, cfg.seed)
    train_mod.apply_auto_epochs(cfg, len(names))
    print(f"scratch (shared vocab) on {len(names)} names | {len(train_names)} train / "
          f"{len(val_names)} val | {cfg.epochs} epochs, patience {cfg.early_stop_patience}")
    train_mod.seed_for_init(cfg)
    model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id)
    t0 = time.time()
    with capture() as cap:
        train_mod.fit(model, vocab, train_names, cfg, device="cpu", val_names=val_names)
    wall = time.time() - t0
    ckpt = train_mod.save_checkpoint(
        os.path.join(CKPT_DIR, f"{PREFIX}{target}_scratch_sv.pt"),
        model, cfg, vocab, train_names, val_names)
    return {"checkpoint": ckpt, "wall_seconds": round(wall, 1),
            "learning_rate": cfg.learning_rate, "epoch_budget": cfg.epochs,
            "patience": cfg.early_stop_patience, "vocab": "shared",
            **summarize(cap.report)}


def arm_finetune(target: str, base_variant: str, lr: float | None) -> dict:
    """`python -m src.finetune --base <base> --data <t> --val-fraction 0.15 ...`."""
    base = os.path.join(CKPT_DIR, f"{PREFIX}base_{base_variant}.pt")
    if not os.path.exists(base):
        raise SystemExit(f"missing base checkpoint {base}; run `pretrain` first")
    suffix = f"ft_{base_variant}" + ("" if lr is None else "_lr")
    name = f"{PREFIX}{target}_{suffix}"
    t0 = time.time()
    with capture() as cap:
        ckpt = finetune_mod.finetune(
            base_path=base, data_path=target_path(target), out_name=name,
            learning_rate=lr, checkpoint_dir=CKPT_DIR, device="cpu",
            val_fraction=VAL_FRACTION, auto_epochs=True,
        )
    return {"checkpoint": ckpt, "wall_seconds": round(time.time() - t0, 1),
            "base": base, "base_variant": base_variant,
            "learning_rate": lr if lr is not None else finetune_mod.FINETUNE_LR,
            "vocab": "shared", **summarize(cap.report)}


ARMS = {
    "scratch": lambda t: arm_scratch(t),
    "scratch_sv": lambda t: arm_scratch_shared_vocab(t),
    "ft_clean": lambda t: arm_finetune(t, "clean", None),
    "ft_clean_lr": lambda t: arm_finetune(t, "clean", Config().learning_rate),
    "ft_leaky": lambda t: arm_finetune(t, "leaky", None),
}


def cmd_arms(args) -> None:
    targets = [args.target] if args.target else TARGETS
    arms = [args.arm] if args.arm else list(ARMS)
    for target in targets:
        names = load_names(target_path(target))
        _, val = split_names(names, VAL_FRACTION, SEED)
        for arm in arms:
            out_name = f"{target}__{arm}"
            if not args.force and os.path.exists(os.path.join(OUT_DIR, f"{out_name}.json")):
                print(f"[skip] {out_name} already recorded")
                continue
            print(f"\n=== {target} · {arm} " + "=" * 40)
            result = ARMS[arm](target)
            write_json(out_name, {
                "run": out_name, "kind": "arm", "arm": arm, "dataset": target,
                "path": target_path(target), "total_names": len(names),
                "val_names": len(val), "train_names": len(names) - len(val),
                **result,
            })


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="verify split identity and measure leakage")
    c.set_defaults(func=cmd_check)

    pt = sub.add_parser("pretrain", help="train a base model")
    pt.add_argument("--variant", choices=("clean", "leaky"), required=True,
                    help="clean = target val names removed from the corpus; "
                         "leaky = the shipped src.pretrain corpus, verbatim")
    pt.set_defaults(func=cmd_pretrain)

    ar = sub.add_parser("arms", help="run the per-target arms")
    ar.add_argument("--target", choices=TARGETS, default=None)
    ar.add_argument("--arm", choices=list(ARMS), default=None)
    ar.add_argument("--force", action="store_true", help="re-run arms already recorded")
    ar.set_defaults(func=cmd_arms)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
