"""Sweep model capacity (hidden_dim x num_layers x arch) across dataset sizes.

Lane WS-18 - wave 4. The driver behind `reports/CAPACITY.md`.

Why this exists
---------------
`reports/BENCHMARK.md` found held-out loss improving monotonically with dataset size
while the train/val gap refused to shrink and the returns flattened.
`reports/ARCH.md` found the smaller GRU beating the LSTM on both datasets under 500
names and losing on both above. Two independent measurements pointing at the same
suspect: the stock model -- `hidden_dim=256`, `num_layers=2`, ~839k parameters -- is
too large for the fifteen datasets in `data/` that hold fewer than 500 names. Nobody
had ever swept capacity on this repo; 256x2 is an untested default inherited from the
first commit. This script is the sweep, written as a script rather than run by hand so
the numbers in the report can be regenerated rather than believed.

What one cell is
----------------
One (dataset, arch, hidden_dim, num_layers) combination, trained from scratch with
everything else held at the repo default: `embedding_dim=32`, `dropout=0.2`,
`lr=3e-3`, `batch_size=32`, `seed=1337`, `--val-fraction 0.15`, and the `--auto-epochs`
budget/patience pair derived from the dataset size (`src/train.derive_epochs`). Only
the three swept fields move.

A cell records, into one JSON file:

* `params` - `sum(p.numel())` over the built model
* `best_val_loss` / `best_epoch` - from the training loop's own bookkeeping
* `train_loss_at_best` - the running training loss of that same epoch (dropout ON,
  which is how `src/train.py` reports it)
* `train_nll` / `val_nll` - the restored best-epoch weights re-scored over the train
  and val halves with dropout OFF and no gradient. `val_nll` should equal
  `best_val_loss` to numerical noise; it is recomputed anyway as a self-check, and
  `gap = val_nll - train_nll` (positive = overfitting) is then on the same footing
  as `reports/ARCH.md`'s gap column, which was also measured dropout-off from a
  checkpoint.
* `seconds` - wall-clock for the fit, excluding data loading and the final re-scoring

Reproducibility
---------------
`Config.seed_init` defaults to True since WS-10, so the RNG is seeded before the model
is constructed and an identical command yields bitwise identical weights. One seed per
cell is therefore a statement about this seed, not about seed variance -- see the
caveats section of `reports/CAPACITY.md`.

Resumability
------------
Every finished cell is written to disk the moment it completes, and a cell whose JSON
already exists is skipped. An interrupted sweep is resumed by re-running the same
command; `--force` re-runs cells that already have output.

Usage
-----
    # the main grid behind reports/CAPACITY.md
    OMP_NUM_THREADS=1 python scripts/sweep_capacity.py \
        --datasets motorcycle_brands aircraft typefaces car_models pharma_drugs \
                   english_words \
        --hidden-dims 64 128 256 384 --layers 1 2 --archs lstm gru

    # replicates behind the noise floor in finding 2: --seed alone redraws the
    # validation split too, --split-seed pins it so initialization varies alone
    OMP_NUM_THREADS=1 python scripts/sweep_capacity.py \
        --datasets motorcycle_brands typefaces --archs lstm gru --seed 7
    OMP_NUM_THREADS=1 python scripts/sweep_capacity.py \
        --datasets motorcycle_brands typefaces --archs lstm gru --seed 7 \
        --split-seed 1337

    # views: the report's tables, regenerated from reports/_capacity/
    OMP_NUM_THREADS=1 python scripts/sweep_capacity.py --pivot
    OMP_NUM_THREADS=1 python scripts/sweep_capacity.py --summary
    OMP_NUM_THREADS=1 python scripts/sweep_capacity.py --summary --all-seeds
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config                                    # noqa: E402
from src.data import Vocab, load_names, make_pairs, split_names  # noqa: E402
from src.model import CharRNN                                    # noqa: E402
from src.train import (                                          # noqa: E402
    derive_epochs,
    derive_patience,
    evaluate_loss,
    fit,
    save_checkpoint,
)

# Other lanes share this 4-core box; one thread per process keeps the timings
# comparable between cells and keeps this lane from starving the others.
torch.set_num_threads(1)

DEFAULT_OUT_DIR = "reports/_capacity"
DEFAULT_DATA_DIR = "data"
CHECKPOINT_PREFIX = "ws18_"   # so this lane's checkpoints never collide with another's


DEFAULT_SEED = 1337


def cell_id(dataset: str, arch: str, hidden_dim: int, num_layers: int,
            seed: int = DEFAULT_SEED, split_seed: int | None = None) -> str:
    """The stable filename stem for one cell. Also its key in the summary table.

    The seeds are only appended when they are not the repo default, so the main grid's
    filenames stay short and the replicate runs are obvious at a glance. ``__s<n>`` is
    the training seed; ``__x<n>`` is a split seed pinned away from it.
    """
    stem = f"{dataset}__{arch}__h{hidden_dim}_l{num_layers}"
    if seed != DEFAULT_SEED:
        stem += f"__s{seed}"
    if split_seed is not None and split_seed != seed:
        stem += f"__x{split_seed}"
    return stem


def run_cell(
    dataset: str,
    arch: str,
    hidden_dim: int,
    num_layers: int,
    *,
    data_dir: str = DEFAULT_DATA_DIR,
    out_dir: str = DEFAULT_OUT_DIR,
    epochs: int | None = None,
    patience: int | None = None,
    val_fraction: float = 0.15,
    seed: int = DEFAULT_SEED,
    split_seed: int | None = None,
    save_checkpoints: bool = False,
    checkpoint_dir: str = "checkpoints",
) -> dict:
    """Train one capacity cell and return (and persist) its result dict."""
    data_path = os.path.join(data_dir, f"{dataset}.txt")
    names = load_names(data_path)
    vocab = Vocab(names)

    cfg = Config(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        arch=arch,
        val_fraction=val_fraction,
        seed=seed,
        dataset_label=dataset,
        dataset_path=data_path,
    )
    # The epoch budget is the --auto-epochs rule, derived from the *whole* dataset the
    # same way src/train.train does it, so a cell here matches what a user would get
    # from `python -m src.train --auto-epochs`. Early stopping is what actually ends
    # these runs; the budget is only a ceiling.
    cfg.epochs = epochs if epochs is not None else derive_epochs(len(names))
    cfg.early_stop_patience = (
        patience if patience is not None else derive_patience(cfg.epochs)
    )
    # No live sample previews: they cost generation time in every cell and print
    # nothing this sweep reads. (`fit` still previews once on the final epoch.)
    cfg.sample_every = 10 ** 9

    # `seed` drives *both* the train/val split and the initialization, because
    # src/train.train passes cfg.seed to split_names. That conflates two very different
    # uncertainties on a small dataset: "would another initialization have said this?"
    # and "would another 46-name validation set have said this?". `split_seed` pins the
    # split so a replicate can vary the initialization alone -- which is the only way to
    # tell those two apart, and on the small datasets they are not the same size.
    effective_split_seed = cfg.seed if split_seed is None else split_seed
    train_names, val_names = split_names(names, cfg.val_fraction, effective_split_seed)

    # Seed before construction so the initial weights are reproducible (WS-10). Not
    # calling src.train.seed_for_init only because that would hide the one line that
    # makes every number in the report repeatable.
    if cfg.seed_init:
        torch.manual_seed(cfg.seed)
    model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id)
    params = sum(p.numel() for p in model.parameters())

    label = cell_id(dataset, arch, hidden_dim, num_layers, cfg.seed, split_seed)
    print(f"\n=== {label} | {len(train_names)} train / {len(val_names)} val "
          f"| {params:,} params | budget {cfg.epochs}/patience {cfg.early_stop_patience}",
          flush=True)

    report: dict = {}
    started = time.time()
    fit(model, vocab, train_names, cfg, device="cpu", log_prefix="  ",
        val_names=val_names, report=report)
    seconds = time.time() - started

    # Re-score the restored best-epoch weights with dropout off, so `gap` here means
    # the same thing as the gap column in reports/ARCH.md.
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    train_nll = evaluate_loss(model, vocab, make_pairs(train_names, vocab), cfg,
                              criterion, "cpu")
    val_nll = evaluate_loss(model, vocab, make_pairs(val_names, vocab), cfg,
                            criterion, "cpu")

    best_epoch = report.get("best_epoch") or 0
    train_losses = report.get("train_losses") or []
    train_at_best = train_losses[best_epoch - 1] if 0 < best_epoch <= len(train_losses) else None

    result = {
        "cell": label,
        "dataset": dataset,
        "arch": arch,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "embedding_dim": cfg.embedding_dim,
        "dropout": cfg.dropout,
        "vocab_size": len(vocab),
        "total_names": len(names),
        "train_names": len(train_names),
        "val_names": len(val_names),
        "params": params,
        "params_per_train_name": round(params / max(1, len(train_names)), 1),
        "epoch_budget": cfg.epochs,
        "patience": cfg.early_stop_patience,
        "epochs_run": report.get("epochs_run"),
        "stopped_early": report.get("stopped_early"),
        "best_epoch": best_epoch,
        "best_val_loss": report.get("best_val_loss"),
        "train_loss_at_best": train_at_best,
        "train_nll": train_nll,
        "val_nll": val_nll,
        # Sign convention matches reports/ARCH.md: positive = held-out is worse than
        # train, i.e. the amount of overfitting.
        "gap": val_nll - train_nll,
        "seconds": round(seconds, 1),
        "seed": cfg.seed,
        "split_seed": effective_split_seed,
        "val_fraction": cfg.val_fraction,
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{label}.json")
    # Write the instant the cell finishes. A sweep killed by a session limit that left
    # thirty JSON files behind is fully recoverable; one holding them in memory is not.
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"  -> best val {result['best_val_loss']:.4f} @ epoch {best_epoch} "
          f"| gap {result['gap']:+.3f} | {seconds:.0f}s -> {out_path}", flush=True)

    if save_checkpoints:
        save_checkpoint(
            os.path.join(checkpoint_dir, f"{CHECKPOINT_PREFIX}{label}.pt"),
            model, cfg, vocab, train_names, val_names,
        )
    return result


def load_results(out_dir: str = DEFAULT_OUT_DIR) -> list[dict]:
    """Every cell written so far, sorted by dataset size then params.

    ``gap`` is recomputed from the two NLLs rather than trusted, so a file written
    before the sign convention was pinned down still reads correctly here.
    """
    if not os.path.isdir(out_dir):
        return []
    rows = []
    for fname in sorted(os.listdir(out_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(out_dir, fname), "r", encoding="utf-8") as fh:
            row = json.load(fh)
        row["gap"] = row["val_nll"] - row["train_nll"]
        rows.append(row)
    rows.sort(key=lambda r: (r["total_names"], r["arch"], r["params"]))
    return rows


def normalize_gaps(out_dir: str = DEFAULT_OUT_DIR) -> int:
    """Rewrite every cell file with the canonical ``gap = val_nll - train_nll``.

    Only needed once, for cells written before the convention was fixed. Idempotent.
    """
    fixed = 0
    for row in load_results(out_dir):
        path = os.path.join(out_dir, f"{row['cell']}.json")
        with open(path, "r", encoding="utf-8") as fh:
            on_disk = json.load(fh)
        if on_disk.get("gap") == row["gap"]:
            continue
        on_disk["gap"] = row["gap"]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(on_disk, fh, indent=2, sort_keys=True)
            fh.write("\n")
        fixed += 1
    return fixed


def print_summary(out_dir: str = DEFAULT_OUT_DIR, seed: int | None = DEFAULT_SEED) -> None:
    """Dump finished cells as one markdown table, ready to paste into the report.

    Defaults to the main grid (training seed and split seed both at the repo default).
    Pass ``seed=None`` for every cell including the replicate sweeps, which outnumber the
    main grid roughly two to one and are what `reports/_capacity/` is for.
    """
    rows = load_results(out_dir)
    if seed is not None:
        rows = [r for r in rows
                if r["seed"] == seed and r.get("split_seed", r["seed"]) == seed]
    if not rows:
        print("no results yet")
        return
    print("| dataset | names | arch | hidden | layers | params | best val | best ep | "
          "train NLL | gap | sec |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| `{r['dataset']}` | {r['train_names']} | {r['arch']} | {r['hidden_dim']} "
              f"| {r['num_layers']} | {r['params']:,} | {r['best_val_loss']:.4f} "
              f"| {r['best_epoch']} | {r['train_nll']:.4f} | {r['gap']:+.3f} "
              f"| {r['seconds']:.0f} |")


def print_pivot(out_dir: str = DEFAULT_OUT_DIR, seed: int = DEFAULT_SEED) -> None:
    """Best held-out loss as a hidden_dim x num_layers grid, one block per dataset+arch.

    This is the shape the question is actually asked in -- "for a dataset of N names,
    what capacity minimizes held-out loss" -- so it is worth having as its own view
    rather than reading it off the flat table by eye.
    """
    rows = [r for r in load_results(out_dir)
            if r["seed"] == seed and r.get("split_seed", r["seed"]) == seed]
    if not rows:
        print("no results yet")
        return
    hidden = sorted({r["hidden_dim"] for r in rows})
    layers = sorted({r["num_layers"] for r in rows})
    keys, seen = [], set()
    for r in rows:
        k = (r["total_names"], r["dataset"], r["arch"])
        if k not in seen:
            seen.add(k)
            keys.append(k)
    for total, dataset, arch in keys:
        block = {(r["hidden_dim"], r["num_layers"]): r
                 for r in rows if r["dataset"] == dataset and r["arch"] == arch}
        best = min(block.values(), key=lambda r: r["best_val_loss"], default=None)
        print(f"\n### {dataset} ({total} names, {arch}, seed {seed}) — best held-out loss")
        print("| layers | " + " | ".join(f"h={h}" for h in hidden) + " |")
        print("|---|" + "---:|" * len(hidden))
        for l in layers:
            cells = []
            for h in hidden:
                r = block.get((h, l))
                if r is None:
                    cells.append("—")
                elif best is not None and r["cell"] == best["cell"]:
                    cells.append(f"**{r['best_val_loss']:.4f}**")
                else:
                    cells.append(f"{r['best_val_loss']:.4f}")
            print(f"| {l} | " + " | ".join(cells) + " |")
        if best is not None:
            print(f"best: h={best['hidden_dim']} l={best['num_layers']} "
                  f"({best['params']:,} params, "
                  f"{best['params'] / best['train_names']:.0f} params/name)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datasets", nargs="+",
                        default=["motorcycle_brands", "typefaces", "pharma_drugs",
                                 "english_words"],
                        help="Dataset stems under --data-dir (without .txt).")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[64, 128, 256, 384])
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--archs", nargs="+", default=["lstm"],
                        choices=("lstm", "gru", "transformer"))
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the --auto-epochs budget for every cell.")
    parser.add_argument("--patience", type=int, default=None,
                        help="Override the derived early-stop patience.")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Replicate the grid under a different seed to measure "
                             "seed variance; the cell filename gains an __s<seed> suffix.")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="Pin the train/val split to this seed while --seed still "
                             "drives initialization and batch order. Use it to separate "
                             "initialization noise from which-names-were-held-out noise; "
                             "the cell filename gains an __x<seed> suffix.")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help=f"Where --save-checkpoints writes, as "
                             f"{CHECKPOINT_PREFIX}<cell>.pt. Default 'checkpoints'.")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help=f"Also write checkpoints/{CHECKPOINT_PREFIX}<cell>.pt. Off "
                             "by default: the sweep needs the metrics, not the weights, "
                             "and the full grid is hundreds of megabytes.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run cells whose JSON already exists.")
    parser.add_argument("--summary", action="store_true",
                        help="Print a markdown table of finished cells and exit.")
    parser.add_argument("--all-seeds", action="store_true",
                        help="--summary only: include the replicate sweeps, not just the "
                             "main grid at --seed.")
    parser.add_argument("--pivot", action="store_true",
                        help="Print best held-out loss as a hidden x layers grid per "
                             "dataset and arch (for --seed), and exit.")
    parser.add_argument("--normalize-gaps", action="store_true",
                        help="Rewrite existing cell files with gap = val_nll - train_nll "
                             "and exit. Idempotent; only cells written before the sign "
                             "convention was pinned need it.")
    args = parser.parse_args()

    if args.normalize_gaps:
        print(f"normalized {normalize_gaps(args.out_dir)} cell file(s)")
        return
    if args.summary:
        print_summary(args.out_dir, None if args.all_seeds else args.seed)
        return
    if args.pivot:
        print_pivot(args.out_dir, args.seed)
        return

    planned = [(d, a, h, l)
               for d in args.datasets
               for a in args.archs
               for l in args.layers
               for h in args.hidden_dims]
    done = skipped = 0
    for dataset, arch, hidden_dim, num_layers in planned:
        label = cell_id(dataset, arch, hidden_dim, num_layers, args.seed,
                        args.split_seed)
        out_path = os.path.join(args.out_dir, f"{label}.json")
        if os.path.exists(out_path) and not args.force:
            print(f"skip (exists) {label}", flush=True)
            skipped += 1
            continue
        run_cell(
            dataset, arch, hidden_dim, num_layers,
            data_dir=args.data_dir, out_dir=args.out_dir,
            epochs=args.epochs, patience=args.patience,
            val_fraction=args.val_fraction, seed=args.seed,
            split_seed=args.split_seed,
            save_checkpoints=args.save_checkpoints,
            checkpoint_dir=args.checkpoint_dir,
        )
        done += 1
    print(f"\n{done} cells run, {skipped} skipped, {len(planned)} planned.")


if __name__ == "__main__":
    main()
