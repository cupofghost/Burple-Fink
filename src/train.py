"""Train the char-RNN on a list of names and save a checkpoint.

Watching the live samples during training is the fun part: they start as noise,
become almost-words, then settle into plausible-but-invented names -- exactly the
progression Janelle Shane described with her paint colors.

Since WS-6 the loop can also hold names back and tell you the truth about them: pass
``--val-fraction 0.15`` and every epoch reports a held-out loss, the best epoch's weights
are the ones that get saved, and ``--patience N`` stops the run once it stops improving.
All of that is opt-in; without those flags this trains exactly as it always has.

WS-10 added four more knobs and fixed one bug:

* ``--weight-decay`` / ``--label-smoothing`` / ``--warmup-epochs`` — regularization and
  LR warmup, all off by default (see :func:`fit`).
* ``--auto-epochs`` — derive the epoch budget from the dataset size instead of making
  the user read a table (see :func:`derive_epochs`).
* ``seed_init`` — the RNG is now seeded *before* the model is built, so a run is
  reproducible from its initial weights on. This is the one WS-10 change that alters
  the default trajectory; ``--no-seed-init`` restores the old, unseeded initialization.

Usage:
    python -m src.train --data data/car_manufacturers.txt --epochs 300 --name manufacturers
    python -m src.train --data data/aircraft.txt --val-fraction 0.15 --patience 20 --name aircraft
    python -m src.train --data data/aircraft.txt --auto-epochs --val-fraction 0.15 --name aircraft
"""

from __future__ import annotations

import argparse
import copy
import math
import os

import torch
import torch.nn as nn

from .config import Config
from .data import Vocab, load_names, make_pairs, make_batches, split_names
from .model import CharRNN
from . import sample as sampling

# How many stalled epochs the "plateau" LR schedule waits before cutting the LR, when
# early stopping is off. When early stopping IS on we halve its patience instead, so the
# LR always gets a chance to drop before the run gives up entirely.
PLATEAU_PATIENCE = 10

# --- --auto-epochs (WS-10): the measured size -> budget rule -------------------------
# What WS-10 measured, on nine datasets spanning the whole shipped range, 159 to 8,631
# names (15% holdout, patience 25, otherwise default config), best epoch of the held-out
# loss:
#
#    159 -> 13    309 -> 10    435 -> 26    590 ->  9   863 -> 13
#   1218 -> 10   1691 ->  8   2223 -> 10   8631 ->  7
#
# The bottom does NOT move later as a dataset grows — if anything it arrives sooner. It
# sits between epoch 7 and 26 across a 54x range of dataset sizes. That is not a surprise
# once stated: an epoch on 8,631 names is fifty times as many gradient steps as an epoch
# on 159 names, so "epochs to convergence" holds roughly constant while the work done per
# epoch grows with the data. The README's old table, which read "bigger dataset -> more
# epochs" off two data points, was over-reading them.
#
# So the budget is NOT an estimate of where the bottom is. It is a ceiling on how far
# past the bottom it is safe to keep going, and *that* does scale with dataset size:
# running to epoch 300 costs car_manufacturers@159 +130% held-out loss but aircraft@435
# only +35% (WS-6), because the larger the dataset the less damage the extra epochs do.
# A sqrt curve gives a monotone ceiling that is ~4x the worst measured bottom at the
# small end and grows gently from there, and is clamped so it can never be smaller than
# the README's smallest suggestion or larger than today's default.
#
# Early stopping is what actually ends a run; the ceiling only has to be comfortably
# clear of the bottom, which at every size measured it is by 4x or more.
AUTO_EPOCH_COEFF = 1.2       # curve shape: COEFF * sqrt(dataset size)
AUTO_EPOCH_HEADROOM = 4.0    # budget = headroom * that, i.e. ~4x the worst bottom seen
AUTO_EPOCH_MIN = 60          # never below the README's smallest suggested budget
AUTO_EPOCH_MAX = 300         # never above today's default, which is already too long
AUTO_PATIENCE_DIVISOR = 6    # README pairs 60/10 and 120/20; both are budget/6
AUTO_PATIENCE_MIN = 10
AUTO_PATIENCE_MAX = 30


def seed_for_init(cfg: Config) -> None:
    """Seed the global RNG *before* a model is constructed, when ``cfg.seed_init`` is on.

    Why this exists (WS-10, STATUS.md **Known issues**): :func:`fit` seeds the RNG as its
    first act, but by then :func:`train` has already called ``CharRNN(...)`` and drawn the
    initial weights from whatever state the process happened to start in. Training was
    reproducible; *initialization* was not, and three identical ``--val-fraction 0.15``
    runs on ``car_manufacturers`` put the best epoch at 12, 16 and 19.

    ``cfg.seed_init`` defaults to True as of WS-10, which means the default training
    trajectory of every command changed once — not for the worse, just deterministically
    seeded from epoch 0 instead of from wherever the process started. Pass
    ``--no-seed-init`` (or ``Config(seed_init=False)``) for the old behavior.

    Deliberately a no-op when off, rather than seeding with something else: an unseeded
    default has to stay *exactly* unseeded for the flag to prove anything.
    """
    if cfg.seed_init:
        torch.manual_seed(cfg.seed)


def derive_epochs(num_names: int) -> int:
    """Turn a dataset size into an epoch budget (``--auto-epochs``).

    The README used to ship a lookup table — 60 epochs under 200 names, 120 for 200-500,
    300 for 1,000+ — that a human had to apply by hand, and that silently goes stale the
    moment a dataset grows. Three wave-3 lanes just grew ``data/`` from 12 datasets to
    28, so it went stale immediately. This is the rule as a function instead.

    The rule is ``4 x (1.2 * sqrt(N))``, rounded *up* to the next 10 and clamped to
    60..300. Read the comment on ``AUTO_EPOCH_COEFF`` for what that is and is not: it is
    a *ceiling* that grows with dataset size because over-training hurts a big dataset
    less than a small one, not a prediction of where the held-out loss bottoms out. The
    bottom, measured on nine datasets from 159 to 8,631 names, does not move with size at
    all — it lands between epoch 7 and 26 throughout.

    Worked examples across the shipped range: 309 names -> 90 epochs, 435 -> 110,
    590 -> 120, 863 -> 150, 1,218 -> 170, 2,223 -> 230, 8,631 -> 300. Every one of those
    is at least 4x the epoch at which that dataset's held-out loss actually bottomed.

    The budget is a ceiling, not a promise: pair it with ``--val-fraction`` and
    ``--patience`` (see :func:`derive_patience`) and the run stops when it stops
    improving, whatever the ceiling says.
    """
    if num_names <= 0:
        return AUTO_EPOCH_MIN
    raw = AUTO_EPOCH_HEADROOM * AUTO_EPOCH_COEFF * math.sqrt(num_names)
    # Round a *ceiling* up, never down: rounding 100.1 down to 100 would quietly shave
    # the headroom on exactly the dataset whose bottom is latest (aircraft, epoch 26).
    rounded = math.ceil(raw / 10.0) * 10
    return max(AUTO_EPOCH_MIN, min(AUTO_EPOCH_MAX, rounded))


def derive_patience(epochs: int) -> int:
    """The companion early-stop patience for an ``--auto-epochs`` budget.

    Kept in the same ratio the README's table used (it paired 60 epochs with patience 10
    and 120 with 20 — both budget/6), clamped to 10..30 so a tiny dataset still gets a
    fair chance to improve and a huge one doesn't coast for a hundred stalled epochs.
    """
    return max(AUTO_PATIENCE_MIN,
               min(AUTO_PATIENCE_MAX, int(round(epochs / AUTO_PATIENCE_DIVISOR))))


def apply_auto_epochs(cfg: Config, num_names: int, log_prefix: str = "") -> None:
    """Set ``cfg.epochs`` (and, if unset, ``cfg.early_stop_patience``) from the data size.

    Mutates ``cfg`` in place and prints the derived numbers *and the reason*, because a
    budget the user didn't type is a budget the user has to be able to audit. Patience is
    only filled in when the caller left it at 0 — an explicit ``--patience`` always wins —
    and is only announced when a validation split exists to make it meaningful.
    """
    cfg.epochs = derive_epochs(num_names)
    note = ""
    if cfg.early_stop_patience <= 0:
        cfg.early_stop_patience = derive_patience(cfg.epochs)
        note = f", patience {cfg.early_stop_patience}"
    print(f"{log_prefix}auto-epochs: {cfg.epochs} epochs{note} for {num_names} names")
    # Say what the number is, not more than it is. It is a size-scaled ceiling; the
    # measured val-loss bottoms sit at epoch 8-26 regardless of dataset size, so on a
    # split run --patience is what should actually end this.
    print(f"{log_prefix}  ceiling scaled from dataset size; held-out loss bottomed at "
          f"epoch 7-26 on every dataset measured, so --patience should stop this first")
    if cfg.val_fraction <= 0:
        print(f"{log_prefix}  note: nothing is held out, so --patience cannot fire and "
              f"this budget IS the run — add --val-fraction 0.15 to make it a ceiling")


def evaluate_loss(
    model: CharRNN,
    vocab: Vocab,
    pairs: list,
    cfg: Config,
    criterion: nn.Module,
    device: str = "cpu",
) -> float:
    """Mean next-character loss over ``pairs`` with no gradients and no dropout.

    Kept separate from the training loop so the held-out number is computed by
    demonstrably the same criterion as the training number — the only difference being
    ``model.eval()`` (dropout off) and ``torch.no_grad()``. Batching is unshuffled, so
    this draws nothing from any RNG: a run with validation enabled and a run without it
    see the identical random stream.
    """
    was_training = model.training
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for inputs, targets, _lengths in make_batches(
            pairs, cfg.batch_size, vocab.pad_id, shuffle=False
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            total_loss += loss.item()
            num_batches += 1
    if was_training:
        model.train()
    return total_loss / max(num_batches, 1)


def fit(
    model: CharRNN,
    vocab: Vocab,
    names: list[str],
    cfg: Config,
    device: str = "cpu",
    log_prefix: str = "",
    *,
    val_names: list[str] | None = None,
    report: dict | None = None,
) -> CharRNN:
    """Run the training loop *in place* on an already-built model and vocab.

    Factored out of :func:`train` so pretraining (a fresh base model) and fine-tuning
    (a loaded base model) share exactly one optimizer / loss / gradient-clipping /
    live-preview implementation. It never touches the checkpoint on disk — callers
    save with :func:`save_checkpoint` — because fine-tuning wants a different
    ``training_names`` list than the names it trained the base on.

    Validation path (WS-6, opt-in). Pass ``val_names`` — names *not* in ``names`` — and
    each epoch also reports the loss on them, the best-scoring epoch's weights are kept
    in memory and restored before returning, and ``cfg.early_stop_patience`` epochs
    without improvement end the run. Without ``val_names`` every one of those branches
    is skipped and the loop is byte-for-byte the pre-WS-6 loop; that is deliberate, and
    ``tests/test_training_quality.py`` pins the identical loss trajectory to prove it.

    ``cfg.lr_schedule`` ("none" | "plateau" | "cosine") is independent of the split;
    "plateau" steers on val loss when there is one and on train loss otherwise.
    ``"none"`` — the default — constructs no scheduler at all.

    Training regimen (WS-10, all three off by default):

    ``cfg.weight_decay > 0`` switches the optimizer from ``Adam`` to ``AdamW``. This is
    deliberate and not interchangeable with ``Adam(weight_decay=...)``: Adam implements
    decay as an L2 term *added to the gradient*, which then goes through Adam's per-
    parameter adaptive scaling, so parameters with small gradients get decayed far less
    than parameters with large ones — the decay strength ends up depending on the
    gradient history. AdamW decouples it, subtracting ``lr * wd * w`` from the weight
    directly, so every parameter is pulled toward zero at the same rate. On a model this
    small the difference is the difference between a knob that means something and a
    knob that quietly does something else.

    ``cfg.label_smoothing > 0`` softens the next-character target in the **training**
    criterion only. Validation is always scored with an unsmoothed
    ``CrossEntropyLoss``, because smoothing raises the loss floor (a smoothed criterion
    cannot reach 0 even on a perfect prediction): if the val criterion were smoothed too,
    "best val loss" would not be comparable between two settings, nor to any number this
    repo has already recorded. So with smoothing on, the *train* loss is on a different
    scale from the val loss and from previous runs — the val loss is the one to compare.

    ``cfg.warmup_epochs > 0`` ramps the LR linearly from ``lr/N`` to ``lr`` over the
    first N epochs. It composes with the schedules rather than fighting them: the
    ``plateau``/``cosine`` scheduler is not built (and so cannot step) until warmup
    finishes, and ``cosine`` then anneals over the *remaining* ``epochs - warmup_epochs``
    so it still reaches ``lr_min`` at the end of the run instead of being cut off
    part-way down.

    Optional ``report`` dict is filled in with ``train_losses``, ``val_losses``, ``lrs``
    (the learning rate each epoch actually ran at), ``best_epoch``, ``best_val_loss``,
    ``stopped_early`` and ``epochs_run`` so callers and tests can inspect the run without
    parsing stdout. The return value stays the model, unchanged, because three call sites
    rely on that.
    """
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    pairs = make_pairs(names, vocab)
    val_pairs = make_pairs(val_names, vocab) if val_names else []

    if cfg.weight_decay > 0:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    else:
        # Plain Adam, constructed exactly as before, so the default path provably never
        # goes near AdamW's different update rule.
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    # ignore_index=pad_id => padding positions contribute nothing to the loss.
    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_id, label_smoothing=cfg.label_smoothing)
    # The held-out number stays on the unsmoothed scale — see the docstring.
    val_criterion = (nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
                     if cfg.label_smoothing > 0 else criterion)

    scheduler = _make_scheduler(optimizer, cfg)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    if cfg.warmup_epochs > 0:
        # Hold the schedule back until the ramp is done, then build it fresh so it starts
        # from the full LR (and, for cosine, anneals over the epochs that are left).
        scheduler = None
        print(f"{log_prefix}warmup: lr ramps linearly to {base_lrs[0]:.2e} over the "
              f"first {cfg.warmup_epochs} epochs")

    train_losses: list[float] = []
    val_losses: list[float] = []
    lrs: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    best_state: dict | None = None
    stalled = 0
    stopped_early = False
    epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        if cfg.warmup_epochs > 0 and epoch <= cfg.warmup_epochs + 1:
            _apply_warmup(optimizer, cfg, epoch, base_lrs)
            if epoch == cfg.warmup_epochs + 1:
                scheduler = _make_scheduler(optimizer, cfg)
        lrs.append(optimizer.param_groups[0]["lr"])

        model.train()
        total_loss = 0.0
        num_batches = 0

        for inputs, targets, _lengths in make_batches(
            pairs, cfg.batch_size, vocab.pad_id, shuffle=True, generator=generator
        ):
            inputs, targets = inputs.to(device), targets.to(device)

            logits, _ = model(inputs)  # (batch, time, vocab)
            # Flatten batch & time so CrossEntropyLoss sees (N, vocab) vs (N,).
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping keeps recurrent training from exploding.
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        train_losses.append(avg_loss)

        # --- held-out loss, best-weight bookkeeping, early stopping (opt-in) ---
        val_loss = None
        if val_pairs:
            val_loss = evaluate_loss(model, vocab, val_pairs, cfg, val_criterion, device)
            val_losses.append(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                # Deep-copy to CPU: state_dict() hands back live references to the
                # model's own tensors, which the next optimizer.step() would overwrite.
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stalled = 0
            else:
                stalled += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            if val_loss is None:
                print(f"{log_prefix}epoch {epoch:4d}/{cfg.epochs} | loss {avg_loss:.4f}")
            else:
                print(f"{log_prefix}epoch {epoch:4d}/{cfg.epochs} "
                      f"| train {avg_loss:.4f} | val {val_loss:.4f}")

        # Peek at what the model can produce so far.
        if epoch % cfg.sample_every == 0 or epoch == cfg.epochs:
            previews = sampling.generate_many(
                model, vocab, cfg, num=5, temperature=cfg.temperature,
                training_names=set(names), only_novel=False, device=device,
            )
            print("   samples:", ", ".join(previews) if previews else "(none)")

        if scheduler is not None and epoch > cfg.warmup_epochs:
            _step_scheduler(scheduler, cfg, val_loss, avg_loss, optimizer, log_prefix)

        if val_pairs and cfg.early_stop_patience > 0 and stalled >= cfg.early_stop_patience:
            stopped_early = True
            print(f"{log_prefix}early stop at epoch {epoch}: no val improvement for "
                  f"{stalled} epochs (best val {best_val:.4f} at epoch {best_epoch})")
            break

    # Restore the best epoch's weights. Without this the caller would checkpoint
    # whatever the *last* epoch produced, which on these tiny datasets is routinely
    # a worse model than epoch 40 was.
    if best_state is not None:
        model.load_state_dict(best_state)
        if not stopped_early and best_epoch < epoch:
            # Only worth saying when it actually rewound. The early-stop message above
            # already named the best epoch, and "restored epoch 40 of 40" is just noise.
            print(f"{log_prefix}restored best weights from epoch {best_epoch} "
                  f"(val {best_val:.4f}); last epoch was {val_losses[-1]:.4f}")

    if report is not None:
        report.update(
            train_losses=train_losses,
            val_losses=val_losses,
            lrs=lrs,
            best_epoch=best_epoch,
            best_val_loss=best_val if best_state is not None else None,
            stopped_early=stopped_early,
            epochs_run=epoch,
        )

    return model


def _make_scheduler(optimizer: torch.optim.Optimizer, cfg: Config):
    """Build the LR scheduler named by ``cfg.lr_schedule``, or ``None`` for "none".

    Returning ``None`` rather than a no-op scheduler matters: it guarantees the default
    path never calls into ``torch.optim.lr_scheduler`` at all, so the learning rate is
    provably the same constant it was before WS-6.

    With ``cfg.warmup_epochs > 0`` this is called a second time, at the end of the ramp,
    so the schedule starts from the full learning rate. Cosine's ``T_max`` therefore
    counts only the post-warmup epochs; with warmup off that expression is
    ``max(1, cfg.epochs)``, exactly as it was.
    """
    if cfg.lr_schedule == "none":
        return None
    if cfg.lr_schedule == "plateau":
        patience = (max(1, cfg.early_stop_patience // 2)
                    if cfg.early_stop_patience > 0 else PLATEAU_PATIENCE)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.lr_factor,
            patience=patience, min_lr=cfg.lr_min,
        )
    if cfg.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, cfg.epochs - max(0, cfg.warmup_epochs)),
            eta_min=cfg.lr_min,
        )
    raise ValueError(
        f"Unknown lr_schedule {cfg.lr_schedule!r}; expected 'none', 'plateau' or 'cosine'."
    )


def _apply_warmup(optimizer: torch.optim.Optimizer, cfg: Config, epoch: int,
                  base_lrs: list[float]) -> None:
    """Set this epoch's warmup learning rate, in place, before the epoch runs.

    Epoch ``e`` of ``N`` trains at ``base_lr * e / N``, so the first epoch is gentle and
    epoch ``N`` is already at full LR; the extra call at ``N + 1`` restores the exact base
    LR and is the hand-off point where :func:`_make_scheduler` takes over.

    Why warm up at all on a char-RNN: the first few updates at 3e-3 on freshly random
    weights are the ones most likely to throw the recurrent state somewhere the run never
    recovers from, and they are also the updates where Adam's second-moment estimate is
    least trustworthy. Warmup is the standard answer; whether it *helps here* is a
    measurement, not a promise — see the WS-10 report.
    """
    scale = min(1.0, epoch / max(1, cfg.warmup_epochs))
    for group, base in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base * scale


def _step_scheduler(scheduler, cfg: Config, val_loss, train_loss, optimizer, log_prefix):
    """Advance the schedule by one epoch and announce any LR change.

    "plateau" needs a metric: the held-out loss when there is one, otherwise the train
    loss. Steering a plateau schedule on train loss is weaker (train loss rarely
    plateaus) but it is well-defined, so ``--lr-schedule plateau`` without a split
    degrades rather than crashes.
    """
    before = optimizer.param_groups[0]["lr"]
    if cfg.lr_schedule == "plateau":
        scheduler.step(val_loss if val_loss is not None else train_loss)
    else:
        scheduler.step()
    after = optimizer.param_groups[0]["lr"]
    if after != before and cfg.lr_schedule == "plateau":
        print(f"{log_prefix}  lr {before:.2e} -> {after:.2e}")


def save_checkpoint(
    out_path: str,
    model: CharRNN,
    cfg: Config,
    vocab: Vocab,
    names: list[str],
    val_names: list[str] | None = None,
) -> str:
    """Write the checkpoint dict — the format contract every stage depends on (§2).

    ``val_names`` adds the one **additive** key WS-6 introduces (HANDOFF §2): the held-out
    names, or ``[]`` when the model trained on everything. It is written unconditionally
    so new checkpoints are self-describing, but every reader must use
    ``ckpt.get("val_names", [])`` — no checkpoint written before WS-6 has the key.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.to_dict(),
            "vocab": vocab.to_dict(),
            "training_names": names,
            "val_names": list(val_names or []),
        },
        out_path,
    )
    return out_path


def train(
    data_path: str,
    out_name: str,
    cfg: Config,
    checkpoint_dir: str = "checkpoints",
    device: str = "cpu",
    auto_epochs: bool = False,
) -> str:
    """Train from scratch on one dataset and write ``<checkpoint_dir>/<out_name>.pt``.

    ``auto_epochs`` is a call-time argument rather than a ``Config`` field on purpose:
    the budget can only be derived once the dataset has been read, and what gets written
    into the checkpoint's config should be the epoch count that actually ran.
    """
    # --- data ---
    names = load_names(data_path)
    # The vocabulary is built from *all* names, before the split. Deriving it from the
    # training half only would make some held-out names unencodable, which is a data
    # problem masquerading as a modelling one; a character inventory is not the kind of
    # leakage a held-out set is meant to detect.
    vocab = Vocab(names)
    train_names, val_names = split_names(names, cfg.val_fraction, cfg.seed)
    split_note = f" | {len(train_names)} train / {len(val_names)} val" if val_names else ""
    print(f"Loaded {len(names)} names | vocab size {len(vocab)} | device {device}{split_note}")
    if auto_epochs:
        apply_auto_epochs(cfg, len(names))

    # --- model + training ---
    # Seed *before* construction (WS-10): otherwise the initial weights come from
    # whatever RNG state the process started in and the run is not reproducible.
    seed_for_init(cfg)
    model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id).to(device)
    fit(model, vocab, train_names, cfg, device=device, val_names=val_names or None)

    # --- save checkpoint (weights + everything needed to sample later) ---
    # training_names is the *train* half so novelty is judged against what the model
    # actually saw; val_names rides along so evaluation can report an honest held-out NLL.
    out_path = save_checkpoint(
        os.path.join(checkpoint_dir, f"{out_name}.pt"), model, cfg, vocab,
        train_names, val_names,
    )
    print(f"\nSaved checkpoint -> {out_path}")
    return out_path


# The WS-9/WS-10 flags every entry point shares. Defined once here and reused by
# src/pretrain.py and src/finetune.py so the three CLIs cannot drift apart, and so the
# help text for "what does weight decay do" lives in exactly one place.
REGIMEN_FIELDS = ("arch", "seed_init", "weight_decay", "label_smoothing", "warmup_epochs")


def add_regimen_args(parser: argparse.ArgumentParser,
                     include_seed_init: bool = True) -> None:
    """Add ``--arch`` and the WS-10 training-regimen flags to ``parser``.

    Every default is ``None`` — meaning "not passed" — so the ``Config`` default survives
    unless the flag is actually used. ``--arch`` only sets ``cfg.arch``; the architectures
    themselves are WS-9's (``src/model.py``), and ``lstm`` is today's model.

    ``include_seed_init=False`` for :mod:`src.finetune`, which loads its weights from a
    checkpoint and so has no random initialization to seed. Offering the flag there would
    be offering a switch wired to nothing.
    """
    parser.add_argument("--arch", choices=("lstm", "gru", "transformer"), default=None,
                        help="Model architecture. Default 'lstm' = today's model.")
    if include_seed_init:
        parser.add_argument("--seed-init", action=argparse.BooleanOptionalAction,
                            default=None, dest="seed_init",
                            help="Seed the RNG before building the model, so the initial "
                                 "weights are reproducible too. On by default since "
                                 "WS-10; --no-seed-init restores the old unseeded "
                                 "initialization.")
    parser.add_argument("--weight-decay", type=float, default=None, dest="weight_decay",
                        help="AdamW decoupled weight decay, e.g. 0.01. Default 0 = plain "
                             "Adam. Any value > 0 switches the optimizer to AdamW.")
    parser.add_argument("--label-smoothing", type=float, default=None,
                        dest="label_smoothing",
                        help="Soften the next-character target, e.g. 0.1. Default 0 = "
                             "off. Applies to the training loss only; the held-out loss "
                             "stays unsmoothed so it remains comparable across runs.")
    parser.add_argument("--warmup-epochs", type=int, default=None, dest="warmup_epochs",
                        help="Ramp the LR linearly from lr/N to lr over the first N "
                             "epochs. Default 0 = off. Composes with --lr-schedule: the "
                             "schedule starts after the ramp.")


def apply_regimen_args(cfg: Config, args: argparse.Namespace) -> None:
    """Copy any regimen flag that was actually passed onto ``cfg``, leaving the rest."""
    for field in REGIMEN_FIELDS:
        val = getattr(args, field, None)
        if val is not None:
            setattr(cfg, field, val)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a char-RNN name generator.")
    parser.add_argument("--data", required=True, help="Newline-separated list of training names.")
    parser.add_argument("--name", default="model", help="Checkpoint name (without extension).")
    parser.add_argument("--epochs", type=int, default=None, help="Override the default epoch count.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, dest="learning_rate")
    parser.add_argument("--hidden-dim", type=int, default=None)
    # Depth is the single largest lever reports/CAPACITY.md measured -- 0.143 nats one way
    # on typefaces, +5.7% the other way on aircraft -- and until wave 4 it was the only
    # swept axis with no CLI flag, reachable solely by editing Config. Note the sweep found
    # NO rule mapping dataset size to a best depth: two datasets within 1.5x of each other
    # in size wanted optima 29x apart in parameters. So this is a knob to try, not one to
    # set from a table.
    parser.add_argument("--num-layers", type=int, default=None, dest="num_layers",
                        help="Stacked recurrent layers (default 2). The biggest single "
                             "lever measured; worth trying 1 on small datasets.")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    # --- WS-6 training-quality flags; every default is None so the Config default
    # (i.e. today's behavior) survives unless you actually pass the flag. ---
    parser.add_argument("--val-fraction", type=float, default=None, dest="val_fraction",
                        help="Hold out this share of names for validation (e.g. 0.15). "
                             "Default 0 = train on everything, as before.")
    parser.add_argument("--patience", type=int, default=None, dest="early_stop_patience",
                        help="Stop after N epochs with no validation improvement "
                             "(needs --val-fraction). Default 0 = never stop early.")
    parser.add_argument("--lr-schedule", choices=("none", "plateau", "cosine"),
                        default=None, dest="lr_schedule",
                        help="Learning-rate schedule. Default 'none' = constant LR.")
    add_regimen_args(parser)
    parser.add_argument("--auto-epochs", action="store_true",
                        help="Derive the epoch budget from the dataset size instead of "
                             "reading a table (see derive_epochs). Ignored if --epochs "
                             "is also given.")
    args = parser.parse_args()

    cfg = Config()
    for field in ("epochs", "batch_size", "learning_rate", "hidden_dim", "num_layers",
                  "val_fraction", "early_stop_patience", "lr_schedule"):
        val = getattr(args, field, None)
        if val is not None:
            setattr(cfg, field, val)
    apply_regimen_args(cfg, args)

    auto_epochs = args.auto_epochs and args.epochs is None
    if args.auto_epochs and args.epochs is not None:
        print(f"--epochs {args.epochs} was given explicitly; ignoring --auto-epochs.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(args.data, args.name, cfg, checkpoint_dir=args.checkpoint_dir, device=device,
          auto_epochs=auto_epochs)


if __name__ == "__main__":
    main()
