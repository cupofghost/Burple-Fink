"""Train the char-RNN on a list of names and save a checkpoint.

Watching the live samples during training is the fun part: they start as noise,
become almost-words, then settle into plausible-but-invented names -- exactly the
progression Janelle Shane described with her paint colors.

Since WS-6 the loop can also hold names back and tell you the truth about them: pass
``--val-fraction 0.15`` and every epoch reports a held-out loss, the best epoch's weights
are the ones that get saved, and ``--patience N`` stops the run once it stops improving.
All of that is opt-in; without those flags this trains exactly as it always has.

Usage:
    python -m src.train --data data/car_manufacturers.txt --epochs 300 --name manufacturers
    python -m src.train --data data/aircraft.txt --val-fraction 0.15 --patience 20 --name aircraft
"""

from __future__ import annotations

import argparse
import copy
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

    Optional ``report`` dict is filled in with ``train_losses``, ``val_losses``,
    ``best_epoch``, ``best_val_loss``, ``stopped_early`` and ``epochs_run`` so callers
    and tests can inspect the run without parsing stdout. The return value stays the
    model, unchanged, because three call sites rely on that.
    """
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    pairs = make_pairs(names, vocab)
    val_pairs = make_pairs(val_names, vocab) if val_names else []

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    # ignore_index=pad_id => padding positions contribute nothing to the loss.
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    scheduler = _make_scheduler(optimizer, cfg)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    best_state: dict | None = None
    stalled = 0
    stopped_early = False
    epoch = 0

    for epoch in range(1, cfg.epochs + 1):
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
            val_loss = evaluate_loss(model, vocab, val_pairs, cfg, criterion, device)
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

        if scheduler is not None:
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
            optimizer, T_max=max(1, cfg.epochs), eta_min=cfg.lr_min,
        )
    raise ValueError(
        f"Unknown lr_schedule {cfg.lr_schedule!r}; expected 'none', 'plateau' or 'cosine'."
    )


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
) -> str:
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

    # --- model + training ---
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a char-RNN name generator.")
    parser.add_argument("--data", required=True, help="Newline-separated list of training names.")
    parser.add_argument("--name", default="model", help="Checkpoint name (without extension).")
    parser.add_argument("--epochs", type=int, default=None, help="Override the default epoch count.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, dest="learning_rate")
    parser.add_argument("--hidden-dim", type=int, default=None)
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
    args = parser.parse_args()

    cfg = Config()
    for field in ("epochs", "batch_size", "learning_rate", "hidden_dim",
                  "val_fraction", "early_stop_patience", "lr_schedule"):
        val = getattr(args, field, None)
        if val is not None:
            setattr(cfg, field, val)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(args.data, args.name, cfg, checkpoint_dir=args.checkpoint_dir, device=device)


if __name__ == "__main__":
    main()
