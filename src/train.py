"""Train the char-RNN on a list of names and save a checkpoint.

Watching the live samples during training is the fun part: they start as noise,
become almost-words, then settle into plausible-but-invented names -- exactly the
progression Janelle Shane described with her paint colors.

Usage:
    python -m src.train --data data/car_manufacturers.txt --epochs 300 --name manufacturers
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn

from .config import Config
from .data import Vocab, load_names, make_pairs, make_batches
from .model import CharRNN
from . import sample as sampling


def fit(
    model: CharRNN,
    vocab: Vocab,
    names: list[str],
    cfg: Config,
    device: str = "cpu",
    log_prefix: str = "",
) -> CharRNN:
    """Run the training loop *in place* on an already-built model and vocab.

    Factored out of :func:`train` so pretraining (a fresh base model) and fine-tuning
    (a loaded base model) share exactly one optimizer / loss / gradient-clipping /
    live-preview implementation. It never touches the checkpoint on disk — callers
    save with :func:`save_checkpoint` — because fine-tuning wants a different
    ``training_names`` list than the names it trained the base on.
    """
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    pairs = make_pairs(names, vocab)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    # ignore_index=pad_id => padding positions contribute nothing to the loss.
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

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

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(f"{log_prefix}epoch {epoch:4d}/{cfg.epochs} | loss {avg_loss:.4f}")

        # Peek at what the model can produce so far.
        if epoch % cfg.sample_every == 0 or epoch == cfg.epochs:
            previews = sampling.generate_many(
                model, vocab, cfg, num=5, temperature=cfg.temperature,
                training_names=set(names), only_novel=False, device=device,
            )
            print("   samples:", ", ".join(previews) if previews else "(none)")

    return model


def save_checkpoint(
    out_path: str,
    model: CharRNN,
    cfg: Config,
    vocab: Vocab,
    names: list[str],
) -> str:
    """Write the checkpoint dict — the format contract every stage depends on (§2)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.to_dict(),
            "vocab": vocab.to_dict(),
            "training_names": names,
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
    vocab = Vocab(names)
    print(f"Loaded {len(names)} names | vocab size {len(vocab)} | device {device}")

    # --- model + training ---
    model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id).to(device)
    fit(model, vocab, names, cfg, device=device)

    # --- save checkpoint (weights + everything needed to sample later) ---
    out_path = save_checkpoint(
        os.path.join(checkpoint_dir, f"{out_name}.pt"), model, cfg, vocab, names
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
    args = parser.parse_args()

    cfg = Config()
    for field in ("epochs", "batch_size", "learning_rate", "hidden_dim"):
        val = getattr(args, field, None)
        if val is not None:
            setattr(cfg, field, val)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(args.data, args.name, cfg, checkpoint_dir=args.checkpoint_dir, device=device)


if __name__ == "__main__":
    main()
