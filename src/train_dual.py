"""Train the char-RNN with a second regression head (WS-4 dual-output).

Shane's original char-rnn didn't just invent paint-color names, it also predicted
each color's RGB value from the same network. This module generalizes that trick
to any ``name<TAB>value`` dataset: the model still learns to spell (the existing
next-character loss from ``src.train``) *and* learns to regress one scalar
attribute per name from a second head, trained jointly in one optimizer step.

Usage:
    python -m src.train_dual --data data/paint_colors.tsv --name paint_colors --epochs 300
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn

from .config import Config
from .data import Vocab, load_name_value_pairs, make_pairs
from .model import CharRNN
from .train import save_checkpoint
from . import sample as sampling


def fit_dual(
    model: CharRNN,
    vocab: Vocab,
    names: list[str],
    values: list[float],
    cfg: Config,
    device: str = "cpu",
    log_prefix: str = "",
) -> CharRNN:
    """Joint next-char + value-regression loop.

    Mirrors ``src.train.fit``'s optimizer / gradient-clipping / live-preview shape,
    but each step also computes ``model.regress_value`` on the same batch and adds
    its MSE (weighted by ``cfg.value_loss_weight``) to the char loss before the
    backward pass, so one set of gradients improves spelling and the attribute
    prediction together. Batching is done inline here (rather than reusing
    ``src.data.make_batches``) because the value loss needs to know which original
    name each padded row came from.
    """
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    pairs = make_pairs(names, vocab)  # pairs[i] lines up with names[i] / values[i]

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    mse_loss_fn = nn.MSELoss()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_ce, total_mse, num_batches = 0.0, 0.0, 0
        order = torch.randperm(len(pairs), generator=generator).tolist()

        for start in range(0, len(order), cfg.batch_size):
            idxs = order[start:start + cfg.batch_size]
            batch = [pairs[i] for i in idxs]
            lengths = [len(inp) for inp, _ in batch]
            max_len = max(lengths)

            inputs = torch.full((len(batch), max_len), vocab.pad_id, dtype=torch.long)
            targets = torch.full((len(batch), max_len), vocab.pad_id, dtype=torch.long)
            for row, (inp, tgt) in enumerate(batch):
                inputs[row, : len(inp)] = torch.tensor(inp, dtype=torch.long)
                targets[row, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)
            inputs, targets = inputs.to(device), targets.to(device)
            lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
            target_values = torch.tensor(
                [values[i] for i in idxs], dtype=torch.float32, device=device
            )

            logits, _ = model(inputs)
            loss_ce = ce_loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            pred_values = model.regress_value(inputs, lengths_t)
            loss_mse = mse_loss_fn(pred_values, target_values)

            loss = loss_ce + cfg.value_loss_weight * loss_mse

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            total_ce += loss_ce.item()
            total_mse += loss_mse.item()
            num_batches += 1

        avg_ce = total_ce / max(num_batches, 1)
        avg_mse = total_mse / max(num_batches, 1)

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(f"{log_prefix}epoch {epoch:4d}/{cfg.epochs} | ce {avg_ce:.4f} | value-mse {avg_mse:.4f}")

        if epoch % cfg.sample_every == 0 or epoch == cfg.epochs:
            previews = sampling.generate_many(
                model, vocab, cfg, num=5, temperature=cfg.temperature,
                training_names=set(names), only_novel=False, device=device,
            )
            print("   samples:", ", ".join(previews) if previews else "(none)")

    return model


def train_dual(
    data_path: str,
    out_name: str,
    cfg: Config,
    checkpoint_dir: str = "checkpoints",
    device: str = "cpu",
) -> str:
    # --- data ---
    pairs = load_name_value_pairs(data_path)
    names = [n for n, _ in pairs]
    values = [v for _, v in pairs]
    vocab = Vocab(names)
    print(f"Loaded {len(names)} name/value pairs | vocab size {len(vocab)} | device {device}")

    # --- model + training ---
    cfg.dual_output = True
    model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id, predict_value=True).to(device)
    fit_dual(model, vocab, names, values, cfg, device=device)

    # --- save checkpoint (same contract as src.train; cfg.dual_output marks the extra head) ---
    out_path = save_checkpoint(
        os.path.join(checkpoint_dir, f"{out_name}.pt"), model, cfg, vocab, names
    )
    print(f"\nSaved dual-output checkpoint -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a char-RNN with a name + numeric-attribute dual output (WS-4).")
    parser.add_argument("--data", required=True, help="TSV file of `name<TAB>value` rows.")
    parser.add_argument("--name", default="model", help="Checkpoint name (without extension).")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, dest="learning_rate")
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--value-loss-weight", type=float, default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    cfg = Config()
    for field in ("epochs", "batch_size", "learning_rate", "hidden_dim", "value_loss_weight"):
        val = getattr(args, field, None)
        if val is not None:
            setattr(cfg, field, val)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dual(args.data, args.name, cfg, checkpoint_dir=args.checkpoint_dir, device=device)


if __name__ == "__main__":
    main()
