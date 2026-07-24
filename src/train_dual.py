"""Train a DualCharRNN on name<TAB>value data (WS-4: name + numeric attribute).

Shane's original char-rnn learned paint names *and* their RGB values from one
network; this generalizes that to any scalar attribute. Trains a combined loss —
next-character cross-entropy (the usual generative task) plus MSE on a
z-score-normalized attribute — so the same LSTM encoding serves both heads.

Usage:
    python -m src.train_dual --data data/dual/paint_colors.tsv --epochs 300 --name paint_colors
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn

from .config import Config
from .data import Vocab
from .dual_data import (
    load_name_value_pairs,
    make_dual_batches,
    make_dual_pairs,
    normalize_values,
)
from .model import DualCharRNN
from . import sample as sampling


def fit_dual(
    model: DualCharRNN,
    vocab: Vocab,
    triples: list,
    cfg: Config,
    attr_loss_weight: float = 1.0,
    device: str = "cpu",
) -> DualCharRNN:
    """Combined-loss training loop, mirroring ``train.fit``'s shape."""
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    char_criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    attr_criterion = nn.MSELoss()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_char_loss = 0.0
        total_attr_loss = 0.0
        num_batches = 0

        for inputs, targets, values, lengths in make_dual_batches(
            triples, cfg.batch_size, vocab.pad_id, shuffle=True, generator=generator
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            values, lengths = values.to(device), lengths.to(device)

            logits, _ = model(inputs)
            char_loss = char_criterion(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
            pred_values = model.forward_attr(inputs, lengths)
            attr_loss = attr_criterion(pred_values, values)
            loss = char_loss + attr_loss_weight * attr_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            total_char_loss += char_loss.item()
            total_attr_loss += attr_loss.item()
            num_batches += 1

        avg_char = total_char_loss / max(num_batches, 1)
        avg_attr = total_attr_loss / max(num_batches, 1)

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(f"epoch {epoch:4d}/{cfg.epochs} | char_loss {avg_char:.4f} | attr_mse {avg_attr:.4f}")

        if epoch % cfg.sample_every == 0 or epoch == cfg.epochs:
            previews = sampling.generate_many(
                model, vocab, cfg, num=5, temperature=cfg.temperature, only_novel=False, device=device,
            )
            print("   samples:", ", ".join(previews) if previews else "(none)")

    return model


def save_dual_checkpoint(
    out_path: str,
    model: DualCharRNN,
    cfg: Config,
    vocab: Vocab,
    names: list,
    value_mean: float,
    value_std: float,
    attr_label: str,
) -> str:
    """Checkpoint dict — a superset of the base format (HANDOFF §2): the extra
    ``value_mean``/``value_std``/``attr_label`` keys are additive, so anything that
    only reads the original four keys (``model_state``, ``config``, ``vocab``,
    ``training_names``) keeps working unchanged.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.to_dict(),
            "vocab": vocab.to_dict(),
            "training_names": names,
            "value_mean": value_mean,
            "value_std": value_std,
            "attr_label": attr_label,
        },
        out_path,
    )
    return out_path


def train_dual(
    data_path: str,
    out_name: str,
    cfg: Config,
    attr_label: str,
    attr_loss_weight: float = 1.0,
    checkpoint_dir: str = "checkpoints",
    device: str = "cpu",
) -> str:
    pairs = load_name_value_pairs(data_path)
    names = [name for name, _ in pairs]
    raw_values = [value for _, value in pairs]
    norm_values, value_mean, value_std = normalize_values(raw_values)

    vocab = Vocab(names)
    print(f"Loaded {len(names)} name/value pairs | vocab size {len(vocab)} | device {device}")

    model = DualCharRNN(len(vocab), cfg, pad_id=vocab.pad_id).to(device)
    triples = make_dual_pairs(list(zip(names, norm_values)), vocab)
    fit_dual(model, vocab, triples, cfg, attr_loss_weight=attr_loss_weight, device=device)

    out_path = save_dual_checkpoint(
        os.path.join(checkpoint_dir, f"{out_name}.pt"),
        model, cfg, vocab, names, value_mean, value_std, attr_label,
    )
    print(f"\nSaved dual-output checkpoint -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a name + numeric-attribute char-RNN (WS-4 dual output).")
    parser.add_argument("--data", required=True, help="A name<TAB>value file, e.g. data/dual/paint_colors.tsv")
    parser.add_argument("--name", default="dual_model", help="Checkpoint name (without extension).")
    parser.add_argument("--attr-label", default="value",
                        help="Human-readable description of the attribute, saved in the checkpoint.")
    parser.add_argument("--attr-loss-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=None)
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
    train_dual(
        args.data, args.name, cfg, args.attr_label,
        attr_loss_weight=args.attr_loss_weight,
        checkpoint_dir=args.checkpoint_dir, device=device,
    )


if __name__ == "__main__":
    main()
