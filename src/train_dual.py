"""Train a dual-output char-RNN: emit a name AND regress a numeric attribute (WS-4).

Janelle Shane's original paint-color net paired every invented name with an RGB
triple. Here a second linear head reads the same LSTM's per-name hidden state and
regresses one scalar (e.g. a car brand's founding year), trained jointly with the
usual next-character loss: ``combined_loss = ce_loss + value_weight * mse_loss``.
The value head (``CharRNN.value_head``, see src/model.py) only exists when
``cfg.dual_output=True``, so ordinary single-output checkpoints and every other
stage (sample/evaluate/serve/export_web) are unaffected by this module.

Usage:
    python -m src.train_dual --data data/car_manufacturers_founding_year.tsv \
        --epochs 300 --name manufacturers_founding_year --value-label "founding year"
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn

from .config import Config
from .data import load_name_value_pairs, Vocab, make_pairs
from .model import CharRNN
from . import sample as sampling


def _make_dual_batches(pairs, values, batch_size, pad_id, generator):
    """Like ``data.make_batches``, but keeps each name's target value aligned to
    its batch row. Kept local rather than extending the shared ``make_batches``
    (used by the single-output train/pretrain/finetune paths) to keep this
    workstream's footprint on shared files to the bare additive minimum.
    """
    order = torch.randperm(len(pairs), generator=generator).tolist()
    for start in range(0, len(order), batch_size):
        idxs = order[start:start + batch_size]
        batch = [pairs[i] for i in idxs]
        vals = [values[i] for i in idxs]
        lengths = [len(inp) for inp, _ in batch]
        max_len = max(lengths)

        inputs = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        targets = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        for row, (inp, tgt) in enumerate(batch):
            inputs[row, : len(inp)] = torch.tensor(inp, dtype=torch.long)
            targets[row, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)

        yield (
            inputs,
            targets,
            torch.tensor(lengths, dtype=torch.long),
            torch.tensor(vals, dtype=torch.float32),
        )


def fit_dual(
    model: CharRNN,
    vocab: Vocab,
    names: list[str],
    values_z: list[float],
    cfg: Config,
    device: str = "cpu",
    value_weight: float = 0.3,
) -> CharRNN:
    """Joint next-char cross-entropy + value-regression MSE, one shared LSTM."""
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    pairs = make_pairs(names, vocab)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    mse_loss_fn = nn.MSELoss()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_ce, total_mse, num_batches = 0.0, 0.0, 0

        for inputs, targets, lengths, values in _make_dual_batches(
            pairs, values_z, cfg.batch_size, vocab.pad_id, generator
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            lengths, values = lengths.to(device), values.to(device)

            out, _hidden = model.encode(inputs)
            logits = model.head(out)
            ce_loss = ce_loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            # Gather each row's LSTM output at its own last non-pad timestep so
            # padding never leaks into the value head's summary vector.
            last_idx = (lengths - 1).clamp(min=0)
            state = out[torch.arange(out.size(0), device=device), last_idx]
            pred_value = model.predict_value(state)
            value_loss = mse_loss_fn(pred_value, values)

            loss = ce_loss + value_weight * value_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            total_ce += ce_loss.item()
            total_mse += value_loss.item()
            num_batches += 1

        avg_ce = total_ce / max(num_batches, 1)
        avg_mse = total_mse / max(num_batches, 1)

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(f"epoch {epoch:4d}/{cfg.epochs} | ce {avg_ce:.4f} | value_mse {avg_mse:.4f}")

        if epoch % cfg.sample_every == 0 or epoch == cfg.epochs:
            previews = sampling.generate_many(
                model, vocab, cfg, num=5, temperature=cfg.temperature,
                training_names=set(names), only_novel=False, device=device,
                return_value=True,
            )
            label = cfg.value_label or "value"
            rendered = ", ".join(f"{n} ({label}: {v:.1f})" for n, v in previews)
            print("   samples:", rendered if rendered else "(none)")

    return model


def train_dual(
    data_path: str,
    out_name: str,
    cfg: Config,
    value_weight: float = 0.3,
    value_label: str = "",
    checkpoint_dir: str = "checkpoints",
    device: str = "cpu",
) -> str:
    # --- data: names + a z-scored numeric attribute for stable regression ---
    pairs = load_name_value_pairs(data_path)
    names = [n for n, _ in pairs]
    raw_values = torch.tensor([v for _, v in pairs], dtype=torch.float32)
    value_mean = raw_values.mean().item()
    value_std = raw_values.std().item() or 1.0
    values_z = ((raw_values - value_mean) / value_std).tolist()

    vocab = Vocab(names)
    cfg.dual_output = True
    cfg.value_mean = value_mean
    cfg.value_std = value_std
    cfg.value_label = value_label
    print(f"Loaded {len(names)} name/value pairs | vocab size {len(vocab)} | device {device}")

    # --- model + training ---
    model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id).to(device)
    fit_dual(model, vocab, names, values_z, cfg, device=device, value_weight=value_weight)

    # --- save checkpoint: same 4-key format (§2) every stage relies on; the
    # dual-output fields all live inside the already-extensible `config` dict ---
    os.makedirs(checkpoint_dir, exist_ok=True)
    out_path = os.path.join(checkpoint_dir, f"{out_name}.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.to_dict(),
            "vocab": vocab.to_dict(),
            "training_names": names,
        },
        out_path,
    )
    print(f"\nSaved checkpoint -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a dual-output char-RNN (name + numeric attribute)."
    )
    parser.add_argument("--data", required=True, help="A name<TAB>value file (WS-4 format).")
    parser.add_argument("--name", default="dual_model", help="Checkpoint name (without extension).")
    parser.add_argument("--value-label", default="", help="Human label for the attribute, e.g. 'founding year'.")
    parser.add_argument("--value-weight", type=float, default=0.3,
                        help="Weight of the MSE term in the combined loss.")
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
        args.data, args.name, cfg,
        value_weight=args.value_weight, value_label=args.value_label,
        checkpoint_dir=args.checkpoint_dir, device=device,
    )


if __name__ == "__main__":
    main()
