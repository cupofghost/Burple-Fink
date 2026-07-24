"""Generate names *and* their predicted numeric attribute from a dual checkpoint.

Reuses ``src.sample``'s generation loop as-is (``DualCharRNN.forward`` is inherited
unchanged from ``CharRNN``, so it's a drop-in model for ``generate_one``/
``generate_many``) and adds the attribute prediction on top.

Usage:
    python -m src.sample_dual --checkpoint checkpoints/paint_colors.pt --num 10 --temperature 0.9
"""

from __future__ import annotations

import argparse

import torch

from .config import Config
from .data import Vocab
from .model import DualCharRNN
from .sample import generate_many


def load_dual_checkpoint(path: str, device: str = "cpu"):
    """Like ``sample.load_checkpoint`` but for the dual-output superset format."""
    ckpt = torch.load(path, map_location=device)
    cfg = Config.from_dict(ckpt["config"])
    vocab = Vocab.from_dict(ckpt["vocab"])
    model = DualCharRNN(len(vocab), cfg, pad_id=vocab.pad_id).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    training_names = set(ckpt.get("training_names", []))
    value_mean = ckpt["value_mean"]
    value_std = ckpt["value_std"]
    attr_label = ckpt.get("attr_label", "value")
    return model, vocab, cfg, training_names, value_mean, value_std, attr_label


@torch.no_grad()
def predict_attr(model: DualCharRNN, vocab: Vocab, name: str, value_mean: float, value_std: float, device: str = "cpu") -> float:
    """Predict a generated name's attribute, mapped back to the original scale."""
    ids = vocab.encode(name)[:-1]  # [START, ...chars] -- matches training's input side
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    length = torch.tensor([len(ids)], dtype=torch.long, device=device)
    normalized = model.forward_attr(inp, length).item()
    return normalized * value_std + value_mean


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate names + predicted attribute from a WS-4 dual-output checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a dual .pt checkpoint.")
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, vocab, cfg, training_names, value_mean, value_std, attr_label = load_dual_checkpoint(args.checkpoint)
    temperature = args.temperature if args.temperature is not None else cfg.temperature

    names = generate_many(
        model, vocab, cfg,
        num=args.num,
        temperature=temperature,
        training_names=training_names,
        only_novel=not args.allow_existing,
        prefix=args.prefix,
    )

    print(f"\n=== {len(names)} names @ temperature {temperature} (attribute: {attr_label}) ===")
    for name in names:
        value = predict_attr(model, vocab, name, value_mean, value_std)
        print(f"  {name:<24} {attr_label}: {value:.3f}")


if __name__ == "__main__":
    main()
