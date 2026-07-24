"""Generate names with a predicted numeric attribute (WS-4 dual-output checkpoints).

Mirrors ``src.sample`` but, after sampling each name's characters, also reads the
value head off the same hidden state — Shane's name+RGB trick, in reverse.

Usage:
    python -m src.sample_dual --checkpoint checkpoints/paint_colors.pt --num 10
"""

from __future__ import annotations

import argparse
from typing import List, Set, Tuple

import torch

from .config import Config
from .data import Vocab
from .model import CharRNN
from .sample import generate_one, load_checkpoint


@torch.no_grad()
def generate_one_with_value(
    model: CharRNN,
    vocab: Vocab,
    temperature: float,
    max_length: int,
    prefix: str = "",
    device: str = "cpu",
) -> Tuple[str, float]:
    """Sample one name, then predict its attribute from the same hidden state
    ``src.train_dual.fit_dual`` trained on: START + characters, no END token."""
    name = generate_one(model, vocab, temperature, max_length, prefix, device)
    if not name:
        return name, 0.0
    ids = [vocab.start_id] + [vocab.stoi[c] for c in name]
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    length = torch.tensor([len(ids)], dtype=torch.long, device=device)
    value = model.regress_value(inp, length).item()
    return name, value


def generate_many_with_value(
    model: CharRNN,
    vocab: Vocab,
    cfg: Config,
    num: int,
    temperature: float,
    training_names: Set[str] | None = None,
    only_novel: bool = True,
    min_length: int = 2,
    prefix: str = "",
    device: str = "cpu",
    max_attempts_factor: int = 40,
) -> List[Tuple[str, float]]:
    """Return ``num`` (name, predicted_value) pairs, de-duplicated and (optionally) novel."""
    training_names = training_names or set()
    results: List[Tuple[str, float]] = []
    seen: Set[str] = set()
    attempts = 0
    cap = num * max_attempts_factor

    while len(results) < num and attempts < cap:
        attempts += 1
        name, value = generate_one_with_value(model, vocab, temperature, cfg.max_length, prefix, device)
        if len(name) < min_length or name in seen:
            continue
        if only_novel and name in training_names:
            continue
        seen.add(name)
        results.append((name, value))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate names + a predicted numeric attribute from a dual-output checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint trained with src.train_dual.")
    parser.add_argument("--num", type=int, default=20, help="How many names to generate.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Creativity knob; higher = weirder. Defaults to the trained config.")
    parser.add_argument("--prefix", default="", help="Force names to start with this string.")
    parser.add_argument("--allow-existing", action="store_true",
                        help="Allow names that already appear in the training data.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility.")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, vocab, cfg, training_names = load_checkpoint(args.checkpoint)
    if not cfg.dual_output:
        parser.error(f"{args.checkpoint} has no value head — train it with `python -m src.train_dual` first.")
    temperature = args.temperature if args.temperature is not None else cfg.temperature

    results = generate_many_with_value(
        model, vocab, cfg,
        num=args.num,
        temperature=temperature,
        training_names=training_names,
        only_novel=not args.allow_existing,
        prefix=args.prefix,
    )

    print(f"\n=== {len(results)} names @ temperature {temperature} ===")
    for name, value in results:
        print(f"  {name:30s} value={value:.3f}")


if __name__ == "__main__":
    main()
