#!/usr/bin/env python3
"""Convenience wrapper: train (if needed) and generate in one command.

This is a thin friendly front-end over ``src.train`` and ``src.sample`` for people
who just want names without learning the module flags.

Examples:
    # Train on the bundled manufacturers, then print 20 novel names:
    python generate.py --data data/car_manufacturers.txt --train --num 20

    # Just generate from an existing checkpoint at high creativity:
    python generate.py --checkpoint checkpoints/manufacturers.pt --num 15 --temperature 1.2
"""

from __future__ import annotations

import argparse
import os

import torch

from src.config import Config
from src import train as trainer
from src import sample as sampling


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and/or generate car names.")
    parser.add_argument("--data", help="Training-name list (required with --train).")
    parser.add_argument("--checkpoint", help="Existing checkpoint to sample from.")
    parser.add_argument("--train", action="store_true", help="Train before generating.")
    parser.add_argument("--name", default="model", help="Checkpoint name when training.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num", type=int, default=20, help="How many names to generate.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--prefix", default="", help="Force a starting string.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs

    checkpoint = args.checkpoint
    if args.train:
        if not args.data:
            parser.error("--train requires --data")
        checkpoint = trainer.train(args.data, args.name, cfg, device=device)
    elif not checkpoint:
        parser.error("Provide --checkpoint, or use --train with --data.")

    if not os.path.exists(checkpoint):
        parser.error(f"Checkpoint not found: {checkpoint}")

    model, vocab, loaded_cfg, training_names = sampling.load_checkpoint(checkpoint, device)
    temperature = args.temperature if args.temperature is not None else loaded_cfg.temperature

    names = sampling.generate_many(
        model, vocab, loaded_cfg,
        num=args.num,
        temperature=temperature,
        training_names=training_names,
        only_novel=True,
        prefix=args.prefix,
        device=device,
    )

    print(f"\n=== {len(names)} fresh names @ temperature {temperature} ===")
    for name in names:
        print(f"  {name}")


if __name__ == "__main__":
    main()
