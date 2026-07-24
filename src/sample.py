"""Generate new names from a trained checkpoint.

Generation is the mirror image of training: start from the START token and, one
character at a time, ask the model for the next character, sample from its output
(scaled by ``temperature`` -- the creativity knob), feed that character back in,
and stop when the model emits the END token.

Usage:
    python -m src.sample --checkpoint checkpoints/manufacturers.pt --num 20 --temperature 0.9
"""

from __future__ import annotations

import argparse
from typing import List, Set

import torch
import torch.nn.functional as F

from .config import Config
from .data import Vocab
from .model import CharRNN


def load_checkpoint(path: str, device: str = "cpu"):
    """Rebuild the vocab, config and model exactly as they were at training time."""
    ckpt = torch.load(path, map_location=device)
    cfg = Config.from_dict(ckpt["config"])
    vocab = Vocab.from_dict(ckpt["vocab"])
    model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    training_names: Set[str] = set(ckpt.get("training_names", []))
    return model, vocab, cfg, training_names


@torch.no_grad()
def generate_one(
    model: CharRNN,
    vocab: Vocab,
    temperature: float,
    max_length: int,
    prefix: str = "",
    device: str = "cpu",
    return_value: bool = False,
):
    """Sample a single name, optionally forced to start with ``prefix``.

    When ``return_value`` is True, returns ``(name, value)`` instead of just
    ``name``: ``value`` is the dual-output model's (WS-4) regressed scalar
    attribute, denormalized via the checkpoint's config, or ``None`` for an
    ordinary (non-dual) model. Default behavior/return type is unchanged.
    """
    model.eval()

    # Prime the LSTM with START (+ any requested prefix characters).
    tokens: List[int] = [vocab.start_id] + [vocab.stoi[c] for c in prefix]
    inp = torch.tensor([tokens], dtype=torch.long, device=device)
    logits, hidden = model(inp)

    out_ids: List[int] = [vocab.stoi[c] for c in prefix]
    # Use the last position's logits as the distribution for the next character.
    next_logits = logits[:, -1, :]

    for _ in range(max_length):
        # Temperature scaling: <1 sharpens (safe), >1 flattens (weird).
        probs = F.softmax(next_logits / max(temperature, 1e-6), dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)  # (1, 1)
        nxt_id = nxt.item()
        if nxt_id == vocab.end_id:
            break
        # Never emit PAD/START mid-name; resample would be overkill, so just stop.
        if nxt_id in (vocab.pad_id, vocab.start_id):
            break
        out_ids.append(nxt_id)

        logits, hidden = model(nxt, hidden)
        next_logits = logits[:, -1, :]

    name = vocab.decode(out_ids)
    if not return_value:
        return name
    if not model.cfg.dual_output:
        return name, None
    value = model.predict_value(hidden[0][-1]).item()
    value = value * model.cfg.value_std + model.cfg.value_mean
    return name, value


def generate_many(
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
    return_value: bool = False,
) -> List:
    """Return ``num`` names, de-duplicated and (optionally) novel vs the training set.

    With ``return_value=True`` each result is a ``(name, value)`` tuple (WS-4
    dual-output); default is unchanged, a plain list of name strings.
    """
    training_names = training_names or set()
    results: List = []
    seen: Set[str] = set()
    attempts = 0
    cap = num * max_attempts_factor

    while len(results) < num and attempts < cap:
        attempts += 1
        sample = generate_one(
            model, vocab, temperature, cfg.max_length, prefix, device,
            return_value=return_value,
        )
        name = sample[0] if return_value else sample
        if len(name) < min_length:
            continue
        if name in seen:
            continue
        if only_novel and name in training_names:
            continue
        seen.add(name)
        results.append(sample)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate names from a trained char-RNN.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
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
    temperature = args.temperature if args.temperature is not None else cfg.temperature

    names = generate_many(
        model, vocab, cfg,
        num=args.num,
        temperature=temperature,
        training_names=training_names,
        only_novel=not args.allow_existing,
        prefix=args.prefix,
        return_value=cfg.dual_output,
    )

    print(f"\n=== {len(names)} names @ temperature {temperature} ===")
    if cfg.dual_output:
        label = cfg.value_label or "value"
        for name, value in names:
            print(f"  {name}  ({label}: {value:.1f})")
    else:
        for name in names:
            print(f"  {name}")


if __name__ == "__main__":
    main()
