"""Generate new names from a trained checkpoint.

Generation is the mirror image of training: start from the START token and, one
character at a time, ask the model for the next character, sample from its output
(scaled by ``temperature`` -- the creativity knob), feed that character back in,
and stop when the model emits the END token.

Temperature alone is a blunt instrument: turn it up for variety and every
implausible character stays reachable, so variety comes with junk. `top_k` /
`top_p` (nucleus) truncate the tail so temperature can be pushed up for variety
without also sampling from the garbage end of the distribution; `repetition_penalty`
separately targets the "Bylfgoammm" character-repeat failure. All three default
to off (0 / 1.0 / 1.0), reproducing the plain-temperature path exactly.

Usage:
    python -m src.sample --checkpoint checkpoints/manufacturers.pt --num 20 --temperature 0.9
    python -m src.sample --checkpoint checkpoints/manufacturers.pt --num 20 \\
        --temperature 1.2 --top-k 8 --repetition-penalty 1.3
"""

from __future__ import annotations

import argparse
from typing import List, Set

import torch
import torch.nn.functional as F

from .config import Config
from .data import Vocab
from .model import CharRNN


def _apply_repetition_penalty(logits: torch.Tensor, generated_ids: List[int],
                               penalty: float) -> torch.Tensor:
    """Discourage characters already emitted in the current name.

    CTRL-style: divide a positive logit by ``penalty`` but *multiply* a negative
    one, so a repeat is always pushed down regardless of the raw logit's sign
    (dividing a negative logit would make it larger, rewarding the repeat --
    the opposite of what a "penalty" should do). ``penalty == 1.0`` is a no-op
    and returns ``logits`` unchanged (not just numerically equivalent) so the
    plain-temperature path stays exactly what it was before this option existed.
    """
    if penalty == 1.0 or not generated_ids:
        return logits
    logits = logits.clone()
    for tid in set(generated_ids):
        val = logits[..., tid]
        logits[..., tid] = torch.where(val > 0, val / penalty, val * penalty)
    return logits


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out (via -inf) every logit outside the k likeliest. ``k <= 0`` = off."""
    if k <= 0 or k >= logits.size(-1):
        return logits
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filter: keep the smallest set of characters whose cumulative
    probability is >= p, always keeping at least the single likeliest
    character. ``p >= 1.0`` = off."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Drop a candidate once the running total *before* it already reached p,
    # so the kept set's cumulative probability is >= p rather than just under it.
    remove_sorted = (cumulative - sorted_probs) > p
    remove_sorted[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove_sorted, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(-1, sorted_idx, sorted_logits)
    return out


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
    *,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    min_length: int = 0,
):
    """Sample a single name, optionally forced to start with ``prefix``.

    When ``return_value`` is True, returns ``(name, value)`` instead of just
    ``name``: ``value`` is the dual-output model's (WS-4) regressed scalar
    attribute, denormalized via the checkpoint's config, or ``None`` for an
    ordinary (non-dual) model. Default behavior/return type is unchanged.

    Decoding controls (WS-7), keyword-only, each off by default so calling
    this with none of them reproduces the old plain-temperature path exactly:
      - ``top_k`` > 0 keeps only the k likeliest next characters.
      - ``top_p`` < 1.0 keeps the smallest set of characters whose cumulative
        probability is >= p (nucleus sampling).
      - ``repetition_penalty`` > 1.0 discourages characters already emitted
        *in this name* (see ``_apply_repetition_penalty`` for why positive and
        negative logits are handled differently).
      - ``min_length`` > 0 masks END (and, since either also ends the name,
        PAD/START) out of the distribution until the name has reached this
        many characters -- enforced during generation, not just discarded
        afterward by the caller.

    Applied in this order every step: repetition penalty -> temperature ->
    top-k -> top-p -> min-length masking -> softmax -> sample. Order matters:
    each filter narrows the distribution the next one reads, and min-length
    masking is applied last (right before softmax) so it always wins even if
    an earlier filter had already suppressed everything else.
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
        scaled = _apply_repetition_penalty(next_logits, out_ids, repetition_penalty)
        scaled = scaled / max(temperature, 1e-6)
        scaled = _top_k_filter(scaled, top_k)
        scaled = _top_p_filter(scaled, top_p)
        if min_length > 0 and len(out_ids) < min_length:
            scaled = scaled.clone()
            # Any of these three ends the name early (see the checks below),
            # so all three must stay unreachable, not just END.
            scaled[..., vocab.end_id] = float("-inf")
            scaled[..., vocab.pad_id] = float("-inf")
            scaled[..., vocab.start_id] = float("-inf")
        probs = F.softmax(scaled, dim=-1)
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
    *,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> List:
    """Return ``num`` names, de-duplicated and (optionally) novel vs the training set.

    With ``return_value=True`` each result is a ``(name, value)`` tuple (WS-4
    dual-output); default is unchanged, a plain list of name strings.

    ``top_k``/``top_p``/``repetition_penalty`` (WS-7) are keyword-only and
    passed straight through to ``generate_one`` on every attempt; their
    defaults (0 / 1.0 / 1.0) are all "off", so callers that don't pass them
    (``src/train.py``'s live previews, ``src/serve.py``'s per-request calls)
    are unaffected. ``min_length`` is now also enforced *during* generation
    (via ``generate_one``'s END-masking) rather than only checked here after
    the fact -- the check below stays as a cheap safety net, but it should
    rarely trigger anymore.
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
            top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
            min_length=min_length,
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
    parser.add_argument("--top-k", type=int, default=None,
                        help="Keep only the k likeliest next characters (0 = off). "
                             "Defaults to the checkpoint's config.")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Nucleus sampling: smallest set with cumulative prob >= p "
                             "(1.0 = off). Defaults to the checkpoint's config.")
    parser.add_argument("--repetition-penalty", type=float, default=None,
                        help="Penalize characters already emitted in the name (1.0 = off). "
                             "Defaults to the checkpoint's config.")
    parser.add_argument("--min-length", type=int, default=None,
                        help="Discard/never emit names shorter than this. "
                             "Defaults to the checkpoint's config.")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, vocab, cfg, training_names = load_checkpoint(args.checkpoint)
    temperature = args.temperature if args.temperature is not None else cfg.temperature
    top_k = args.top_k if args.top_k is not None else cfg.top_k
    top_p = args.top_p if args.top_p is not None else cfg.top_p
    repetition_penalty = (args.repetition_penalty if args.repetition_penalty is not None
                           else cfg.repetition_penalty)
    min_length = args.min_length if args.min_length is not None else cfg.min_length

    names = generate_many(
        model, vocab, cfg,
        num=args.num,
        temperature=temperature,
        training_names=training_names,
        only_novel=not args.allow_existing,
        min_length=min_length,
        prefix=args.prefix,
        return_value=cfg.dual_output,
        top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
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
