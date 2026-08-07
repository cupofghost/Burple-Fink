# Wave 4 — act on what wave 3 measured (2026-08-05)

Wave 3 doubled the dataset library and, in doing so, disproved half of its own premise.
Wave 4 follows the evidence rather than the plan.

## What wave 3 actually established

Two lanes converged on the same conclusion from opposite directions:

- **`reports/BENCHMARK.md`** — a controlled within-dataset size ladder. Held-out loss improves
  monotonically with data in 5 of 5 domains, **but** the train/val gap at the best epoch does
  not shrink, and returns flatten (`car_models` gains 0.143 nats going 218→400 training names
  and only 0.049 going 700→1035).
- **`reports/ARCH.md`** — a three-way architecture comparison. The **GRU beats the LSTM on both
  small datasets with 25% fewer parameters** and loses on both large ones. The LSTM overfits
  more on 4 of 4.

The shared conclusion: the stock model — 2 layers, `hidden_dim=256`, ~839k parameters — is
**too large for half the library**. Fifteen of the thirty datasets have under 500 names. That
default is untested and inherited from the first commit.

So wave 4 is not "more data". It is capacity, and the two claims nobody has checked.

## Lane map (strict file ownership — no two lanes share a file)

| Lane | Owns | Goal |
|---|---|---|
| **WS-18 · capacity** | `reports/CAPACITY.md`, `reports/_capacity/`, `scripts/sweep_capacity.py` | Sweep `hidden_dim` × `num_layers` × dataset size. Find the right capacity per size — and test whether "use a GRU under 500 names" is really just "use a smaller model" |
| **WS-19 · budgeting** | `docs/AGENT_BUDGETING.md` | Research how the rolling usage window actually works, and turn it into operational rules for this repo's orchestration pattern |
| **WS-20 · transfer** | `reports/TRANSFER.md`, `reports/_transfer/`, `scripts/bench_transfer.py` | Measure the README's oldest unverified claim: does pretrain-then-finetune beat from-scratch? Watch for validation leakage through `load_all_names` |
| **WS-21 · dual-output** | new `data/*.tsv` + sidecars | Six new `name<TAB>value` datasets. The repo's most distinctive feature has 3 demo files against 30 plain ones |

`src/` is owned by **no** lane this wave — every lane measures the engine rather than changing
it. Findings that imply a code change get written down and applied at consolidation, so a
default never moves in the same commit as the evidence for moving it.

`README.md`, `STATUS.md`, `HANDOFF.md` and `AGENTS.md` remain orchestrator-only.

## Two claims wave 4 exists to test

1. **"Pretrain one base, then fine-tune per domain."** The README has recommended this since
   wave 1 and it has never been measured — not once in three waves. It is a plausible story
   about transfer learning, asserted as fact. WS-20 checks it, and checks first whether the
   base's pretraining corpus leaks the target's validation names (it plausibly does, via
   `load_all_names`, which would make any transfer "win" an artifact).
2. **"Use a GRU under 500 names."** `reports/ARCH.md` recommends this, but the GRU is also
   simply *smaller*. If a correctly-sized LSTM wins everywhere, the gating mechanism was a red
   herring and that recommendation needs revising. WS-18 is explicitly asked to try to
   overturn it.

Both are cases where the repo would rather be corrected than flattered.

## Process change: survive the usage window

Waves 2 and 3 lost real work to session limits. The pattern was consistent and avoidable:

- Two data lanes held ~2,000 curated names in context and were killed before writing anything.
  One had built 1,139 pharmaceutical names. All lost.
- The architecture lane finished all 17 of its training runs, then was killed before writing
  its table — and was recoverable **only** because checkpoints happened to persist on disk.
  The orchestrator rebuilt the table from them into `reports/ARCH.md`.
- Lanes that wrote each artifact as they finished it lost nothing.

Wave 4 therefore runs **four concurrent lanes, not nine** (nine parallel Opus agents exhausted
a window in about twenty minutes), and every brief carries the same two rules:

1. Write each finished artifact to disk *before* starting the next one.
2. Write a first complete draft of the deliverable early, then improve it in place.

WS-19 turns this into documented guidance; `docs/AGENT_BUDGETING.md` is the durable version.
