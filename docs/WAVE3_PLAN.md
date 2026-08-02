# Wave 3 — build-out sprint (2026-08-01)

Wave 2 made the existing engine honest (held-out validation, decoding sweeps, CI).
Wave 3 acts on what wave 2 *measured*, and widens the platform.

The single loudest finding from wave 2 is in `STATUS.md`:

> A 135-name dataset cannot support a 2-layer, 256-wide LSTM. This is an argument for
> more data, not for a smaller epoch default.

So wave 3 is **data-first**, with four parallel data lanes, plus five engineering lanes
that were unblocked by wave 2's instrumentation.

## Lane map (strict file ownership — no two lanes share a file)

| Lane | Owns | Goal |
|---|---|---|
| **WS-9 · architecture** | `src/arch/` (new), `src/model.py`, `tests/test_arch.py` | GRU + transformer decoder behind `--arch`; measure all three on the same split |
| **WS-10 · training regimen** | `src/train.py`, `src/pretrain.py`, `src/finetune.py`, `tests/test_training_quality.py` | Fix the RNG-init bug; add weight decay / label smoothing / LR warmup; size-aware epoch defaults |
| **WS-11 · evaluation** | `src/evaluate.py`, `tests/test_evaluate.py` | Use the `val_names` WS-6 now writes; markdown `--report`; benchmark every dataset |
| **WS-12 · product** | `web/`, `src/export_web.py`, `src/serve.py`, `tests/test_web.py` | Multi-model gallery, decoding controls in the UI, shareable output |
| **WS-13 · infra** | `scripts/`, `.github/`, `tests/test_repo_hygiene.py` | Fix the red hygiene job; dataset validator; widen CI |
| **WS-14/15/16 · new data** | new files in `data/` only | ~18 new datasets, ≥350 names each, disjoint domains per lane |
| **WS-17 · data growth** | the four thin existing `data/` files | Grow `car_manufacturers`, `car_models`, `motorcycle_brands`, `spacecraft` past 350 |

`src/config.py` is **not** owned by any lane. Every field wave 3 needs was pre-declared
there on 2026-08-01, before the lanes started — the same trick wave 2 used to avoid the
collision that caused the 2026-07-24 consolidation. Lanes read those fields; nobody edits
the file.

`README.md`, `STATUS.md` and `HANDOFF.md` are likewise unowned: the orchestrating session
merges them once at the end, so eighteen lanes don't fight over one catalog table.

## Rules for every lane

1. Stay inside your owned files. If you truly need something outside them, say so in your
   report instead of editing it.
2. Do not run `git` at all. The orchestrating session commits and pushes.
3. Test only what you changed (`AGENTS.md` §4). Use a unique checkpoint name so parallel
   lanes don't overwrite each other under `checkpoints/`.
4. Report measurements, not adjectives. "Transformer beat the LSTM" is worthless without
   the two numbers and the dataset they came from. A negative result is a real result —
   wave 2's top-k finding was one, and it saved the next agent from a dead end.
5. Sign your work: `Signed: <program> | <model> | <effort>`.
