# STATUS

Last consolidation: (never)

## Active work
| Date | Area / files | Task & state (≤3 lines) | Signature |
|------|--------------|-------------------------|-----------|
| 2026-07-24 | WS-1 dataset: `data/world_cities.txt` | Added 671 real world city/capital names, smoke-tested (train+sample), registered in README/HANDOFF. Done, PR open. | Signed: Claude Code \| Sonnet 5 \| medium |
| 2026-07-24 | WS-4 dual-output: `src/train_dual.py`, `data/car_manufacturers_founding_year.tsv` | Added optional value-regression head (additive Config fields + CharRNN.value_head), new dual-output training script, seed dataset (66 mfrs + founding year), `sample.py` auto-prints values. Checkpoint format unchanged; 4 new tests pass, full suite (18) green. Done, PR open. | Signed: Claude Code \| Sonnet 5 \| medium |

## Shared-file touches
- `src/config.py`: added 4 new fields (`dual_output`, `value_mean`, `value_std`,
  `value_label`), all with backward-compatible defaults. `from_dict` already ignored
  unknown keys, so old checkpoints/configs are unaffected. — Signed: Claude Code | Sonnet 5 | medium
- `src/model.py`: added optional `CharRNN.value_head` (None unless `cfg.dual_output`),
  a new `encode()` method, and `predict_value()`. `forward(x, hidden)` signature/behavior
  unchanged. — Signed: Claude Code | Sonnet 5 | medium
- `src/data.py`: added one new function `load_name_value_pairs()`; nothing existing
  changed. — Signed: Claude Code | Sonnet 5 | medium
- `src/sample.py`: `generate_one`/`generate_many` gained an opt-in `return_value` param
  (default False = old behavior/return type exactly); `main()` prints the value for
  dual-output checkpoints only. — Signed: Claude Code | Sonnet 5 | medium

## Known issues
(one line each: what, where, date noticed, signature)

## Archive
(completed entries moved here during consolidation — one line each)
