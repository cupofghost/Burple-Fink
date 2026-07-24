# STATUS

Last consolidation: (never)

## Active work
| Date | Area / files | Task & state (≤3 lines) | Signature |
|------|--------------|-------------------------|-----------|
| 2026-07-24 | WS-4 dual-output: `src/model.py`, `src/data.py`, `src/config.py`, new `src/train_dual.py`/`src/sample_dual.py`, `data/paint_colors.tsv` | Done: value-regression head + `fit_dual`/`sample_dual`, demoed on a real CSS-named-colors dataset (name→hex is real, value is luminance computed from it). Tests pass (18/18), smoke-trained + sampled cleanly. Docs updated. | Claude Code \| Sonnet 5 \| high |

## Shared-file touches
- `src/model.py`: added optional `predict_value` flag + `value_head` + `regress_value()`. `forward(x, hidden)` signature/behavior unchanged. — Claude Code | Sonnet 5 | high
- `src/config.py`: added `dual_output` (default False) and `value_loss_weight` fields. Backward-compatible via existing `from_dict` unknown-key handling. — Claude Code | Sonnet 5 | high
- `src/data.py`: added `load_name_value_pairs()`; no existing functions touched. — Claude Code | Sonnet 5 | high
- `src/sample.py`: one-line change in `load_checkpoint` to pass `predict_value=cfg.dual_output` to `CharRNN(...)`. — Claude Code | Sonnet 5 | high

## Known issues
(one line each: what, where, date noticed, signature)

## Archive
(completed entries moved here during consolidation — one line each)
