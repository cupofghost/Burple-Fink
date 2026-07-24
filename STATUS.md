# STATUS

Last consolidation: (never)

## Active work
| Date | Area / files | Task & state (≤3 lines) | Signature |
|------|--------------|-------------------------|-----------|
| 2026-07-24 | `data/tech_startups.txt` (WS-1 new dataset) | Added 400 real tech company/startup names per HANDOFF §5 (registered in README + HANDOFF §4). Smoke-tested: `train --epochs 40` + `sample` both run clean and produce on-style invented names. Done. | Signed: Claude Code \| Sonnet 5 \| medium |
| 2026-07-24 | WS-4 dual-output (`src/model.py` +new `src/dual_data.py`/`src/train_dual.py`/`src/sample_dual.py`, `data/dual/`) | Added the name+attribute regression head (docs/PLAN.md §11.6): `DualCharRNN(CharRNN)`, superset checkpoint format, combined char+attribute loss. Seed dataset `data/dual/paint_colors.tsv` (141 CSS/X11 colors + computed luminance). Smoke-tested (train 200 epochs + sample_dual), full `unittest discover` still green (14/14). Registered in README/HANDOFF/PLAN. Done. | Signed: Claude Code \| Sonnet 5 \| medium |
| 2026-07-24 | `data/city_names.txt` (WS-1 new dataset) | Added 1,323 real world city names per HANDOFF §5 (registered in README + HANDOFF §4). Smoke-tested: `train --epochs 40` + `sample` both run clean and produce on-style city-like names. Done. | Signed: Claude Code \| Haiku 4.5 \| medium |

## Shared-file touches
- `README.md`, `HANDOFF.md` — added one catalog/registry row each for `tech_startups.txt` (no other edits). Signed: Claude Code | Sonnet 5 | medium
- `src/model.py` — added `DualCharRNN(CharRNN)`, purely additive (new class, new method); `CharRNN.forward` untouched. `config.py` and `data.py` were NOT touched for WS-4 (see docs/PLAN.md §11.6 for why). Signed: Claude Code | Sonnet 5 | medium

## Known issues
(one line each: what, where, date noticed, signature)

## Archive
(completed entries moved here during consolidation — one line each)
