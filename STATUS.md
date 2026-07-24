# STATUS

Last consolidation: 2026-07-24 — Signed: Claude Code | Sonnet 5 | medium

## Active work
| Date | Area / files | Task & state (≤3 lines) | Signature |
|------|--------------|-------------------------|-----------|
| 2026-07-24 | `data/dual/periodic_elements.tsv` (WS-4 new dataset) | Added all 118 IUPAC elements + atomic number (docs/PLAN.md §11.6). Smoke-tested: `train_dual --epochs 300` (char loss 3.7→0.55, attr MSE →0.001) + `sample_dual` produce plausible name+number pairs. Registered in README/HANDOFF/PLAN. Done. | Signed: Claude Code \| Sonnet 5 \| medium |
| 2026-07-24 | `data/motorcycle_brands.txt` (WS-1 new dataset) | Added 63 real motorcycle manufacturer brands (Harley-Davidson through Zongshen, globally sourced). Smoke-tested: train 40 epochs (loss 3.95→1.04) + sample produce 10 plausible bike-style names. Registered in README/HANDOFF/branch-table. Done. | Signed: Claude Code \| Haiku 4.5 \| medium |

## Shared-file touches
(list file + what changed + signature)

## Known issues
(one line each: what, where, date noticed, signature)

## Archive
- 2026-07-24 `data/tech_startups.txt` — 400 real tech company/startup names (WS-1). Signed: Claude Code | Sonnet 5 | medium
- 2026-07-24 WS-4 dual-output — `DualCharRNN` regression head + `data/dual/paint_colors.tsv` seed dataset (docs/PLAN.md §11.6). Signed: Claude Code | Sonnet 5 | medium
- 2026-07-24 `data/city_names.txt` — 1,323 real world city names (WS-1); deduped post-hoc (326 raw duplicate lines removed). Signed: Claude Code | Haiku 4.5 | medium / Claude Code | Sonnet 5 | low
