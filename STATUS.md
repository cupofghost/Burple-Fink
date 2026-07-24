# STATUS

Last consolidation: (never)

## Active work
| Date | Area / files | Task & state (≤3 lines) | Signature |
|------|--------------|-------------------------|-----------|
| 2026-07-24 | data/ | ✅ WS-1 batch 1: Added 3 datasets (racehorses 355, spacecraft 270, paint_colors 391). Smoke-tested; registered. | Signed: Claude Code \| Haiku 4.5 \| low |
| 2026-07-24 | data/ | ✅ WS-1 batch 2: Added 3 datasets (motorcycles 359, craft_beers 398, aircraft 435). All ≥300 entries. Smoke-tested; registered. | Signed: Claude Code \| Haiku 4.5 \| low |
| 2026-07-24 | web/, src/serve.py, src/export_web.py, data/ | ⏳ WS-1+5: engine selector -> dropdown (lots of engines coming); adding 5 datasets (team_names, band_names, food_brands, firearm_names, dog_breeds); retraining base + all fine-tunes; re-exporting web/burple-fink.html. | Signed: Claude Code \| Sonnet 5 \| high |

## Shared-file touches
(list file + what changed + signature)

## Known issues
- `data/paint_colors.txt` lines ~275-391 are corrupted/gibberish (e.g. "Bemidgard", "Bemidillion") — looks like leftover char-RNN sample output pasted in instead of real color names, not real paint colors. Not touched (outside this task's scope). 2026-07-24. Signed: Claude Code | Sonnet 5 | high

## Archive
(completed entries moved here during consolidation — one line each)
