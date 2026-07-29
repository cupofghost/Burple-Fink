# STATUS

Last consolidation: 2026-07-24 — Signed: Claude Code | Sonnet 5 | high

## Active work
| Date | Area / files | Task & state (≤3 lines) | Signature |
|------|--------------|-------------------------|-----------|
| 2026-07-24 | data/ | ✅ WS-1 batch 1: Added 3 datasets (racehorses 355, spacecraft 270, paint_colors 391). Smoke-tested; registered. | Signed: Claude Code \| Haiku 4.5 \| low |
| 2026-07-24 | data/ | ✅ WS-1 batch 2: Added 3 datasets (motorcycles 359, craft_beers 398, aircraft 435). All ≥300 entries. Smoke-tested; registered. | Signed: Claude Code \| Haiku 4.5 \| low |
| 2026-07-29 | docs/, src/config.py | ✅ Wave-2 upgrade plan + workspace prep: three non-overlapping workstreams (WS-6/7/8) specced in `docs/UPGRADE_PLAN.md` + `docs/upgrade/AGENT-{A,B,C}.md`; `src/config.py` fields pre-declared so no two agents share a file. No behavior change. | Signed: Claude Code \| Opus 5 \| high |
| 2026-07-29 | WS-6 · `src/train.py`, `pretrain.py`, `finetune.py`, `data.py` | 🔒 Reserved for Agent A (`claude/ws6-training-quality`): held-out validation split, early stopping, best-epoch weights, LR schedule. Brief: `docs/upgrade/AGENT-A.md`. | Reserved by: Claude Code \| Opus 5 \| high — Agent A re-signs on claim |
| 2026-07-29 | WS-7 · `src/sample.py`, `evaluate.py` | 🔒 Reserved for Agent B (`claude/ws7-decoding-quality`): top-k / nucleus / repetition penalty, near-duplicate metric, decoding sweep. Brief: `docs/upgrade/AGENT-B.md`. | Reserved by: Claude Code \| Opus 5 \| high — Agent B re-signs on claim |
| 2026-07-29 | WS-8 · `.github/`, `scripts/`, `src/serve.py`, `web/` | 🔒 Reserved for Agent C (`claude/ws8-ci-and-hygiene`): CI on every PR, registry/secret drift checks, phone-UI error handling + knobs. Brief: `docs/upgrade/AGENT-C.md`. | Reserved by: Claude Code \| Opus 5 \| high — Agent C re-signs on claim |

## Shared-file touches
- `src/model.py`, `src/config.py`, `src/data.py`, `src/sample.py`: now carry the WS-4 dual-output implementation chosen during consolidation (originally from `claude/scope-vs-please-yrlsll`) — `value_head`/`encode()`/`predict_value()` on `CharRNN`, `dual_output`/`value_mean`/`value_std`/`value_label` on `Config`, `load_name_value_pairs()` on data, opt-in `return_value` on `sample.generate_one`/`generate_many`. All additive/backward-compatible; `forward(x, hidden)` unchanged. — Claude Code | Sonnet 5 | high

- `src/config.py`: wave-2 prep pre-declared the fields WS-6 and WS-7 will consume
  (`val_fraction`/`early_stop_patience`/`lr_schedule`/`lr_factor`/`lr_min`,
  `top_k`/`top_p`/`repetition_penalty`/`min_length`). All additive with defaults that
  reproduce current behavior, verified via `Config.to_dict`/`from_dict` round-trip incl. an
  old-checkpoint dict. Done deliberately so the three wave-2 agents never edit this shared
  file concurrently — the collision that caused the 2026-07-24 consolidation. — Claude Code | Opus 5 | high

## Known issues
- `data/car_manufacturers_founding_year.tsv` founding years are sourced from each brand's commonly-cited history, not cross-checked against a primary source — treat as approximate, not authoritative. Flagged 2026-07-24 — Claude Code | Sonnet 5 | high
- `data/world_cities.txt` was merged (2026-07-24) from two independently-built lists via case-insensitive dedupe + alphabetical sort; the original curation order/rationale of both source files was not preserved, only the names. — Claude Code | Sonnet 5 | high

## Archive
- 2026-07-24 **Consolidation**: three sessions (`claude/next-item-v4te8p`, `claude/scope-vs-please-yrlsll`, `claude/next-task-tnbsmq`) independently built WS-4 dual-output in parallel without visibility into each other's branches (a git-branch-per-agent blind spot — STATUS.md claims don't cross branches). All three worked and were verified (18/18 tests on two; the third was hand-smoke-tested, had no automated tests). Owner chose `scope-vs-please-yrlsll`'s design (value normalization + `sample.py` integration, already had an open PR). Consolidated onto `claude/next-item-v4te8p`: adopted the chosen WS-4 code, discarded the other two implementations, kept and re-pointed all three branches' demo datasets at the winning API (`car_manufacturers_founding_year.tsv`, `paint_colors.tsv`, `periodic_elements.tsv`), and merged the two independently-built city datasets (671 + 1,323 names, ~45% overlap) into one deduped `world_cities.txt` (1,691 names) rather than dropping either. Also folded in `tech_startups.txt` (400) and `motorcycle_brands.txt` (63), both uncontested WS-1 additions. No secrets/PII found in any branch. Full suite green (18/18) + smoke-trained every dataset post-merge. — Signed: Claude Code | Sonnet 5 | high
- 2026-07-24 WS-4 dual-output (`next-item-v4te8p` design: `predict_value` flag + `regress_value()`, separate `train_dual.py`/`sample_dual.py`) — superseded by consolidation above, code discarded, `paint_colors.tsv` kept. Signed: Claude Code | Sonnet 5 | high
- 2026-07-24 WS-4 dual-output (`scope-vs-please-yrlsll` design: `value_head`/`encode()`/`predict_value()`, value normalization) — chosen as canonical by the owner during consolidation. Signed: Claude Code | Sonnet 5 | medium
- 2026-07-24 WS-1 `world_cities.txt` (671 names, `scope-vs-please-yrlsll`) — superseded by the merged file above, content preserved. Signed: Claude Code | Sonnet 5 | medium
- 2026-07-24 WS-4 dual-output (`next-task-tnbsmq` design: `DualCharRNN` subclass, no tests) — superseded by consolidation, code discarded, `periodic_elements.tsv` + its `paint_colors.tsv` kept (the latter deduped against the near-identical `next-item-v4te8p` copy; one kept). Signed: Claude Code | Sonnet 5 | medium
- 2026-07-24 WS-1 `tech_startups.txt` (400 names, `next-task-tnbsmq`) — Signed: Claude Code | Sonnet 5 | medium
- 2026-07-24 WS-1 `motorcycle_brands.txt` (63 names, `next-task-tnbsmq`) — Signed: Claude Code | Haiku 4.5 | medium
- 2026-07-24 WS-1 `city_names.txt` (1,323 names, `next-task-tnbsmq`; deduped post-hoc, 326 raw duplicate lines removed) — superseded by the merged `world_cities.txt` above, content preserved. Signed: Claude Code | Haiku 4.5 | medium / Claude Code | Sonnet 5 | low
