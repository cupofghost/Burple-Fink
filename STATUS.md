# STATUS

Last consolidation: 2026-07-24 — Signed: Claude Code | Sonnet 5 | high

## Active work
| Date | Area / files | Task & state (≤3 lines) | Signature |
|------|--------------|-------------------------|-----------|
| 2026-07-24 | data/ | ✅ WS-1 batch 1: Added 3 datasets (racehorses 355, spacecraft 270, paint_colors 391). Smoke-tested; registered. | Signed: Claude Code \| Haiku 4.5 \| low |
| 2026-07-24 | data/ | ✅ WS-1 batch 2: Added 3 datasets (motorcycles 359, craft_beers 398, aircraft 435). All ≥300 entries. Smoke-tested; registered. | Signed: Claude Code \| Haiku 4.5 \| low |
| 2026-07-29 | docs/, src/config.py | ✅ Wave-2 upgrade plan + workspace prep: three non-overlapping workstreams (WS-6/7/8) specced in `docs/UPGRADE_PLAN.md` + `docs/upgrade/AGENT-{A,B,C}.md`; `src/config.py` fields pre-declared so no two agents share a file. No behavior change. | Signed: Claude Code \| Opus 5 \| high |
| 2026-07-29 | WS-6 · `src/train.py`, `pretrain.py`, `finetune.py`, `data.py`, `tests/test_training_quality.py` | ✅ Held-out split + per-epoch val loss + early stopping + best-epoch weight restore + `plateau`/`cosine` LR schedules, all opt-in via `--val-fraction`/`--patience`/`--lr-schedule`. **Measured (300 epochs, 15% holdout, current defaults):** `car_manufacturers` (159) val bottoms ~epoch 12–19 then gets **130% worse** by 300 (best 2.98 → 6.87; train 0.72, gap 6.15 nats). `aircraft` (435) bottoms ~epoch 26, **+30%** worse by 300 (0.78 → 1.02; gap 0.64). So ~95% of the 300-epoch budget is actively harmful on both. 34 new tests + 18 existing, 52/52 green. | Signed: Claude Code \| Opus 5 \| high |
| 2026-07-29 | WS-7 · `src/sample.py`, `evaluate.py` | ✅ Done: `top_k`/`top_p`/`repetition_penalty`/`min_length` on `generate_one`/`generate_many` (keyword-only, off by default, exact-reproduction proof in `tests/test_sampling.py`); `evaluate.py` gained near-duplicate rate (edit-dist ≤1/≤2), honest held-out NLL (degrades gracefully, no `val_names` yet), and `--sweep`/`--compare`. Measured on 2 checkpoints (159 car brands vs. 8,631 English words): **plain temperature @ 1.1–1.3 beat every top-k/nucleus setting tried on both** — truncation pushed sampling *toward* memorized names (novelty 38%→32%, near-dup 72%→80% at top_k=10 on the small checkpoint) rather than away from junk, since junk wasn't the failure mode at this model scale. `repetition_penalty` still helps independently (targets char-repeats, which top-k/nucleus don't touch). 16 new tests + existing 18 all green (34/34). Did not touch `src/train.py`/`config.py`/`data.py` or wait on WS-6's `val_names`. Brief: `docs/upgrade/AGENT-B.md`. | Signed: Claude Code \| Sonnet 5 \| high |
| 2026-07-29 | WS-8 · `.github/`, `scripts/`, `src/serve.py`, `web/` | ✅ Agent C (`claude/ws8-ci-and-hygiene-w4f3tb`): added `.github/workflows/ci.yml` (torch-free `hygiene` job + `test` job with full suite + CLI smoke train/sample, ~6s locally); `scripts/check_repo.py` (stdlib-only registry-drift/weights/secrets checks) + `tests/test_repo_hygiene.py` (15 tests, no torch); `src/serve.py` gained `/api/health` + real JSON error responses (400/404/500) shown inline in the UI. 33/33 repo tests green. Did not wire decoding knobs (WS-7 not on `main` yet) or touch `export_web.py`/`burple-fink.html`. | Signed: Claude Code \| Sonnet 5 \| medium |

## Shared-file touches
- `src/model.py`, `src/config.py`, `src/data.py`, `src/sample.py`: now carry the WS-4 dual-output implementation chosen during consolidation (originally from `claude/scope-vs-please-yrlsll`) — `value_head`/`encode()`/`predict_value()` on `CharRNN`, `dual_output`/`value_mean`/`value_std`/`value_label` on `Config`, `load_name_value_pairs()` on data, opt-in `return_value` on `sample.generate_one`/`generate_many`. All additive/backward-compatible; `forward(x, hidden)` unchanged. — Claude Code | Sonnet 5 | high

- `src/config.py`: wave-2 prep pre-declared the fields WS-6 and WS-7 will consume
  (`val_fraction`/`early_stop_patience`/`lr_schedule`/`lr_factor`/`lr_min`,
  `top_k`/`top_p`/`repetition_penalty`/`min_length`). All additive with defaults that
  reproduce current behavior, verified via `Config.to_dict`/`from_dict` round-trip incl. an
  old-checkpoint dict. Done deliberately so the three wave-2 agents never edit this shared
  file concurrently — the collision that caused the 2026-07-24 consolidation. — Claude Code | Opus 5 | high

## Known issues
- **`fit()` seeds the RNG *after* `train()` has already built the model**, so initial weights
  are not reproducible: three identical `--val-fraction 0.15` runs on `car_manufacturers`
  put the best epoch at 12, 16 and 19 (best val 2.92–2.98). Training itself is seeded, only
  the initialization isn't. Found while measuring WS-6; NOT fixed, because seeding it would
  change the default training trajectory for every existing command — that is a deliberate
  decision the owner should make, not a side effect of adding a validation split. The fix is
  one line in `train()`/`pretrain()`. Flagged 2026-07-29 — Claude Code | Opus 5 | high
- **The right answer for the small datasets is more data, not fewer epochs.** WS-6 can now
  see that `car_manufacturers` (135 training names) reaches its best held-out loss around
  epoch 12–19, which is *before* the model has learned to spell — early-stopped samples are
  noticeably rougher than 300-epoch ones (`Si wte`, `Towc`). Both readings are honest: the
  300-epoch model generalizes far worse, and the val-optimal model isn't pretty either. A
  135-name dataset cannot support a 2-layer 256-wide LSTM. This is an argument for WS-1
  (more datasets, ≥300 names) rather than for a smaller epoch default.
  Flagged 2026-07-29 — Claude Code | Opus 5 | high
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
