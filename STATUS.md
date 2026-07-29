# STATUS

Last consolidation: 2026-07-29 — Signed: Claude Code | Opus 5 | high

## Active work
| Date | Area / files | Task & state (≤3 lines) | Signature |
|------|--------------|-------------------------|-----------|
| — | — | _Nothing in flight._ Wave 2 (WS-6/7/8) is merged and consolidated. Next up is **WS-1: more and bigger datasets** — see the wave-2 outcome note in `HANDOFF.md §3` for why that outranks the deferred model work. | — |

## Shared-file touches
- `src/model.py`, `src/config.py`, `src/data.py`, `src/sample.py`: carry the WS-4 dual-output
  implementation chosen during the 2026-07-24 consolidation — `value_head`/`encode()`/
  `predict_value()` on `CharRNN`, `dual_output`/`value_mean`/`value_std`/`value_label` on
  `Config`, `load_name_value_pairs()` on data, opt-in `return_value` on `sample.generate_one`/
  `generate_many`. All additive; `forward(x, hidden)` unchanged. — Claude Code | Sonnet 5 | high
- `src/config.py`: wave-2 prep pre-declared every field WS-6 and WS-7 would consume
  (`val_fraction`/`early_stop_patience`/`lr_schedule`/`lr_factor`/`lr_min`, `top_k`/`top_p`/
  `repetition_penalty`/`min_length`) so three concurrent agents never edited it at once. It
  worked — wave 2 shipped with zero file collisions. Reuse the pattern for the next parallel
  wave. — Claude Code | Opus 5 | high
- `scripts/check_repo.py` + `tests/test_repo_hygiene.py` (2026-07-29 consolidation): the
  secret scanner now skips RFC 2606 documentation domains (`example.com`/`.org`/`.net`) and
  its own fixture file, which is why CI was red on main. Agent C's file, changed by the
  consolidation because it blocked everyone; C's 15 tests still pass, plus 2 pinning the
  fix. — Claude Code | Opus 5 | high

## Known issues
- **`fit()` seeds the RNG *after* `train()` has already built the model**, so initial weights
  are not reproducible: three identical `--val-fraction 0.15` runs on `car_manufacturers` put
  the best epoch at 12, 16 and 19 (best val 2.92–2.98). Training itself is seeded, only the
  initialization isn't. NOT fixed — the one-line fix changes the default training trajectory
  for every existing command, which is the owner's call, not a side effect of another task.
  **Owner decision pending.** Flagged 2026-07-29 — Claude Code | Opus 5 | high
- **The datasets are too small for the model reading them.** `car_manufacturers` (135 training
  names) reaches its best held-out loss around epoch 12–19 — before the model can spell — and
  72–80% of "novel" generated names are within one edit of a real training name. Both readings
  are honest and they point the same way: this is an argument for WS-1 (bigger datasets, ≥300
  and ideally ≥3,000 names), not for a smaller epoch default. See `docs/PLAN.md §12`.
  Flagged 2026-07-29 — Claude Code | Opus 5 | high
- `src/train_dual.py` (WS-4) has its own training loop and did **not** receive WS-6's
  validation split / early stopping — dual-output training is still measured on training data
  only. Deliberately out of scope for wave 2. Flagged 2026-07-29 — Claude Code | Opus 5 | high
- The phone UI does not expose WS-7's decoding knobs: WS-8 shipped before WS-7 landed on main.
  Given WS-7 measured that top-k/nucleus made output *worse* at this scale, wiring them is not
  obviously worth doing — owner's call. Flagged 2026-07-29 — Claude Code | Opus 5 | high
- `data/car_manufacturers_founding_year.tsv` founding years come from each brand's commonly
  cited history, not a primary source — approximate, not authoritative. Flagged 2026-07-24 —
  Claude Code | Sonnet 5 | high
- `data/world_cities.txt` was merged (2026-07-24) from two independently built lists via
  case-insensitive dedupe + alphabetical sort; only the names were preserved, not either
  file's original curation order. — Claude Code | Sonnet 5 | high

## Archive
- 2026-07-29 **Consolidation (wave 2)**: three agents (WS-6 training quality, WS-7 decoding
  quality, WS-8 CI & hygiene) ran concurrently off pre-partitioned briefs and merged with
  **zero file collisions and zero duplicated work** — the failure mode of the 2026-07-24
  consolidation did not recur. Consolidation fixed the one thing that was actually broken on
  main (the new CI hygiene job was red: `scripts/check_repo.py` flagged 6 secrets/PII, all of
  them its own planted test fixtures) and reconciled the docs the three lanes had each edited
  separately: HANDOFF §2 state + module table, §7 branch statuses, a wave-2 outcome note in
  §3, README stage table + wave-2 summary, and a new `docs/PLAN.md §12` recording what the
  wave measured. No code behavior changed beyond the scanner fix; no duplicate implementations
  to arbitrate. Secrets/PII scan clean (`check_repo`: all clear). 85/85 tests — 17 hygiene
  tests re-run here, the other 68 on the signed word of the lanes that wrote them (AGENTS.md
  §4). — Signed: Claude Code | Opus 5 | high
- 2026-07-29 WS-6 training quality (`claude/ws6-training-quality`, PR #11): held-out split,
  per-epoch val loss, early stopping, best-epoch restore, plateau/cosine LR schedules, additive
  `val_names` checkpoint key; 34 tests. Measured ~90–95% of the 300-epoch default to be
  actively harmful. Signed: Claude Code | Opus 5 | high
- 2026-07-29 WS-7 decoding quality (`claude/ws7-decoding-quality-fcmaoj`, PR #10): keyword-only
  `top_k`/`top_p`/`repetition_penalty`/`min_length`, near-duplicate memorization metric, honest
  held-out NLL, `--sweep`/`--compare`; 16 tests. Negative result: plain temperature beat every
  truncation setting tried. Signed: Claude Code | Sonnet 5 | high
- 2026-07-29 WS-8 CI & repo hygiene (`claude/ws8-ci-and-hygiene-w4f3tb`): first CI workflow
  (torch-free hygiene job + full suite + CLI smoke), `scripts/check_repo.py`, `/api/health` and
  real JSON errors in the UI; 15 tests. Signed: Claude Code | Sonnet 5 | medium
- 2026-07-29 Wave-2 plan + workspace prep (`claude/burple-fink-upgrade-plan-m7ndof`, PR #8):
  three-lane partition with disjoint file ownership, per-agent briefs, one shared launch
  prompt, `src/config.py` pre-wiring. Signed: Claude Code | Opus 5 | high
- 2026-07-24 WS-1 batch 1: racehorses (355), spacecraft (270), paint_colors (391). Signed:
  Claude Code | Haiku 4.5 | low
- 2026-07-24 WS-1 batch 2: motorcycles (359), craft_beers (398), aircraft (435). Signed:
  Claude Code | Haiku 4.5 | low
- 2026-07-24 **Consolidation**: three sessions independently built WS-4 dual-output on separate
  branches without visibility into each other (STATUS.md claims don't cross branches). Owner
  chose `scope-vs-please-yrlsll`'s design; the other two implementations were discarded but all
  their datasets were kept and repointed at the winning API, and two independently built city
  lists were merged into `world_cities.txt` (1,691 names). No secrets/PII found. 18/18 green.
  — Signed: Claude Code | Sonnet 5 | high
- 2026-07-24 WS-4 dual-output, three parallel designs (`next-item-v4te8p` `predict_value` flag;
  `scope-vs-please-yrlsll` `value_head`/`encode()` — **chosen**; `next-task-tnbsmq` `DualCharRNN`
  subclass). Two superseded, code discarded, demo datasets kept. Signed: Claude Code | Sonnet 5
  | high / medium / medium
- 2026-07-24 WS-1 datasets from the superseded branches, all preserved: `world_cities.txt`
  (671, merged), `city_names.txt` (1,323, merged), `tech_startups.txt` (400),
  `motorcycle_brands.txt` (63). Signed: Claude Code | Sonnet 5 & Haiku 4.5 | medium
