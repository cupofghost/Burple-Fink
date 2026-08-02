# STATUS

Last consolidation: 2026-08-02 — Signed: Claude Code | Opus 5 | high

## Active work
| Date | Area / files | Task & state (≤3 lines) | Signature |
|------|--------------|-------------------------|-----------|
| 2026-08-01 | wave-3 prep · `src/config.py`, `docs/WAVE3_PLAN.md` | ✅ Pre-declared every field the nine wave-3 lanes consume, before they started, so none of them had to edit the shared config concurrently. Lane map published. | Signed: Claude Code \| Opus 5 \| high |
| 2026-08-01 | WS-14 · `data/` | ✅ Six nature datasets, 3,546 names: birds 863, plants_flowers 634, minerals_gems 624, mountains 523, dog_breeds 461, mushrooms 441. All validate clean. | Signed: Claude Code \| Opus 5 \| medium |
| 2026-08-01 | WS-15 · `data/` | ✅ Six culture datasets, 3,520 names: video_games 753, metal_bands 663, perfumes 639, board_games 585, cocktails 459, cheeses 421. ~25 real bands deliberately excluded as unsuitable for a public repo. | Signed: Claude Code \| Opus 5 \| medium |
| 2026-08-01 | WS-16 · `data/` | ✅ Six "named made things" datasets, 4,807 names: pharma_drugs 2223, greek_myth 755, stars_constellations 503, sailing_ships 468, typefaces 463, locomotives 395. Cross-checked its own character usage against `shared_vocab.json` before reporting. | Signed: Claude Code \| Opus 5 \| medium |
| 2026-08-01 | WS-17 · `data/` (4 thin files) | ✅ Grew the four datasets wave 2 called too small: car_manufacturers 159→590, car_models 257→1218, motorcycle_brands 63→309, spacecraft 270→593. Weighted toward defunct/historical marques. Normalized `Chang'e` and `SSM/I`; dropped 3 duplicate lines. | Signed: Claude Code \| Opus 5 \| high |
| 2026-08-01 | WS-13 · `scripts/`, `.github/`, `tests/test_repo_hygiene.py`, `tests/test_data_hygiene.py` | ✅ Fixed the red `hygiene` job (see Known issues, now resolved) with a line-scoped `# check_repo: allow` pragma + a one-entry `(path, string)` allowlist — no blanket skips, and `test_new_secret_in_a_non_fixture_file_is_still_caught` pins the invariant. New `scripts/check_data.py`. Hygiene tests 15 → 67. | Signed: Claude Code \| Opus 5 \| high |
| 2026-08-02 | consolidation · `README.md`, `HANDOFF.md`, `data/*.meta.json`, `data/shared_vocab.json` | ✅ Normalized 4 lanes' inconsistent domain labels onto one 8-domain taxonomy; rebuilt both catalogs from the sidecars; re-armed the drift check (`--strict`) in CI. Paid off the entire `KNOWN_NONCONFORMING` list (see Resolved). Regenerated the shared vocab 66→67 symbols. | Signed: Claude Code \| Opus 5 \| high |
| 2026-08-02 | WS-9 · `src/arch/`, `src/model.py`, `tests/test_arch.py` | ⏳ **Handoff.** `CharRNN` now dispatches on `cfg.arch` to lstm/gru/transformer cores; tests written. Interrupted by the session limit while running the three-way measurement — **the measurement table does not exist yet**, so no architecture recommendation has been earned. Resume by training the three archs on one held-out split. | Signed: Claude Code \| Opus 5 \| medium |
| 2026-08-02 | WS-10 · `src/train.py`, `pretrain.py`, `finetune.py`, `tests/test_training_quality.py` | ⏳ **Handoff.** `seed_init` implemented and flipped to `True` (see Shared-file touches); `weight_decay`/`label_smoothing`/`warmup_epochs`/`--arch` wired. Interrupted while writing tests. Outstanding: does regularization actually close the train/val gap, and the `--auto-epochs` rule. | Signed: Claude Code \| Opus 5 \| medium |
| 2026-08-02 | WS-11 · `src/evaluate.py`, `tests/test_evaluate.py`, `reports/` | ⏳ **Handoff.** Held-out NLL against `val_names`, markdown `--report`, and per-dataset benchmark JSON for **29 of 30 datasets** in `reports/_bench/`. Interrupted before writing `reports/BENCHMARK.md`. See Known issues for what the raw numbers do and don't say. | Signed: Claude Code \| Opus 5 \| medium |
| 2026-08-02 | WS-12 · `web/`, `src/export_web.py`, `src/serve.py` | ⏳ **Handoff.** Exporter reworked for multi-model output; server holds at 70 KB regardless of model count. Interrupted before finishing the gallery UI and `tests/test_web.py`. | Signed: Claude Code \| Opus 5 \| medium |

## Shared-file touches
- `src/config.py`: wave-3 fields pre-declared 2026-08-01 before any lane started (`arch`/`num_heads`/`ff_dim`/`max_position`, `seed_init`/`weight_decay`/`label_smoothing`/`warmup_epochs`, `dataset_label`/`dataset_path`). Deliberate, for the same reason as wave 2 — so parallel agents never edit this file concurrently. — Claude Code | Opus 5 | high
- `src/config.py`: **`seed_init` default flipped `False` → `True`** by WS-10 with the owner's authorization. This is the one wave-3 default that does *not* reproduce prior behavior: every default training trajectory changed once, differently seeded rather than worse. Pass `--no-seed-init` for the old behavior. — Claude Code | Opus 5 | medium
- `data/shared_vocab.json`: regenerated at consolidation, 66 → 67 symbols (gained `1`, lost nothing). Existing checkpoints are stale; checkpoints are gitignored and regenerable. — Claude Code | Opus 5 | high
- `data/*.meta.json`: `domain` normalized across all 30 sidecars onto one taxonomy. — Claude Code | Opus 5 | high

## Known issues
- **The wave-3 benchmark does not yet prove "more data fixes overfitting", and the raw
  numbers must not be read as if it does.** All 29 runs in `reports/_bench/` early-stopped
  (best epoch 6–23, patience 20), so their train/val gaps (0.07–0.79) are *gaps at the best
  epoch*. Wave 2's notorious 6.15-nat gap was measured at **epoch 300**. Those are different
  measurements and comparing them directly is wrong. The one honest apples-to-apples
  comparison available today: `car_manufacturers` best val loss was **2.98 at 135 training
  names** (wave 2) and is **2.566 at 502** (now) — a real improvement from 3.7× the data.
  Cross-dataset gap comparisons are confounded by domain entropy: `aircraft` reaches val
  0.774 and `motorcycle_brands` 2.731, which reflects how predictable each domain is, not
  how much either overfits. Flagged 2026-08-02 — Claude Code | Opus 5 | high
- **No architecture recommendation has been earned yet.** `src/arch/` implements gru and
  transformer, and the tests cover the contracts, but WS-9 was interrupted before the
  three-way measurement ran. Do not describe the transformer as better or worse than the
  LSTM anywhere until that table exists. Flagged 2026-08-02 — Claude Code | Opus 5 | high
- Every dataset added in wave 3 carries `"verified": false` — entries were recalled from
  model knowledge and are believed real, but were not cross-checked against a primary
  source. `periodic_elements.tsv` is the sole verified file. Good training data; not a
  reference work. Flagged 2026-08-01 — Claude Code | Opus 5 | high
- The letters-and-digits alphabet costs real surface forms: `Wilsons Warbler` lost its
  apostrophe, `Popocatepetl` its accents, and `K2` is absent from `mountains.txt` entirely.
  The names are real; some spellings are not canonical. Flagged 2026-08-01 — Claude Code | Opus 5 | medium
- `data/car_manufacturers_founding_year.tsv` founding years are sourced from each brand's
  commonly-cited history, not cross-checked against a primary source — treat as
  approximate. Flagged 2026-07-24 — Claude Code | Sonnet 5 | high

## Resolved this wave
- ✅ **`scripts/check_repo.py` failed on `main`, so CI's `hygiene` job was red** (flagged
  2026-07-29). Fixed by WS-13 without weakening the scanner. Its own first attempt — a
  broad "`@example.com` is never PII" rule — was caught and reverted for swallowing a real
  detection.
- ✅ **`fit()` seeded the RNG after the model was built**, so initial weights were not
  reproducible and three identical commands put the best epoch at 12, 16 and 19 (flagged
  2026-07-29). Fixed by WS-10; `seed_init` now defaults to `True`.
- ✅ **"The right answer for the small datasets is more data, not fewer epochs"** (flagged
  2026-07-29). Acted on: 18 new datasets and the four thin ones grown. 12 datasets / 13,412
  names → 30 / 27,226.
- ✅ **`data/world_cities.txt`'s provenance caveat** (flagged 2026-07-24) turned out to hide
  a real defect. The 2026-07-24 merge deduplicated case-insensitively, which could not see
  that 17 cities were present **twice** — once accented, once not (`Belém` and `Belem` as
  separate lines). Normalization collapsed them: 1,691 → 1,674 rows, no city lost.
- ✅ **117 lines across four datasets carried characters the model's alphabet has no symbol
  for**, including a U+00AD soft hyphen in `world_cities.txt:397` invisible in every editor.
  `filter_to_vocab` was dropping those names at fine-tune time silently. All normalized;
  `KNOWN_NONCONFORMING` is now empty and a test pins it that way.

## Archive
- 2026-07-29 **Wave 2** (WS-6/7/8): held-out validation + early stopping + best-epoch weights; top-k/nucleus/repetition-penalty decoding with a sweep; the repo's first CI. Its two headline measurements both still stand: ~95% of a 300-epoch budget is actively harmful on small datasets, and plain temperature at 1.1–1.3 beat every top-k/nucleus setting tried. Signed: Claude Code | Opus 5 / Sonnet 5 | high
- 2026-07-24 **Consolidation**: three sessions independently built WS-4 dual-output in parallel without visibility into each other's branches (a git-branch-per-agent blind spot — STATUS.md claims don't cross branches). Owner chose `scope-vs-please-yrlsll`'s design; the other two were discarded but all three branches' demo datasets were kept and re-pointed at the winning API. Two independently-built city lists were merged rather than either being dropped. Signed: Claude Code | Sonnet 5 | high
- 2026-07-24 WS-4 dual-output, three competing designs — superseded by the consolidation above. Signed: Claude Code | Sonnet 5 | medium
- 2026-07-24 WS-1 seed datasets (`tech_startups` 400, `motorcycle_brands` 63, `city_names` 1,323, `world_cities` 671) — all folded into the merged library. Signed: Claude Code | Sonnet 5 / Haiku 4.5 | medium
