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
| 2026-08-02 | WS-9 · `src/arch/`, `src/model.py`, `tests/test_arch.py` | ✅ `CharRNN` dispatches on `cfg.arch` to lstm/gru/transformer; stepwise decoding proven equivalent to a full forward pass. Lane was cut off by a session limit after its 17 runs finished but before writing the table; the orchestrator reconstructed it from the checkpoints into `reports/ARCH.md`. **GRU beats LSTM below ~500 names with 25% fewer params; transformer is last on all four despite 30% more.** | Signed: Claude Code \| Opus 5 \| high |
| 2026-08-02 | WS-10 · `src/train.py`, `pretrain.py`, `finetune.py`, `tests/test_training_quality.py` | ✅ Seeded init proven: three identical runs now give **bitwise identical** checkpoints (`torch.equal` on every tensor); unseeded reproduced wave 2's 12/16/19 spread. Regularization measured on 3 datasets — **none of weight decay, label smoothing or warmup improves best held-out loss**; recommendation is to leave all three off. `--auto-epochs` added, and it corrected the README's premise (see Known issues). 79 tests green. | Signed: Claude Code \| Opus 5 \| high |
| 2026-08-02 | WS-11 · `src/evaluate.py`, `tests/test_evaluate.py`, `reports/` | ✅ `reports/BENCHMARK.md`: all 30 datasets, plus a **controlled within-dataset size ladder** (validation set split once and held fixed across arms, nested training prefixes) and a re-run of wave 2's exact 300-epoch protocol with `aircraft` as an untouched control. Answers the wave's question — see Known issues → Resolved. Also caught a determinism bug in its own `evaluate()` (seeding before `load_checkpoint`, which consumes RNG). 42 tests. | Signed: Claude Code \| Opus 5 \| high |
| 2026-08-02 | WS-12 · `web/`, `src/export_web.py`, `src/serve.py`, `tests/test_web.py` | ✅ All 30 datasets ship in one **5.25 MB** offline HTML: float16 encoding buys 3.1× at worst logit error 2.1e-4 (24× inside tolerance), falling back to float32 per-model on failure — bigger, never wronger. Gallery grouped by the 8 sidecar domains; decoding controls framed by wave 2's measurements, with truncation in a drawer labelled *measured worse here*. Fidelity check tested by **breaking** it six ways. 72 tests. | Signed: Claude Code \| Opus 5 \| high |

## Shared-file touches
- `src/config.py`: wave-3 fields pre-declared 2026-08-01 before any lane started (`arch`/`num_heads`/`ff_dim`/`max_position`, `seed_init`/`weight_decay`/`label_smoothing`/`warmup_epochs`, `dataset_label`/`dataset_path`). Deliberate, for the same reason as wave 2 — so parallel agents never edit this file concurrently. — Claude Code | Opus 5 | high
- `src/config.py`: **`seed_init` default flipped `False` → `True`** by WS-10 with the owner's authorization. This is the one wave-3 default that does *not* reproduce prior behavior: every default training trajectory changed once, differently seeded rather than worse. Pass `--no-seed-init` for the old behavior. — Claude Code | Opus 5 | medium
- `data/shared_vocab.json`: regenerated at consolidation, 66 → 67 symbols (gained `1`, lost nothing). Existing checkpoints are stale; checkpoints are gitignored and regenerable. — Claude Code | Opus 5 | high
- `data/*.meta.json`: `domain` normalized across all 30 sidecars onto one taxonomy. — Claude Code | Opus 5 | high

## Known issues
- **More data does NOT reduce overfitting per epoch — only the 300-epoch collapse.** This is
  the wave's central result and it is easy to overstate in the other direction, so state it
  precisely. `reports/BENCHMARK.md` §1 ran a controlled ladder (one validation set split
  once and held fixed across arms, nested training prefixes, same vocab and seed): best
  held-out loss fell monotonically with training-set size in **5 of 5 domains**, but the
  train/val gap at the best epoch shrank in only 1 of 5, and on `english_words` it grew
  (+0.276 at 500 names → +0.500 at 2,500). Early stopping already halts before the
  divergence, so the surviving gap mostly measures domain entropy. Returns also flatten
  (`car_models` gains 0.143 nats from 218→400 training names but only 0.049 from 700→1035).
  Wave 2's diagnosis was **half** right: the datasets were too small *and* a 2-layer
  256-wide LSTM is too large for them. Flagged 2026-08-02 — Claude Code | Opus 5 | high
- **The `--epochs` guidance in every doc before 2026-08-02 was built on a false premise.**
  It said bigger datasets need more epochs. The val-loss bottom measured across nine
  datasets from 159 to 8,631 names is 13, 10, 26, 9, 13, 10, 8, 10, 7 — it does not move
  with size, and arrives marginally *earlier*, because one epoch over 8,631 names is fifty
  times the gradient steps of one epoch over 159. The old table was two data points
  over-read. `--auto-epochs` now derives a **ceiling on over-training damage** (which does
  scale with size) rather than predicting the bottom. Flagged 2026-08-02 — Claude Code | Opus 5 | high
- **Regularization does not help this model.** Weight decay, label smoothing and warmup were
  measured on 159/590/2,223-name datasets; none improves best held-out loss. Label smoothing
  cuts 300-epoch over-training damage (+91% → +27%) but that protects against a failure mode
  early stopping already removed. Leave all three off. Flagged 2026-08-02 — Claude Code | Opus 5 | high
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
- ✅ **"No architecture recommendation has been earned"** (flagged earlier today). Earned:
  `reports/ARCH.md`. The GRU wins below ~500 training names with 25% fewer parameters and
  overfits less on 4 of 4 datasets; the LSTM wins above; the transformer is last on all four
  while carrying 30% more parameters than the LSTM. Default stays `lstm`.
- ✅ **Two independent measurements agree that the model, not just the data, is the problem.**
  `reports/BENCHMARK.md` found flattening returns as data grew; `reports/ARCH.md` found the
  *smaller* architecture winning on exactly the small datasets. Wave 4's obvious first move is
  a capacity sweep (`hidden_dim`, `num_layers`) rather than more data.
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
