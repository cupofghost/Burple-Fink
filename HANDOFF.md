# HANDOFF — Burple-Fink multi-agent guide

**Read this before doing anything.** Burple-Fink is built in stages by multiple agents
working in parallel. This document is the source of truth for *what to do next*, *how to
do it without colliding with other agents*, and *when a piece of work is "done."*

- Project overview & quick start: [`README.md`](README.md)
- Design rationale & pipelines: [`docs/PLAN.md`](docs/PLAN.md)
- Repo conventions & commands: [`CLAUDE.md`](CLAUDE.md)

---

## 1. The mission in one sentence

Build **one reusable char-RNN engine** and train/fine-tune it on **many datasets** so it
can generate a *variety* of invented names — starting with automotive names and growing
outward — the way Janelle Shane's char-rnn generated paint colors.

---

## 2. Current state (Stages 0, 2, 3, 5 complete)

A working character-level LSTM that trains **per-dataset from scratch** and samples with a
temperature ("creativity") knob. Verified end-to-end on CPU: loss drops from ~3.7 to
<1.0 and output matures from noise → plausible names.

On top of that MVP, the fine-tuning core (**WS-2**), evaluation harness (**WS-3**), both
serving front-ends (**WS-5**), and the dual-output name+attribute head (**WS-4**) are now
implemented — see the ✅ markers in §3. A `tests/` suite
(`python -m unittest discover -s tests`) pins the vocab, training, model, and export
invariants. Still open: more datasets (**WS-1**, always open-ended by design).

Modules and their responsibilities:

| File | Responsibility | Safe to extend? |
|------|----------------|-----------------|
| `src/config.py` | All hyperparameters + special tokens (`PAD`/`START`/`END`) | Add fields; keep `from_dict` backward-compatible |
| `src/data.py` | Name loading, `Vocab`, `(input, target)` next-char pairs, batching | Yes — see the shared-vocab note in §6 |
| `src/model.py` | `Embedding → LSTM → Linear` char-RNN (+ optional value head, WS-4) | Yes — keep the `forward(x, hidden)` signature |
| `src/train.py` | Training loop, live previews, checkpoint save | Yes |
| `src/sample.py` | Checkpoint load + temperature generation (+ predicted value for dual checkpoints) | Yes |
| `src/train_dual.py` | WS-4: joint next-char + value-regression training | Yes |
| `generate.py` | Friendly one-command wrapper | Yes |

**Checkpoint format** (a single `torch.save` dict) — do not break these keys, other
stages depend on them:

```python
{
    "model_state":    <state_dict>,
    "config":         <Config.to_dict()>,
    "vocab":          <Vocab.to_dict()>,      # {"itos": [...]}
    "training_names": [<str>, ...],           # used to filter for novelty
}
```

> **WS-6 adds one additive key: `"val_names"`** (a list of the held-out names, `[]` when the
> model trained on everything). The four keys above are unchanged. Every reader must use
> `ckpt.get("val_names", [])` — no checkpoint written before 2026-07-29 has the key, and a
> bare `ckpt["val_names"]` would crash on all of them. When it is non-empty, `training_names`
> is the *training half only*, so novelty is judged against what the model actually saw, and
> the held-out names are available for an honest (non-training) NLL.

> WS-4's dual-output head (§3) does not add or rename any of these keys — it only adds
> new `Config` fields (`dual_output`/`value_mean`/`value_std`/`value_label`) inside the
> existing `config` dict, plus extra `value_head.*` tensors inside `model_state` for
> checkpoints trained with `src/train_dual.py`. Ordinary checkpoints are unaffected.

---

## 3. Workstreams (claim one, work in parallel)

Each workstream maps to a stage in the [README status table](README.md#where-the-project-is-now).
They are deliberately decoupled so multiple agents can run at once. **Claim a workstream
by adding your branch to the table in §7 before you start.**

### WS-1 · Dataset library  *(low coupling — great for parallel agents)*
Add clean, documented datasets under `data/`. Each new dataset is independent, so many
agents can do this simultaneously with near-zero merge risk.
- Follow [§5 Adding a new dataset](#5-adding-a-new-dataset) exactly.
- Target ≥300 names per dataset (more is much better — Shane used ~7,700).
- Update the catalog table in `README.md` and §4 below.

### WS-2 · Shared base model + fine-tuning  *(the "fine-tuning" core)* — ✅ **done** (`claude/program-development-q4fcp7`)
Implemented: `src/data.py` grew a shared-vocabulary layer (`build/save/load_shared_vocab`,
`filter_to_vocab`, persisted to `data/shared_vocab.json`); `src/train.py` was refactored to
expose a reusable `fit()` loop + `save_checkpoint()`; `src/pretrain.py` trains a base model
on all datasets; `src/finetune.py` specializes it per dataset (gentler LR, fewer epochs) into
`checkpoints/<dataset>_ft.pt`. Checkpoint format (§2) unchanged. Design write-up in `docs/PLAN.md §11`.

Original brief — turn the from-scratch trainer into **pretrain-then-fine-tune** (transfer learning):
1. Introduce a **shared vocabulary** (see §6) so one base model can be fine-tuned on any
   dataset without resizing the embedding/head.
2. Add `src/pretrain.py`: train a base model on the concatenation of *all* datasets.
3. Add `src/finetune.py`: load the base checkpoint and continue training on one dataset
   (typically a lower learning rate, fewer epochs). Reuse `src/train.py`'s loop.
4. Save fine-tuned checkpoints as `checkpoints/<dataset>_ft.pt` in the same format (§2).
- **Deliverable:** a fine-tuned checkpoint that reaches lower loss / better samples in
  fewer epochs than the from-scratch baseline on the same dataset.

### WS-3 · Evaluation harness — ✅ **done** (`claude/program-development-q4fcp7`)
Implemented in `src/evaluate.py`: `python -m src.evaluate --checkpoint …` prints novelty,
uniqueness, mean pairwise edit distance, a character-bigram log-likelihood plausibility
(generated vs. training reference), and the model's training NLL. Metrics confirm the
temperature knob trades novelty for typicality (e.g. car-models novelty ~16% @ 0.8 → ~56% @ 1.4
while the plausibility ratio stays ~1.0).

Original brief — add `src/evaluate.py` computing, for a checkpoint + its dataset:
- **Novelty** — % of generated names not in the training set.
- **Plausibility** — a cheap heuristic (e.g. character-bigram log-likelihood vs the
  training distribution) and/or held-out validation loss.
- **Diversity** — unique-name rate and average edit distance.
- **Deliverable:** `python -m src.evaluate --checkpoint … ` prints a metrics table;
  numbers are how WS-2 proves fine-tuning helps.

### WS-4 · Dual-output (name + attribute) — ✅ **done** (`claude/scope-vs-please-yrlsll`, consolidated onto `claude/next-item-v4te8p`)
> **Consolidation note (2026-07-24):** three sessions independently built this workstream
> on separate branches (`claude/next-item-v4te8p`, `claude/scope-vs-please-yrlsll`,
> `claude/next-task-tnbsmq`) without seeing each other's work — an unavoidable risk of
> the git-branch-per-agent setup, since STATUS.md claims on one branch aren't visible to
> a session working from another. The owner picked this branch's design (see STATUS.md
> Archive for why); the other two implementations were discarded, but their **datasets**
> were kept and repointed at this API — no dataset work was thrown away, only the
> duplicate model/training code.

Implemented Shane's name+number trick: `src/config.py` gained `dual_output` /
`value_mean` / `value_std` / `value_label` (all default "off", so ordinary checkpoints
are unaffected); `src/model.py`'s `CharRNN` gained an optional `value_head` (`None`
unless `cfg.dual_output`) plus `encode()` (embedding+LSTM, factored out of `forward`)
and `predict_value(state)`; `forward(x, hidden)` itself is unchanged. `src/data.py`
gained `load_name_value_pairs()` for `name<TAB>value` TSVs (`#`-comments and blank
lines skipped). `src/train_dual.py` z-scores the raw values (`value_mean`/`value_std`
saved to the checkpoint) and trains `combined_loss = ce_loss + value_weight * mse_loss`
jointly on one shared LSTM. `src/sample.py`'s `generate_one`/`generate_many` gained an
opt-in `return_value` param (default `False` = old behavior/return type exactly), and
`python -m src.sample` auto-prints the denormalized value for dual checkpoints.
Checkpoint format (§2) unchanged.

Demo datasets (`name<TAB>value`, not part of the shared-vocab `*.txt` glob):
- `data/car_manufacturers_founding_year.tsv` — 66 real car brands + founding year.
- `data/paint_colors.tsv` — ~140 real CSS/X11 named colors + relative luminance
  (computed by `scripts/build_paint_colors.py` from each color's hex — a direct homage
  to Shane's original paint-color + RGB experiment). Closed, standardized list (CSS only
  defines that many keywords), so it's smaller than the usual ≥300 target on purpose.
- `data/periodic_elements.tsv` — all 118 IUPAC chemical elements + atomic number.

```bash
python -m src.train_dual --data data/car_manufacturers_founding_year.tsv \
    --name manufacturers_founding_year --epochs 300 --value-label "founding year"
python -m src.sample --checkpoint checkpoints/manufacturers_founding_year.pt --num 10
```

Original brief — turn Shane's name+RGB trick into a general name+number regression:
1. Add a second head that regresses a numeric attribute (for cars: horsepower / price
   tier / a learned "sportiness" score). Requires datasets with a value column —
   coordinate with WS-1 on a `name<TAB>value` format.

### WS-5 · Serving — ✅ **done** (`claude/program-development-q4fcp7`)
Two front-ends, both a mobile-friendly "instrument panel" UI (temperature dial, prefix,
live novelty flags), sharing one CSS/markup source in `web/app_template.html`:
- `src/serve.py` — a **stdlib-only** HTTP server (no new deps) that serves the UI and a
  `/api/generate` endpoint backed by the live PyTorch checkpoints. Binds `0.0.0.0` for phones.
- `src/export_web.py` — bakes weights into one self-contained `web/burple-fink.html` that
  runs the char-RNN **in the browser**; it verifies the JS forward pass matches the torch
  model's logits before writing, so the in-browser net is faithful.

No coupling to WS-2/3/4 beyond the checkpoint format.

---

### Wave 2 — WS-6 / WS-7 / WS-8 (planned 2026-07-29, three parallel agents)

Stages 0–5 are built; wave 2 upgrades their *quality and durability*. Full rationale, the
file-ownership matrix, the contracts, and the merge protocol are in
[`docs/UPGRADE_PLAN.md`](docs/UPGRADE_PLAN.md). Each lane has a self-contained brief under
`docs/upgrade/`. The lanes were partitioned so **no two agents edit the same file** — the
failure mode that caused the 2026-07-24 consolidation.

#### WS-6 · Training quality — ✅ **done** (`claude/ws6-training-quality`, brief: `docs/upgrade/AGENT-A.md`)
Implemented: `data.split_names(names, val_fraction, seed)` gives a deterministic, disjoint,
order-preserving train/val split; `train.fit()` grew a keyword-only `val_names=` path that
computes a held-out loss each epoch (`model.eval()` + `no_grad()`, unshuffled, so it draws
nothing from any RNG), logs `train X.XXXX | val X.XXXX`, keeps the best epoch's `state_dict`
in memory and restores it before returning, and stops after `cfg.early_stop_patience` stalled
epochs. `cfg.lr_schedule` adds `"plateau"`/`"cosine"` (`"none"` builds no scheduler at all).
`train.evaluate_loss()` is exposed for scoring any name list with the training criterion, and
`fit(..., report=dict)` fills in the per-epoch curves plus `best_epoch`/`stopped_early` so the
run is inspectable without parsing stdout. `--val-fraction` / `--patience` / `--lr-schedule`
are wired through `train.py`, `pretrain.py` and `finetune.py`; all default to off. Adds one
**additive** checkpoint key, `"val_names"` (readers must use `ckpt.get("val_names", [])`);
the four keys in §2 are unchanged. 34 tests in `tests/test_training_quality.py`, including a
golden pre-WS-6 loss trajectory that pins the default path. Measured result in `STATUS.md`.

Deliberately not done: `src/train_dual.py`'s `fit_dual()` has its own loop and gets none of
this — it is the obvious follow-up, and it was frozen for wave 2.

#### WS-7 · Decoding quality — ⏳ open (`claude/ws7-decoding-quality`, brief: `docs/upgrade/AGENT-B.md`)
top-k / nucleus / repetition-penalty sampling as keyword-only params on
`generate_one`/`generate_many` (defaults = today's plain temperature sampling), plus a
near-duplicate (edit-distance) memorization metric, honest held-out NLL when a checkpoint
carries `val_names`, and a `--sweep` grid that picks decoding settings with numbers.
Owns `src/sample.py`, `src/evaluate.py`.

#### WS-8 · CI & repo hygiene — ⏳ open (`claude/ws8-ci-and-hygiene`, brief: `docs/upgrade/AGENT-C.md`)
First CI for the repo (`.github/workflows/ci.yml`: unittest suite + CLI smoke train/sample),
a stdlib-only `scripts/check_repo.py` that catches dataset-registry drift, committed weights,
and secrets/PII (§3), plus `/api/health` and real error handling in the phone UI.
Owns `.github/`, `scripts/`, `src/serve.py`, `web/`.

> `src/config.py` was pre-wired with all fields WS-6 and WS-7 need (defaults reproduce
> current behavior), so **no wave-2 agent edits it**. `src/model.py` and `src/train_dual.py`
> are frozen for the wave. Domain conditioning (`docs/PLAN.md §9.4`) and alternative
> backbones (§9.6) are deliberately deferred to wave 3 — they need the same four files at
> once and cannot be parallelized safely.

---

## 4. Dataset registry

Keep this in sync with `data/` and the README catalog. One row per dataset file.

| File | Domain | Count | Owner (branch) | Notes |
|------|--------|-------|----------------|-------|
| `car_manufacturers.txt` | Auto brands | ~150 | seed | Real worldwide brands |
| `car_models.txt` | Car model names | ~250 | seed | Real model names |
| `english_words.txt` | Common English words | ~8,600 | `claude/plausible-words-dataset-53qyyg` | Frequency-ranked common words (Google Web Trillion Word Corpus via `first20hours/google-10000-english`, MIT). Curated to lowercase a–z, length 3–10, vowel-bearing. Teaches general English spelling → plausible word-shapes; good base-model fuel. |
| `world_cities.txt` | World city/capital names | ~1,690 | `claude/scope-vs-please-yrlsll` + `claude/next-task-tnbsmq` | Consolidated (2026-07-24) from two independently-built city lists (671 + 1,323 real names, ~45% overlap) into one case-insensitive-deduped, alphabetized file — no real names lost from either. |
| `tech_startups.txt` | Real tech company/startup names | ~400 | `claude/next-task-tnbsmq` | Well-known startups/tech companies. |
| `motorcycle_brands.txt` | Real motorcycle manufacturers | ~60 | `claude/next-task-tnbsmq` | Harley-Davidson through Zongshen, globally sourced. |
| `motorcycles.txt` | Motorcycle brands & models | 359 | `claude/next-three-items-nj7dyj` | Real motorcycle and brand names from Honda, Yamaha, Kawasaki, Harley-Davidson, Ducati, BMW, KTM, Royal Enfield, and others — broader scope than `motorcycle_brands.txt` (brands only), includes model names (Rebel, Street Glide, Ninja). Both kept; not merged, since they cover different scopes. |
| `racehorses.txt` | Racehorse names | 355 | `claude/next-three-items-nj7dyj` | Real thoroughbred racehorse names spanning classic legends (Secretariat, Man O' War) to modern winners. Creative, varied naming patterns; good for testing diverse English word-building. |
| `spacecraft.txt` | NASA/ESA spacecraft & satellites | 270 | `claude/next-three-items-nj7dyj` | Real spacecraft, satellites, and mission names from NASA, ESA, JAXA, CNSA. Includes rovers, orbiters, landers, and space stations. Mix of acronyms (GOES, TESS) and full names (Voyager, Cassini). |
| `craft_beers.txt` | Craft brewery & beer names | 398 | `claude/next-three-items-nj7dyj` | Creative beer names from US and international craft breweries. Includes descriptive names (IPA, Porter, Stout), whimsical names (Hoppy Beer, Belly Dance), and branded brewery ales (Stone IPA, Founders All Day). |
| `aircraft.txt` | Aircraft models | 435 | `claude/next-three-items-nj7dyj` | Real aircraft and helicopter model names from Boeing, Airbus, Bombardier, Embraer, Gulfstream, Cessna, Piper, Learjet, Saab, Dassault, and others. Mix of alphanumeric (Boeing 737, A320) and named models (Dreamliner, Super King Air). |
| `car_manufacturers_founding_year.tsv` | Car brands + founding year | 66 | `claude/scope-vs-please-yrlsll` | WS-4 dual-output demo, `name<TAB>value`. Founding years sourced from each brand's commonly-cited history; treat as approximate. |
| `paint_colors.tsv` | CSS/X11 named colors + relative luminance | ~140 | `claude/next-item-v4te8p` | WS-4 dual-output demo, `name<TAB>value` format. Names + hex are the standardized CSS Color Module Level 4 keyword list (source of truth in `scripts/build_paint_colors.py`); value is luminance computed from the hex, not an external fact. **Not the same file as** `paint_colors.txt` below (WS-1, plain name list, no values) — kept both; extension disambiguates. |
| `paint_colors.txt` | Paint color names | 391 | `claude/next-three-items-nj7dyj` | Real and whimsical paint color names from major manufacturers. Ranges from descriptive (Alabaster, Apricot) to poetic (Aurora Red, Crystal Clear). Homage to Janelle Shane's original neural-network paint-color project. WS-1 plain-name-list sibling of the `.tsv` WS-4 demo above. |
| `periodic_elements.tsv` | Chemical elements + atomic number | 118 | `claude/next-task-tnbsmq` | WS-4 dual-output demo, `name<TAB>value`. All 118 IUPAC elements — a closed, exactly-verifiable list. |

None of the `.tsv` dual-output datasets are part of the shared-vocab `*.txt` glob (§6);
they're only consumed by `src.train_dual`.

---

## 5. Adding a new dataset

Do these steps in order; they keep every dataset uniform and mergeable.

1. **Create** `data/<domain>.txt` — one name per line, UTF-8, no header row.
2. **Clean it:**
   - Strip leading/trailing whitespace; drop blank lines.
   - Remove duplicates (the loader also de-dupes, but keep files clean).
   - Prefer canonical casing (Title Case for brands); be consistent within a file.
   - Drop obvious noise: years, trim levels, SKUs, marketing suffixes.
3. **Aim for ≥300 entries.** Small sets produce charming nonsense — fine for a demo,
   weak for fine-tuning. Note the count honestly.
4. **Smoke-test it:**
   ```bash
   python -m src.train --data data/<domain>.txt --epochs 40 --name <domain>
   python -m src.sample --checkpoint checkpoints/<domain>.pt --num 10
   ```
5. **Register it:** add a row to the catalog in `README.md` **and** to §4 above.
6. **Do not commit checkpoints** — `checkpoints/` and `*.pt` are gitignored. Commit only
   the `.txt` dataset and doc updates.

**Sourcing & licensing:** only use openly available / factual name lists (brand names,
model names are facts). Note the source in the registry `Notes` column when relevant. Do
not scrape sites that forbid it.

---

## 6. Key design decision for fine-tuning: shared vocabulary

Today `Vocab` is built **per dataset** from the characters that appear in it. That is fine
for from-scratch training but **breaks fine-tuning**: a base model's embedding and output
layers are sized to its vocab, so loading it against a different dataset's vocab fails.

**WS-2 must introduce a shared vocab.** Recommended approach:
- Define a fixed character set once (e.g. printable ASCII, or the union of characters
  across all `data/*.txt`), plus the three special tokens, and persist it to
  `data/shared_vocab.json`.
- Build `Vocab` from that file for *both* pretraining and every fine-tune, so all
  checkpoints share identical embedding/head dimensions.
- Keep `Vocab.from_dict` / `to_dict` unchanged so the checkpoint format in §2 still holds.

Document the choice in `docs/PLAN.md` when you implement it.

> ✅ **Implemented** on `claude/program-development-q4fcp7`. `src/data.py` gained
> `build_shared_vocab` / `save_shared_vocab` / `load_shared_vocab` / `filter_to_vocab`;
> `src/pretrain.py` builds and persists `data/shared_vocab.json` on first run and every
> fine-tune loads it. `Vocab.from_dict` / `to_dict` are unchanged, so the checkpoint
> format in §2 still holds. Rationale is written up in `docs/PLAN.md §11`.

---

## 7. Coordination & branch strategy

- **`main` is the integration branch.** It always trains and samples cleanly.
- **One workstream = one branch.** Name it `claude/ws<N>-<slug>` (e.g.
  `claude/ws1-boat-names`, `claude/ws2-finetune`). Branch from the latest `main`.
- **Open a PR into `main`** per workstream; keep PRs focused on a single workstream.
- **Claim your work here** so others don't duplicate it:

  | Branch | Workstream | Agent / date | Status |
  |--------|-----------|--------------|--------|
  | `claude/rnn-auto-name-generator-bqbpnv` | WS-0 MVP + docs | initial | ✅ merged to main |
  | `claude/program-development-q4fcp7` | WS-2 fine-tuning + WS-3 eval + WS-5 serving/web app | 2026-07-23 | ⏳ in review |
  | `claude/plausible-words-dataset-53qyyg` | WS-1 `english_words.txt` (plausible-words dataset) | 2026-07-23 | ⏳ in review |
  | `claude/next-item-v4te8p` | **Consolidated PR**: WS-4 dual-output (chosen implementation, originally from `scope-vs-please-yrlsll`) + all salvaged WS-1/WS-4 datasets from the three parallel branches below | 2026-07-24 | ⏳ in review |
  | `claude/scope-vs-please-yrlsll` | WS-1 `world_cities.txt` (superseded by the merged file) + WS-4 dual-output (chosen implementation, ported to `next-item-v4te8p`) | 2026-07-24 | ⚠️ superseded by consolidation, PR #5 to be closed |
  | `claude/next-task-tnbsmq` | WS-1 `tech_startups.txt`/`motorcycle_brands.txt`/`city_names.txt` (superseded by the merged file) + WS-4 dual-output (discarded design, its `periodic_elements.tsv`/`paint_colors.tsv` ported to `next-item-v4te8p`) | 2026-07-24 | ⚠️ superseded by consolidation |
  | `claude/burple-fink-upgrade-plan-m7ndof` | Wave-2 plan + workspace prep (`docs/UPGRADE_PLAN.md`, per-agent briefs, `src/config.py` pre-wiring) | 2026-07-29 | ⏳ in review |
  | `claude/ws6-training-quality` | WS-6 training quality (Agent A): held-out split, val loss, early stopping, best-epoch restore, LR schedules; additive `val_names` checkpoint key | 2026-07-29 | ⏳ in review |
  | `claude/ws7-decoding-quality` | WS-7 decoding quality (Agent B) | 2026-07-29 | 🔒 reserved, not started |
  | `claude/ws8-ci-and-hygiene` | WS-8 CI & repo hygiene (Agent C) | 2026-07-29 | 🔒 reserved, not started |

- **Low-collision zones** (edit freely): new files under `data/`, new modules under
  `src/` (e.g. `pretrain.py`, `finetune.py`, `evaluate.py`), your own docs.
- **High-collision zones** (coordinate / keep diffs minimal): `src/config.py`,
  `src/data.py`, the checkpoint format, and these three docs. If you must change a shared
  file, make the smallest change that works and call it out in your PR description.

### GitHub repository settings (already configured — work within these, don't try to change them)

- `main` is protected: PRs are required before merging; direct pushes are rejected.
- Required approvals: **0** — you're authorized to merge your own PR once your branch is
  current with `main`.
- Merge method: **squash only** (merge commit and rebase are disabled).
- Linear history is required.
- Force pushes are blocked.
- Head branches auto-delete after merge.
- No CI/status checks are configured — don't wait on checks that don't exist.
- No required signed commits, no required reviewers, no code owners.
- Branch naming: `<agent-name>/<short-feature-description>`, branched off the latest
  `main` (the `claude/ws<N>-<slug>` scheme above is a specific case of this pattern).

---

## 8. Definition of done (per workstream)

A workstream is done when:
1. `python -m src.train --data <a dataset> --epochs 40` runs clean on CPU.
2. `python -m src.sample --checkpoint <ckpt> --num 10` produces on-style names.
3. Any new module has a docstring explaining *why*, not just *what*.
4. The README status table, the dataset registry (§4), and the branch table (§7) are
   updated to reflect your change.
5. The checkpoint format (§2) is unchanged **or** the change is documented here and in
   `docs/PLAN.md`.
6. **A pull request into `main` is open.** Always open a PR when your task is complete —
   every task ends this way, no exceptions. Push your branch, then open the PR with a
   summary of what changed and how you verified it. A task with no PR is not done.

---

## 9. Guardrails

- **Environment:** install PyTorch from the **default PyPI index** — the custom pytorch
  download index (`download.pytorch.org`) is blocked here. `numpy` is optional; the code
  runs without it (Torch prints a harmless warning).
- **Don't commit large binaries** — no checkpoints, no datasets over a few MB without
  discussion.
- **Keep it small and CPU-friendly.** These models must train in minutes on a laptop.
- **Don't break the public surfaces:** the CLI flags of `train.py`/`sample.py`/
  `generate.py` and the checkpoint dict keys are contracts other stages rely on.
- **Be honest in the docs.** If a dataset is tiny or results are weak, say so.
