# Agent A — WS-6 · Train honestly

Branch: `claude/ws6-training-quality` · Suggested model/effort: **Opus 5 | high**
Read first: `AGENTS.md`, `STATUS.md`, `docs/UPGRADE_PLAN.md`, then only the files below.

## The problem you are fixing

`src/train.py:fit()` trains on **100% of the names** for a fixed `cfg.epochs` (default 300)
and keeps whatever the final epoch produced. There is no held-out set anywhere in the repo —
`src/evaluate.py` even labels its NLL "not held-out" because it has nothing else to report.

The datasets are 66–435 names against a 2-layer, 256-wide LSTM. That is a model with far more
capacity than data, trained past the point of memorization with no way to notice. The existing
novelty metric doesn't catch it either: it only checks exact string equality, so a model that
has memorized `Corvette` and emits `Corvete` scores as 100% novel.

You are adding the instrument that tells the truth about this, and the two cheap fixes that
follow from it.

## Deliverables

1. **`data.split_names(names, val_fraction, seed)`** in `src/data.py` — deterministic,
   returns `(train, val)`, disjoint, stable across runs for a given seed. Same-seed
   reproducibility matters more than shuffling cleverness.

2. **`fit()` grows an optional validation path** (`src/train.py`):
   - accepts `val_names` (default `None` → today's behavior, exactly);
   - computes validation loss each epoch with the same criterion, `model.eval()` +
     `torch.no_grad()`;
   - prints `train X.XXXX | val X.XXXX` on the existing log cadence;
   - tracks the best val loss, keeps the best weights **in memory** (a deep-copied
     `state_dict`), and restores them before returning — no extra files on disk;
   - stops early after `cfg.early_stop_patience` epochs with no improvement (`0` = never),
     printing why it stopped and at which epoch the best was found;
   - optional LR schedule per `cfg.lr_schedule` (`"none" | "plateau" | "cosine"`, honoring
     `cfg.lr_factor` / `cfg.lr_min`). `"none"` is the default and must behave identically to
     today.

3. **Wire it through** `src/train.py:train()`, `src/pretrain.py`, `src/finetune.py`:
   `--val-fraction`, `--patience`, `--lr-schedule` CLI flags, all defaulting to the config
   defaults (i.e. off). Fine-tuning benefits most — it's the one place where "fewer epochs,
   gentler LR" is currently a guess.

4. **Checkpoint gains `"val_names"`** (list, may be empty) via `save_checkpoint`. Additive
   only — do not rename or remove any existing key. Document it in `HANDOFF.md §2` as an
   additive extension, and note that readers must use `ckpt.get("val_names", [])`.
   Agent B reads this key to report an honest held-out NLL; it is the one cross-agent
   contract in this wave and it flows one way only.

5. **`tests/test_training_quality.py`** (new file — do not edit the existing test files):
   - split is deterministic, disjoint, and covers every name;
   - val loss is actually computed and differs from train loss on a contrived set;
   - early stopping fires and reports the right best epoch;
   - best-weights restore returns the best epoch's weights, not the last epoch's
     (construct a case where the last epoch is worse);
   - `val_fraction=0` reproduces the current loss trajectory for a fixed seed —
     this is the backward-compatibility proof.

6. **The measurement that makes this worth doing.** Train two datasets at the current
   defaults with a 15% holdout — a small one (`data/car_manufacturers.txt`, ~150) and a
   larger one (`data/aircraft.txt`, 435) — and report in `STATUS.md` + your PR:
   - the epoch where val loss bottomed out vs. the 300 you were paying for;
   - train vs. val loss at epoch 300 (the overfitting gap);
   - your recommended default `epochs` / `early_stop_patience` per dataset size.

   If the honest answer is "the defaults were fine," say that. A negative result here is
   still the most useful thing anyone has learned about this model.

## Rules

- **Files you own:** `src/train.py`, `src/pretrain.py`, `src/finetune.py`, `src/data.py`
  (add the split helper; don't refactor the rest), `tests/test_training_quality.py`,
  your `STATUS.md` row, your `HANDOFF.md` §3/§7 entries, `README.md` only if the owner's
  commands change.
- **Do not touch:** `src/config.py` (your fields are already declared — `val_fraction`,
  `early_stop_patience`, `lr_schedule`, `lr_factor`, `lr_min`), `src/sample.py`,
  `src/evaluate.py`, `src/model.py`, `src/train_dual.py`, `src/serve.py`, `web/`,
  `scripts/`, `.github/`, and the three existing test files.
- `src/train_dual.py` has its own loop (`fit_dual`) and is **out of scope** — leave it
  alone even though it looks like it wants the same treatment. Note it as follow-up work
  rather than doing it.
- Install torch from the **default PyPI index** (`pip install -r requirements.txt`);
  `download.pytorch.org` is blocked here.
- Sign every commit and `STATUS.md` entry: `Signed: Claude Code | Opus 5 | high`.
