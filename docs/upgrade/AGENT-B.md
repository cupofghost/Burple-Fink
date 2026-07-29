# Agent B — WS-7 · Sample better, and prove it

Branch: `claude/ws7-decoding-quality` · Suggested model/effort: **Sonnet 5 | high**
Read first: `AGENTS.md`, `STATUS.md`, `docs/UPGRADE_PLAN.md`, then only the files below.

## The problem you are fixing

`src/sample.py:generate_one()` samples from the **entire vocabulary** at every step, scaled
only by temperature. Every implausible character stays reachable, so the sole way to reduce
junk is to lower the temperature — which also drags output back toward the training set. The
README's own table admits it: 0.2–0.5 is "safe, boring, close to real training names" and
1.1–1.5 is "*Bylfgoam Glosd* chaos". There is no setting that is both varied *and* clean.

Truncating the tail (top-k / nucleus) decouples those two things: keep the temperature high
for variety, but never sample from the garbage tail. This is the cheapest real quality win
available in the repo, and the sweep you build is what proves it rather than asserting it.

## Deliverables

1. **Decoding controls in `src/sample.py`**, added to `generate_one` and `generate_many` as
   **keyword-only parameters at the end of the signature**, each defaulting to today's exact
   behavior (`src/train.py` and `src/serve.py` both call these — do not break them):
   - `top_k=0` — keep only the k likeliest next characters (0 = off);
   - `top_p=1.0` — nucleus: the smallest set whose cumulative probability ≥ p (1.0 = off);
   - `repetition_penalty=1.0` — divide the logits of characters already emitted in this name
     (1.0 = off). Char-level repetition is the "Bylfgoammm" failure, so penalize *emitted
     characters in the current name*, and say in the docstring what you chose and why;
   - `min_length` / `max_length` honored at generation time, not only as a post-filter.

   Apply the order: repetition penalty → temperature → top-k → top-p → sample. Document that
   order in the docstring; it changes the result and future readers will need to know.

2. **CLI flags** on `python -m src.sample`: `--top-k`, `--top-p`, `--repetition-penalty`,
   `--min-length`, each defaulting to the checkpoint's config so old checkpoints behave as
   they always did.

3. **`src/evaluate.py` — three additions:**
   - **Honest held-out NLL.** Agent A's WS-6 adds a `"val_names"` key to new checkpoints.
     If `ckpt.get("val_names")` is non-empty, report NLL on it and label it *held-out*; keep
     the existing training-NLL line labeled as it is. Every checkpoint in existence today
     lacks that key — degrade gracefully, never require it, and don't wait for A to land.
   - **A memorization metric.** Current novelty only catches exact copies. Add
     *near-duplicate rate*: the share of generated names within edit distance 1 (and report
     ≤2 separately) of any training name. `_edit_distance` already exists in this file. This
     is the number that reveals whether the model learned the style or the list.
   - **`--sweep`**: a grid over temperature × (top-k or top-p) that prints one table —
     novelty, near-duplicate rate, plausibility ratio, uniqueness, mean pairwise edit
     distance per setting — and names a recommended setting with a one-line justification.
     Optionally `--compare ckpt-a ckpt-b …` to put several checkpoints in one table.

4. **`tests/test_sampling.py`** (new file — do not edit the existing test files):
   - `top_k=k` restricts the sampled character to the k likeliest (assert on the support,
     not on a lucky draw);
   - `top_p=1.0, top_k=0, repetition_penalty=1.0` reproduces the old path **exactly** for a
     fixed seed — this is the backward-compatibility proof;
   - repetition penalty demonstrably reduces repeated characters;
   - `min_length` is honored;
   - `--sweep` returns one row per grid point with all metrics populated.

5. **The recommendation.** Run the sweep on two checkpoints of different data sizes and put
   the winning settings in `STATUS.md` + your PR + the README temperature table (that table
   is currently the only guidance the owner has, and it will be out of date the moment you
   land). Report honestly if nucleus sampling turns out not to help at this model scale.

## Rules

- **Files you own:** `src/sample.py`, `src/evaluate.py`, `tests/test_sampling.py`, your
  `STATUS.md` row, your `HANDOFF.md` §3/§7 entries, and the README sampling/temperature
  section.
- **Do not touch:** `src/config.py` (your fields are already declared — `top_k`, `top_p`,
  `repetition_penalty`, `min_length`), `src/train.py`, `src/pretrain.py`, `src/finetune.py`,
  `src/data.py`, `src/model.py`, `src/train_dual.py`, `src/serve.py`, `src/export_web.py`,
  `web/`, `.github/`, and the three existing test files.
- `src/train.py:fit()` calls `generate_many` for its live previews and `src/serve.py` calls it
  per request. Both must keep working **unchanged** — that is why the new parameters are
  keyword-only with today's defaults. Don't "improve" the callers; Agent A and Agent C own
  those files this wave.
- Install torch from the **default PyPI index**; `download.pytorch.org` is blocked here.
- Sign every commit and `STATUS.md` entry: `Signed: Claude Code | Sonnet 5 | high`.
