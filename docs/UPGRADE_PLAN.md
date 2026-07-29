# Upgrade Plan — Wave 2 (2026-07-29)

**For the owner, in one paragraph.** Stages 0–5 are built: the engine trains, fine-tunes,
scores, and serves. What's missing is *honesty and control*. Today the model trains on
100% of every dataset for a fixed 300 epochs and keeps whatever the last epoch produced,
so nothing in this repo can tell you whether it learned a style or just memorized 66 car
brands. And when it generates, the only knob is temperature — turn it up for variety and
you also invite junk letters, turn it down for quality and you get the training set back.
Wave 2 fixes both, and adds the automated check that would have caught the duplicated-work
mess of 2026-07-24. Three agents, three lanes, no shared files.

| Workstream | Agent | What it buys you |
|---|---|---|
| **WS-6 · Train honestly** | A | A held-out set, early stopping, and best-epoch weights → you find out how much the model memorizes, and stop paying for epochs that make it worse. |
| **WS-7 · Sample better, prove it** | B | top-k / nucleus / repetition penalty → high variety *without* the junk; plus a sweep that picks the best settings with numbers instead of vibes. |
| **WS-8 · Make it safe to ship** | C | CI that runs the tests on every PR, a registry-drift check, and the phone UI wired to the new knobs. |

Not in this wave (deliberately): domain-conditioned generation (`docs/PLAN.md §9.4`),
alternative backbones (§9.6), and new datasets (WS-1). Reasons in §5.

---

## 1. Why these three, and why they can run at once

The 2026-07-24 consolidation cost a full session because three agents built the same
feature on three branches and could not see each other's `STATUS.md` claims. This wave is
designed so that **no two agents can touch the same file**:

| File / area | A (WS-6) | B (WS-7) | C (WS-8) |
|---|---|---|---|
| `src/train.py`, `src/pretrain.py`, `src/finetune.py` | ✅ owns | ✋ | ✋ |
| `src/data.py` (add split helper only) | ✅ owns | ✋ | ✋ |
| `src/sample.py`, `src/evaluate.py` | ✋ | ✅ owns | ✋ |
| `src/serve.py`, `src/export_web.py`, `web/` | ✋ | ✋ | ✅ owns |
| `.github/`, `scripts/` | ✋ | ✋ | ✅ owns |
| `src/config.py` | **pre-wired — nobody edits it** (see §2) | | |
| `src/model.py`, `src/train_dual.py` | **frozen this wave** — no one touches them | | |
| `tests/` | new file only: `test_training_quality.py` | new file only: `test_sampling.py` | new file only: `test_repo_hygiene.py` |
| `STATUS.md`, `HANDOFF.md`, `README.md` | own pre-created row/section only | same | same |

Existing test files (`test_data.py`, `test_engine.py`, `test_dual_output.py`) are **read-only
this wave**. If your change breaks one, that is a signal your change broke a contract — fix
your code, don't edit the test.

## 2. Pre-wiring (already done on the prep branch)

`src/config.py` already declares every field this wave needs, with defaults that reproduce
today's behavior byte-for-byte:

- WS-6: `val_fraction=0.0`, `early_stop_patience=0`, `lr_schedule="none"`, `lr_factor=0.5`, `lr_min=0.0`
- WS-7: `top_k=0`, `top_p=1.0`, `repetition_penalty=1.0`, `min_length=2`

So **no agent edits `src/config.py`**. If you genuinely need a field that isn't there, add it
inside your own workstream's comment block and flag it in `STATUS.md` under *Shared-file
touches* — but check first that an existing field doesn't already cover it.

## 3. Contracts nobody may break

1. **Checkpoint dict keys** (HANDOFF §2) — `model_state`, `config`, `vocab`, `training_names`.
   WS-6 may **add** one key, `"val_names"` (a list, possibly empty). Additive only; every
   reader must use `ckpt.get("val_names", [])`. Document the addition in HANDOFF §2.
2. **`sample.generate_one` / `generate_many` signatures** — `src/train.py` and `src/serve.py`
   call these. New parameters go **at the end, keyword-only, defaulting to today's behavior**.
   `forward(x, hidden)` on `CharRNN` stays as-is.
3. **CLI flags** of `train.py` / `sample.py` / `generate.py` — additive only.
4. **One cross-agent contract, one direction:** A writes `val_names` into checkpoints; B reads
   it *if present* to report an honest held-out NLL. B must degrade gracefully when it's
   absent (every checkpoint that exists today lacks it). Neither waits on the other.

## 4. Merge protocol

Branch off `main` once the prep branch is merged (see the prompt for the fallback).

Preferred merge order — **C, then A, then B** — so CI is guarding the repo before the two
code-heavy PRs land. It is a preference, not a dependency: any order works.

Before merging your own PR: `git fetch origin main && git merge origin/main`, resolve, push.
`main` is squash-merge + linear history, 0 required approvals, so you merge your own PR
(HANDOFF §7). If `STATUS.md` conflicts, **keep both rows** — that's never a real conflict.

## 5. Explicitly deferred to wave 3

- **Domain conditioning** (`docs/PLAN.md §9.4`) — one model, `--domain aircraft`. It needs
  `data.py` + `model.py` + `train.py` + `sample.py` changed together, which is precisely the
  file set this wave splits across three agents. Run it alone, after this wave, on a model
  that finally has a validation set to prove it helps.
- **GRU / char-Transformer backbone** (§9.6) — a comparison is only meaningful once WS-6's
  held-out loss and WS-7's sweep exist to compare *with*. Sequencing, not scope-cutting.
- **WS-1 more datasets** — always open, zero collision risk, but it competes for the same
  three agent slots and the quality levers above are worth more per token right now.

## 6. Definition of done (in addition to HANDOFF §8)

1. Your new test file passes, and the three pre-existing test files still pass untouched.
2. Behavior with default settings is **unchanged** — prove it in your report, don't assert it.
3. Your `STATUS.md` row is updated with a real result, not "done".
4. Docs updated: your HANDOFF §3 workstream entry, your §7 branch row, and the README
   line for your feature if it changes what the owner types.
5. A PR into `main` is open (HANDOFF §8.6).
6. Report back to the owner in ≤10 lines: what changed, the numbers you measured, what you
   deliberately did *not* do, and the AGENTS.md §7 consolidation check.

## 7. The per-agent briefs

- Agent A → [`docs/upgrade/AGENT-A.md`](upgrade/AGENT-A.md) — WS-6, branch `claude/ws6-training-quality`
- Agent B → [`docs/upgrade/AGENT-B.md`](upgrade/AGENT-B.md) — WS-7, branch `claude/ws7-decoding-quality`
- Agent C → [`docs/upgrade/AGENT-C.md`](upgrade/AGENT-C.md) — WS-8, branch `claude/ws8-ci-and-hygiene`

## 8. Recommended model & effort per lane

| Agent | Model | Effort | Why |
|---|---|---|---|
| A | Opus 5 | high | Touches the training loop every other stage depends on; the failure mode (silent data leakage between train and val, or a "best weights" restore that quietly loads the wrong epoch) is invisible in a smoke test. Needs the most careful reasoning and the most honest reporting. |
| B | Sonnet 5 | high | Well-specified, self-contained numerics. Nucleus/top-k sampling is standard and the file is 173 lines — but the defaults-unchanged proof and the sweep interpretation need care. |
| C | Sonnet 5 | medium | Mostly plumbing: YAML, a linting script, UI wiring. Low algorithmic risk, high file count. Don't send Haiku — CI YAML that can't be run locally needs judgment about what will actually pass. |

After all three land, run **one consolidation session** (AGENTS.md §8).

---

## 9. Outcome (filled in by the consolidation, 2026-07-29)

All three lanes shipped and merged the same day, in the preferred order (C → B → A), with
**zero file collisions and zero duplicated work**. The pre-wiring in §2 is what did it: no
agent ever needed to open `src/config.py`. Reuse that pattern for the next parallel wave.

What each lane delivered, and what it measured, is recorded in `docs/PLAN.md §12` and
summarized in `HANDOFF.md §3`. The one-line version: **both code lanes independently found
that the datasets are too small for the model reading them**, which is why §5's deferred
items (conditioning, alternative backbones) stay deferred and **WS-1 — more and bigger
datasets — is the recommended wave 3**.

Consolidation found one real defect and fixed it: WS-8's secret scanner flagged its own
planted test fixtures, so the CI hygiene job was red on `main` from the moment it landed.
`scripts/check_repo.py` now skips RFC 2606 documentation domains and its own fixture file.

Two decisions were left to the owner rather than made here (both in STATUS.md → Known
issues): whether to seed model initialization in `fit()` (it would change the default
training trajectory for every existing command), and whether to wire WS-7's decoding knobs
into the phone UI (WS-7's own measurements suggest they'd make output worse at this scale).

A third is worth naming: CI is **advisory** until branch protection requires it —
Settings → Branches → `main` → Require status checks → select the `ci` checks.
