# Agent C — WS-8 · Make it safe to ship

Branch: `claude/ws8-ci-and-hygiene` · Suggested model/effort: **Sonnet 5 | medium**
Read first: `AGENTS.md`, `STATUS.md`, `docs/UPGRADE_PLAN.md`, then only the files below.

## The problem you are fixing

There is no CI. `tests/` exists and passes, but nothing runs it — a broken `main` is
discovered by the next agent, at the cost of a whole session. HANDOFF §7 states plainly:
"No CI/status checks are configured." Meanwhile this repo's actual failure mode is
bookkeeping drift: on 2026-07-24 three agents duplicated a workstream, and the dataset
registry (`HANDOFF.md §4`) and README catalog have to be hand-synced with `data/` on every
single dataset PR. Both are mechanically checkable.

You are also the only agent touching what the owner actually uses: the phone UI.

## Deliverables

1. **`.github/workflows/ci.yml`** — on `push` and `pull_request`:
   - Python 3.11, `pip install -r requirements.txt` (CPU torch from the **default PyPI
     index** — `download.pytorch.org` is blocked), with pip caching so reruns are quick;
   - `python -m unittest discover -s tests -v`;
   - a smoke train + sample (e.g. `--epochs 20` on `data/car_manufacturers.txt`, then
     `python -m src.sample --checkpoint … --num 5`) so the CLI contract is exercised, not
     just the library;
   - a fast job that runs `scripts/check_repo.py` **without** installing torch, so
     bookkeeping errors fail in seconds instead of after a torch install.
   - Keep the whole thing under ~10 minutes.

2. **`scripts/check_repo.py`** — the hygiene checks, importable as functions and runnable as
   a CLI (`python scripts/check_repo.py`), stdlib only:
   - **registry drift:** every `data/*.txt` and `data/*.tsv` appears in the `HANDOFF.md §4`
     table *and* the README catalog, and every row in those tables points at a file that
     exists (`data/shared_vocab.json` is not a dataset — exclude it);
   - **no committed weights:** no `*.pt` / `*.pth` tracked by git;
   - **no secrets or PII** (AGENTS.md §3): scan tracked text files for email addresses and
     obvious key patterns (`sk-…`, `ghp_…`, `AKIA…`, `-----BEGIN … PRIVATE KEY-----`).
     Report findings — never auto-delete anything, per AGENTS.md §3.
   - Exit non-zero with a message that says which file and which table are out of sync.
     The owner is not a coder: the failure message must say what to do, not just what broke.

3. **`tests/test_repo_hygiene.py`** (new file — do not edit the existing test files):
   unit-test the check functions against fixtures (a fake data dir, a fake registry table),
   including the drift-detected and clean cases. Must run **without torch installed**.

4. **The phone UI catches up** (`src/serve.py`, `web/app_template.html`):
   - a `/api/health` endpoint returning the loaded checkpoints and their labels;
   - real error responses: today a bad request or a failed generate is opaque — return JSON
     with a message the UI can display, and show it in the UI;
   - wire the UI to **whatever decoding knobs exist on `main` when you rebase**. Agent B
     (WS-7) is adding `top_k` / `top_p` / `repetition_penalty` to `sample.generate_many`. If
     B's PR has landed by the time you get here, add the controls; if not, ship without them
     and say so in your PR — do **not** import or guess at an API that isn't on `main` yet,
     and do not edit `src/sample.py` yourself.
   - **Do not touch the JavaScript LSTM forward pass** in `web/burple-fink.html` or the
     verification logic in `src/export_web.py`. That code is verified against PyTorch to
     5e-3 and is not worth the risk this wave. `web/burple-fink.html` is a build artifact —
     regenerate it only if your template change requires it, and say so in the PR.

5. **Honest note on enforcement.** Branch protection currently requires **no** status checks,
   and HANDOFF §7 tells agents not to change repo settings. So your CI is advisory until the
   owner flips one switch. End your report with the exact setting to change ("Settings →
   Branches → main → Require status checks → select `ci`") and let them decide.

## Rules

- **Files you own:** `.github/**`, `scripts/**`, `src/serve.py`, `src/export_web.py` (only
  if the UI wiring truly requires it), `web/app_template.html`, `web/README.md`,
  `tests/test_repo_hygiene.py`, your `STATUS.md` row, your `HANDOFF.md` §3/§7 entries, and
  the README serving section.
- **Do not touch:** `src/config.py`, `src/train.py`, `src/sample.py`, `src/evaluate.py`,
  `src/data.py`, `src/model.py`, `src/pretrain.py`, `src/finetune.py`, `src/train_dual.py`,
  `data/**`, and the three existing test files.
- If `scripts/check_repo.py` finds pre-existing drift in the registry tables, **fix the
  tables** (that's bookkeeping, and it's yours) but do not touch dataset files themselves.
  If it finds a secret or PII, stop and flag it to the owner — AGENTS.md §3 is explicit that
  removing it properly requires cleaning git history.
- Sign every commit and `STATUS.md` entry: `Signed: Claude Code | Sonnet 5 | medium`.
