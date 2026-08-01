# scripts/

Helper scripts for repo/GitHub administration (not part of the name-generation engine).
All of them are stdlib-only and torch-free, so CI's `hygiene` job runs them in seconds
without installing anything.

## check_repo.py

Bookkeeping and safety checks: registry drift, committed model weights, secrets/PII.

```bash
python scripts/check_repo.py            # weights/secrets/PII block; drift is advisory
python scripts/check_repo.py --strict    # drift blocks too
```

**Exit codes.** A committed checkpoint, an API key or an email address always fails the
build. Registry drift — a `data/` file with no row in the HANDOFF §4 registry or the
README catalog — is *advisory* by default and prints without failing, because during a
data wave the catalog tables are deliberately merged once, at the end, by the
orchestrating session. The finding lists every missing row pre-filled from the dataset's
`.meta.json` sidecar, ready to paste into either table. Run `--strict` after
consolidation to confirm the merge was complete.

**Suppressing a known-fake match.** The scanner used to flag its own test fixtures,
which kept the `hygiene` job red. Two narrow mechanisms exist, and nothing broader:

1. `# check_repo: allow` on the same line as the fixture. Scope is that one line, so a
   real secret elsewhere in the same file is still caught.
2. `KNOWN_FIXTURE_MATCHES` in the script, keyed on `(repo-relative path, exact string)`,
   for the one unowned file (`STATUS.md`) that cannot carry a pragma.

There is no file-level, directory-level or domain-level exemption. In particular
`tests/` is not skipped, and `@example.com` is not blanket-allowed — the hygiene suite
has tests pinning both down.

## check_data.py

Validates every `data/*.txt` before it reaches a training run.

```bash
python scripts/check_data.py             # summary table + findings
python scripts/check_data.py --strict    # warnings (missing sidecars) become failures
python scripts/check_data.py --verbose   # list every grandfathered offence
```

Checks: the character set `^[A-Za-z0-9][A-Za-z0-9 -]*[A-Za-z0-9]$` (letters, digits,
spaces, hyphens — digits are in because real names use them, `ATR 42`, `RAV4`);
case-insensitive duplicates; blank and one-character lines; and the
`data/<stem>.meta.json` sidecar (required keys, `name` matches the stem, `count` matches
the file's real line count).

A **missing** sidecar is a warning — lanes are still landing them. A sidecar whose
`count` disagrees with the file is an **error**: a stale count misreports dataset size
in the README catalog and in every evaluation run.

`KNOWN_NONCONFORMING` grandfathers a named list of pre-existing files whose findings are
demoted to warnings. It is a record of known debt, not an amnesty: `--strict` still
fails on them, `--verbose` still lists every offending line, and a new dataset gets no
such treatment. Do not add your own file to it — normalise the data instead.

## setup-branch-protection.sh

Stamps the **correct** branch protection on a repo (or every repo you own),
idempotently. Use it to (re)build protection after a ruleset is deleted or
misconfigured.

### The lesson it encodes

A previous ruleset targeted **all branches** with *require linear history*. On a
brand-new branch's first push, its entire ancestry counts as "new commits", so a
single legacy merge commit in `main`'s history (`bed4b06`) made GitHub reject
**every** feature-branch push with *"must not contain merge commits."* That
commit couldn't be removed (force-push to `main` was also blocked), so every
agent paid a manual-rebase tax on every push, indefinitely.

Root cause was **scope**, not the commit. The fix — baked into this script — is
to protect the **default branch only** (`~DEFAULT_BRANCH`). Feature branches are
then never evaluated, so an old merge commit deep in `main`'s history is inert,
and agents push / force-push / delete their own branches freely. `main` keeps
full protection; *require linear history* only inspects newly-introduced commits,
and squash-merges never add a merge commit, so the legacy commit stays
grandfathered.

### What it sets

- Ruleset `default-branch-protection` on `~DEFAULT_BRANCH` (enforcement active):
  require PR (0 approvals → self-merge OK), require linear history, block force
  pushes, block deletion.
- Repo settings: squash-merge only, auto-delete head branches on merge.

### Usage

```bash
# DRY RUN (default) — prints what it would do, changes nothing
./scripts/setup-branch-protection.sh --owner <owner> <repo>
./scripts/setup-branch-protection.sh --all            # preview across all your repos

# APPLY
./scripts/setup-branch-protection.sh --apply <repo>
./scripts/setup-branch-protection.sh --apply --all    # every non-archived repo
```

Requires `gh` (authenticated, admin on the targets) and `jq`. Re-running updates
the existing ruleset in place rather than duplicating it.
