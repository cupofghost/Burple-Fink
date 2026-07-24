# scripts/

Helper scripts for repo/GitHub administration (not part of the name-generation engine).

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
