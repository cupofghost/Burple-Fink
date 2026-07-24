#!/usr/bin/env bash
#
# setup-branch-protection.sh — stamp the *correct* branch protection on a repo (or all repos).
#
# Why this exists
# ---------------
# An earlier ruleset targeted **all branches** with "require linear history".
# Because a brand-new branch's entire ancestry counts as "new commits" on its
# first push, one legacy merge commit baked into main's history (bed4b06) made
# GitHub reject *every* feature-branch push with "must not contain merge
# commits" — a manual-rebase tax paid on every push, forever, with no way to
# excise the commit (force-push to main was also blocked).
#
# The fix, captured here: protect the **default branch only** (~DEFAULT_BRANCH).
# Feature branches are then never evaluated, so an old merge commit in main's
# history is harmless, and agents can push / force-push / delete their own
# branches freely. main keeps full protection; "require linear history" only
# ever inspects newly-introduced commits, and squash-merges never introduce a
# merge commit, so the legacy commit stays grandfathered and inert.
#
# What it configures
# ------------------
#   Ruleset "default-branch-protection" on ~DEFAULT_BRANCH, enforcement=active:
#     - pull_request           require a PR before merging, 0 approvals (self-merge OK)
#     - required_linear_history no merge commits added to the default branch
#     - non_fast_forward        block force pushes to the default branch
#     - deletion                block deleting the default branch
#   Repo merge settings:
#     - squash merges only (merge-commit and rebase-merge buttons off)
#     - auto-delete head branches on merge
#
# Idempotent: re-running updates the existing ruleset in place (matched by name)
# instead of creating duplicates.
#
# Requirements: gh (authenticated with admin on the target repos) and jq.
#
# Usage:
#   ./setup-branch-protection.sh                      # DRY RUN, current repo dir's owner, all repos
#   ./setup-branch-protection.sh --owner cupofghost Burple-Fink    # DRY RUN, one repo
#   ./setup-branch-protection.sh --apply --all        # APPLY to every non-archived repo
#   ./setup-branch-protection.sh --apply Burple-Fink  # APPLY to one repo
#
set -euo pipefail

RULESET_NAME="default-branch-protection"
APPLY=false
ALL=false
OWNER=""
REPOS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) APPLY=true; shift ;;
    --all)   ALL=true; shift ;;
    --owner) OWNER="$2"; shift 2 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *)  REPOS+=("$1"); shift ;;
  esac
done

command -v gh >/dev/null || { echo "error: gh CLI not found" >&2; exit 1; }
command -v jq >/dev/null || { echo "error: jq not found" >&2; exit 1; }

[[ -n "$OWNER" ]] || OWNER="$(gh api user --jq .login)"

if [[ ${#REPOS[@]} -eq 0 ]]; then
  # No explicit repos. Applying to *every* repo is only allowed with --all, so a
  # bare `--apply` can't fan out by accident.
  if $APPLY && ! $ALL; then
    echo "error: --apply with no repos requires --all (refusing to touch every repo implicitly)" >&2
    exit 2
  fi
  mapfile -t REPOS < <(gh repo list "$OWNER" --limit 200 --json name,isArchived \
    --jq '.[] | select(.isArchived | not) | .name')
fi

$APPLY || echo ">>> DRY RUN — no changes will be made. Re-run with --apply to execute. <<<"
echo ">>> owner=$OWNER  repos=${#REPOS[@]}"

# The ruleset body (identical for create and update).
ruleset_body() {
  jq -n --arg name "$RULESET_NAME" '{
    name: $name,
    target: "branch",
    enforcement: "active",
    conditions: { ref_name: { include: ["~DEFAULT_BRANCH"], exclude: [] } },
    rules: [
      { type: "pull_request", parameters: {
          required_approving_review_count: 0,
          dismiss_stale_reviews_on_push: false,
          require_code_owner_review: false,
          require_last_push_approval: false,
          required_review_thread_resolution: false } },
      { type: "required_linear_history" },
      { type: "non_fast_forward" },
      { type: "deletion" }
    ]
  }'
}

merge_settings() {
  jq -n '{
    allow_squash_merge: true,
    allow_merge_commit: false,
    allow_rebase_merge: false,
    delete_branch_on_merge: true
  }'
}

for REPO in "${REPOS[@]}"; do
  echo "=== $OWNER/$REPO ==="
  existing_id="$(gh api "repos/$OWNER/$REPO/rulesets" --jq \
    ".[] | select(.name==\"$RULESET_NAME\") | .id" 2>/dev/null | head -1 || true)"

  if [[ -n "$existing_id" ]]; then
    echo "  ruleset '$RULESET_NAME' exists (id=$existing_id) -> update"
    if $APPLY; then
      ruleset_body | gh api -X PUT "repos/$OWNER/$REPO/rulesets/$existing_id" --input - >/dev/null
      echo "  ruleset updated"
    fi
  else
    echo "  ruleset '$RULESET_NAME' missing -> create"
    if $APPLY; then
      ruleset_body | gh api -X POST "repos/$OWNER/$REPO/rulesets" --input - >/dev/null
      echo "  ruleset created"
    fi
  fi

  echo "  merge settings -> squash-only + auto-delete head branches"
  if $APPLY; then
    merge_settings | gh api -X PATCH "repos/$OWNER/$REPO" --input - >/dev/null
    echo "  merge settings applied"
  fi
done

$APPLY && echo ">>> done." || echo ">>> dry run complete — nothing changed."
