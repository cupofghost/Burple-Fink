# The wave-2 agent prompt

Paste this into three separate sessions. **Change only the first line** (`AGENT: A` → `B` → `C`).

```text
AGENT: A          ← set to A, B, or C. Nothing else in this prompt changes.

You are one of three agents upgrading the Burple-Fink repo in parallel. The letter above
fixes your lane; the other two agents are working at the same time in files you must not open.

1. SET UP YOUR BRANCH.
   A -> claude/ws6-training-quality
   B -> claude/ws7-decoding-quality
   C -> claude/ws8-ci-and-hygiene

   git fetch origin
   BASE=origin/main
   git cat-file -e origin/main:docs/UPGRADE_PLAN.md 2>/dev/null || BASE=origin/claude/burple-fink-upgrade-plan-m7ndof
   git checkout -B <your branch from the list above> "$BASE"

   (The wave-2 prep may not be merged to main yet; the check above bases you off the prep
   branch if not. Say which base you used in your final report.)

2. READ, IN THIS ORDER, AND NOTHING ELSE: AGENTS.md, STATUS.md, docs/UPGRADE_PLAN.md,
   docs/upgrade/AGENT-<your letter>.md. The brief is your spec; the plan is the ground rules.
   Then read only the source files your brief says you own. Do not survey the repo.

3. CLAIM YOUR WORK. Your row is already reserved in the STATUS.md "Active work" table —
   replace its "Reserved by ..." signature with your own real one:
   Signed: <program> | <model you are actually running> | <effort>

4. BUILD EXACTLY WHAT YOUR BRIEF LISTS. Hard rules:
   - Never edit a file your brief doesn't list as yours. src/config.py is already pre-wired
     with your fields — do not touch it. src/model.py and src/train_dual.py are frozen.
   - New tests go in your own new test file. tests/test_data.py, tests/test_engine.py and
     tests/test_dual_output.py are READ-ONLY: if one breaks, your change broke a contract.
   - Every new option defaults to today's behavior. Prove it with a seeded test; don't assert it.
   - Treat main as working. Don't re-verify, re-read, or refactor other agents' code (AGENTS.md §4).
   - Install torch from the default PyPI index; download.pytorch.org is blocked here.

5. FINISH. Run your own tests; run the existing suite once as well if your change touches
   code it covers (it does for A and B). Update STATUS.md, HANDOFF.md §3/§7, and the README
   per your brief's definition of done. Commit with your signature line, then:
   git push -u origin <your branch>
   Open a PR into main (HANDOFF §8.6 — a task with no PR is not done). Before merging:
   git fetch origin main && git merge origin/main, resolve, push, then merge your own PR
   (main is squash-only, linear history, 0 required approvals). If STATUS.md conflicts with
   another lane, KEEP BOTH ROWS — that is never a real conflict.

6. REPORT BACK in 10 lines or fewer: what changed, the numbers you actually measured, what
   you deliberately did not do, and the AGENTS.md §7 consolidation check.

If your brief and the code disagree, say so and use your judgment — but do not silently
expand scope into another agent's files. If you finish early, stop; do not pick up more work.
```
