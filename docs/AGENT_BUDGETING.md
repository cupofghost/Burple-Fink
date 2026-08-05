# Agent budgeting — surviving the rolling usage window (v1 — 2026-08-05)

Waves 2 and 3 lost real work to usage limits. This file is the durable version of that
lesson. It is operational guidance for *this repo's* orchestration pattern, not a general
essay about tokens.

Everything below is marked **[fact]** (read directly from Anthropic docs on 2026-08-05),
**[reported]** (from support.claude.com via search extracts — those pages returned 403 to
direct fetch, so treat as slightly weaker), or **[inference]** (ours). Nothing here is a
guessed number. Where we do not know, section 9 says so.

---

## 1. The rules, if you read nothing else

1. Write each finished artifact to disk **before** starting the next one.
2. Write a first complete draft of your deliverable **early**, then improve it in place.
3. Never accumulate a large deliverable in context. Many small permanent files beat one
   big final write.
4. Pick model and effort at session start and do not change them mid-task.
5. Bulk generation runs on Sonnet or Haiku. Opus is for architecture, measurement design,
   and hard debugging.
6. Four concurrent lanes, staggered. Not nine.
7. Check `/usage` before any long uninterrupted run.

---

## 2. How the limits actually work

### 2.1 Two windows, both live at once
Subscription plans meter against a **rolling 5-hour window** and a **weekly window**
simultaneously. **[fact]** — the Claude Code statusline exposes both as
[`rate_limits.five_hour` and `rate_limits.seven_day`](https://code.claude.com/docs/en/statusline#rate-limit-usage),
and [manage costs](https://code.claude.com/docs/en/costs) describes the allowance as
resetting "on a rolling five-hour window and a weekly window".

The weekly window resets at a **fixed day and time assigned to the account**, not 7 days
after you started. **[reported]** This matches the `resets Mon 12:00am` example in the
[usage limit errors](https://code.claude.com/docs/en/errors#youve-hit-your-session-limit).

A single burst can exhaust the **weekly** allowance before the 5-hour one resets. **[fact]**
That is the failure mode that ends a whole wave, not just an afternoon.

### 2.2 Exhaustion is a hard stop
> "Claude Code **blocks further requests** until the reset time shown in the message."
> — [usage limit errors](https://code.claude.com/docs/en/errors#youve-hit-your-session-limit) **[fact]**

No degrade. No queue. No smaller model fallback. The three messages are:

```
You've hit your session limit · resets 3:45pm
You've hit your weekly limit  · resets Mon 12:00am
You've hit your Opus limit    · resets 3:45pm
```

Session and weekly limits are **shared across all models** — `/model` does not rescue you.
The **Opus limit is the exception**: it applies only to Opus requests, so switching to
Sonnet keeps you working. **[fact]** Max plans also carry a separate Opus-only weekly
limit alongside the all-models one. **[reported]**

Plan for a hard stop that arrives without warning, mid-sentence, with whatever is in
context lost.

### 2.3 What draws from the pool
Everything on the account: claude.ai web, desktop, mobile, and Claude Code share one
allowance. **[reported]** Within a session, `/usage` attributes recent usage to skills,
**subagents**, plugins, and individual MCP servers as percentages of your plan usage
**[fact]** — which settles the question directly: **subagents draw from the same budget as
the main session.**

### 2.4 Models
Official guidance: Sonnet is the default and right for the large majority of coding work;
Opus "uses meaningfully more of your quota, so switch to it when you need it rather than
leaving it on by default". **[reported]** The Claude Code docs add: "Reserve Opus for
complex architectural decisions or multi-step reasoning… For simple subagent tasks,
specify `model: haiku`." **[fact]**
([reduce token usage](https://code.claude.com/docs/en/costs#reduce-token-usage))

Anthropic does **not** publish how subscription budget is metered per model. As a rough
*ordering only*, API list prices per million tokens (input/output) are:
[Opus 5 $5/$25, Sonnet 5 $2/$10 (introductory, through 2026-08-31; $3/$15 after),
Haiku 4.5 $1/$5](https://platform.claude.com/docs/en/about-claude/pricing). **[fact for the
prices]** Reading that as a ~5:2:1 burn ratio (~5:3:1 from September) is **[inference]**.
Use it to rank lanes, never to compute a budget.

### 2.5 Subagents multiply cost more than they multiply work
Three things compound:

- Each subagent runs its **own context window** with its own system prompt and its own
  copy of CLAUDE.md and project context. **[fact]**
- Each subagent **starts with a cold cache** and uses the **5-minute** cache TTL, even on
  a subscription where the main conversation gets the 1-hour TTL. **[fact]**
  ([subagents and the cache](https://code.claude.com/docs/en/prompt-caching#subagents-and-the-cache))
- [Agent teams use ~7x the tokens](https://code.claude.com/docs/en/costs#agent-team-token-costs)
  of a standard session when teammates run in plan mode, roughly proportional to team
  size. **[fact]**

So N parallel agents cost meaningfully **more** than the same work done serially, not the
same amount compressed into less wall-clock. **[inference]**

The built-in guards are counts, not budget: **20 concurrent** subagents and **200 spawns
per session** by default
([concurrent subagent limit](https://code.claude.com/docs/en/sub-agents#concurrent-subagent-limit)).
**[fact]** Nothing in Claude Code stops you from spending a window. You have to.

### 2.6 Caching, and what quietly throws it away
A cache hit bills at **0.1x** the input rate; the 5-minute write is 1.25x and the 1-hour
write 2x ([pricing](https://platform.claude.com/docs/en/about-claude/pricing#prompt-caching)).
On a subscription Claude Code requests the **1-hour TTL automatically**
([cache lifetime](https://code.claude.com/docs/en/prompt-caching#cache-lifetime)). **[fact]**

The cache is an **exact prefix match** — a change anywhere early recomputes everything
after it. These throw it away mid-task
([full list](https://code.claude.com/docs/en/prompt-caching#actions-that-invalidate-the-cache)):
switching model, changing `/effort`, enabling fast mode, an MCP server connecting or
disconnecting when its tools sit in the prefix, `/compact`, and upgrading Claude Code.
**[fact]**

Two consequences for this repo:

- **Choose model and effort at session start.** A mid-lane `/model` or `/effort` switch
  re-reads the entire conversation uncached.
- **Worktrees do not share a cache.** Cache scope is effectively one machine and one
  directory, and "that includes worktrees of the same repository"
  ([cache scope](https://code.claude.com/docs/en/prompt-caching#cache-scope)). **[fact]**
  Parallel sessions in the *same* directory do share. Worktree isolation buys safety and
  costs cache.

And the reason long lanes get expensive on their own: Claude Code re-sends the full
conversation every turn, so
["a one-line question in a session that has been open all day still draws usage for the
whole conversation"](https://code.claude.com/docs/en/costs#why-usage-climbs-in-a-long-session).
**[fact]**

---

## 3. Checkpoint early and often — the primary defence

This is the lesson wave 2 and 3 actually paid for. Everything else in this file is
secondary.

1. **Write each finished artifact to disk before starting the next.** Not at the end of
   the lane. When item 3 of 6 is done, it is on disk before item 4 begins.
2. **Write a first complete draft of your deliverable early, then improve it in place.**
   A thin, honest v1 on disk beats a perfect v1 that never gets written. Do not research
   for an hour and write at the end.
3. **Never accumulate a deliverable in context.** If you are holding more than about a
   screenful of generated content that exists nowhere else, stop and write it.
4. **Prefer many small permanent files to one big final write.** `reports/_capacity/run-07.json`
   is recoverable; the same numbers in a transcript are not.
5. **Append, don't rewrite.** For long-running generation, append to the output file as
   you go. A partial file with 700 of 1,139 names is 700 names saved.
6. **Emit a `PROGRESS` line into the artifact itself** — what is done, what is next —
   so a resumer reads state from the file rather than from you.

Sizing rule of thumb: **no more than ~15 minutes of work should exist only in context**
at any moment. **[inference]**

---

## 4. Design work so an interruption is cheap

The wave-3 architecture lane finished 17 training runs and was killed before writing its
results table. It was recovered **only** because the training checkpoints had landed on
disk as a side-effect, and the orchestrator rebuilt `reports/ARCH.md` from them.

The two data lanes held ~2,000 curated names in context, one of them 1,139 pharmaceutical
names, and were killed before writing anything. All of it was lost.

The difference was not effort, luck, or model. It was **durable side-effects**.

Before starting a lane, answer one question: *if this agent dies right now, what is on
disk?* If the answer is "nothing until the end", restructure:

| Shape | Interruption cost | Verdict |
|---|---|---|
| Generate 1,139 names in context, write once at the end | total loss | never |
| Append names to the `.tsv` in batches of 50 | ≤50 names | correct |
| Run 17 experiments, tabulate at the end | recoverable *if* checkpoints persist | fragile — do not rely on it |
| Run 17 experiments, write `_runs/NN.json` after each | one run | correct |
| Research for an hour, then write the doc | total loss | never |
| Draft the doc at 20% research, improve in place | one edit | correct |

Corollary: **make the durable write the cheap part**. If writing the artifact requires
first re-deriving something expensive, the design is wrong.

**And check the lane can write at all.** The wave-4 budgeting lane was dispatched to a
read-only agent type and did all its research before discovering it had no `Write` tool;
the orchestrator had to transcribe the result by hand. Confirm the toolset matches the
deliverable before dispatch — a lane that cannot produce durable output is the most
expensive shape there is.

---

## 5. How many parallel lanes

**Observed in this repo:** nine parallel Opus agents exhausted a 5-hour window in about
twenty minutes. Wave 4 runs **four** concurrent lanes (`docs/WAVE4_PLAN.md`).

The trade-off is explicit: **parallelism buys wall-clock and sells survivability.** Nine
lanes finish in a fifth of the time *if they finish*. They do not: they hit the wall
together, and every lane dies at once, at whatever point it happens to be in.

Working rules:

1. **Default to 4 concurrent lanes.** Raise only when every lane is Sonnet/Haiku and
   checkpoints frequently.
2. **Stagger starts by 10–15 minutes.** Staggered lanes hit their expensive phases at
   different times, so a wall clips one or two rather than all of them.
   **[inference]** — no Anthropic doc addresses staggering.
3. **Never start a new lane when the 5-hour window is past ~70%.** Give running lanes the
   remainder to reach their next checkpoint.
4. **Stagger the *survivable* ones last.** If one lane is the wave's critical path, start
   it first with the freshest budget.
5. **Shut lanes down when their work is done.** An idle teammate keeps consuming
   ([agent team costs](https://code.claude.com/docs/en/costs#agent-team-token-costs)).
   **[fact]**

If a lane cannot be made cheaply interruptible, it should run **alone**, not in parallel.

---

## 6. Model selection per lane

Match the model to what the lane's output is *for*, not to how hard it feels.

| Lane type | Model | Why |
|---|---|---|
| Bulk data generation (`data/*.tsv`, name lists, sidecars) | **Sonnet**, Haiku for mechanical expansion | Output is verifiable by inspection and by the training loop. A weaker name is a cheap error; a lost window is not. |
| Mechanical edits, renames, format conversion | **Haiku** | Deterministic work. `model: haiku` in [subagent frontmatter](https://code.claude.com/docs/en/sub-agents#choose-a-model). **[fact]** |
| Running a defined sweep or benchmark | **Sonnet** | The experiment design is the hard part; executing it is not. |
| Designing a measurement, choosing controls, spotting leakage | **Opus** | A wrong control invalidates the whole lane silently. |
| Architecture decisions, cross-cutting refactors, hard debugging | **Opus** | Exactly the case Anthropic names. **[reported]** |
| Interpreting results into a claim the README will carry | **Opus** | This repo's failure mode is confident unverified claims, not slow ones. |

Decision test: **would a subtle error here be caught by the next step, or would it be
believed?** Caught → Sonnet or Haiku. Believed → Opus.

Two mechanics:
- Set the model in the **subagent's frontmatter** (`model: sonnet`), not by switching
  mid-session. Mid-session switches invalidate the cache. **[fact]**
- Set effort at session start too, for the same reason. **[fact]**

---

## 7. Resumption

### 7.1 The mechanics
- `claude --continue` resumes the most recent session in the directory; `claude --resume`
  opens the picker; `claude --resume <name|session-id>` goes straight there. **[fact]**
- Sessions started with `claude -p` or the Agent SDK **do not appear in the picker** but
  resume by ID — **run it from the directory the session started in**. **[fact]**
- Transcripts live at `~/.claude/projects/<project>/<session-id>.jsonl`, 30-day retention.
  **[fact]**
- Restored: full history including tool calls and results, model, agent, permission mode.
  **Not** restored: `--mcp-config`, `--settings`, `--plugin-dir`, `--fallback-model`,
  `--add-dir`. Pass them again. **[fact]**
- On Pro/Max, resuming a session idle >~1h and >100k tokens offers **resume from summary**.
  The cache has expired either way, so that first request reprocesses the full history
  once regardless of which option you pick. **[fact]**
  ([sessions](https://code.claude.com/docs/en/sessions#resume-from-a-summary))
- **Resuming after a Claude Code upgrade re-reads the whole conversation uncached** — the
  first turn back into a long lane can be the most expensive request of the day. **[fact]**
- **Subagents:** `general-purpose` and custom subagents can be resumed with their agent ID.
  The built-in **Explore and Plan agents are one-shot and return no agent ID, so they
  cannot be resumed.** **[fact]**
  ([resume subagents](https://code.claude.com/docs/en/sub-agents#resume-subagents))
  Use `general-purpose` or a custom subagent for any lane whose work must be continuable.

### 7.2 What the resume message must contain
Resuming replays the transcript, so the agent knows what *it* did. It does **not** know
what changed on disk while it was dead, and it will happily redo expensive work. Every
resume message states all four:

```
Resuming lane WS-XX. You were stopped by a usage limit, not by an error.

ON DISK AND COMMITTED since you stopped:
  - <path> — <one line>
COMMITTED BY OTHER LANES (do not re-verify, per AGENTS.md §4):
  - <path> — <one line>
UNCOMMITTED WORKING-TREE CHANGES:
  - <path> — <one line>

ALREADY DONE — do not re-run:
  - <expensive step> → output at <path>

STILL OWED, in order:
  1. <deliverable>
  2. <deliverable>

Start by reading <paths>, then continue at step <N>. Write each artifact to disk
before starting the next.
```

The "do not re-run" block is the one that saves the most budget. A resumed agent that
re-runs a completed sweep spends the new window reproducing the old one.

---

## 8. Knowing where you stand

- **`/usage`** — plan usage bars, plus a breakdown attributing recent usage to skills,
  **subagents**, plugins, and MCP servers as a percentage of total, flagging any behaviour
  (long context, cache misses) that accounts for ≥10%. `d`/`w` toggles 24h/7d. **[fact]**
  Caveat, from the docs: figures are approximate and computed from **local session history
  on this machine**, so other devices and claude.ai are not included.
- **`/status`**, **`/context`**, **`/cost`** (alias of `/usage`). **[fact]**
- **Statusline**, for continuous visibility — the most useful thing to set up before a
  wave. Fields: `rate_limits.five_hour.used_percentage`, `.resets_at`, and the same under
  `seven_day`. Present only for Pro/Max after the first API response in a session, and may
  be absent — handle that
  ([rate limit usage](https://code.claude.com/docs/en/statusline#rate-limit-usage)). **[fact]**
- **`/usage-credits`** buys usage past the limit. Note that drawing on credits drops the
  cache TTL from one hour to five minutes, which makes everything after it more expensive.
  **[fact]**

Orchestrator habit: record the 5-hour and weekly percentages in `STATUS.md` when a wave
starts and when it stops. Two numbers, and the next wave can be planned instead of guessed.

---

## 9. What we could not verify

Stated plainly, because a confident wrong number here is worse than an admitted gap.

1. **Whether the 5-hour window is anchored to your first message and then expires, or
   slides continuously.** Anthropic calls it "rolling" and says it "resets every five
   hours", and the error shows a concrete reset time — compatible with both readings.
   Community sources assert both and contradict each other. **Do not plan around either.**
   Plan around the reset time `/usage` reports.
2. **The size of the allowance**, in messages, tokens, or hours, for Pro or Max. Anthropic
   states there is no fixed message count. Circulating community figures are unverified;
   ignore them.
3. **What happens to an in-flight background subagent when the window is exhausted
   mid-run.** Docs say requests are blocked; nothing addresses partial completions. Assume
   the worst and checkpoint.
4. **Whether subscription budget is metered as a dollar-equivalent of API list prices.**
   The 5:2:1 ordering in §2.4 is our inference from list prices, not a documented formula.
5. **Publication dates for the support.claude.com articles.** Those pages returned 403 to
   direct fetch on 2026-08-05; their claims here are marked **[reported]**.

**Superseded-source warning:** weekly limits post-date the original 5-hour-only scheme.
Any source that describes only a 5-hour limit is out of date and should not be trusted on
anything else either.

---

## 10. Sources

Fetched 2026-08-05.

- [Manage costs effectively](https://code.claude.com/docs/en/costs) — `/usage`, subagent
  attribution, agent team costs, why usage climbs in a long session
- [Usage limit errors](https://code.claude.com/docs/en/errors#youve-hit-your-session-limit)
  — the hard stop, the three messages, model-sharing behaviour
- [How Claude Code uses prompt caching](https://code.claude.com/docs/en/prompt-caching) —
  invalidation, TTL, cache scope, subagents and the cache
- [Create custom subagents](https://code.claude.com/docs/en/sub-agents) — concurrency and
  spawn limits, `model:` frontmatter, resuming subagents
- [Manage sessions](https://code.claude.com/docs/en/sessions) — resume, what is and is not
  restored, resume from summary, transcript paths
- [Customize your status line](https://code.claude.com/docs/en/statusline#rate-limit-usage)
  — `rate_limits` fields
- [Pricing](https://platform.claude.com/docs/en/about-claude/pricing) — per-model rates,
  cache multipliers
- [Models, usage, and limits in Claude Code](https://support.claude.com/en/articles/14552983-models-usage-and-limits-in-claude-code)
  and [How do usage and length limits work?](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work)
  — **[reported]**, 403 to direct fetch

Recheck this file at the start of each wave. Anthropic has changed these policies more
than once.
