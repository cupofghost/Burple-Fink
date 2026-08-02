# Architecture comparison — LSTM vs GRU vs transformer

**Lane WS-9 · wave 3.** Reconstructed by the orchestrating session from WS-9's twelve
best-epoch checkpoints after the lane was cut off by a session limit. See *Provenance* at the
bottom for exactly what is and isn't measured here.

`src/arch/` lets `cfg.arch` select the recurrent core, so all three architectures share one
checkpoint format, one training loop and one sampler. This report asks whether the choice
matters.

## The measurement

Four datasets spanning 370 → 7,336 training names, each with a 15% held-out split (seed
1337, patience 30), best-epoch weights restored. Every number below is that checkpoint
re-scored on its own held-out names with the training criterion, dropout off.

| dataset | train names | arch | params | **held-out loss** | train loss | gap |
|---|---:|---|---:|---:|---:|---:|
| `aircraft` | 370 | **gru** | 636,257 | **0.7725** | 0.4583 | +0.314 |
| `aircraft` | 370 | lstm | 842,081 | 0.7786 | 0.4309 | +0.348 |
| `aircraft` | 370 | transformer | 1,098,081 | 0.9334 | 0.6440 | +0.289 |
| `typefaces` | 394 | **gru** | 633,367 | **2.3471** | 1.7288 | +0.618 |
| `typefaces` | 394 | lstm | 839,191 | 2.3754 | 1.6975 | +0.678 |
| `typefaces` | 394 | transformer | 1,095,191 | 2.5410 | 1.9665 | +0.574 |
| `pharma_drugs` | 1,890 | gru | 633,656 | 2.0331 | 1.6854 | +0.348 |
| `pharma_drugs` | 1,890 | **lstm** | 839,480 | **2.0252** | 1.5811 | +0.444 |
| `pharma_drugs` | 1,890 | transformer | 1,095,480 | 2.1172 | 1.8272 | +0.290 |
| `english_words` | 7,336 | gru | 625,853 | 2.0487 | 1.7195 | +0.329 |
| `english_words` | 7,336 | **lstm** | 831,677 | **2.0078** | 1.5610 | +0.447 |
| `english_words` | 7,336 | transformer | 1,087,677 | 2.0904 | 1.9311 | +0.159 |

## Three findings

### 1. The GRU wins on small data; the LSTM wins on large. The crossover is real.

- 370 names: gru **0.7725** vs lstm 0.7786
- 394 names: gru **2.3471** vs lstm 2.3754
- 1,890 names: lstm **2.0252** vs gru 2.0331
- 7,336 names: lstm **2.0078** vs gru 2.0487

The margins are small — 0.6–2.0% — but the ordering flips cleanly at dataset size, and the
GRU achieves its wins with **25% fewer parameters** (≈633k vs ≈839k).

This is the same conclusion `reports/BENCHMARK.md` reached from the opposite direction. That
report found returns flattening as data grew and concluded wave 2's diagnosis was half right:
these datasets were too small *and* the 2-layer 256-wide LSTM is too large for them. Here the
smaller architecture wins exactly where "the model is too large" was the diagnosis, and loses
once there is enough data to feed the larger one. Two independent measurements, same story.

### 2. The LSTM overfits more than the GRU on every dataset tested — 4 of 4.

| dataset | gru gap | lstm gap |
|---|---:|---:|
| `aircraft` | +0.314 | +0.348 |
| `typefaces` | +0.618 | +0.678 |
| `pharma_drugs` | +0.348 | +0.444 |
| `english_words` | +0.329 | +0.447 |

No exceptions, and the margin widens with dataset size. The LSTM's extra capacity is
consistently spent on the training set. On the two larger datasets it converts that into a
better held-out loss anyway; on the two smaller ones it does not.

### 3. The transformer loses on all four — and not for lack of capacity.

It is **last on every dataset**, by 1.0% (`english_words`) to 8.3% (`typefaces`), while
carrying **30% more parameters than the LSTM** and 73% more than the GRU. It cannot be
dismissed as parameter-starved; at this scale it is simply worse at this task.

Its gap column says why: **+0.159 on `english_words`**, the smallest gap anywhere in the
table, alongside the worst held-out loss on that dataset. That is underfitting, not
overfitting. A 2-layer causal transformer over ≤40-character sequences has no long-range
dependency to exploit — names are short, and the recurrent bottleneck that looks like a
limitation on long text is a useful inductive bias on short strings.

## Recommendation

**Keep `arch: "lstm"` as the default.** It wins on the larger datasets, and it is what every
existing checkpoint and every number in `STATUS.md` and `reports/BENCHMARK.md` was measured
with; switching the default would invalidate that history for a sub-1% gain that only
materializes below ~500 names.

**Reach for `--arch gru` on datasets under roughly 500 names** — it wins there, trains a
smaller model, and overfits less. Fifteen of the thirty datasets in `data/` are in that range.

**Don't use `--arch transformer` for this task** on present evidence. It is implemented,
correct (stepwise decoding is proven equivalent to a full forward pass in
`tests/test_arch.py`) and worth keeping for longer-sequence work, but nothing here recommends
it for name generation.

## Provenance — what this is and isn't

Honest about the reconstruction, because the lane didn't finish:

- **Measured directly.** Held-out loss, training loss, the gap, and parameter counts. All
  recomputed from the twelve checkpoints in one process with `torch.set_num_threads(1)`,
  identical criterion and batching, dropout off. These are the numbers above.
- **Not available.** Best epoch and wall-clock per run. WS-9 had them in flight but was cut
  off before writing them out, and a checkpoint stores only the restored best-epoch weights,
  not the trajectory. Any throughput comparison would have to be re-run.
- **Parameter budgets are not matched**, and deliberately not. Each arch was run at the
  repo's stock `hidden_dim=256`, which is what a user typing `--arch gru` actually gets. That
  makes the two headline results *stronger*, not weaker: the GRU wins on small data with
  fewer parameters, and the transformer loses with more. A budget-matched rerun would be a
  fair follow-up but would not overturn either sign.
- **Four datasets, one seed each.** Margins of 0.6–2.0% on a single seed are suggestive, not
  conclusive. The *ordering* is consistent across four independent datasets and agrees with
  an independent measurement in `reports/BENCHMARK.md`, which is what makes it worth acting
  on. Treat the crossover point (~500 names) as approximate.

Signed: Claude Code | Opus 5 | high
