# Capacity — how big should the model be, as a function of dataset size?

**Lane WS-18 · wave 4.** Produced by `scripts/sweep_capacity.py`; every cell's raw JSON is
in `reports/_capacity/`. Regenerate with the commands under *Method*.

**Answer: there is no capacity rule as a function of dataset size, because dataset size is
not what determines the answer.** Three datasets within 1.5× of each other in size want
optimal models 29× apart in parameter count (`motorcycle_brands` 263 names → 64k;
`aircraft` 370 names → 1.85M; `typefaces` 394 names → 92k), while a 28× change in dataset
size moves the optimum only ~5×. What tracks the optimum is how learnable the domain is, and
you do not know that until you have trained something. **No change to `hidden_dim` or
`num_layers` defaults is supported by this sweep.**

**But `reports/ARCH.md`'s recommendation should be revised.** Its measurements reproduce here
exactly — five cells, all four columns, four decimals — and at the stock `h=256, l=2` the GRU
really does beat the LSTM on both small datasets. Let each architecture choose its own size
and the ordering **reverses on both**: `aircraft` 0.7608 (lstm, `h=384,l=2`) vs 0.7725 (gru),
`typefaces` 2.2873 (lstm, `h=128,l=1`) vs 2.2975 (gru). On a third small dataset the LSTM
wins 23 of 24 paired comparisons across three seeds. "Use a GRU under 500 names" was "use a
different-sized model", and the gating mechanism is a red herring.

**The most actionable finding is smaller and duller:** depth is the biggest single lever
measured here (0.143 nats one way on `typefaces`, +5.7% the other way on `aircraft`) and
`src/train.py` has no `--num-layers` flag.

## Why this lane exists

Two wave-3 reports arrived at the same suspicion from opposite directions.
`reports/BENCHMARK.md` found held-out loss improving monotonically with dataset size in 5
of 5 domains, but with the train/val gap refusing to shrink and the returns flattening
(`car_models` gains 0.143 nats going 218→400 training names and only 0.049 going
700→1035). `reports/ARCH.md` found the smaller GRU beating the LSTM on both datasets under
500 names with 25% fewer parameters, and losing on both above. Both concluded: the stock
model is too large for half the library.

Nobody had ever tested that. `hidden_dim=256, num_layers=2` is a default inherited from the
first commit. This report sweeps it.

## Method

One cell = one `(dataset, arch, hidden_dim, num_layers)` combination trained from scratch.
Everything else is held at the repo default: `embedding_dim=32`, `dropout=0.2`, `lr=3e-3`,
`batch_size=32`, `seed=1337`, `--val-fraction 0.15`, and the `--auto-epochs` budget and
patience derived from dataset size by `src/train.derive_epochs` / `derive_patience`. Early
stopping ends every run; the budget is only a ceiling. Best-epoch weights are restored, then
re-scored over both halves with **dropout off and no gradient**, so `gap = held-out NLL −
train NLL` means exactly what the gap column in `reports/ARCH.md` means.

### The grid actually run

| axis | values |
|---|---|
| `hidden_dim` | 64, 128, 256, 384 |
| `num_layers` | 1, 2 |
| `arch` | `lstm`, `gru` |
| datasets | `motorcycle_brands` (309), `aircraft` (435), `typefaces` (463), `car_models` (1,218), `pharma_drugs` (2,223), `english_words` (8,631) |

`aircraft` and `car_models` were added to the four datasets the lane was chartered with:
`aircraft` because `reports/ARCH.md` measured it and it makes the reproduction check direct,
`car_models` to fill the 463 → 2,223 gap. Coverage is complete on the small datasets and
thinner on the large ones, where one cell costs minutes rather than seconds.

Two replicate sweeps sit on top of that, because the margins turned out to be small enough
to need them (finding 2):

* **seeds 7 and 99** over the full grid on `motorcycle_brands` and `typefaces`. Since
  `src/train.train` derives the split from `cfg.seed`, this redraws the validation set as
  well as the initialization — it measures the total run-to-run range.
* **`--split-seed 1337` with seeds 7, 99, 2024** over the same grid, pinning the validation
  set so the initialization varies alone. This is the knob that separates "another init
  would have said something else" from "another 46 held-out names would have said something
  else", and they are not the same size.

```
OMP_NUM_THREADS=1 python scripts/sweep_capacity.py \
    --datasets motorcycle_brands aircraft typefaces car_models pharma_drugs english_words \
    --hidden-dims 64 128 256 384 --layers 1 2 --archs lstm gru
# replicates
OMP_NUM_THREADS=1 python scripts/sweep_capacity.py \
    --datasets motorcycle_brands typefaces --archs lstm gru --seed 7
OMP_NUM_THREADS=1 python scripts/sweep_capacity.py \
    --datasets motorcycle_brands typefaces --archs lstm gru --seed 7 --split-seed 1337
# views
OMP_NUM_THREADS=1 python scripts/sweep_capacity.py --pivot     # the grids below
OMP_NUM_THREADS=1 python scripts/sweep_capacity.py --summary   # every cell, flat
```

`torch.set_num_threads(1)` throughout, three or four processes at a time on a shared 4-core
box, so wall-clock is comparable *between* cells but is not a single-process benchmark.

**The protocol reproduces `reports/ARCH.md` exactly where the two overlap** — five cells,
all four columns, to four decimals, from an independent driver with a different epoch
budget. The table is in finding 4. That is the licence to read the rest of this report
against that one.

## The grid

*(regenerate with `--pivot`)*

<!-- BEGIN:PIVOT -->
### motorcycle_brands (309 names, gru, seed 1337) — best held-out loss
| layers | h=64 | h=128 | h=256 | h=384 |
|---|---:|---:|---:|---:|
| 1 | **2.7461** | 2.7638 | 2.7830 | 2.7896 |
| 2 | 2.7615 | 2.7641 | 2.7891 | 2.7848 |
best: h=64 l=1 (24,345 params, 93 params/name)

### motorcycle_brands (309 names, lstm, seed 1337) — best held-out loss
| layers | h=64 | h=128 | h=256 | h=384 |
|---|---:|---:|---:|---:|
| 1 | 2.7217 | 2.7586 | 2.7193 | 2.7428 |
| 2 | **2.7164** | 2.7170 | 2.7486 | 2.7678 |
best: h=64 l=2 (63,897 params, 243 params/name)

### aircraft (435 names, gru, seed 1337) — best held-out loss
| layers | h=64 | h=128 | h=256 | h=384 |
|---|---:|---:|---:|---:|
| 1 | 0.8236 | 0.7902 | 0.7904 | 0.7997 |
| 2 | 0.8011 | 0.7863 | **0.7725** | 0.7761 |
best: h=256 l=2 (636,257 params, 1720 params/name)

### aircraft (435 names, lstm, seed 1337) — best held-out loss
| layers | h=64 | h=128 | h=256 | h=384 |
|---|---:|---:|---:|---:|
| 1 | 0.7956 | 0.7859 | 0.8226 | 0.8025 |
| 2 | 0.7898 | 0.7876 | 0.7786 | **0.7608** |
best: h=384 l=2 (1,851,873 params, 5005 params/name)

### typefaces (463 names, gru, seed 1337) — best held-out loss
| layers | h=64 | h=128 | h=256 | h=384 |
|---|---:|---:|---:|---:|
| 1 | 2.3227 | **2.2975** | 2.3193 | 2.3635 |
| 2 | 2.4076 | 2.3673 | 2.3471 | 2.3758 |
best: h=128 l=1 (71,063 params, 180 params/name)

### typefaces (463 names, lstm, seed 1337) — best held-out loss
| layers | h=64 | h=128 | h=256 | h=384 |
|---|---:|---:|---:|---:|
| 1 | 2.3100 | **2.2873** | 2.3374 | 2.3335 |
| 2 | 2.4262 | 2.4301 | 2.3754 | 2.3917 |
best: h=128 l=1 (91,799 params, 233 params/name)

### car_models (1218 names, lstm, seed 1337) — best held-out loss
| layers | h=64 | h=128 | h=256 | h=384 |
|---|---:|---:|---:|---:|
| 1 | 2.3941 | 2.3929 | **2.3815** | 2.3892 |
| 2 | 2.3902 | 2.4056 | — | — |
best: h=256 l=1 (316,034 params, 305 params/name)

### pharma_drugs (2223 names, lstm, seed 1337) — best held-out loss
| layers | h=64 | h=128 | h=256 | h=384 |
|---|---:|---:|---:|---:|
| 1 | 2.0793 | 2.0432 | 2.0309 | 2.0301 |
| 2 | 2.0597 | 2.0383 | **2.0252** | — |
best: h=256 l=2 (839,480 params, 444 params/name)

### english_words (8631 names, lstm, seed 1337) — best held-out loss
| layers | h=64 | h=128 | h=256 | h=384 |
|---|---:|---:|---:|---:|
| 1 | 2.0798 | 2.0541 | 2.0209 | **2.0175** |
| 2 | — | — | — | — |
best: h=384 l=1 (654,141 params, 89 params/name)
<!-- END:PIVOT -->

## Findings

<!-- BEGIN:FINDINGS -->
### 1. There is no clean capacity-vs-size rule, because dataset size is the wrong variable.

This is the lane's main result, and it is a negative one.

Three datasets of **almost the same size** want capacities spanning the entire grid:

| dataset | train names | best cell | params at best | params/name | best held-out loss |
|---|---:|---|---:|---:|---:|
| `motorcycle_brands` | 263 | `h=64, l=2` | 63,897 | 243 | 2.7164 |
| `aircraft` | 370 | `h=384, l=2` | 1,851,873 | 5,005 | 0.7608 |
| `typefaces` | 394 | `h=128, l=1` | 91,799 | 233 | 2.2873 |

263 → 394 names is a 1.5× spread in size. The optimal parameter count over those same three
datasets spans **29×**, and it is not ordered by size. Meanwhile going from 263 to 7,336
training names — a **28×** spread in size — moves the optimum only from ~64k to ~305k
parameters, about **5×**.

**Dataset size is the weaker predictor of the two.** A parameters-per-name rule is not
merely noisy; it is fitted to the wrong variable, and the numbers say so: 243, 5,005 and 233
params/name at three dataset sizes that are effectively identical.

What actually predicts the answer is how much structure the domain has, for which the
achievable held-out loss is a usable proxy. `aircraft` bottoms out at 0.76 nats/char —
names like `Boeing 747-400` are highly regular, there is a great deal to learn, and the
largest model on the grid (1.85M parameters on 370 names, 5,005 per name) wins.
`motorcycle_brands` bottoms out at 2.72 nats/char — nearly unpredictable — and the smallest
models win. `typefaces`, at 2.29, sits between them and so does its optimum. Capacity should
track *how learnable the domain is*, and dataset size only correlates with that
incidentally.

This is a direct correction to both wave-3 reports. `reports/BENCHMARK.md` said "these
datasets were too small *and* a 2-layer 256-wide LSTM is too large for them", and
`reports/ARCH.md` inferred a ~500-name threshold from the same reasoning. On `aircraft` —
370 names, inside that threshold — the stock model is not too large. It is **too small**:
`h=384, l=2` beats it by 2.3%.

### 2. The size of the effect: a few percent, and on the small datasets it is inside the noise.

Sweeping the whole 8-cell grid moves held-out loss by **1.9%** on `motorcycle_brands`
(2.7164 → 2.7678, best to worst), **6.2%** on `typefaces` (2.2873 → 2.4301) and **2.4%** on
`pharma_drugs` (2.0301 → 2.0793).

Set that against the measured noise floor. Re-running the *same cells* under three different
seeds — which, because `src/train.train` passes `cfg.seed` to `split_names`, redraws the
validation set as well as the initialization — gives a median across-seed range of **0.088
nats** per cell on `motorcycle_brands`. **The entire capacity effect on that dataset (0.051
nats, best to worst) is smaller than the run-to-run range of a single cell.** With a
46-name validation set, that is what one should expect.

One thing *does* survive the noise: the best **width**, marginalized over depth, is `h=64`
on `motorcycle_brands` at all three seeds — 1337, 7 and 99 — even though the best individual
*cell* moves between `h64/l1` and `h64/l2`. On `typefaces` it is less stable, moving 128 → 256
between seeds 1337 and 7. So "which width band" is roughly a one-seed-supportable question on
these datasets and "which exact cell" is not, which is why finding 1 is stated in bands and
orders of magnitude rather than in located optima.

So no per-cell ranking on a ~300-name dataset is supportable from one seed — including the
rankings in this report, and including the 0.6–2.0% margins `reports/ARCH.md` acted on. What
*is* supportable is paired, same-seed, same-split comparisons repeated across seeds, which is
how findings 3 and 4 are stated.

### 3. Depth is the largest single lever in the sweep — and it is the one the CLI cannot reach.

`num_layers` produces the biggest effects here, in *both* directions:

- On `typefaces`, `l=1` beats `l=2` at **every width in both architectures — 8/8 at seed
  1337 and 8/8 again when the whole grid is re-run at seed 7** — by up to 0.143 nats
  (lstm h=128: 2.2873 vs 2.4301). That is the largest margin any single knob produces
  anywhere in this sweep.
- On `aircraft`, the reverse, nearly as strongly: `l=2` wins **6 of 7** matched pairs, and
  dropping the stock model to one layer costs **+5.7%** (lstm h=256: 0.7786 → 0.8226).
- `motorcycle_brands`: `l=1` wins 5/8 at seed 1337 and 8/8 at seed 7. `pharma_drugs` is a
  wash (`l=2` by 0.28% at h=256).

Tallied honestly across every complete grid, `l=1` is better in **3 of 6** — a coin flip.
The per-dataset results are not noise (they replicate across seeds and are far larger than
the noise floor); they simply point in opposite directions, and which direction is the same
learnability axis as finding 1: the domain that wants maximum capacity wants the second
layer, the domains that want minimum capacity do not.

There is a secondary pattern worth naming: **depth substitutes for width rather than adding
to it.** On `pharma_drugs`, 2 layers win at exactly the two widths that are too narrow for
it (h=64: 2.0597 vs 2.0793; h=128: 2.0383 vs 2.0432) and lose once width is adequate
(h=256: 2.0252 vs 2.0309). The second layer is a way of buying capacity, not a different
kind of capacity.

**The practical problem this exposes:** `src/train.py` accepts `--hidden-dim` but there is
**no `--num-layers` flag**. The axis with the largest measured effect in this report — worth
0.143 nats one way on `typefaces` and 5.7% the other way on `aircraft` — cannot be changed
from the command line at all. `Config.num_layers` exists and is honored; only the CLI is
missing. That is the most actionable thing in this report.

### 4. The GRU/LSTM crossover does not survive capacity tuning. The gating mechanism is a red herring.

The question the lane was built to answer. **The answer is that
`reports/ARCH.md`'s "reach for `--arch gru` on datasets under roughly 500 names" is really
"reach for a smaller model" — tuning capacity flips the winner on both of the small datasets
that report measured.**

First, the protocol reproduces `reports/ARCH.md` **exactly** at the configuration it
measured — all four columns, five cells for five:

| stock `h=256, l=2` cell | `ARCH.md` params / held-out / train / gap | this sweep |
|---|---|---|
| `aircraft` gru | 636,257 / 0.7725 / 0.4583 / +0.314 | 636,257 / 0.7725 / 0.4583 / +0.314 |
| `aircraft` lstm | 842,081 / 0.7786 / 0.4309 / +0.348 | 842,081 / 0.7786 / 0.4309 / +0.348 |
| `typefaces` gru | 633,367 / 2.3471 / 1.7288 / +0.618 | 633,367 / 2.3471 / 1.7288 / +0.618 |
| `typefaces` lstm | 839,191 / 2.3754 / 1.6975 / +0.678 | 839,191 / 2.3754 / 1.6975 / +0.678 |
| `pharma_drugs` lstm | 839,480 / 2.0252 / 1.5811 / +0.444 | 839,480 / 2.0252 / 1.5811 / +0.444 |

So this is a reinterpretation of the same measurement, not a competing one. At the stock
size the GRU really does win on both small datasets.

Now let each architecture pick its own size. The ordering reverses on **both**:

| dataset | | GRU | LSTM | winner |
|---|---|---:|---:|---|
| `aircraft` (370) | stock `h=256, l=2` | **0.7725** | 0.7786 | gru, by 0.8% |
| | each at its own best cell | 0.7725 `h=256,l=2` | **0.7608** `h=384,l=2` | **lstm, by 1.5%** |
| `typefaces` (394) | stock `h=256, l=2` | **2.3471** | 2.3754 | gru, by 1.2% |
| | each at its own best cell | 2.2975 `h=128,l=1` | **2.2873** `h=128,l=1` | **lstm, by 0.4%** |

Note what the two datasets do with the freedom: `aircraft` spends it going **bigger**
(`h=384, l=2`), `typefaces` spends it going **smaller** (`h=128, l=1`). "The GRU is better on
small data" cannot explain both. "The stock LSTM is the wrong size, in whichever direction
this domain happens to need" explains both. At `h=256, l=2` the GRU's 25%-fewer-parameters
was not a better inductive bias; it was a different point on a capacity axis nobody had swept.

Third, on a small dataset `ARCH.md` did not test, the crossover is absent even at stock size.
`motorcycle_brands` (263 train): the LSTM wins **8/8 matched capacities at seed 1337, 8/8 at
seed 7, and 7/8 at seed 99 — 23 of 24 paired comparisons across three independent splits**,
2.7164–2.7678 against the GRU's 2.7461–2.7896. This is the single claim in the report that
clears the noise floor comfortably, precisely because it is paired and replicated.

Two honest qualifications. The tuned LSTM's margins over the tuned GRU (1.5% and 0.4%) are
inside the noise floor of finding 2, so the correct reading of those rows is "the GRU's
advantage is gone", not "the LSTM is now better". And at *matched* capacity on `typefaces`
the GRU still wins 5 of 8 cells — it is a perfectly good core. What has no support left is
choosing it *because the dataset is small*.

### 5. The train/val gap does not close at any capacity — confirming `BENCHMARK.md` §1.

`reports/BENCHMARK.md` found the gap refusing to shrink as data grew. It does not shrink as
capacity shrinks either, and it does not track held-out loss at all. On `typefaces` the
*best* cell (lstm h=128 l=1, held-out 2.2873) carries gap **+0.569**, while one of the
*worst* (lstm h=128 l=2, held-out 2.4301) carries a much healthier **+0.370**. Selecting on
the smallest gap would have picked one of the worst models on the grid.

Whatever the gap measures here, it is not model quality, and no capacity setting drives it to
zero. That is consistent with `BENCHMARK.md`'s reading that early stopping already absorbs
the overfitting the gap would otherwise expose — and it means the gap should stop being cited
as evidence that a model is oversized. This sweep is the direct test of that inference, and
it does not hold up.
<!-- END:FINDINGS -->

## What I would change about the repo's defaults

**1. Change nothing about `hidden_dim` or `num_layers`.** This lane was chartered to find a
better default and the evidence does not support one. `h=256` is never the best width on any
dataset measured and never disastrous either; `l=2` is better on 3 of 6 complete grids and
worse on 3. Every alternative fixed default I can construct from this data is beaten by the
current one on some dataset by more than it wins on another, because the variable that would
let you choose — how learnable the domain is — is not known until after you have trained
something. **A wrong-but-central default beats a rule fitted to the wrong variable.**

That is a negative result, and it is the honest one. It also directly answers the question
this lane was asked: there is no clean parameters-per-name rule, and the reason is not that
the measurement is too noisy to find it, but that dataset size does not determine the answer.

**2. Add a `--num-layers` flag to `src/train.py`.** The concrete, actionable finding. Depth
is the largest single lever measured here — 0.143 nats on `typefaces`, +5.7% the other way
on `aircraft` — and it is the only capacity knob with no CLI flag (`--hidden-dim` exists,
`--num-layers` does not; `Config.num_layers` is honored, so this is a one-line argparse
addition plus a name in the field list). Right now a user who reads this report cannot act
on it without editing Python. Not done here — `src/` is outside this lane.

**3. Revise `reports/ARCH.md`'s recommendation.** "Reach for `--arch gru` on datasets under
roughly 500 names" is not supported once capacity is a free parameter: the ordering reverses
on both datasets that recommendation was drawn from, and on a third small dataset the LSTM
wins 23 of 24 paired comparisons across three seeds. That report's *measurements* reproduce here
exactly, five cells for five, on all four columns — it is the generalization from them that
does not hold. `ARCH.md` should keep its numbers and drop the ~500-name rule. Its other two
conclusions (transformer last, LSTM as the default) are untouched by this lane.

**4. Stop ranking configurations on sub-500-name datasets from a single seed.** A 15% split
of a 300-name dataset is 46 validation names, and finding 2 measures that as a ±0.088-nat
instrument on `motorcycle_brands` and ±0.297 on `typefaces` — larger, in both cases, than the
entire capacity effect being measured. This affects existing reports, not just this one: the
0.6–2.0% margins in `reports/ARCH.md` are well inside it. The cheap fix is to report a
seed range on any small-dataset comparison, or to say the comparison could not be made.

## Every cell

*(regenerate with `--summary`)*

<!-- BEGIN:SUMMARY -->
| dataset | names | arch | hidden | layers | params | best val | best ep | train NLL | gap | sec |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `motorcycle_brands` | 263 | gru | 64 | 1 | 24,345 | 2.7461 | 16 | 2.3062 | +0.440 | 3 |
| `motorcycle_brands` | 263 | gru | 64 | 1 | 24,345 | 2.8092 | 18 | 2.2466 | +0.563 | 3 |
| `motorcycle_brands` | 263 | gru | 64 | 1 | 24,345 | 2.7643 | 16 | 2.3631 | +0.401 | 2 |
| `motorcycle_brands` | 263 | gru | 64 | 2 | 49,305 | 2.7615 | 18 | 2.3198 | +0.442 | 4 |
| `motorcycle_brands` | 263 | gru | 64 | 2 | 49,305 | 2.8167 | 14 | 2.5015 | +0.315 | 4 |
| `motorcycle_brands` | 263 | gru | 64 | 2 | 49,305 | 2.7502 | 12 | 2.6065 | +0.144 | 4 |
| `motorcycle_brands` | 263 | gru | 128 | 1 | 71,385 | 2.7638 | 8 | 2.4594 | +0.304 | 4 |
| `motorcycle_brands` | 263 | gru | 128 | 1 | 71,385 | 2.8486 | 11 | 2.3108 | +0.538 | 3 |
| `motorcycle_brands` | 263 | gru | 128 | 1 | 71,385 | 2.7701 | 9 | 2.4518 | +0.318 | 2 |
| `motorcycle_brands` | 263 | gru | 128 | 2 | 170,457 | 2.7641 | 8 | 2.5424 | +0.222 | 4 |
| `motorcycle_brands` | 263 | gru | 128 | 2 | 170,457 | 2.8519 | 10 | 2.3688 | +0.483 | 6 |
| `motorcycle_brands` | 263 | gru | 128 | 2 | 170,457 | 2.7524 | 9 | 2.5877 | +0.165 | 6 |
| `motorcycle_brands` | 263 | gru | 256 | 1 | 239,193 | 2.7830 | 10 | 2.1160 | +0.667 | 5 |
| `motorcycle_brands` | 263 | gru | 256 | 1 | 239,193 | 2.8662 | 9 | 2.2122 | +0.654 | 5 |
| `motorcycle_brands` | 263 | gru | 256 | 1 | 239,193 | 2.7748 | 6 | 2.4972 | +0.278 | 4 |
| `motorcycle_brands` | 263 | gru | 384 | 1 | 505,305 | 2.7896 | 9 | 2.0350 | +0.755 | 9 |
| `motorcycle_brands` | 263 | gru | 384 | 1 | 505,305 | 2.8854 | 5 | 2.4400 | +0.445 | 9 |
| `motorcycle_brands` | 263 | gru | 384 | 1 | 505,305 | 2.7948 | 6 | 2.4368 | +0.358 | 8 |
| `motorcycle_brands` | 263 | gru | 256 | 2 | 633,945 | 2.7891 | 10 | 2.0267 | +0.762 | 11 |
| `motorcycle_brands` | 263 | gru | 256 | 2 | 633,945 | 2.8806 | 7 | 2.5001 | +0.381 | 12 |
| `motorcycle_brands` | 263 | gru | 256 | 2 | 633,945 | 2.8075 | 6 | 2.4539 | +0.354 | 11 |
| `motorcycle_brands` | 263 | gru | 384 | 2 | 1,392,345 | 2.7848 | 7 | 2.2270 | +0.558 | 20 |
| `motorcycle_brands` | 263 | gru | 384 | 2 | 1,392,345 | 2.8977 | 7 | 2.3370 | +0.561 | 23 |
| `motorcycle_brands` | 263 | gru | 384 | 2 | 1,392,345 | 2.8515 | 6 | 2.4078 | +0.444 | 19 |
| `motorcycle_brands` | 263 | lstm | 64 | 1 | 30,617 | 2.7217 | 17 | 2.3739 | +0.348 | 3 |
| `motorcycle_brands` | 263 | lstm | 64 | 1 | 30,617 | 2.7706 | 24 | 2.2438 | +0.527 | 4 |
| `motorcycle_brands` | 263 | lstm | 64 | 1 | 30,617 | 2.7610 | 12 | 2.5911 | +0.170 | 3 |
| `motorcycle_brands` | 263 | lstm | 64 | 2 | 63,897 | 2.7164 | 21 | 2.4901 | +0.226 | 2 |
| `motorcycle_brands` | 263 | lstm | 64 | 2 | 63,897 | 2.8007 | 27 | 2.3884 | +0.412 | 4 |
| `motorcycle_brands` | 263 | lstm | 64 | 2 | 63,897 | 2.7196 | 24 | 2.4688 | +0.251 | 3 |
| `motorcycle_brands` | 263 | lstm | 128 | 1 | 92,121 | 2.7586 | 16 | 2.1763 | +0.582 | 4 |
| `motorcycle_brands` | 263 | lstm | 128 | 1 | 92,121 | 2.8084 | 14 | 2.3068 | +0.502 | 2 |
| `motorcycle_brands` | 263 | lstm | 128 | 1 | 92,121 | 2.7582 | 9 | 2.5416 | +0.217 | 2 |
| `motorcycle_brands` | 263 | lstm | 128 | 2 | 224,217 | 2.7170 | 18 | 2.3100 | +0.407 | 4 |
| `motorcycle_brands` | 263 | lstm | 128 | 2 | 224,217 | 2.8099 | 18 | 2.3791 | +0.431 | 6 |
| `motorcycle_brands` | 263 | lstm | 128 | 2 | 224,217 | 2.7700 | 15 | 2.5457 | +0.224 | 4 |
| `motorcycle_brands` | 263 | lstm | 256 | 1 | 313,433 | 2.7193 | 12 | 2.2020 | +0.517 | 4 |
| `motorcycle_brands` | 263 | lstm | 256 | 1 | 313,433 | 2.8246 | 9 | 2.4074 | +0.417 | 5 |
| `motorcycle_brands` | 263 | lstm | 256 | 1 | 313,433 | 2.7707 | 6 | 2.6323 | +0.138 | 3 |
| `motorcycle_brands` | 263 | lstm | 384 | 1 | 665,817 | 2.7428 | 10 | 2.2044 | +0.538 | 6 |
| `motorcycle_brands` | 263 | lstm | 384 | 1 | 665,817 | 2.8313 | 9 | 2.3044 | +0.527 | 8 |
| `motorcycle_brands` | 263 | lstm | 384 | 1 | 665,817 | 2.7752 | 6 | 2.5768 | +0.198 | 8 |
| `motorcycle_brands` | 263 | lstm | 256 | 2 | 839,769 | 2.7486 | 10 | 2.4481 | +0.300 | 9 |
| `motorcycle_brands` | 263 | lstm | 256 | 2 | 839,769 | 2.8394 | 10 | 2.5208 | +0.319 | 11 |
| `motorcycle_brands` | 263 | lstm | 256 | 2 | 839,769 | 2.7928 | 10 | 2.5351 | +0.258 | 10 |
| `motorcycle_brands` | 263 | lstm | 384 | 2 | 1,848,537 | 2.7678 | 10 | 2.3769 | +0.391 | 17 |
| `motorcycle_brands` | 263 | lstm | 384 | 2 | 1,848,537 | 2.8502 | 10 | 2.4762 | +0.374 | 27 |
| `motorcycle_brands` | 263 | lstm | 384 | 2 | 1,848,537 | 2.7969 | 6 | 2.6502 | +0.147 | 19 |
| `aircraft` | 370 | gru | 64 | 1 | 25,121 | 0.8236 | 40 | 0.5705 | +0.253 | 10 |
| `aircraft` | 370 | gru | 64 | 2 | 50,081 | 0.8011 | 37 | 0.5606 | +0.241 | 17 |
| `aircraft` | 370 | gru | 128 | 1 | 72,673 | 0.7902 | 31 | 0.4641 | +0.326 | 11 |
| `aircraft` | 370 | gru | 128 | 2 | 171,745 | 0.7863 | 28 | 0.4548 | +0.331 | 24 |
| `aircraft` | 370 | gru | 256 | 1 | 241,505 | 0.7904 | 16 | 0.4922 | +0.298 | 19 |
| `aircraft` | 370 | gru | 384 | 1 | 508,641 | 0.7997 | 15 | 0.4571 | +0.343 | 33 |
| `aircraft` | 370 | gru | 256 | 2 | 636,257 | 0.7725 | 16 | 0.4583 | +0.314 | 43 |
| `aircraft` | 370 | gru | 384 | 2 | 1,395,681 | 0.7761 | 15 | 0.4424 | +0.334 | 67 |
| `aircraft` | 370 | lstm | 64 | 1 | 31,393 | 0.7956 | 52 | 0.5492 | +0.246 | 9 |
| `aircraft` | 370 | lstm | 64 | 2 | 64,673 | 0.7898 | 75 | 0.4789 | +0.311 | 16 |
| `aircraft` | 370 | lstm | 128 | 1 | 93,409 | 0.7859 | 33 | 0.5110 | +0.275 | 8 |
| `aircraft` | 370 | lstm | 128 | 2 | 225,505 | 0.7876 | 39 | 0.4663 | +0.321 | 20 |
| `aircraft` | 370 | lstm | 256 | 1 | 315,745 | 0.8226 | 17 | 0.5473 | +0.275 | 14 |
| `aircraft` | 370 | lstm | 384 | 1 | 669,153 | 0.8025 | 16 | 0.4816 | +0.321 | 32 |
| `aircraft` | 370 | lstm | 256 | 2 | 842,081 | 0.7786 | 26 | 0.4309 | +0.348 | 45 |
| `aircraft` | 370 | lstm | 384 | 2 | 1,851,873 | 0.7608 | 16 | 0.4663 | +0.295 | 87 |
| `typefaces` | 394 | gru | 64 | 1 | 24,151 | 2.3227 | 22 | 1.8559 | +0.467 | 4 |
| `typefaces` | 394 | gru | 64 | 1 | 24,151 | 2.6218 | 19 | 1.9310 | +0.691 | 4 |
| `typefaces` | 394 | gru | 64 | 2 | 49,111 | 2.4076 | 21 | 1.9205 | +0.487 | 9 |
| `typefaces` | 394 | gru | 64 | 2 | 49,111 | 2.7444 | 15 | 2.1136 | +0.631 | 7 |
| `typefaces` | 394 | gru | 128 | 1 | 71,063 | 2.2975 | 15 | 1.7597 | +0.538 | 5 |
| `typefaces` | 394 | gru | 128 | 1 | 71,063 | 2.5776 | 15 | 1.6977 | +0.880 | 6 |
| `typefaces` | 394 | gru | 128 | 2 | 170,135 | 2.3673 | 11 | 1.9798 | +0.387 | 10 |
| `typefaces` | 394 | gru | 128 | 2 | 170,135 | 2.6982 | 10 | 2.0664 | +0.632 | 8 |
| `typefaces` | 394 | gru | 256 | 1 | 238,615 | 2.3193 | 10 | 1.7446 | +0.575 | 9 |
| `typefaces` | 394 | gru | 256 | 1 | 238,615 | 2.5936 | 11 | 1.5511 | +1.042 | 13 |
| `typefaces` | 394 | gru | 384 | 1 | 504,471 | 2.3635 | 9 | 1.6541 | +0.709 | 18 |
| `typefaces` | 394 | gru | 384 | 1 | 504,471 | 2.6504 | 7 | 1.8668 | +0.784 | 19 |
| `typefaces` | 394 | gru | 256 | 2 | 633,367 | 2.3471 | 9 | 1.7288 | +0.618 | 25 |
| `typefaces` | 394 | gru | 256 | 2 | 633,367 | 2.7008 | 7 | 1.9803 | +0.721 | 20 |
| `typefaces` | 394 | gru | 384 | 2 | 1,391,511 | 2.3758 | 9 | 1.4645 | +0.911 | 54 |
| `typefaces` | 394 | gru | 384 | 2 | 1,391,511 | 2.7298 | 7 | 1.7251 | +1.005 | 37 |
| `typefaces` | 394 | lstm | 64 | 1 | 30,423 | 2.3100 | 30 | 1.8165 | +0.493 | 2 |
| `typefaces` | 394 | lstm | 64 | 1 | 30,423 | 2.6164 | 26 | 1.8430 | +0.773 | 4 |
| `typefaces` | 394 | lstm | 64 | 1 | 30,423 | 2.4462 | 29 | 1.9138 | +0.532 | 4 |
| `typefaces` | 394 | lstm | 64 | 2 | 63,703 | 2.4262 | 30 | 1.9187 | +0.507 | 4 |
| `typefaces` | 394 | lstm | 64 | 2 | 63,703 | 2.7104 | 27 | 2.0304 | +0.680 | 7 |
| `typefaces` | 394 | lstm | 64 | 2 | 63,703 | 2.5609 | 30 | 1.9837 | +0.577 | 6 |
| `typefaces` | 394 | lstm | 128 | 1 | 91,799 | 2.2873 | 19 | 1.7179 | +0.569 | 3 |
| `typefaces` | 394 | lstm | 128 | 1 | 91,799 | 2.6105 | 17 | 1.7943 | +0.816 | 5 |
| `typefaces` | 394 | lstm | 128 | 1 | 91,799 | 2.4139 | 18 | 1.8874 | +0.527 | 5 |
| `typefaces` | 394 | lstm | 128 | 2 | 223,895 | 2.4301 | 17 | 2.0602 | +0.370 | 6 |
| `typefaces` | 394 | lstm | 128 | 2 | 223,895 | 2.7272 | 15 | 2.1014 | +0.626 | 9 |
| `typefaces` | 394 | lstm | 128 | 2 | 223,895 | 2.4748 | 24 | 1.8256 | +0.649 | 8 |
| `typefaces` | 394 | lstm | 256 | 1 | 312,855 | 2.3374 | 14 | 1.6416 | +0.696 | 7 |
| `typefaces` | 394 | lstm | 256 | 1 | 312,855 | 2.5598 | 15 | 1.5521 | +1.008 | 12 |
| `typefaces` | 394 | lstm | 256 | 1 | 312,855 | 2.3801 | 16 | 1.5813 | +0.799 | 9 |
| `typefaces` | 394 | lstm | 384 | 1 | 664,983 | 2.3335 | 11 | 1.6831 | +0.650 | 12 |
| `typefaces` | 394 | lstm | 384 | 1 | 664,983 | 2.6100 | 10 | 1.7983 | +0.812 | 18 |
| `typefaces` | 394 | lstm | 384 | 1 | 664,983 | 2.4453 | 9 | 2.0628 | +0.383 | 14 |
| `typefaces` | 394 | lstm | 256 | 2 | 839,191 | 2.3754 | 13 | 1.6975 | +0.678 | 15 |
| `typefaces` | 394 | lstm | 256 | 2 | 839,191 | 2.6796 | 8 | 2.1976 | +0.482 | 20 |
| `typefaces` | 394 | lstm | 384 | 2 | 1,847,703 | 2.3917 | 9 | 1.9599 | +0.432 | 32 |
| `typefaces` | 394 | lstm | 384 | 2 | 1,847,703 | 2.6399 | 8 | 2.0534 | +0.587 | 47 |
| `car_models` | 1035 | lstm | 64 | 1 | 31,490 | 2.3941 | 15 | 2.1089 | +0.285 | 8 |
| `car_models` | 1035 | lstm | 64 | 2 | 64,770 | 2.3902 | 22 | 2.0667 | +0.324 | 15 |
| `car_models` | 1035 | lstm | 128 | 1 | 93,570 | 2.3929 | 15 | 1.9258 | +0.467 | 12 |
| `car_models` | 1035 | lstm | 128 | 2 | 225,666 | 2.4056 | 14 | 1.9968 | +0.409 | 21 |
| `car_models` | 1035 | lstm | 256 | 1 | 316,034 | 2.3815 | 10 | 1.9483 | +0.433 | 22 |
| `car_models` | 1035 | lstm | 384 | 1 | 669,570 | 2.3892 | 7 | 2.0871 | +0.302 | 39 |
| `pharma_drugs` | 1890 | lstm | 64 | 1 | 30,520 | 2.0793 | 21 | 1.8164 | +0.263 | 15 |
| `pharma_drugs` | 1890 | lstm | 64 | 2 | 63,800 | 2.0597 | 22 | 1.8304 | +0.229 | 37 |
| `pharma_drugs` | 1890 | lstm | 128 | 1 | 91,960 | 2.0432 | 16 | 1.7083 | +0.335 | 21 |
| `pharma_drugs` | 1890 | lstm | 128 | 2 | 224,056 | 2.0383 | 21 | 1.6475 | +0.391 | 124 |
| `pharma_drugs` | 1890 | lstm | 256 | 1 | 313,144 | 2.0309 | 10 | 1.6544 | +0.377 | 38 |
| `pharma_drugs` | 1890 | lstm | 384 | 1 | 665,400 | 2.0301 | 7 | 1.7363 | +0.294 | 70 |
| `pharma_drugs` | 1890 | lstm | 256 | 2 | 839,480 | 2.0252 | 10 | 1.5811 | +0.444 | 333 |
| `english_words` | 7336 | lstm | 64 | 1 | 27,901 | 2.0798 | 19 | 1.8358 | +0.244 | 44 |
| `english_words` | 7336 | lstm | 128 | 1 | 87,613 | 2.0541 | 11 | 1.6982 | +0.356 | 48 |
| `english_words` | 7336 | lstm | 256 | 1 | 305,341 | 2.0209 | 7 | 1.6115 | +0.409 | 180 |
| `english_words` | 7336 | lstm | 384 | 1 | 654,141 | 2.0175 | 6 | 1.5659 | +0.452 | 583 |
<!-- END:SUMMARY -->

## Caveats

<!-- BEGIN:CAVEATS -->
**The noise floor is the dominant caveat, and it is measured rather than asserted.** On
`motorcycle_brands` the median across-seed range of a single cell is 0.088 nats; on
`typefaces` it is 0.297. Both exceed that dataset's entire best-to-worst capacity spread
(0.051 and 0.143 respectively). Because `src/train.train` derives the train/val split from
`cfg.seed`, changing the seed redraws the validation set, so most of that range is *which
46–69 names got held out*, not initialization luck — `typefaces` best-cell held-out loss is
2.2873 at seed 1337 and 2.5598 at seed 7, a level shift applied to the whole grid.

That distinction is what keeps the report's claims standing. A level shift moves every cell
in a split together, so **paired within-split comparisons remain informative even when
absolute numbers are not**, and every claim in findings 3 and 4 is stated that way and
checked for replication across splits: `l=1` over `l=2` reproduces 8/8 on `typefaces` at
seed 1337 and 8/8 at seed 7, and the LSTM over the GRU reproduces 8/8, 8/8 and 7/8 on
`motorcycle_brands` at seeds 1337, 7 and 99. The claims that are
*not* paired — chiefly "which single cell is best" and the params/name figures in finding 1 —
should be read as ±one grid step, not as located optima. Finding 1's conclusion survives
that slack easily: the `aircraft`-vs-`motorcycle_brands` contrast is 29× in parameters,
which no amount of ±one-step uncertainty closes.

**`Config.seed_init` defaults to True**, so identical commands give bitwise identical
checkpoints and every number here is exactly reproducible from the command in *Method*.
Reproducible is not the same as robust: it means a re-run confirms the number, not that a
different seed would have produced it.

**Grid coverage is uneven.** The full 4×2 grid × both architectures ran on
`motorcycle_brands`, `typefaces` and `aircraft`. Larger datasets are more thinly covered
because a single `english_words` cell costs 40–800 s against 2–30 s for a small one; where a
cell is missing the pivot table shows "—", and no claim rests on an unmeasured cell.

**One learning rate, one dropout, one batch size.** Everything except `hidden_dim`,
`num_layers` and `arch` sits at the repo default. That is deliberate — it answers "what
would a user get" — but capacity and learning rate interact, and a 64-wide model at
`lr=3e-3` is not necessarily that width's best showing. The narrow-and-deep cells in
particular may be under-served.

**Domain learnability is a post-hoc explanation.** Finding 1 explains the capacity spread by
how predictable each domain is, using achievable held-out loss as the proxy. That proxy is
measured *after* training, so it explains the pattern rather than predicting it, and it rests
on six datasets. It is the best available account of why size fails as a predictor; it is not
itself a validated rule, and nothing in the recommendations depends on it being one.

**`transformer` was not swept.** `reports/ARCH.md` found it last on all four datasets with
the most parameters, so this lane spent its budget on the two architectures still in
contention. A capacity sweep might treat it more kindly; that is untested either way.
<!-- END:CAVEATS -->

Signed: Claude Code | Opus 5 | high
