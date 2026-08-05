# Capacity — how big should the model be, as a function of dataset size?

**Lane WS-18 · wave 4.** Produced by `scripts/sweep_capacity.py`; every cell's raw JSON is
in `reports/_capacity/`. Regenerate with the commands under *Method*.

> **Status: in progress.** The grid is still filling in. Tables below are built from the
> cells on disk at the time of writing and are updated in place as more land. Anything
> marked "—" has not run yet.

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

```
OMP_NUM_THREADS=1 python scripts/sweep_capacity.py \
    --datasets motorcycle_brands typefaces pharma_drugs english_words \
    --hidden-dims 64 128 256 384 --layers 1 2 --archs lstm gru
OMP_NUM_THREADS=1 python scripts/sweep_capacity.py --pivot     # the grids below
OMP_NUM_THREADS=1 python scripts/sweep_capacity.py --summary   # every cell, flat
```

`torch.set_num_threads(1)` throughout, three or four processes at a time on a shared 4-core
box, so wall-clock is comparable *between* cells but is not a single-process benchmark.

**The protocol reproduces `reports/ARCH.md` exactly where the two overlap.** `typefaces`,
LSTM, `h=256, l=2` — the stock config — gives held-out 2.3754, train 1.6975, gap +0.678
here; `reports/ARCH.md` reports 2.3754 / 1.6975 / +0.678. Same three numbers to four
decimals, from an independent driver with a different epoch budget. That is the licence to
read the rest of this table against that one.

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

### typefaces (463 names, gru, seed 1337) — best held-out loss
| layers | h=64 | h=128 | h=256 | h=384 |
|---|---:|---:|---:|---:|
| 1 | 2.3227 | **2.2975** | — | — |
| 2 | — | — | — | — |
best: h=128 l=1 (71,063 params, 180 params/name)

### typefaces (463 names, lstm, seed 1337) — best held-out loss
| layers | h=64 | h=128 | h=256 | h=384 |
|---|---:|---:|---:|---:|
| 1 | 2.3100 | **2.2873** | 2.3374 | 2.3335 |
| 2 | 2.4262 | 2.4301 | 2.3754 | 2.3917 |
best: h=128 l=1 (91,799 params, 233 params/name)

### pharma_drugs (2223 names, lstm, seed 1337) — best held-out loss
| layers | h=64 | h=128 | h=256 | h=384 |
|---|---:|---:|---:|---:|
| 1 | 2.0793 | 2.0432 | 2.0309 | **2.0301** |
| 2 | 2.0597 | — | — | — |
best: h=384 l=1 (665,400 params, 352 params/name)

### english_words (8631 names, lstm, seed 1337) — best held-out loss
| layers | h=64 | h=128 | h=256 | h=384 |
|---|---:|---:|---:|---:|
| 1 | 2.0798 | **2.0541** | — | — |
| 2 | — | — | — | — |
best: h=128 l=1 (87,613 params, 12 params/name)
<!-- END:PIVOT -->

## Findings

<!-- BEGIN:FINDINGS -->
<!-- END:FINDINGS -->

## Every cell

*(regenerate with `--summary`)*

<!-- BEGIN:SUMMARY -->
| dataset | names | arch | hidden | layers | params | best val | best ep | train NLL | gap | sec |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `motorcycle_brands` | 263 | gru | 64 | 1 | 24,345 | 2.7461 | 16 | 2.3062 | +0.440 | 3 |
| `motorcycle_brands` | 263 | gru | 64 | 2 | 49,305 | 2.7615 | 18 | 2.3198 | +0.442 | 4 |
| `motorcycle_brands` | 263 | gru | 128 | 1 | 71,385 | 2.7638 | 8 | 2.4594 | +0.304 | 4 |
| `motorcycle_brands` | 263 | gru | 128 | 2 | 170,457 | 2.7641 | 8 | 2.5424 | +0.222 | 4 |
| `motorcycle_brands` | 263 | gru | 256 | 1 | 239,193 | 2.7830 | 10 | 2.1160 | +0.667 | 5 |
| `motorcycle_brands` | 263 | gru | 384 | 1 | 505,305 | 2.7896 | 9 | 2.0350 | +0.755 | 9 |
| `motorcycle_brands` | 263 | gru | 256 | 2 | 633,945 | 2.7891 | 10 | 2.0267 | +0.762 | 11 |
| `motorcycle_brands` | 263 | gru | 384 | 2 | 1,392,345 | 2.7848 | 7 | 2.2270 | +0.558 | 20 |
| `motorcycle_brands` | 263 | lstm | 64 | 1 | 30,617 | 2.7217 | 17 | 2.3739 | +0.348 | 3 |
| `motorcycle_brands` | 263 | lstm | 64 | 2 | 63,897 | 2.7164 | 21 | 2.4901 | +0.226 | 2 |
| `motorcycle_brands` | 263 | lstm | 128 | 1 | 92,121 | 2.7586 | 16 | 2.1763 | +0.582 | 4 |
| `motorcycle_brands` | 263 | lstm | 128 | 2 | 224,217 | 2.7170 | 18 | 2.3100 | +0.407 | 4 |
| `motorcycle_brands` | 263 | lstm | 256 | 1 | 313,433 | 2.7193 | 12 | 2.2020 | +0.517 | 4 |
| `motorcycle_brands` | 263 | lstm | 384 | 1 | 665,817 | 2.7428 | 10 | 2.2044 | +0.538 | 6 |
| `motorcycle_brands` | 263 | lstm | 256 | 2 | 839,769 | 2.7486 | 10 | 2.4481 | +0.300 | 9 |
| `motorcycle_brands` | 263 | lstm | 384 | 2 | 1,848,537 | 2.7678 | 10 | 2.3769 | +0.391 | 17 |
| `typefaces` | 394 | gru | 64 | 1 | 24,151 | 2.3227 | 22 | 1.8559 | +0.467 | 4 |
| `typefaces` | 394 | gru | 128 | 1 | 71,063 | 2.2975 | 15 | 1.7597 | +0.538 | 5 |
| `typefaces` | 394 | lstm | 64 | 1 | 30,423 | 2.3100 | 30 | 1.8165 | +0.493 | 2 |
| `typefaces` | 394 | lstm | 64 | 2 | 63,703 | 2.4262 | 30 | 1.9187 | +0.507 | 4 |
| `typefaces` | 394 | lstm | 128 | 1 | 91,799 | 2.2873 | 19 | 1.7179 | +0.569 | 3 |
| `typefaces` | 394 | lstm | 128 | 2 | 223,895 | 2.4301 | 17 | 2.0602 | +0.370 | 6 |
| `typefaces` | 394 | lstm | 256 | 1 | 312,855 | 2.3374 | 14 | 1.6416 | +0.696 | 7 |
| `typefaces` | 394 | lstm | 384 | 1 | 664,983 | 2.3335 | 11 | 1.6831 | +0.650 | 12 |
| `typefaces` | 394 | lstm | 256 | 2 | 839,191 | 2.3754 | 13 | 1.6975 | +0.678 | 15 |
| `typefaces` | 394 | lstm | 384 | 2 | 1,847,703 | 2.3917 | 9 | 1.9599 | +0.432 | 32 |
| `pharma_drugs` | 1890 | lstm | 64 | 1 | 30,520 | 2.0793 | 21 | 1.8164 | +0.263 | 15 |
| `pharma_drugs` | 1890 | lstm | 64 | 2 | 63,800 | 2.0597 | 22 | 1.8304 | +0.229 | 37 |
| `pharma_drugs` | 1890 | lstm | 128 | 1 | 91,960 | 2.0432 | 16 | 1.7083 | +0.335 | 21 |
| `pharma_drugs` | 1890 | lstm | 256 | 1 | 313,144 | 2.0309 | 10 | 1.6544 | +0.377 | 38 |
| `pharma_drugs` | 1890 | lstm | 384 | 1 | 665,400 | 2.0301 | 7 | 1.7363 | +0.294 | 70 |
| `english_words` | 7336 | lstm | 64 | 1 | 27,901 | 2.0798 | 19 | 1.8358 | +0.244 | 44 |
| `english_words` | 7336 | lstm | 128 | 1 | 87,613 | 2.0541 | 11 | 1.6982 | +0.356 | 48 |
<!-- END:SUMMARY -->

## Caveats

<!-- BEGIN:CAVEATS -->
<!-- END:CAVEATS -->

Signed: Claude Code | Opus 5 | high
