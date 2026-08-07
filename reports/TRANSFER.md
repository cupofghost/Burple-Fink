# Transfer learning — does pretrain-then-finetune actually beat from-scratch?

**Lane WS-20 · wave 4.** Driver: `scripts/bench_transfer.py`. Raw per-run JSON: `reports/_transfer/`.

The README has recommended transfer learning since wave 1:

> Training every dataset from scratch relearns "how names are spelled" each time. Instead,
> pretrain **one** base model on all datasets, then fine-tune cheap specialized copies from it.

Three waves have shipped that claim without measuring it. This report measures it. It is worth
measuring now and not before, because wave 3 took `data/` from 12 datasets / 13,412 names to
**30 / 27,226** (26,591 after cross-file de-duplication) — a base pretrained on 27k names is a
materially different proposition from one pretrained on 13k.

**Status: draft — bases in flight. From-scratch arms complete; fine-tune arms pending.**

---

## 0. Two pre-flight checks, before any comparison is worth making

Run them with `python scripts/bench_transfer.py check`; output is `reports/_transfer/preflight.json`.

### 0.1 The validation splits are identical between arms ✅

`src.train` splits `load_names(path)`. `src.finetune` splits `filter_to_vocab(load_names(path),
shared_vocab)`. Those are the same list only if the shared vocab drops nothing — and it drops
nothing, on all 30 datasets. Both paths then call `split_names(names, 0.15, 1337)`, which is
deterministic. Verified name-for-name:

| target | names | own vocab | dropped by shared vocab | splits identical | val names |
|---|---:|---:|---:|:--:|---:|
| `motorcycle_brands` | 309 | 57 | 0 | ✅ | 46 |
| `typefaces` | 463 | 55 | 0 | ✅ | 69 |
| `spacecraft` | 593 | 67 | 0 | ✅ | 89 |
| `pharma_drugs` | 2,223 | 56 | 0 | ✅ | 333 |

Every loss in this report is measured on literally the same held-out names in every arm.

### 0.2 The shipped base leaks the entire validation set ❌

`src.pretrain` trains the base on `load_all_names` over **every** dataset, then splits 15% off the
*combined* corpus for its own early stopping. The target's validation names are in that corpus —
all of them — and the base's random 15% holdout is not aligned with the target's:

| target | val names | in base corpus | in base's **training** half |
|---|---:|---:|---:|
| `motorcycle_brands` | 46 | 46 (100%) | 39 (85%) |
| `typefaces` | 69 | 69 (100%) | 59 (86%) |
| `spacecraft` | 89 | 89 (100%) | 76 (85%) |
| `pharma_drugs` | 333 | 333 (100%) | 280 (84%) |

So a fine-tune of the shipped base is scored on names the base read during pretraining, ~85% of
them with gradient updates. **As shipped, `pretrain` → `finetune` → "held-out loss" is not a
held-out loss.** Any transfer win measured that way is uninterpretable.

This is a measurement-protocol problem, not a bug in `src/pretrain.py` — pretraining on all data is
the correct thing for the *product*, and only becomes leakage when you then evaluate a fine-tune of
it against a split drawn from that same data. Nothing in `src/` needs to change; what needs to
change is the README's unmeasured claim, and this report.

**What we ran because of it.** Two bases, identical in every other respect:

- **clean base** — corpus = all 30 datasets **minus the union of the four targets' validation
  names** (537 names, 2.0% of the corpus), removed *by string* so a name that also appears in
  another dataset can't slip back in under that file's flag. This is the honest arm B.
- **leaky base** — `src.pretrain`'s corpus verbatim. Run only to *measure the size of the
  artifact*, so the report can say how much of a naive transfer win is real.

---

## 1. Method

Per target, every arm sees the identical 15%/seed-1337 split, the identical `--auto-epochs` budget
and the identical derived patience (both entry points derive them from the same dataset size), on
stock `Config` — 2-layer, 256-wide LSTM, batch 32, `seed_init=True` so runs are reproducible.

| arm | what it is |
|---|---|
| `scratch` | `python -m src.train --data <t> --val-fraction 0.15 --auto-epochs --patience N` — the shipped from-scratch path, per-dataset vocab, lr 3e-3 |
| `scratch_sv` | the same, built against the 67-token **shared** vocab — a control, see below |
| `ft_clean` | `python -m src.finetune --base ws20_base_clean.pt --data <t> --val-fraction 0.15 --auto-epochs` at `src.finetune`'s default lr 5e-4 |
| `ft_clean_lr` | the same at lr 3e-3, matching from-scratch |
| `ft_leaky` | fine-tune from the leaking base — the number the README's claim produces if nobody checks |

**Why `scratch_sv` exists.** `src.train` sizes the softmax to the dataset's own characters (55–67
here); every fine-tune inherits the base's 67-token shared vocab. That is a difference in the
measurement, not in transfer, and it lands in the same loss number. `scratch_sv` removes it, so the
*only* difference between `scratch_sv` and `ft_clean` is whether the weights start random or
pretrained. It turns out not to matter (§2), which is itself worth knowing — it means the
headline `scratch` vs `ft_clean` comparison is not confounded by vocabulary size.

**Base budget.** The base sees 26,591 names, so one epoch costs ~43 s and patience dominates the
bill. We used `--auto-epochs` (ceiling 300) with **patience 20 rather than the derived 30**. This
is the one budget cut in the report, and it is defensible: `fit` restores the best epoch's weights
whatever ends the run, so a shorter patience can only cost us a *later* bottom, never a worse
checkpoint at the bottom we found; WS-10 measured the held-out bottom at epoch 7 on 8,631 names and
never later than 26 across a 54× size range; and with 3,900+ validation names the base's val curve
is far smoother than a 46-name one. Both bases got the identical budget, so the clean/leaky
comparison is unaffected either way.

---

## 2. From-scratch baseline, and the vocabulary control

| target | arm | best val loss | best epoch | epochs run | train loss @ best | wall |
|---|---|---:|---:|---:|---:|---:|
| `motorcycle_brands` | `scratch` | **2.7486** | 10 | 25 | 2.5147 | 13.1 s |
| `motorcycle_brands` | `scratch_sv` | 2.7554 | 14 | 29 | 2.4619 | 18.0 s |
| `typefaces` | `scratch` | **2.3754** | 13 | 31 | 1.8277 | 22.8 s |
| `typefaces` | `scratch_sv` | 2.3775 | 14 | 32 | 2.1144 | 30.9 s |
| `spacecraft` | `scratch` | **2.5050** | 13 | 33 | 1.8657 | 42.6 s |
| `spacecraft` | `scratch_sv` | 2.5050 | 13 | 33 | 1.8657 | 53.6 s |
| `pharma_drugs` | `scratch` | **2.0252** | 10 | 40 | 1.6961 | 151.0 s |
| `pharma_drugs` | `scratch_sv` | 2.0268 | 10 | 40 | 1.8311 | 152.8 s |

**The shared vocabulary costs nothing.** The largest gap between the two from-scratch arms is
0.0068 nats (`motorcycle_brands`), and on `spacecraft` the two arms are bit-identical — `spacecraft`
happens to use all 67 shared characters, so its own vocab *is* the shared vocab, which is a free
end-to-end check that the control arm is doing what it claims. Widening the softmax from 55 to 67
tokens is not a measurable handicap at this scale, so arm B carries no hidden vocabulary tax and
`scratch` is a fair baseline for it.

These baselines also line up with WS-11's independently-run numbers (`spacecraft` 2.481 vs 2.505,
`motorcycle_brands` 2.762 vs 2.749 under a slightly different budget), which is the cheapest
available evidence that this lane's harness measures the same thing `reports/BENCHMARK.md` does.

---

## 3. Fine-tuned arms

*Pending — both bases are still pretraining. This section fills in as
`reports/_transfer/*__ft_*.json` land.*

---

## 4. Verdict

*Pending.*
