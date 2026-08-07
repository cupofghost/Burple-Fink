# Burple-Fink 🚗🤖

A character-level RNN **name-generation platform**. One reusable model, trained and
fine-tuned on many different datasets, to invent new names in a variety of styles —
car brands, model names, and (over time) whatever else we point it at.

Inspired by [Janelle Shane's neural-network paint-color experiment](https://aiweirdness.com/),
where a char-rnn trained on ~7,700 Sherwin-Williams paint colors learned to hallucinate
new colors with names like *"Dondarf"*, *"Sane Green"*, and *"Stoomy Brown"*. Like hers,
this model predicts **one character at a time**, so it can synthesize brand-new words
that no dictionary contains.

> The name *"Burple-Fink"* is exactly the kind of thing this project emits — a perfectly
> product-shaped word that means nothing at all.

---

## 📌 This is a multi-stage, multi-agent project

Burple-Fink is being built in stages by **multiple agents working in parallel**. The end
state is a single char-RNN engine plus a growing library of training datasets and
per-domain fine-tuned checkpoints, so the same engine can generate many *kinds* of names.

**If you are an agent (or human) picking up work here, start with
[`HANDOFF.md`](HANDOFF.md).** It defines the stages, the open workstreams, the
conventions every contributor must follow, and how to add a new dataset without
colliding with other agents. Repo-wide conventions and commands for agents also live in
[`CLAUDE.md`](CLAUDE.md).

### Where the project is now

| Stage | Description | Status |
|-------|-------------|--------|
| **0. MVP engine** | Char-RNN (LSTM) that trains per-dataset from scratch + temperature sampling | ✅ Done |
| **1. Dataset library** | Many clean, documented name datasets under a shared convention | ⏳ In progress |
| **2. Shared base + fine-tuning** | Pretrain one base model on all corpora, then fine-tune per domain (transfer learning) | ✅ Done |
| **3. Evaluation harness** | Automatic novelty / plausibility / diversity metrics to compare checkpoints | ✅ Done |
| **4. Dual-output** | Emit a name **and** an attribute (Shane's name+RGB trick) | ✅ Done |
| **5. Serving** | CLI ✅ → live server ✅ → in-browser web app ✅ | ✅ Done |
| **6. Training quality** | Held-out split, per-epoch val loss, early stopping, best-epoch weights, LR schedules | ✅ Done |
| **7. Dataset library at scale** | 30 datasets / 27,226 names, each with a `.meta.json` sidecar and a validator | ✅ Done |
| **8. Architecture options** | LSTM / GRU / transformer behind `--arch`, sharing one checkpoint format | ✅ Done |
| **9. Reproducibility** | Seeded initialization — identical commands give bitwise-identical checkpoints | ✅ Done |

**Wave 2 (2026-07-29)** upgraded quality rather than adding stages: honest held-out
validation + early stopping (WS-6), top-k/nucleus sampling and a decoding sweep (WS-7), and
the repo's first CI (WS-8). See [`docs/UPGRADE_PLAN.md`](docs/UPGRADE_PLAN.md).

**Wave 3 (2026-08-01/02)** acted on wave 2's loudest finding — *"a 135-name dataset cannot
support a 2-layer, 256-wide LSTM"* — with nine parallel lanes. The dataset library went from
12 files / 13,412 names to **30 / 27,226**; the four thinnest datasets were grown in place;
`src/arch/` added GRU and transformer cores behind `--arch`; and
[`reports/BENCHMARK.md`](reports/BENCHMARK.md) settles the size question with a controlled
within-dataset ladder. See [`docs/WAVE3_PLAN.md`](docs/WAVE3_PLAN.md).

> **The benchmark's answer splits in two.** More data reliably makes a *better model* —
> held-out loss improved monotonically with training-set size in 5 of 5 domains. But it does
> **not** make the model overfit less per epoch: the train/val gap at the best epoch didn't
> shrink, and on `english_words` it moved the wrong way. Returns also flatten. So wave 2's
> diagnosis was half right — these datasets were too small *and* this model is too large for
> them.

The full rationale and design for each stage lives in [`docs/PLAN.md`](docs/PLAN.md).

---

## Why char-RNN?

A char-RNN doesn't know what a "word" is. It only learns:

> *"Given the characters I've seen so far, what character probably comes next?"*

Made-up product names aren't real words, so a **word-level** model literally cannot
produce them. A **character-level** model works letter-by-letter and can therefore
synthesize entirely new tokens like `Corvella` or `Burple-Fink`. The price is that it
must also *learn to spell* — which is what makes the early-training output so gloriously
broken.

We use an **LSTM** variant, which trains far more reliably than a vanilla RNN on exactly
the same principle. See [`docs/PLAN.md`](docs/PLAN.md) for the architecture.

---

## Quick start

```bash
# 1. Install dependencies (PyTorch). Note: install from the default PyPI index —
#    the custom pytorch download index is not reachable in this environment.
pip install -r requirements.txt

# 2. Train on a dataset (writes a checkpoint to ./checkpoints)
python -m src.train --data data/car_manufacturers.txt --epochs 300 --name manufacturers

# 3. Generate 20 brand-new manufacturer names
python -m src.sample --checkpoint checkpoints/manufacturers.pt --num 20 --temperature 0.8

# …or do both in one shot:
python generate.py --data data/car_models.txt --train --name models --num 20
```

Point `--data` at any newline-separated list of names to train a new generator.

### Transfer learning: pretrain once, fine-tune per domain

Training every dataset from scratch relearns "how names are spelled" each time. Instead,
pretrain **one** base model on all datasets (a shared vocabulary keeps the weights
compatible), then fine-tune cheap specialized copies from it:

```bash
# 1. Pretrain a base model on every data/*.txt (writes data/shared_vocab.json + base.pt)
python -m src.pretrain --epochs 300 --name base

# 2. Fine-tune the base onto one domain (gentler LR, fewer epochs)
python -m src.finetune --base checkpoints/base.pt --data data/car_models.txt --name car_models
#    -> checkpoints/car_models_ft.pt

# 3. Score a checkpoint: novelty / plausibility / diversity
python -m src.evaluate --checkpoint checkpoints/car_models_ft.pt --temperature 1.0
```

See [`HANDOFF.md §6`](HANDOFF.md#6-key-design-decision-for-fine-tuning-shared-vocabulary)
for the shared-vocabulary design that makes fine-tuning possible.

### Training honestly: hold some names back

A 256-wide, 2-layer LSTM has far more capacity than 150 car brands. Trained on 100% of a
dataset for a fixed 300 epochs, there is no way to tell whether it learned a *style* or
simply memorized the list. Three opt-in flags fix that — they work on `src.train`,
`src.pretrain` and `src.finetune` alike:

```bash
# Hold back 15% of the names, report a real held-out loss every epoch, keep the
# best-scoring epoch's weights (not the last epoch's), and stop once it stops improving.
python -m src.train --data data/aircraft.txt --val-fraction 0.15 --patience 20 --name aircraft

# Optional learning-rate schedule: 'none' (default) | 'plateau' | 'cosine'
python -m src.finetune --base checkpoints/base.pt --data data/car_models.txt \
    --name car_models --val-fraction 0.15 --patience 10 --lr-schedule plateau
```

The log gains a second number — `epoch  120/300 | train 0.8123 | val 1.4470` — and the run
tells you which epoch it kept. Checkpoints trained this way carry the held-out names under
an additive `val_names` key ([HANDOFF §2](HANDOFF.md#2-current-state-stages-0-2-3-5-complete)),
so evaluation can score against names the model never saw.

Every flag defaults to off, so leaving them out trains exactly as before —
`tests/test_training_quality.py` pins the default loss trajectory against the pre-WS-6
one to prove it.

**What this measured, and what to type instead of `--epochs 300`.** Held-out loss bottoms
out early and then gets *worse* for the rest of the budget. Wave 2 measured this on two
datasets; wave 3 re-ran the same protocol after growing the data:

| Dataset | Names | Val loss bottoms at | Val loss by epoch 300 | Train/val gap at 300 |
|---|---|---|---|---|
| `car_manufacturers.txt` (wave 2) | 159 | epoch **12–19** | 2.98 → **6.87** (130% worse) | 6.15 nats |
| `car_manufacturers.txt` (grown) | 590 | epoch **10** | 2.58 → **4.79** (86% worse) | 3.95 nats |
| `aircraft.txt` — *unchanged file, protocol control* | 435 | epoch **21** | 0.81 → **1.08** (34% worse) | 0.71 nats |

`aircraft.txt` is the control: wave 3's data lanes never touched it, and re-running wave 2's
protocol reproduces wave 2's numbers within noise. That is what licenses reading the
`car_manufacturers` improvement as an effect of more data rather than of a changed
measurement.

Best epochs are now single values, not ranges: since wave 3, initialization is seeded before
the model is built (`Config.seed_init`, default `True`), so identical commands produce
**bitwise identical** checkpoints. Pass `--no-seed-init` for the old, unreproducible
behavior.

### Stop guessing the epoch budget: `--auto-epochs`

```bash
python -m src.train --data data/motorcycle_brands.txt --auto-epochs --val-fraction 0.15
#  -> auto-epochs: 90 epochs, patience 15 for 309 names
#  -> early stop at epoch 25 (best val 2.7486 at epoch 10)
```

This replaces a hand-applied lookup table, **and it corrects the premise that table was
built on.** That table said bigger datasets need more epochs. Measuring the val-loss bottom
across nine datasets from 159 to 8,631 names gives: 13, 10, 26, 9, 13, 10, 8, 10, **7**. The
bottom does not move with dataset size — if anything it arrives *earlier*, because one epoch
over 8,631 names is fifty times the gradient steps of one epoch over 159.

So the derived budget is **not** a prediction of where the bottom is. It is a ceiling on how
far past the bottom it is safe to run, and *that* does scale with size: over-training damage
was +130% at 159 names versus +35% at 435. Every derived budget clears its measured bottom
by at least 4×, and `--patience` is expected to stop the run first.

`--epochs N` on its own is unchanged. Passing both prints a notice and honors `--epochs`.

### Regularization: measured, and it doesn't help here

`--weight-decay` (switches the optimizer to AdamW), `--label-smoothing` and
`--warmup-epochs` are all available and all default to off. Wave 3 measured them on three
datasets (159, 590 and 2,223 names) and the honest result is that **none of them improves
the best achievable held-out loss**:

| setting | @159 | @590 | `pharma_drugs` @2223 |
|---|---|---|---|
| baseline | 2.9696 | **2.5756** | **2.0252** |
| `--weight-decay 0.01` | 2.9698 | 2.5759 | 2.0279 |
| `--label-smoothing 0.1` | **2.9358** | 2.6157 | 2.0745 |
| `--warmup-epochs 10` | 2.9747 | 2.5823 | 2.0464 |

Weight decay is inert at 0.01 and mildly harmful at 0.1 — on a model that reaches its best
epoch in single digits, decay has had roughly 150 updates to act and hasn't. Label smoothing
wins only on the smallest set and loses on both larger ones. Warmup never helps.

Label smoothing *does* cut over-training damage at a full 300-epoch budget (+91% → +27% on
`car_manufacturers`), but since wave 2 the trainer already keeps the best epoch's weights and
stops early — so that protects against a failure mode that no longer exists. **Leave all
three off** unless you have a reason the numbers above don't cover.

Note validation is always scored *unsmoothed*, so best-val stays comparable across settings
and against every number recorded in `STATUS.md`.

### Choosing an architecture: `--arch lstm | gru | transformer`

`src/arch/` implements all three behind one checkpoint format, one training loop and one
sampler, so `--arch` is the only thing that changes. Measured on four datasets from 370 to
7,336 training names ([`reports/ARCH.md`](reports/ARCH.md)):

| dataset | train names | gru | lstm | transformer |
|---|---:|---:|---:|---:|
| `aircraft` | 370 | **0.7725** | 0.7786 | 0.9334 |
| `typefaces` | 394 | **2.3471** | 2.3754 | 2.5410 |
| `pharma_drugs` | 1,890 | 2.0331 | **2.0252** | 2.1172 |
| `english_words` | 7,336 | 2.0487 | **2.0078** | 2.0904 |

(held-out loss, lower is better)

**A wave-3 report read this as "the GRU wins below roughly 500 training names", and
wave 4 disproved it.** The numbers above are all measured at the stock `hidden_dim=256,
num_layers=2` — so they compare a GRU against an LSTM that is 25% *larger*, not against a
better LSTM. [`reports/CAPACITY.md`](reports/CAPACITY.md) let each architecture pick its own
size and **the ordering reverses on both small datasets**:

| dataset | gru at its best cell | lstm at its best cell | winner |
|---|---|---|---|
| `aircraft` (370) | 0.7725 `h=256,l=2` | **0.7608** `h=384,l=2` | lstm, by 1.5% |
| `typefaces` (394) | 2.2975 `h=128,l=1` | **2.2873** `h=128,l=1` | lstm, by 0.4% |

On a third small dataset the LSTM wins **23 of 24 paired comparisons across three
independent splits**. "Use a GRU under 500 names" was really "use a different-sized model",
and the gating mechanism was a red herring.

So: **`--arch lstm` is the right default at every size measured**, and `--arch gru` is worth
trying only as one more knob, not as a rule keyed to dataset size.

The transformer is last on all four *while carrying 30% more parameters than the LSTM*, and
posts the smallest train/val gap in the table next to its worst held-out loss — it underfits.
Names are short, so there is no long-range dependency for attention to exploit, and the
recurrent bottleneck turns out to be a useful inductive bias. It is implemented and correct
(stepwise decoding is proven equivalent to a full forward pass), just not recommended here.

`--arch lstm` remains the default.

### Dual-output: name + numeric attribute (Shane's paint-color trick)

Shane's original char-rnn didn't just name paint colors — it predicted their RGB
value too. `CharRNN` gains an optional second head (`value_head`, only present when
`cfg.dual_output=True`) that regresses one numeric attribute per name, trained
jointly with the usual next-character loss via `src/train_dual.py`. The value is
z-scored internally (`Config.value_mean`/`value_std`) so it works for any scale —
a 0–1 color brightness or a four-digit founding year — and `python -m src.sample`
auto-prints the (denormalized) predicted value for dual checkpoints:

```bash
# Train on real car brands + their founding year
python -m src.train_dual --data data/car_manufacturers_founding_year.tsv \
    --name manufacturers_founding_year --epochs 300 --value-label "founding year"

# Generate invented brand names with a predicted founding year
python -m src.sample --checkpoint checkpoints/manufacturers_founding_year.pt --num 10
#  -> Fordia  (founding year: 1921.4)   Motoza  (founding year: 1998.7)
```

Also demoed on `data/paint_colors.tsv` (CSS named colors + luminance) and
`data/periodic_elements.tsv` (elements + atomic number). See
[`HANDOFF.md §3 WS-4`](HANDOFF.md#ws-4--dual-output-name--attribute) for the design.

### The web app (open it on your phone)

There are two ways to run the mobile-friendly UI — an instrument panel with a
cold→hot "temperature" dial, a prefix box, and live novelty flags:

```bash
# A) Live server — the UI talks to the real PyTorch model in this process.
#    Stdlib only; binds 0.0.0.0 so a phone on the same Wi-Fi can open it.
python -m src.serve
#    -> http://localhost:8000   (or http://<your-computer-ip>:8000 from a phone)

# B) Static export — bake the weights into ONE self-contained HTML file that runs the
#    net entirely in the browser (no server, shareable, works offline).
python -m src.export_web \
    --model checkpoints/car_manufacturers_ft.pt:"Car brands" \
    --model checkpoints/car_models_ft.pt:"Car models" \
    --out web/burple-fink.html
```

The exporter **verifies** its JavaScript forward pass matches the trained PyTorch model's
logits before writing the file, so the in-browser net is faithful to the real one.

The live server also exposes `GET /api/health` (which checkpoints are loaded, and their
labels) and returns real JSON error messages — with an HTTP status code and a message the
UI displays inline — on a bad request or a failed generation, instead of a silent failure.

---

## Sampling temperature (the "creativity" knob)

| Temperature | Behavior                                             |
|-------------|------------------------------------------------------|
| `0.2–0.5`   | Safe, boring, close to real training names           |
| `0.7–0.9`   | The sweet spot — plausible but novel                 |
| `1.1–1.5`   | *Dondarf* / *Bylfgoam Glosd* chaos                   |

Sample runs from the current engine:

- **Manufacturers @ 0.8:** Sabarg, Jaguat, Mercuber, Tovaso, Chewo
- **Manufacturers @ 1.3:** Muhkelveo, Volvoz, UVismann, Driza
- **Car models @ 0.8:** Sentaza, Carlare, Ventora, Chezla

### Decoding controls (WS-7): `--top-k`, `--top-p`, `--repetition-penalty`

`python -m src.sample` and `python -m src.evaluate` both take `--top-k N` (keep only the
N likeliest next characters), `--top-p P` (nucleus: smallest set with cumulative
probability ≥ P), and `--repetition-penalty R` (discourage characters already used in the
current name). All three default to off (0 / 1.0 / 1.0), so nothing changes unless you
pass them. `--min-length` is now enforced *during* generation, not only filtered after.

`python -m src.evaluate --checkpoint <ckpt> --sweep [--compare <ckpt2> …]` grid-searches
temperature × decoding setting and prints novelty, **near-duplicate rate** (share of
generated names within edit distance 1, and separately ≤2, of a training name — the
number that reveals memorization plain novelty misses), plausibility ratio, uniqueness,
and mean edit distance, then recommends a setting.

**Measured** on two checkpoints of very different data sizes (159 real car brands vs.
8,631 English words, 150–200 samples/setting, temperatures 0.7–1.3, `top_k∈{5,10}`,
`top_p∈{0.8,0.9}`): **plain temperature sampling at 1.1–1.3 beat every top-k/nucleus
setting tried, on both checkpoints.** Truncating the tail didn't reduce junk here (these
small char-RNNs weren't producing much at these temperatures — plausibility ratio stayed
~1.0–1.03) but it did shrink the pool of reachable characters enough to push sampling back
toward memorized training names: at temperature 1.3 on the manufacturers checkpoint,
novelty dropped from 38% (plain) to 32% (`top_k=10`) and near-duplicate rate rose from 72%
to 80%. `repetition_penalty` still helps independently — it targets character-repeat
junk (*"Bylfgoammm"*) that top-k/nucleus don't touch. Honest conclusion: at this model
scale, reach for a higher temperature and `repetition_penalty` before top-k/nucleus; don't
assume truncating the tail is free.

---

## Repository layout

```
Burple-Fink/
├── README.md                 # you are here — the project overview
├── HANDOFF.md                # ← START HERE if you're picking up work
├── CLAUDE.md                 # repo conventions & commands for agents
├── requirements.txt
├── docs/
│   └── PLAN.md               # full design rationale, pipelines, and roadmap
├── data/                     # 30 datasets + a .meta.json sidecar each (see catalog below)
│   ├── english_words.txt     # 8,631 — base-model fuel
│   ├── pharma_drugs.txt      # 2,223 — the best char-RNN signal in the repo
│   ├── world_cities.txt      # 1,691
│   ├── car_models.txt        # 1,218
│   ├── …                     # 26 more, from birds to typefaces
│   ├── *.meta.json           # label / domain / count / provenance per dataset
│   ├── *.tsv                 # WS-4 dual-output demos: name<TAB>value
│   └── shared_vocab.json     # fixed char set shared by base + all fine-tunes
├── scripts/
│   ├── build_paint_colors.py # regenerates data/paint_colors.tsv from its hex source
│   ├── check_repo.py         # registry drift, committed weights, secrets/PII
│   └── check_data.py         # dataset charset / duplicates / sidecar validation
├── src/
│   ├── config.py             # hyperparameters in one place
│   ├── data.py               # read names -> vocab -> tensors (+ shared vocab)
│   ├── model.py              # the char-RNN itself (+ optional value head)
│   ├── arch/                 # lstm / gru / transformer cores behind cfg.arch
│   ├── train.py              # training loop (reusable `fit`) + checkpointing
│   ├── sample.py             # load checkpoint, generate names (+ predicted value)
│   ├── pretrain.py           # train one base model on all datasets
│   ├── finetune.py           # specialize the base onto one dataset
│   ├── train_dual.py         # WS-4: joint name + numeric-attribute training
│   ├── evaluate.py           # novelty / plausibility / diversity metrics
│   ├── export_web.py         # bake a checkpoint into a browser-runnable UI
│   └── serve.py              # local server wiring the UI to the live model
├── web/
│   ├── app_template.html     # the UI (CSS + markup + in-browser char-RNN)
│   └── burple-fink.html      # built self-contained app (open on a phone)
├── tests/                    # unittest suite (vocab, training, dual-output, export)
└── generate.py               # one-command train-and-generate wrapper
```

---

## Dataset catalog

The whole point of the platform is *variety* of names, so datasets are first-class.
Each lives in `data/` as a newline-separated `.txt` file with a `data/<stem>.meta.json`
sidecar carrying its label, domain, count and provenance. `python scripts/check_data.py`
validates every one of them.

**30 datasets, 27,226 names.**

### Vehicles & Transport — 3,774 names

| Dataset file | Contents | Count |
|---|---|---|
| `car_models.txt` | Car models | 1,218 |
| `car_manufacturers.txt` | Car manufacturers | 590 |
| `sailing_ships.txt` | Sailing Ships and Naval Vessels | 468 |
| `aircraft.txt` | Aircraft models | 435 |
| `locomotives.txt` | Locomotives | 395 |
| `motorcycles.txt` | Motorcycle brands and models | 359 |
| `motorcycle_brands.txt` | Motorcycle brands | 309 |

### Space & Sky — 1,096 names

| Dataset file | Contents | Count |
|---|---|---|
| `spacecraft.txt` | Spacecraft | 593 |
| `stars_constellations.txt` | Stars and Constellations | 503 |

### Nature — 3,546 names

| Dataset file | Contents | Count |
|---|---|---|
| `birds.txt` | Bird species | 863 |
| `plants_flowers.txt` | Plants and flowers | 634 |
| `minerals_gems.txt` | Minerals and gemstones | 624 |
| `mountains.txt` | Mountains and peaks | 523 |
| `dog_breeds.txt` | Dog breeds | 461 |
| `mushrooms.txt` | Mushrooms and fungi | 441 |

### Food & Drink — 1,278 names

| Dataset file | Contents | Count |
|---|---|---|
| `cocktails.txt` | Cocktails | 459 |
| `cheeses.txt` | Cheeses | 421 |
| `craft_beers.txt` | Craft brewery and beer names | 398 |

### Games & Music — 2,001 names

| Dataset file | Contents | Count |
|---|---|---|
| `video_games.txt` | Video games | 753 |
| `metal_bands.txt` | Metal and rock bands | 663 |
| `board_games.txt` | Board games | 585 |

### Design & Brands — 1,893 names

| Dataset file | Contents | Count |
|---|---|---|
| `perfumes.txt` | Perfumes | 639 |
| `typefaces.txt` | Typefaces | 463 |
| `tech_startups.txt` | Tech company and startup names | 400 |
| `paint_colors.txt` | Paint color names | 391 |

### Science & Medicine — 2,223 names

| Dataset file | Contents | Count |
|---|---|---|
| `pharma_drugs.txt` | Pharmaceutical Drugs | 2,223 |

### Words, Names & Places — 11,432 names

| Dataset file | Contents | Count |
|---|---|---|
| `english_words.txt` | Common English words | 8,631 |
| `world_cities.txt` | World city and capital names | 1,674 |
| `greek_myth.txt` | Greek and Roman Mythology | 755 |
| `racehorses.txt` | Racehorse names | 355 |

### Dual-output datasets (`name<TAB>value`)

Wave 4 took this from three demo files to nine real ones — 3,246 pairs. Each sidecar
records an honest **signal assessment**: whether the numeric value is actually predictable
from the spelling of the name, which is the only thing a char-RNN can learn from. A value
statistically independent of the characters trains the head to predict the mean and teaches
nothing.

| Dataset file | Contents | Count |
|---|---|---|
| `car_manufacturers_founding_year.tsv` | Car brands + founding year | 66 |
| `paint_colors.tsv` | CSS named colors + luminance | 141 |
| `periodic_elements.tsv` | Chemical elements + atomic number | 118 |
| `birds_wingspan.tsv` | Bird species + typical wingspan (cm) | 617 |
| `dinosaurs_length.tsv` | Dinosaur genera + body length (m) | 405 |
| `dog_breeds_weight.tsv` | Dog breeds + typical adult weight (kg) | 348 |
| `mountains_height.tsv` | Mountains + elevation (m) | 386 |
| `pharma_drugs_year.tsv` | Drug INNs + year of first approval | 795 |
| `spacecraft_year.tsv` | Spacecraft + launch year | 370 |

**Provenance caveat.** Every dataset added in wave 3 carries `"verified": false`
in its sidecar: the entries were recalled from model knowledge and are believed real,
but were not cross-checked against a primary source. `periodic_elements.tsv` is the
one exception — its 118 rows were checked. Treat the rest as good training data,
not as a reference work.

Five pre-wave datasets contain characters outside the model's alphabet (accents,
apostrophes) and are grandfathered in `scripts/check_data.py`'s `KNOWN_NONCONFORMING`.
New datasets must stay within `[A-Za-z0-9]`, space and hyphen.

---

## Contributing / handoff

This repo is worked on by multiple agents. **Read [`HANDOFF.md`](HANDOFF.md) first** — it
covers the branch strategy, how to claim a workstream, dataset conventions, the
fine-tuning design, evaluation criteria, and the definition of done for each stage.
