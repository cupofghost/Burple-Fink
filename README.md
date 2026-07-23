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
| **2. Shared base + fine-tuning** | Pretrain one base model on all corpora, then fine-tune per domain (transfer learning) | 🔜 Planned |
| **3. Evaluation harness** | Automatic novelty / plausibility / diversity metrics to compare checkpoints | 🔜 Planned |
| **4. Dual-output** | Emit a name **and** an attribute (Shane's name+RGB trick) | 🔜 Planned |
| **5. Serving** | CLI ✅ → API → tiny web demo | ⏳ CLI done |

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
├── data/                     # training datasets, one name per line
│   ├── car_manufacturers.txt # ~150 real auto brands
│   └── car_models.txt        # ~250 real car model names
├── src/
│   ├── config.py             # hyperparameters in one place
│   ├── data.py               # read names -> vocab -> tensors
│   ├── model.py              # the char-RNN (LSTM) itself
│   ├── train.py              # training loop + checkpointing
│   └── sample.py             # load checkpoint, generate names
└── generate.py               # one-command train-and-generate wrapper
```

---

## Dataset catalog

The whole point of the platform is *variety* of names, so datasets are first-class. Each
lives in `data/` as a newline-separated `.txt` file and follows the conventions in
[`HANDOFF.md`](HANDOFF.md#adding-a-new-dataset).

| Dataset file | Domain | Count | Status |
|--------------|--------|-------|--------|
| `car_manufacturers.txt` | Auto brands | ~150 | ✅ seed |
| `car_models.txt` | Car model names | ~250 | ✅ seed |
| _(your dataset here)_ | — | — | 🔜 |

Ideas for future domains (a Shane-style variety): boat & yacht names, motorcycle brands,
aircraft, spacecraft, perfumes, craft beers, racehorses, tech startups, paint colors
(homage), fantasy characters, city names. Claim one in `HANDOFF.md` before you start.

---

## Contributing / handoff

This repo is worked on by multiple agents. **Read [`HANDOFF.md`](HANDOFF.md) first** — it
covers the branch strategy, how to claim a workstream, dataset conventions, the
fine-tuning design, evaluation criteria, and the definition of done for each stage.
