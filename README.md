# Burple-Fink 🚗🤖

A character-level recurrent neural network (char-RNN) that invents **new names** by
learning the "spelling style" of a set of real ones. Inspired by
[Janelle Shane's neural-network paint-color experiment](https://aiweirdness.com/),
where a char-rnn trained on ~7,700 Sherwin-Williams paint colors learned to
hallucinate new colors with names like *"Dondarf"*, *"Sane Green"*, and
*"Stoomy Brown"*.

Just like Shane's model, this one predicts **one character at a time**. Give it a
list of real names, let it read them thousands of times, and it starts to produce
plausible-but-invented names in the same style.

**First job:** generate fake **automobile manufacturer** names and **car model**
names after training on real ones (`Toyota`, `Corvette`, `Lamborghini`, `Wrangler`, ...).

> The name *"Burple-Fink"* is exactly the kind of thing this project is designed to
> emit — a perfectly car-shaped word that means nothing at all.

---

## Why char-RNN?

A char-RNN doesn't know what a "word" is. It only learns:

> *"Given the characters I've seen so far, what character probably comes next?"*

Train it on `Corvette` and thousands of siblings and it internalizes patterns like
"car names love hard C's, double-T's, and Italian-ish `-o`/`-i` endings." Then you
seed it with a starting character and let it babble, sampling one character at a
time until it emits an end-of-name token. Turn up the **sampling temperature**
("creativity" in Shane's telling) and it gets weirder — *Dondarf* territory.

This repo uses an **LSTM** variant of the RNN, which trains more reliably than a
vanilla RNN but works on exactly the same principle.

---

## Quick start

```bash
# 1. Install dependencies (PyTorch)
pip install -r requirements.txt

# 2. Train on the bundled car-manufacturer list (writes a checkpoint to ./checkpoints)
python -m src.train --data data/car_manufacturers.txt --epochs 300 --name manufacturers

# 3. Generate 20 brand-new manufacturer names
python -m src.sample --checkpoint checkpoints/manufacturers.pt --num 20 --temperature 0.8
```

Swap `data/car_manufacturers.txt` for `data/car_models.txt` to generate model names
instead, or point `--data` at any newline-separated list of names (dog breeds, metal
bands, Doctor Who episodes — Shane did them all).

---

## Repository layout

```
Burple-Fink/
├── README.md                 # you are here
├── requirements.txt
├── docs/
│   └── PLAN.md               # the full project plan & roadmap
├── data/
│   ├── car_manufacturers.txt # ~150 real auto brands (training data)
│   └── car_models.txt        # ~250 real car model names (training data)
├── src/
│   ├── config.py             # hyperparameters in one place
│   ├── data.py               # read names -> vocab -> tensors
│   ├── model.py              # the char-RNN (LSTM) itself
│   ├── train.py              # training loop + checkpointing
│   └── sample.py             # load checkpoint, generate names
└── generate.py               # tiny convenience wrapper around sample.py
```

See **[docs/PLAN.md](docs/PLAN.md)** for the design rationale, the training pipeline,
the "creativity knob," and the roadmap toward the RGB-style dual-output extension
that Shane used for paint colors.

---

## Sampling temperature (the "creativity" knob)

| Temperature | Behavior                                             |
|-------------|------------------------------------------------------|
| `0.2–0.5`   | Safe, boring, close to real training names           |
| `0.7–0.9`   | The sweet spot — plausible but novel                 |
| `1.1–1.5`   | *Dondarf* / *Bylfgoam Glosd* chaos                   |

---

## Roadmap (short version)

1. ✅ Char-RNN that generates names from any name list.
2. ⏩ Bundle & clean larger real-world car datasets for better output.
3. ⏩ Add the **dual-output** trick from the paint-color experiment: predict a name
   **and** an associated numeric attribute (for cars: e.g. horsepower or a "sportiness"
   score) at the same time.
4. ⏩ Web demo / API.

Full detail in [docs/PLAN.md](docs/PLAN.md).
