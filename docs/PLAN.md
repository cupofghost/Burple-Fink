# Project Plan — Burple-Fink Name Generator

## 1. Goal

Build an AI tool that **invents new names** in the style of a training set of real
names. The immediate deliverable is a **char-level RNN (char-RNN)** that, after
training on real automobile manufacturer and car model names, produces plausible new
ones.

This directly mirrors Janelle Shane's paint-color experiment: she fed a char-rnn
~7,700 Sherwin-Williams paint colors (names + RGB values) and it learned to generate
new colors and names one character at a time. We start with the simpler "names only"
version and leave a clear path to her dual-output (name + number) trick.

## 2. Why a character-level model (not word-level)?

Made-up product names aren't real words, so a **word-level** model (which picks from a
fixed vocabulary of existing words) literally cannot produce them. A **character-level**
model works at the level of individual letters and therefore can synthesize entirely
new tokens like `Burple-Fink` or `Corvella`. The trade-off is that it must also *learn
to spell*, which is exactly what makes the early-training output so gloriously broken.

## 3. How it works (the core loop)

The model is trained on a single task:

> Given the characters seen so far in a name, predict the **next character**.

- Each name is bracketed with a **start token** and an **end token**, e.g.
  `^Toyota$`.
- Characters are mapped to integer IDs (the **vocabulary**).
- The network reads the sequence one character at a time. Its hidden state is a
  running "memory" of what it has read.
- At each step it outputs a probability distribution over the next character.
- **Training:** compare the predicted next-char distribution to the actual next char
  (cross-entropy loss); backpropagate; repeat over the whole dataset for many epochs.
- **Generation:** start from `^`, sample a character from the model's output, feed it
  back in, and repeat until the model emits `$` (end of name).

### The creativity / temperature knob

When sampling, we divide the output logits by a **temperature** `T` before converting
to probabilities:

- `T < 1` sharpens the distribution → safe, repetitive, close to the training set.
- `T ≈ 1` = the model's raw opinion.
- `T > 1` flattens it → adventurous, misspelled, surreal (Shane's *Dondarf* mode).

This single dial is what produced both `Sane Green` and `Bylfgoam Glosd` in the
original experiment.

## 4. Architecture

```
input chars ──► Embedding ──► LSTM (1–2 layers, hidden≈128–256) ──► Linear ──► logits over vocab
```

- **Embedding layer:** turns each character ID into a small dense vector.
- **LSTM:** the recurrent core; a gated RNN that trains far more stably than a vanilla
  RNN and captures longer-range spelling dependencies. (Shane's `char-rnn` used LSTMs
  under the hood.)
- **Linear head:** projects the LSTM output back to a score per vocabulary character.

Everything is deliberately small so it trains in seconds-to-minutes on a laptop CPU.

## 5. Data pipeline (`src/data.py`)

1. Read a newline-separated list of names from a `.txt` file.
2. Normalize (strip whitespace, drop blanks/duplicates, optionally lowercase).
3. Build the vocabulary from every character that appears, plus `^` (start) and
   `$` (end) tokens.
4. For each name produce an `(input, target)` pair where `target` is `input` shifted
   left by one character — i.e. "predict the next char."
5. Batch by padding to the longest name in the batch.

## 6. Training pipeline (`src/train.py`)

- Load data, build model, pick an optimizer (Adam).
- For each epoch: forward pass → cross-entropy loss (ignoring pad positions) →
  backprop → step.
- Periodically print the loss and a few **live samples** so you can literally watch
  the model go from noise → almost-words → plausible names (the fun part).
- Save a checkpoint (`model weights + vocab + config`) to `./checkpoints/<name>.pt`.

## 7. Generation pipeline (`src/sample.py`)

- Load a checkpoint (weights + vocab).
- Generate N names at a chosen temperature.
- Filter out names that already exist in the training set (we want *new* ones) and
  optionally enforce a length range.

## 8. Datasets included

- `data/car_manufacturers.txt` — ~150 real auto brands worldwide.
- `data/car_models.txt` — ~250 real car model names.

> ⚠️ **Data-size reality check:** Shane used ~7,700 examples. Our seed lists are much
> smaller, so expect charming nonsense rather than polished output at first. The
> single biggest quality lever is **more training data** — see roadmap step 2.

## 9. Roadmap / extensions

1. **✅ MVP:** names-only char-RNN with temperature sampling. *(this repo)*
2. **More & cleaner data:** scrape/compile a few thousand real car names; dedupe,
   normalize casing, remove trim levels and years.
3. **Dual output (the paint-color trick):** have the network emit a name **and** a
   numeric attribute simultaneously — for paint that was RGB; for cars it could be
   horsepower, price tier, or a learned "sportiness"/"luxury" score. Implementation:
   add a second regression head and a combined loss.
4. **Conditioning:** let the user prime generation ("give me a name starting with 'Za'"
   or "an Italian-sounding sports-car name") via a seed prefix or a country/style tag.
5. **Packaging:** simple CLI (done) → optional Flask/FastAPI endpoint → tiny web demo.
6. **Alternative backbones:** swap the LSTM for a GRU or a small character Transformer
   and compare output quality.

## 10. Success criteria for the MVP

- Trains without errors on the bundled data on a CPU in a few minutes.
- After training, ≥50% of generated manufacturer names are (a) novel — not in the
  training set — and (b) pronounceable / "car-shaped."
- The temperature knob visibly changes output from conservative to chaotic.
