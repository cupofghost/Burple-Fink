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
- `data/english_words.txt` — ~8,600 common English words. This is the "just make it
  *plausible*" corpus: unlike the domain-specific brand/model lists, it isn't about any
  one topic — it teaches the model the general shape of English spelling so it emits
  believable word-forms (the *Burple-Fink* effect). Sourced from a frequency-ranked
  common-word list (Google Web Trillion Word Corpus via `first20hours/google-10000-english`,
  MIT) and curated to lowercase `a–z`, length 3–10, vowel-bearing tokens. At ~8.6k entries
  it's the largest corpus here (near Shane's ~7,700), which also makes it strong fuel for
  the shared-vocab base model in §11.2.

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

## 11. Fine-tuning, evaluation, and serving (Stages 2 / 3 / 5)

Stage 0 trained a fresh model per dataset. Everything below turns that into the
platform described in the README: one base model, specialized cheaply per domain,
measured objectively, and served through a phone-friendly UI.

### 11.1 Shared vocabulary — the enabling decision

A char-RNN's embedding and output layers are sized to its vocabulary. If the base
model and a fine-tune disagree on the character↔id mapping, the saved weights simply
don't fit the new model. So **fine-tuning requires a single vocabulary shared by every
checkpoint that exchanges weights.**

Design (implemented in `src/data.py`):

- `build_shared_vocab(paths)` builds one `Vocab` from the **union of characters across
  all datasets** — the existing `Vocab` already derives its char set from the names it's
  given, so feeding it every dataset yields a superset that each per-dataset vocab is
  contained in.
- It is persisted to `data/shared_vocab.json` (via the unchanged `Vocab.to_dict`) the
  first time `src/pretrain.py` runs, and every fine-tune loads that same file. Keeping it
  on disk makes the mapping stable across future runs.
- `filter_to_vocab(names, vocab)` splits a dataset into representable vs. dropped names,
  so a dataset that introduces a brand-new character fails loudly (with a hint to rebuild
  the vocab) instead of crashing deep in encoding.

`Vocab.from_dict` / `to_dict` and the checkpoint format (HANDOFF §2) are **unchanged**;
the shared vocab is just a specific `Vocab` that happens to span every dataset.

### 11.2 Pretrain → fine-tune (transfer learning)

- `src/train.py` was refactored so its epoch loop lives in a reusable
  `fit(model, vocab, names, cfg)` plus a `save_checkpoint(...)` helper. `train()` (and its
  CLI) behave exactly as before; pretraining and fine-tuning reuse the same loop.
- `src/pretrain.py` builds the shared vocab, trains a base model on the concatenation of
  all datasets, and saves `checkpoints/base.pt`.
- `src/finetune.py` loads the base checkpoint (inheriting its architecture **and** shared
  vocab), then continues training on one dataset with a **lower learning rate for fewer
  epochs** (defaults: 5e-4, 60 epochs) — nudging the general speller toward a domain
  rather than overwriting it. Output: `checkpoints/<dataset>_ft.pt`, with `training_names`
  set to the fine-tune dataset so novelty is judged against that domain.

### 11.3 Evaluation harness

`src/evaluate.py` turns "looks fun" into numbers for a checkpoint against its own
training set:

- **Novelty** — fraction of distinct generated names not present in training.
- **Plausibility** — mean character-bigram log-likelihood under an add-1-smoothed model
  of the training names, reported for generated names *and* real names so the ratio is
  interpretable (~1 = as typical as real names). Plus the model's own training NLL as a
  fit sanity-check (labeled not-held-out).
- **Diversity** — uniqueness rate and mean pairwise Levenshtein distance.

These make the temperature trade-off measurable and let WS-2 prove fine-tuning helps.

### 11.4 Serving (two front-ends, one UI)

Both share `web/app_template.html` (the instrument-panel CSS + markup); only the "brain"
differs:

- **`src/export_web.py`** bakes a checkpoint's weights into one self-contained HTML file
  that re-runs `Embedding → LSTM → Linear` **in JavaScript**. Before writing, it verifies
  a pure-Python reference of that same forward pass reproduces the real PyTorch model's
  logits (tolerance 5e-3) — so the browser net can't silently diverge from the trained
  one. Good for sharing / offline / opening on a phone with no server.
- **`src/serve.py`** is a stdlib-only HTTP server that serves the same UI but generates
  by calling the live checkpoints in-process (`/api/generate`). It binds `0.0.0.0` so a
  phone on the same network can reach it, and swapping in a newly trained checkpoint
  changes the output with no re-export.

### 11.5 What's still open

- **WS-1 (more datasets):** the single biggest quality lever is still more data; the
  fine-tuning machinery is ready for any dataset dropped into `data/` (rebuild the shared
  vocab by deleting `data/shared_vocab.json` and re-running pretrain).
- **WS-4 (dual-output):** ✅ done — see §11.6.

### 11.6 Dual-output: name + numeric attribute (WS-4)

Implements the Shane paint-color trick (§9.3): the network emits a name **and**
regresses a scalar attribute from the same LSTM encoder, trained with a combined loss.

- **`Config.dual_output` / `value_mean` / `value_std` / `value_label`** (`src/config.py`)
  — four new fields, all defaulting to "off"/neutral, so every existing (non-dual)
  checkpoint is unaffected. `value_mean`/`value_std` are z-score stats used to
  denormalize the head's output back to the attribute's real scale; `value_label` is a
  human string (e.g. `"founding year"`) purely for display.
- **`CharRNN.value_head`** (`src/model.py`) — an `nn.Linear(hidden_dim, 1)`, created only
  when `cfg.dual_output=True` (`None` otherwise). `forward(x, hidden)` is byte-for-byte
  unchanged; a new `encode(x, hidden)` factors out the embedding+LSTM step so the value
  head can read the per-timestep LSTM output directly, and `predict_value(state)` applies
  the head to a `(batch, hidden_dim)` summary vector.
- **`data.load_name_value_pairs(path)`** (`src/data.py`) — reads a `name<TAB>value` file
  (comments/blank lines skipped), the WS-4 dataset format.
- **`src/train_dual.py`** (new module) — trains a dual-output model:
  `combined_loss = ce_loss + value_weight * mse_loss`. The value target for each
  training example is gathered from the LSTM output at that sequence's own last
  non-pad timestep (via the batch's `lengths`), so padding never leaks into the
  regression signal. Kept as its own module (rather than extending `src/train.py`'s
  `fit()`) to keep the single-output training path untouched.
- **`sample.generate_one`/`generate_many`** (`src/sample.py`) gained an opt-in
  `return_value` parameter; default `False` preserves the exact old return type
  (`str` / `List[str]`). `python -m src.sample` auto-detects `cfg.dual_output` on the
  loaded checkpoint and prints the denormalized value alongside each name.
- **Checkpoint format (§2) is unchanged** — no new top-level keys. The dual-output
  fields all live inside the already-extensible `config` dict, and `model_state` simply
  contains extra `value_head.*` tensors for dual checkpoints.
- **Seed dataset:** `data/car_manufacturers_founding_year.tsv` — 66 manufacturers paired
  with a founding year (sourced from each brand's commonly-cited Wikipedia-infobox
  founding date, including well-known predecessor-company years, e.g. Suzuki 1909 as a
  loom maker; ambiguous/contested cases were left out rather than guessed).
- **Try it:**
  ```bash
  python -m src.train_dual --data data/car_manufacturers_founding_year.tsv \
      --epochs 300 --name manufacturers_founding_year --value-label "founding year"
  python -m src.sample --checkpoint checkpoints/manufacturers_founding_year.pt --num 10
  ```
