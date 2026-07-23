# CLAUDE.md — repo guide for agents

Burple-Fink is a character-level RNN **name-generation platform**: one reusable char-RNN
(LSTM) engine, trained and fine-tuned on many datasets to invent a variety of names. It
is a **multi-stage, multi-agent** project.

## 👉 Before you start
**Read [`HANDOFF.md`](HANDOFF.md).** It defines the stages, the open workstreams, how to
claim work, dataset conventions, the fine-tuning design, and the definition of done.
Overview is in [`README.md`](README.md); design detail in [`docs/PLAN.md`](docs/PLAN.md).

## Commands
```bash
pip install -r requirements.txt                                   # default PyPI index only (see below)
python -m src.train --data data/<x>.txt --epochs 300 --name <x>   # train, saves checkpoints/<x>.pt
python -m src.sample --checkpoint checkpoints/<x>.pt --num 20 --temperature 0.8
python generate.py --data data/<x>.txt --train --name <x> --num 20  # train + generate in one go
```

## Layout
- `src/config.py` — hyperparameters + special tokens (`PAD`/`START`/`END`).
- `src/data.py` — name loading, `Vocab`, next-char `(input, target)` pairs, batching.
- `src/model.py` — the char-RNN: `Embedding → LSTM → Linear`; keep `forward(x, hidden)`.
- `src/train.py` — training loop, live sample previews, checkpoint save.
- `src/sample.py` — checkpoint load + temperature ("creativity") sampling.
- `generate.py` — friendly wrapper.
- `data/*.txt` — datasets, one name per line. Adding one? Follow HANDOFF §5.

## Conventions
- **Checkpoint format is a contract** (`model_state`, `config`, `vocab`, `training_names`).
  Don't break these keys — other stages depend on them. See HANDOFF §2.
- **Branch per workstream:** `claude/ws<N>-<slug>`, branched from `main`, PR into `main`.
  Claim it in the HANDOFF §7 table.
- **Low-collision zones** (edit freely): new files in `data/` and `src/`.
  **High-collision zones** (minimal diffs, call out in PR): `config.py`, `data.py`, the
  checkpoint format, and the top-level docs.
- Docstrings explain *why*, matching the existing style.

## Guardrails
- Install PyTorch from the **default PyPI index** — `download.pytorch.org` is blocked in
  this environment. `numpy` is optional (Torch prints a harmless warning without it).
- Keep models small and **CPU-trainable in minutes**.
- **Never commit checkpoints or large binaries** — `checkpoints/` and `*.pt` are gitignored.
- Verify before you hand off: `train` for ~40 epochs and `sample` should both run clean.
