# web/ — the Burple-Fink front-end

A mobile-friendly "instrument panel" UI: pick an engine, drag the cold→hot **temperature
dial**, optionally set a starting prefix, and tap **Invent names**. Each result is flagged
`new` (never in the training data) or `real`, and tapping it copies the name.

There are two ways to run it — both share the CSS + markup in **`app_template.html`**, so
that single file is the source of truth for the look; only the "brain" differs.

## A) Live server (talks to the real PyTorch model)

```bash
python -m src.serve            # -> http://localhost:8000
```

Serves the UI and a `/api/generate` endpoint backed by the checkpoints in `checkpoints/`.
Stdlib-only (no extra installs) and bound to `0.0.0.0`, so from a phone on the same Wi-Fi
open `http://<your-computer-ip>:8000`.

## B) Static export (runs the net in the browser)

```bash
python -m src.export_web \
    --model checkpoints/car_manufacturers_ft.pt:"Car brands" \
    --model checkpoints/car_models_ft.pt:"Car models" \
    --out web/burple-fink.html
```

Produces **`burple-fink.html`** — one self-contained file with the weights baked in that
re-runs the char-RNN in JavaScript (no server, shareable, works offline). The exporter
checks its JS forward pass against the trained PyTorch model before writing, so the
in-browser network is faithful to the real one. Regenerate it whenever you retrain.

> `burple-fink.html` is a build artifact checked in for convenience. If you change
> `app_template.html` or retrain, rebuild it with the command above.

**Bundling many engines:** every `--model` embeds its full weight matrices as JSON, so
file size scales with `hidden_dim` squared *and* the number of engines — at the default
`hidden_dim=256` a 13-engine export came out to ~92MB (unshippable). Pretrain the shared
base with a smaller `--hidden-dim` (96 gives ~1.1MB/engine, ~14MB for 13 engines, with no
visible quality loss on this project's dataset sizes) when exporting more than a couple
of engines:
```bash
python -m src.pretrain --name base --hidden-dim 96
```
Also see `src/finetune.py`'s `FINETUNE_EPOCHS`/`FINETUNE_LR` comment — a base shared
across many dissimilar domains needs a less gentle fine-tune than the original two
(similar, automotive) domains did, or engines leak each other's vocabulary.
