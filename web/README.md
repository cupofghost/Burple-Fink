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
