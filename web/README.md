# web/ — the Burple-Fink front-end

A mobile-first **gallery** of name generators. Pick a generator from a searchable list
grouped by domain, drag the cold→hot **temperature dial**, tap **Invent names**. Each
result is flagged `new` or `in training data`, tap a name to copy it, tap ★ to keep it.
Kept names survive a reload.

There are two ways to run it, and since WS-12 they share **one** file — `app_template.html`
holds the CSS, the markup *and* the JavaScript. A build differs only in what gets spliced
into two markers:

| marker | static export | live server |
|---|---|---|
| `/*__BURPLE_MODELS__*/` | models **with weights** + dataset catalog | model metadata only |
| `/*__BURPLE_ENGINE__*/` | (empty — the browser runs the net) | `fetch("/api/generate")` |

That is why a feature added to the gallery shows up in both front ends at once. Before
wave 3 the server kept a second copy of the UI script, and the two had already drifted.

## A) Live server (talks to the real PyTorch model)

```bash
python -m src.serve            # -> http://localhost:8000
```

Serves the UI and `/api/generate`, backed by the checkpoints in `checkpoints/`. Stdlib-only
(no extra installs) and bound to `0.0.0.0`, so from a phone on the same Wi-Fi open
`http://<your-computer-ip>:8000`.

**This is the front end that scales.** The weights never leave the Python process, so the
page carries labels, not models: ~70 KB with one checkpoint and ~83 KB with all 30, versus
5.25 MB for the static export of the same 30. Anything the gallery marks `server` instead
of `offline` is available here.

| endpoint | returns |
|---|---|
| `GET /api/health` | `{status, models, checkpoints:[…]}` |
| `GET /api/models` | `{models:[{id,label,domain,verified,provenance,count}]}` |
| `GET /api/generate` | `{names:[{name,novel,exact}]}` |

`/api/generate` takes `engine` (dataset id, e.g. `dog_breeds`), `count`, `temperature`,
`prefix`, `novel_only`, and the WS-7 decoding controls `top_k`, `top_p`,
`repetition_penalty`, `min_length`. Every one defaults to off, so an old client gets
exactly the pre-wave-3 behavior. Bad input returns a JSON `{"error": …}` with a 400; a
model that blows up mid-request returns 500 and leaves the server running.

## B) Static export (runs the net in the browser)

```bash
python -m src.export_web \
    --model checkpoints/ws12_car_manufacturers.pt \
    --model checkpoints/ws12_pharma_drugs.pt \
    --out web/burple-fink.html
```

Produces **`burple-fink.html`** — one self-contained file with the weights baked in that
re-runs the char-RNN in JavaScript. No server, shareable, works offline, no CDN or font
requests. Labels and domains come from `data/<stem>.meta.json` where present and degrade
gracefully where absent, so `--model path.pt` alone is usually enough;
`path.pt:Label:Domain` overrides.

The exporter **verifies** its own forward pass against the trained PyTorch model before
writing, and refuses to export if they disagree by more than 5e-3 in any logit. That check
is what makes the in-browser net trustworthy rather than merely plausible, so
`tests/test_web.py` corrupts weights five different ways and asserts the check *fails*.

---

## The size trade-off (why not all 30 models at full size?)

`data/` holds 30 datasets. Baking all of them into one HTML file at the **default training
config** is not an option, and it is worth being precise about why. Measured on a real
checkpoint (`hidden_dim=256`, `num_layers=2`, `embedding_dim=32` — 838,902 parameters):

| encoding | per model | × 30 |
|---|---|---|
| nested JSON, 5 decimals (pre-wave-3 format) | 6.72 MB | 202 MB |
| base64 float32 | 4.27 MB | 128 MB |
| **base64 float16** (current) | **2.13 MB** | **64 MB** |

There are only two levers, and they cost very different things:

**1. Encoding — a free 3.1x.** Weights now ship as one base64 float16 blob per model
instead of nested JSON arrays. This is free because it is *verified*: on that same
checkpoint the worst logit error against PyTorch moves from 4.9e-5 (JSON, 5 decimals) to
2.1e-4 (float16), still 24x inside the 5e-3 tolerance. If float16 ever does miss the
tolerance for some checkpoint, `export_model` re-packs that one model as float32 and
re-verifies rather than shipping something unfaithful — the file gets bigger, never
wronger. (gzip was measured too and rejected: only 1.09x on float16 weight bytes, which
does not justify depending on `DecompressionStream`.)

**2. Model size — a 12x, and it costs quality.** A checkpoint trained with
`--hidden-dim 64` has 63,897 parameters instead of 838,902:

| training config | params | per model, f16 | × 30 |
|---|---|---|---|
| default (`hidden_dim=256`) | 838,902 | 2.13 MB | 64 MB |
| **gallery (`--hidden-dim 64`)** | **63,897** | **~176 KB** | **~5.2 MB** |

(The per-model figure includes each dataset's training names, which ride along so the
browser can flag memorized output honestly — 4.5–14 KB per dataset.)

So the shipped `burple-fink.html` bundles **gallery-size checkpoints**, and all 30 fit in
about 5 MB — a file a phone opens in a second or two.

### What was chosen, and what it costs

- **`--budget-mb` (default 8.0)** caps the whole page. Models are added in the order you
  pass them until the budget is reached; the first model is never dropped, and if it alone
  overshoots, the exporter says so rather than pretending.
- **Nothing is silently dropped.** Every dataset in `data/` is listed in the gallery.
  Bundled ones get an `offline` chip; the rest get a `server` chip and a disabled row
  reading "not in this file". A 30-dataset repo never looks like a 3-dataset one.
- **`--max-models N`** is a hard count cap for when you want a deliberately small file.

**What the 13x size cut actually costs.** Measured rather than assumed — same data, same
15% held-out split, 60 epochs, best validation loss (lower is better):

| dataset | `hidden_dim=256` | `hidden_dim=64` | cost |
|---|---|---|---|
| `cheeses` (421 names) | 2.4398 (epoch 12) | 2.4425 (epoch 31) | **+0.1%** |
| `dog_breeds` (461 names) | 1.9155 (epoch 18) | 2.0299 (epoch 36) | **+6.0%** |

So the gallery-size model is between "indistinguishable" and "6% worse" on held-out loss
while being 13x smaller. Note *where* the big model peaks: epoch 12–18 out of 60, after
which its validation loss nearly doubles (2.44 → 3.78 on cheeses) while the small model
degrades gently (2.44 → 2.62). That is wave 2's finding again — *"a 135-name dataset
cannot support a 2-layer, 256-wide LSTM"* — showing up as overfitting on 400-name datasets
too. The small model is not merely a compression compromise here; on this data it is also
the better-behaved one.

The cost is still not zero, and it will grow for the larger datasets (`english_words` at
8,631 names, `pharma_drugs` at 2,223). If you want the biggest model for a particular
dataset, run the server (A) instead of the export (B). That is what the server is for.

Rebuilding the shipped file:

```bash
python -m src.export_web $(for f in checkpoints/ws12_*.pt; do echo --model $f; done) \
    --out web/burple-fink.html
```

The exporter prints a per-model size table (params, weight bytes, training-name bytes,
total) and the final file size on every run. `tests/test_web.py` asserts that accounting is
exact rather than estimated, so "don't ship a 40 MB page" is a checkable claim.

> `burple-fink.html` is a build artifact checked in for convenience. If you change
> `app_template.html` or retrain, rebuild it with the command above.

---

## The decoding controls, and why they are ordered this way

`src/sample.py` gained `top_k`, `top_p`, `repetition_penalty` and `min_length` in wave 2.
All four are in the UI now, but **not presented as equals**, because wave 2 measured them:

> Plain temperature sampling at 1.1–1.3 beat every top-k/nucleus setting tried, on both
> checkpoints. Truncating the tail didn't reduce junk here, but it did shrink the pool of
> reachable characters enough to push sampling back toward memorized training names: at
> temperature 1.3, novelty dropped from 38% to 32% (`top_k=10`) and near-duplicate rate
> rose from 72% to 80%.

So the panel puts forward what actually won:

1. **Temperature** — the primary dial, defaulting to **1.15**, inside the measured band,
   with `1.1–1.3 measured best` printed on the scale.
2. **Repetition penalty** — the second dial. It survived the sweep because it targets a
   failure the others don't touch: the `Bylfgoammm` character-repeat stutter.
3. **Top-k / top-p / min length** — in a collapsed drawer labelled *"Truncation — measured
   worse here"*, containing the numbers above and defaulting to off.

They are available because they are real controls and the owner may want to explore them.
They are not the recommendation, and the UI does not imply they are. `min_length` sits with
them but is framed neutrally — it is a filter, not a quality claim.

The order of operations in the browser matches `src/sample.py` exactly: repetition penalty
→ temperature → top-k → top-p → min-length masking → softmax → sample.

---

## Output: copy, keep, and honest novelty

- **Copy** — tap any name; **Copy all** takes the whole batch as newline-separated text.
- **Keep** — tap ★ to add a name to a favorites list persisted in `localStorage`
  (`burple-fink.kept.v1`), which survives a reload and records which generator produced
  each name. **Keep all new** takes the novel ones in one tap. If `localStorage` is
  unavailable (private browsing) the list degrades to memory and says so rather than
  failing silently.
- **Novelty is not flattering.** A generated name that is character-for-character in the
  training data is rendered in muted text with an amber `in training data` badge and an
  amber rule — never as an invention. Case-only matches (`Rolls-royce` vs `Rolls-Royce`)
  are also treated as copies, since calling one an invention because a letter changed case
  would be dishonest. The batch summary reports both halves: *"7 of 12 never existed
  before · 5 copied from training data"*. Both front ends apply the identical rule.
- **Provenance is on the page.** Every wave-3 dataset carries `verified: false` in its
  sidecar — the names were recalled from general knowledge, not cross-checked against a
  primary source. A small amber-dotted *"Unverified training data"* disclosure under the
  generator picker expands to that dataset's `provenance` text, so the page never quietly
  implies its inputs are authoritative.

## Tests

`tests/test_web.py` — 72 tests, ~5 s — covers the weight codec, the fidelity check
*failing* on corrupted weights, the size accounting, the multi-model manifest, sidecar
metadata handling, template invariants (markers consumed, no external resources, every
`getElementById` target exists), and the server's happy paths plus its 400/404/500/503 JSON
error paths. It builds its checkpoints with `torch.save` directly and never runs a training
loop, so it stays fast and independent of the lanes editing `src/train.py`.

Where `node` is available it additionally runs the **actual shipped JavaScript** and
compares it to PyTorch and to `src/sample.py`:

- the browser's forward pass matches PyTorch within the exporter's 5e-3 bar, and matches
  the Python reference pass to ~1e-15 — which is what makes the exporter's guarantee
  meaningful for the browser rather than only for a stand-in;
- all four decoding controls (`top_k`, `top_p`, `repetition_penalty`, `min_length`, plus
  the order they are applied in) reproduce `src/sample.py`'s distribution to within 2e-3,
  recovered exactly by sweeping a stubbed `Math.random`.

These skip rather than fail when node is absent — node is a testing convenience, never a
runtime dependency of the export.
