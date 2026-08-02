"""Export trained checkpoints to a self-contained web UI that runs in the browser.

Why in the browser (and not a server)?
--------------------------------------
The models here are tiny, so the honest way to give someone a name generator they can
*open on their phone* — with no server to host, no Python to install, no network round
trip — is to ship the actual trained weights inside a single HTML file and re-run the
exact ``Embedding -> LSTM -> Linear`` forward pass in JavaScript. This module is the
bridge: it pulls the weights out of a checkpoint, lays them out the way the JS engine
expects, and (critically) **verifies** a reference re-implementation of the forward pass
reproduces the real PyTorch model's logits before trusting them. If that check fails we
refuse to export, so the web UI can never silently diverge from the trained model.

The JS engine in ``web/app_template.html`` mirrors this exact math. Keep the two in sync.

Wave 3 (WS-12) changed *how the weights are written*, not what they mean:

* Weights ship as one base64 **float16** blob per model instead of nested JSON arrays.
  Measured on a default-config checkpoint (H=256, L=2, 838,902 parameters): 6.72 MB of
  JSON text becomes 2.13 MB of base64 — a 3.1x reduction — while the worst-case logit
  error against PyTorch only moves from 4.9e-5 to 2.1e-4, still 24x inside the 5e-3
  tolerance. If float16 ever *does* miss the tolerance for some checkpoint, the exporter
  automatically re-packs that model as float32 and re-verifies rather than shipping it.
* The fidelity check now runs on the **unpacked wire bytes**, not on an intermediate. What
  gets verified is exactly what the browser will decode and execute.
* Forty models do not fit in one HTML file at any encoding (see ``web/README.md``), so the
  export takes a size budget. Models that do not fit are still listed in the catalog and
  shown in the gallery as "not in this file" — never silently dropped.

Usage:
    python -m src.export_web \\
        --model checkpoints/car_manufacturers_ft.pt:"Car brands" \\
        --model checkpoints/car_models_ft.pt:"Car models" \\
        --template web/app_template.html --out web/burple-fink.html
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import math
import os
import struct
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .data import Vocab
from .sample import load_checkpoint

# Marker in the HTML template that the model bundle is spliced into.
TEMPLATE_MARKER = "/*__BURPLE_MODELS__*/"
# Marker where a build may override the in-browser engine. The static export leaves it
# empty (the browser runs the net); src/serve.py replaces it with a fetch to /api/generate
# so both front ends share one copy of the UI instead of two that drift apart.
ENGINE_MARKER = "/*__BURPLE_ENGINE__*/"

# Bundle format version. The JS engine checks this so a stale template and a fresh
# bundle fail loudly instead of mis-decoding weights into plausible-looking garbage.
BUNDLE_FORMAT = 3

# Default size budget for the whole HTML file. A phone opens ~8 MB of HTML comfortably;
# it does not open 85 MB, which is what forty default-config models would cost.
DEFAULT_BUDGET_MB = 8.0

# Largest magnitude IEEE-754 binary16 can represent. Anything past this would encode as
# inf, so a checkpoint containing such a weight is forced to float32.
F16_MAX = 65504.0

DOMAIN_FALLBACK = "Other"


# --------------------------------------------------------------------------------------
# weight packing
# --------------------------------------------------------------------------------------

def pack_floats(values: Sequence[float], dtype: str) -> str:
    """Pack a flat float sequence into little-endian base64. ``dtype`` is f16 or f32."""
    code = "e" if dtype == "f16" else "f"
    raw = struct.pack("<%d%s" % (len(values), code), *values)
    return base64.b64encode(raw).decode("ascii")


def unpack_floats(blob: str, count: int, dtype: str) -> List[float]:
    """Inverse of :func:`pack_floats` — the Python twin of the JS decoder.

    Kept deliberately simple and shape-free: the browser and this function both rebuild
    the tensors purely from the model's declared dimensions, so there is no per-tensor
    metadata that could drift between the two implementations.
    """
    code = "e" if dtype == "f16" else "f"
    raw = base64.b64decode(blob)
    return list(struct.unpack("<%d%s" % (count, code), raw[: count * (2 if dtype == "f16" else 4)]))


def tensor_order(sd: Dict[str, torch.Tensor], num_layers: int) -> List[torch.Tensor]:
    """The one canonical tensor order shared by the packer, the JS engine and the
    reference forward pass below.

    embedding | per layer: w_ih, w_hh, b_ih, b_hh | head_w | head_b
    """
    out = [sd["embedding.weight"]]
    for i in range(num_layers):
        out.append(sd[f"lstm.weight_ih_l{i}"])
        out.append(sd[f"lstm.weight_hh_l{i}"])
        out.append(sd[f"lstm.bias_ih_l{i}"])
        out.append(sd[f"lstm.bias_hh_l{i}"])
    out.append(sd["head.weight"])
    out.append(sd["head.bias"])
    return out


def flat_weights(sd: Dict[str, torch.Tensor], num_layers: int) -> List[float]:
    flat: List[float] = []
    for t in tensor_order(sd, num_layers):
        flat.extend(t.flatten().tolist())
    return flat


def weight_count(vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int) -> int:
    """How many floats the blob must hold, derived only from the declared dimensions."""
    n = vocab_size * embedding_dim
    inp = embedding_dim
    for _ in range(num_layers):
        n += 4 * hidden_dim * inp + 4 * hidden_dim * hidden_dim + 8 * hidden_dim
        inp = hidden_dim
    n += vocab_size * hidden_dim + vocab_size
    return n


class Unpacked:
    """A model's weights as flat lists plus the shapes needed to index them.

    Both the verification forward pass here and the JS engine read weights through this
    same flat layout, so "the browser runs different code" can only ever be a bug in one
    file, not a difference in data layout.
    """

    def __init__(self, m: Dict):
        self.H = H = m["hidden_dim"]
        self.L = m["num_layers"]
        self.E = E = m["embedding_dim"]
        self.V = V = len(m["itos"])
        n = weight_count(V, E, H, self.L)
        w = unpack_floats(m["weights"], n, m["dtype"])
        if len(w) != n:
            raise RuntimeError(
                f"Weight blob for {m['label']!r} holds {len(w)} floats, "
                f"but its declared dimensions need {n}.")
        self.w = w
        # offsets into the flat blob, in tensor_order
        o = 0
        self.emb = o; o += V * E
        self.layers: List[Tuple[int, int, int, int, int]] = []
        inp = E
        for _ in range(self.L):
            w_ih = o; o += 4 * H * inp
            w_hh = o; o += 4 * H * H
            b_ih = o; o += 4 * H
            b_hh = o; o += 4 * H
            self.layers.append((w_ih, w_hh, b_ih, b_hh, inp))
            inp = H
        self.head_w = o; o += V * H
        self.head_b = o; o += V


def _sigmoid(x: float) -> float:
    if x < -60.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def forward_logits(u: Unpacked, ids: Sequence[int]) -> List[float]:
    """Re-run Embedding -> LSTM -> Linear from the *unpacked wire bytes*.

    This is the same algorithm, over the same flat layout, that the browser runs.
    Comparing its output to the real torch model is what lets us guarantee the web UI is
    faithful to the trained net.
    """
    H, L, E, V, w = u.H, u.L, u.E, u.V, u.w
    h = [[0.0] * H for _ in range(L)]
    c = [[0.0] * H for _ in range(L)]

    x: List[float] = [0.0] * H
    for tok in ids:
        x = w[u.emb + tok * E: u.emb + (tok + 1) * E]
        for li in range(L):
            w_ih, w_hh, b_ih, b_hh, inp = u.layers[li]
            hl, cl = h[li], c[li]
            new_h = [0.0] * H
            new_c = [0.0] * H
            for k in range(H):
                # gate rows are stacked [input, forget, cell, output] in the 4H dim
                def pre(off: int) -> float:
                    r = off + k
                    s = w[b_ih + r] + w[b_hh + r]
                    base_i = w_ih + r * inp
                    for j in range(inp):
                        s += w[base_i + j] * x[j]
                    base_h = w_hh + r * H
                    for j in range(H):
                        s += w[base_h + j] * hl[j]
                    return s

                i_g = _sigmoid(pre(0))
                f_g = _sigmoid(pre(H))
                g_g = math.tanh(pre(2 * H))
                o_g = _sigmoid(pre(3 * H))
                new_c[k] = f_g * cl[k] + i_g * g_g
                new_h[k] = o_g * math.tanh(new_c[k])
            h[li], c[li] = new_h, new_c
            x = new_h  # output of this layer is the input to the next

    logits = []
    for v in range(V):
        base = u.head_w + v * H
        s = w[u.head_b + v]
        for j in range(H):
            s += w[base + j] * x[j]
        logits.append(s)
    return logits


# Backwards-compatible alias: the old private name, kept so nothing outside this module
# breaks if it imported it.
_forward_logits = forward_logits


@torch.no_grad()
def verify(m: Dict, torch_model, vocab: Vocab, device: str = "cpu",
           tol: float = 5e-3) -> float:
    """Assert the *packed* weights reproduce the torch model's logits (else refuse).

    Returns the worst observed logit difference so callers can report the margin.
    Raises ``RuntimeError`` if any probe exceeds ``tol``.
    """
    torch_model.eval()
    u = Unpacked(m)
    # Build probe sequences from real characters in *this* vocab (indices 3+ skip the
    # PAD/START/END specials) so the check works for any dataset, not just ones that
    # happen to contain a hardcoded letter. The long probe matters: quantization error
    # compounds through the recurrence, so a single step would not catch drift.
    real = vocab.itos[3:]
    probes = ["", real[0] if real else "", "".join(real[:2]), "".join(real[:6])]
    worst = 0.0
    worst_probe = ""
    for probe in probes:
        ids = [vocab.start_id] + [vocab.stoi[ch] for ch in probe]
        ref = forward_logits(u, ids)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = torch_model(inp)
        torch_last = logits[0, -1, :].tolist()
        max_diff = max(abs(a - b) for a, b in zip(ref, torch_last))
        if max_diff > worst:
            worst, worst_probe = max_diff, probe
    if worst > tol:
        raise RuntimeError(
            f"Web export failed fidelity check for prime {worst_probe!r}: max logit diff "
            f"{worst:.4g} > {tol}. The JS engine would not match the trained model.")
    return worst


# Old private name kept as an alias (signature-compatible with the pre-wave-3 helper).
_verify = verify


# --------------------------------------------------------------------------------------
# dataset metadata (the gallery's grouping and labels)
# --------------------------------------------------------------------------------------

def pretty_label(stem: str) -> str:
    """'car_manufacturers' -> 'Car manufacturers' — the graceful fallback."""
    return stem.replace("_", " ").replace("-", " ").strip().capitalize() or stem


def read_meta(stem: str, data_dir: str = "data") -> Dict:
    """Read ``data/<stem>.meta.json`` if the dataset carries one.

    Wave 3's new datasets ship a sidecar with ``label`` and ``domain``; the older ones do
    not. Missing file, unreadable file and missing keys all degrade to the same graceful
    fallback rather than failing the export.
    """
    path = os.path.join(data_dir, f"{stem}.meta.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        if not isinstance(meta, dict):
            return {}
        return meta
    except (OSError, ValueError):
        return {}


def resolve_dataset(checkpoint_path: str, cfg=None, data_dir: str = "data") -> Dict:
    """Work out which dataset a checkpoint came from, and how to present it.

    Resolution order, most trustworthy first:

    1. ``cfg.dataset_path`` / ``cfg.dataset_label`` — the fields wave 3 pre-wired into
       ``src/config.py`` for exactly this. Empty on checkpoints trained before that.
    2. The checkpoint filename: ``ws12_dog_breeds_ft.pt`` -> ``dog_breeds``, by stripping
       an ``_ft`` suffix and then matching the longest ``data/*.txt`` stem that the
       filename ends with (so lane prefixes like ``ws12_`` do not defeat the lookup).
    3. Give up on the dataset and just prettify the filename.
    """
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    if stem.endswith("_ft"):
        stem = stem[:-3]

    dataset = ""
    if cfg is not None:
        cfg_path = getattr(cfg, "dataset_path", "") or ""
        if cfg_path:
            dataset = os.path.splitext(os.path.basename(cfg_path))[0]

    if not dataset:
        known = [os.path.splitext(os.path.basename(p))[0]
                 for p in glob.glob(os.path.join(data_dir, "*.txt"))]
        # longest match wins: 'car_models' must not shadow 'car_models_electric'
        matches = [k for k in known if stem == k or stem.endswith("_" + k)]
        if matches:
            dataset = max(matches, key=len)

    meta = read_meta(dataset, data_dir) if dataset else {}
    cfg_label = (getattr(cfg, "dataset_label", "") or "") if cfg is not None else ""
    label = cfg_label or meta.get("label") or pretty_label(dataset or stem)
    domain = meta.get("domain") or DOMAIN_FALLBACK
    return {
        "id": dataset or stem,
        "label": str(label),
        "domain": str(domain),
        # Provenance rides along to the UI. Every wave-3 dataset is verified:false --
        # recalled from general knowledge, not cross-checked against a primary source --
        # and a page that quietly implied otherwise would be lying about its own inputs.
        "verified": bool(meta.get("verified", False)),
        "provenance": str(meta.get("provenance", "")),
    }


def dataset_catalog(data_dir: str = "data") -> List[Dict]:
    """Every dataset in ``data/``, labelled and grouped — bundled or not.

    The gallery shows the whole catalog so the file never implies the eleven models it
    contains are all that exist.
    """
    entries = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        meta = read_meta(stem, data_dir)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                count = len({ln.strip() for ln in fh if ln.strip()})
        except OSError:
            count = 0
        entries.append({
            "id": stem,
            "label": str(meta.get("label") or pretty_label(stem)),
            "domain": str(meta.get("domain") or DOMAIN_FALLBACK),
            "count": count,
            "verified": bool(meta.get("verified", False)),
            "provenance": str(meta.get("provenance", "")),
        })
    return entries


# --------------------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------------------

def export_model(checkpoint: str, label: Optional[str] = None, device: str = "cpu",
                 domain: Optional[str] = None, precision: str = "auto",
                 data_dir: str = "data") -> Dict:
    """Turn one checkpoint into a JSON-able description the JS engine can execute.

    ``precision`` is ``auto`` (try float16, fall back to float32 if the fidelity check
    rejects it), or a forced ``f16`` / ``f32``.
    """
    model, vocab, cfg, training_names = load_checkpoint(checkpoint, device)
    sd = model.state_dict()
    info = resolve_dataset(checkpoint, cfg, data_dir)

    embedding_dim = sd["embedding.weight"].shape[1]
    flat = flat_weights(sd, cfg.num_layers)
    expected = weight_count(len(vocab), embedding_dim, cfg.hidden_dim, cfg.num_layers)
    if len(flat) != expected:
        raise RuntimeError(
            f"{checkpoint}: packed {len(flat)} weights but its declared dimensions "
            f"imply {expected}. The exporter and the checkpoint disagree on the "
            f"architecture — refusing to write a bundle the browser would mis-decode.")

    base = {
        "id": info["id"],
        "label": label or info["label"],
        "domain": domain or info["domain"],
        "verified": info["verified"],
        "provenance": info["provenance"],
        "hidden_dim": cfg.hidden_dim,
        "num_layers": cfg.num_layers,
        "embedding_dim": embedding_dim,
        "max_length": cfg.max_length,
        "default_temperature": cfg.temperature,
        "itos": list(vocab.itos),
        "pad_id": vocab.pad_id,
        "start_id": vocab.start_id,
        "end_id": vocab.end_id,
        # The training names ride along so the browser can flag which generated names are
        # genuinely novel vs. memorized copies — the WS-3 novelty metric, live. Stored as
        # one newline-joined string: same information, ~3 bytes per name cheaper than a
        # JSON array of quoted strings.
        "training_names": "\n".join(sorted(training_names)),
    }

    if any(abs(v) > F16_MAX for v in flat):
        order: Iterable[str] = ("f32",)
    elif precision == "auto":
        order = ("f16", "f32")
    else:
        order = (precision,)

    last_error: Optional[Exception] = None
    for dtype in order:
        candidate = dict(base, dtype=dtype, weights=pack_floats(flat, dtype))
        try:
            candidate["max_logit_error"] = round(verify(candidate, model, vocab, device), 8)
        except RuntimeError as exc:            # this encoding is not faithful enough
            last_error = exc
            continue
        candidate["bytes"] = model_bytes(candidate)
        return candidate

    raise RuntimeError(
        f"{checkpoint}: no supported weight encoding passed the fidelity check "
        f"({last_error})")


def model_bytes(m: Dict) -> int:
    """Exact serialized size of one model inside the bundle."""
    return len(json.dumps(m, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))


def plan_bundle(models: List[Dict], budget_mb: float = DEFAULT_BUDGET_MB,
                max_models: int = 0, overhead: int = 0) -> Tuple[List[Dict], List[Dict]]:
    """Split exported models into (bundled, skipped) under a size budget.

    Order is caller order — the first ``--model`` you pass is the first one kept, so the
    budget never silently reorders the owner's priorities. ``budget_mb <= 0`` disables the
    budget; ``max_models <= 0`` disables the count cap.
    """
    budget = int(budget_mb * 1024 * 1024) if budget_mb and budget_mb > 0 else 0
    kept: List[Dict] = []
    skipped: List[Dict] = []
    total = overhead
    for m in models:
        size = m.get("bytes") or model_bytes(m)
        if max_models > 0 and len(kept) >= max_models:
            skipped.append(dict(m, skip_reason="over --max-models"))
            continue
        if budget and total + size > budget and kept:
            skipped.append(dict(m, skip_reason="over --budget-mb"))
            continue
        kept.append(m)
        total += size
    return kept, skipped


def size_report(models: List[Dict], template_bytes: int = 0) -> Dict:
    """Byte accounting for the export, so 'don't ship a 40 MB page' is checkable.

    Every number is measured from the actual serialized bundle, not estimated.
    """
    rows = []
    for m in models:
        weights = len(m["weights"])
        names = len(m["training_names"].encode("utf-8"))
        total = m.get("bytes") or model_bytes(m)
        rows.append({
            "id": m["id"],
            "label": m["label"],
            "domain": m["domain"],
            "dtype": m["dtype"],
            "params": weight_count(len(m["itos"]), m["embedding_dim"],
                                   m["hidden_dim"], m["num_layers"]),
            "weight_bytes": weights,
            "training_name_bytes": names,
            "other_bytes": total - weights - names,
            "bytes": total,
        })
    models_bytes = sum(r["bytes"] for r in rows)
    return {
        "models": rows,
        "model_count": len(rows),
        "models_bytes": models_bytes,
        "template_bytes": template_bytes,
        "total_bytes": models_bytes + template_bytes,
    }


def build_bundle(models: List[Dict], skipped: Optional[List[Dict]] = None,
                 catalog: Optional[List[Dict]] = None) -> Dict:
    """Assemble the object the template gets, including the not-bundled manifest."""
    bundled_ids = {m["id"] for m in models}
    manifest = []
    seen = set()
    for entry in (catalog or []):
        seen.add(entry["id"])
        manifest.append(dict(entry, bundled=entry["id"] in bundled_ids))
    for m in (skipped or []):
        if m["id"] in seen:
            for entry in manifest:
                if entry["id"] == m["id"]:
                    entry["skip_reason"] = m.get("skip_reason", "not bundled")
            continue
        manifest.append({"id": m["id"], "label": m["label"], "domain": m["domain"],
                         "count": 0, "bundled": False,
                         "verified": m.get("verified", False),
                         "provenance": m.get("provenance", ""),
                         "skip_reason": m.get("skip_reason", "not bundled")})
    for m in models:
        if m["id"] not in seen:
            manifest.append({"id": m["id"], "label": m["label"], "domain": m["domain"],
                             "count": len(m["training_names"].split("\n")) if
                             m["training_names"] else 0, "bundled": True,
                             "verified": m.get("verified", False),
                             "provenance": m.get("provenance", "")})
    manifest.sort(key=lambda e: (e["domain"].lower(), e["label"].lower()))
    return {"format": BUNDLE_FORMAT, "models": models, "catalog": manifest}


def build_html(models: List[Dict], template_path: str, out_path: str,
               skipped: Optional[List[Dict]] = None,
               catalog: Optional[List[Dict]] = None) -> str:
    """Splice the model bundle into the HTML template, producing one standalone file."""
    with open(template_path, "r", encoding="utf-8") as fh:
        template = fh.read()
    if TEMPLATE_MARKER not in template:
        raise ValueError(f"Template {template_path!r} is missing marker {TEMPLATE_MARKER}")
    bundle = json.dumps(build_bundle(models, skipped, catalog),
                        ensure_ascii=True, separators=(",", ":"))
    html = template.replace(TEMPLATE_MARKER, bundle)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_path


def _parse_model_arg(spec: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Split ``path.pt[:Label[:Domain]]``; both label and domain fall back to metadata."""
    parts = spec.split(":")
    path = parts[0]
    label = parts[1] if len(parts) > 1 and parts[1] else None
    domain = parts[2] if len(parts) > 2 and parts[2] else None
    return path, label, domain


def _fmt(n: int) -> str:
    return f"{n / 1024:.0f} KB" if n < 1024 * 1024 else f"{n / 1048576:.2f} MB"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export checkpoints into a self-contained browser UI.")
    parser.add_argument("--model", action="append", required=True, dest="models",
                        metavar="CHECKPOINT[:LABEL[:DOMAIN]]",
                        help="A checkpoint, optional display label and domain; repeatable. "
                             "Label and domain default to data/<stem>.meta.json.")
    parser.add_argument("--template", default="web/app_template.html")
    parser.add_argument("--out", default="web/burple-fink.html")
    parser.add_argument("--json-out", default=None,
                        help="Optional: also write the raw model bundle JSON here.")
    parser.add_argument("--data-dir", default="data",
                        help="Where to look for <stem>.meta.json sidecars and the catalog.")
    parser.add_argument("--budget-mb", type=float, default=DEFAULT_BUDGET_MB,
                        help=f"Refuse to grow the page past this (default "
                             f"{DEFAULT_BUDGET_MB} MB; 0 = no budget). Models that do not "
                             f"fit are listed in the gallery as 'not in this file'.")
    parser.add_argument("--max-models", type=int, default=0,
                        help="Hard cap on how many models are bundled (0 = no cap).")
    parser.add_argument("--precision", choices=("auto", "f16", "f32"), default="auto",
                        help="Weight encoding. 'auto' tries float16 and falls back to "
                             "float32 if the fidelity check rejects it.")
    parser.add_argument("--no-catalog", action="store_true",
                        help="Do not list the un-bundled datasets from --data-dir.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exported = []
    for spec in args.models:
        path, label, domain = _parse_model_arg(spec)
        m = export_model(path, label, device, domain=domain,
                         precision=args.precision, data_dir=args.data_dir)
        exported.append(m)
        print(f"  ✓ {m['label']:<24} {m['domain']:<14} {m['dtype']} "
              f"{_fmt(m['bytes']):>9}  (logit error {m['max_logit_error']:.2e})")

    with open(args.template, "r", encoding="utf-8") as fh:
        template_bytes = len(fh.read().encode("utf-8"))

    kept, skipped = plan_bundle(exported, args.budget_mb, args.max_models,
                                overhead=template_bytes)
    catalog = None if args.no_catalog else dataset_catalog(args.data_dir)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(build_bundle(kept, skipped, catalog), fh, separators=(",", ":"))
        print(f"Wrote model bundle -> {args.json_out}")

    out = build_html(kept, args.template, args.out, skipped, catalog)
    report = size_report(kept, template_bytes)
    actual = os.path.getsize(out)

    print("\n  size accounting")
    print(f"    {'model':<26}{'dtype':>6}{'params':>10}{'weights':>11}{'names':>9}{'total':>11}")
    for r in report["models"]:
        print(f"    {r['label'][:25]:<26}{r['dtype']:>6}{r['params']:>10}"
              f"{_fmt(r['weight_bytes']):>11}{_fmt(r['training_name_bytes']):>9}"
              f"{_fmt(r['bytes']):>11}")
    print(f"    {'UI (template)':<26}{'':>6}{'':>10}{'':>11}{'':>9}{_fmt(template_bytes):>11}")
    print(f"    {'':<26}{'':>6}{'':>10}{'':>11}{'TOTAL':>9}{_fmt(actual):>11}")

    if skipped:
        print(f"\n  {len(skipped)} model(s) left out to stay inside the budget:")
        for m in skipped:
            print(f"    - {m['label']} ({_fmt(m['bytes'])}, {m['skip_reason']})")
        print("    They are still listed in the gallery, marked 'not in this file'.")

    budget = args.budget_mb * 1024 * 1024 if args.budget_mb > 0 else 0
    if budget and actual > budget:
        # The template plus one model can exceed the budget; say so rather than pretend.
        print(f"\n  ! {_fmt(actual)} is over the {args.budget_mb} MB budget — the first "
              f"model alone does not fit. Export a smaller checkpoint (see web/README.md).")

    print(f"\nWrote self-contained web UI -> {out}  ({_fmt(actual)}, "
          f"{len(kept)} model(s) playable offline)")


if __name__ == "__main__":
    main()
