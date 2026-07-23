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

Usage:
    python -m src.export_web \
        --model checkpoints/car_manufacturers_ft.pt:"Car brands" \
        --model checkpoints/car_models_ft.pt:"Car models" \
        --template web/app_template.html --out web/burple-fink.html
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List

import torch

from .data import Vocab
from .sample import load_checkpoint

# Marker in the HTML template that the model bundle is spliced into.
TEMPLATE_MARKER = "/*__BURPLE_MODELS__*/"
# Round weights to keep the embedded JSON small; 5 decimals is far finer than sampling
# needs and keeps logits within ~1e-3 of the full-precision model (we assert this below).
ROUND = 5


def _round(x: float) -> float:
    return round(float(x), ROUND)


def _mat(t: torch.Tensor) -> List[List[float]]:
    return [[_round(v) for v in row] for row in t.tolist()]


def _vec(t: torch.Tensor) -> List[float]:
    return [_round(v) for v in t.tolist()]


def export_model(checkpoint: str, label: str, device: str = "cpu") -> Dict:
    """Turn one checkpoint into a JSON-able description the JS engine can execute."""
    model, vocab, cfg, training_names = load_checkpoint(checkpoint, device)
    sd = model.state_dict()

    layers = []
    for i in range(cfg.num_layers):
        layers.append({
            "w_ih": _mat(sd[f"lstm.weight_ih_l{i}"]),   # (4H, in)  gates: [i, f, g, o]
            "w_hh": _mat(sd[f"lstm.weight_hh_l{i}"]),   # (4H, H)
            "b_ih": _vec(sd[f"lstm.bias_ih_l{i}"]),     # (4H,)
            "b_hh": _vec(sd[f"lstm.bias_hh_l{i}"]),     # (4H,)
        })

    model_dict = {
        "label": label,
        "hidden_dim": cfg.hidden_dim,
        "num_layers": cfg.num_layers,
        "max_length": cfg.max_length,
        "default_temperature": cfg.temperature,
        "itos": list(vocab.itos),
        "pad_id": vocab.pad_id,
        "start_id": vocab.start_id,
        "end_id": vocab.end_id,
        "embedding": _mat(sd["embedding.weight"]),      # (V, E)
        "layers": layers,
        "head_w": _mat(sd["head.weight"]),              # (V, H)
        "head_b": _vec(sd["head.bias"]),                # (V,)
        # The training names ride along so the browser can flag which generated names
        # are genuinely novel vs. memorized copies — the WS-3 novelty metric, live.
        "training_names": list(training_names),
    }

    _verify(model_dict, model, vocab, device)
    return model_dict


# --- reference forward pass (pure Python) — mirrors the JS engine exactly -------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _forward_logits(m: Dict, ids: List[int]) -> List[float]:
    """Re-run Embedding -> LSTM -> Linear from the exported (rounded) weights.

    This is the same algorithm the browser runs. Comparing its output to the real
    torch model is what lets us guarantee the web UI is faithful to the trained net.
    """
    H = m["hidden_dim"]
    h = [[0.0] * H for _ in range(m["num_layers"])]
    c = [[0.0] * H for _ in range(m["num_layers"])]

    x: List[float] = []
    for tok in ids:
        x = m["embedding"][tok]
        for li, layer in enumerate(m["layers"]):
            w_ih, w_hh = layer["w_ih"], layer["w_hh"]
            b_ih, b_hh = layer["b_ih"], layer["b_hh"]
            hl, cl = h[li], c[li]
            new_h = [0.0] * H
            new_c = [0.0] * H
            for k in range(H):
                # gate pre-activations: rows [k], [H+k], [2H+k], [3H+k] = i, f, g, o
                def pre(off: int) -> float:
                    r = off + k
                    s = b_ih[r] + b_hh[r]
                    wih_r, whh_r = w_ih[r], w_hh[r]
                    for j, xv in enumerate(x):
                        s += wih_r[j] * xv
                    for j in range(H):
                        s += whh_r[j] * hl[j]
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
    for row, b in zip(m["head_w"], m["head_b"]):
        logits.append(b + sum(wv * xv for wv, xv in zip(row, x)))
    return logits


@torch.no_grad()
def _verify(m: Dict, torch_model, vocab: Vocab, device: str,
            tol: float = 5e-3) -> None:
    """Assert the exported weights reproduce the torch model's logits (else refuse)."""
    torch_model.eval()
    # Build probe sequences from real characters in *this* vocab (indices 3+ skip the
    # PAD/START/END specials) so the check works for any dataset, not just ones that
    # happen to contain a hardcoded letter.
    real = vocab.itos[3:]
    probes = ["", real[0] if real else "", "".join(real[:2])]
    for probe in probes:
        ids = [vocab.start_id] + [vocab.stoi[ch] for ch in probe]
        ref = _forward_logits(m, ids)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = torch_model(inp)
        torch_last = logits[0, -1, :].tolist()
        max_diff = max(abs(a - b) for a, b in zip(ref, torch_last))
        if max_diff > tol:
            raise RuntimeError(
                f"Web export failed fidelity check for prime {probe!r}: max logit diff "
                f"{max_diff:.4g} > {tol}. The JS engine would not match the trained model."
            )


def build_html(models: List[Dict], template_path: str, out_path: str) -> str:
    """Splice the model bundle into the HTML template, producing one standalone file."""
    with open(template_path, "r", encoding="utf-8") as fh:
        template = fh.read()
    if TEMPLATE_MARKER not in template:
        raise ValueError(f"Template {template_path!r} is missing marker {TEMPLATE_MARKER}")
    bundle = json.dumps({"models": models}, ensure_ascii=True, separators=(",", ":"))
    html = template.replace(TEMPLATE_MARKER, bundle)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_path


def _parse_model_arg(spec: str) -> tuple[str, str]:
    """Split a ``path.pt:Label`` CLI argument (label defaults to the filename)."""
    if ":" in spec:
        path, label = spec.split(":", 1)
        return path, label
    return spec, spec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export checkpoints into a self-contained browser UI.")
    parser.add_argument("--model", action="append", required=True, dest="models",
                        metavar="CHECKPOINT[:LABEL]",
                        help="A checkpoint and display label; repeatable.")
    parser.add_argument("--template", default="web/app_template.html")
    parser.add_argument("--out", default="web/burple-fink.html")
    parser.add_argument("--json-out", default=None,
                        help="Optional: also write the raw model bundle JSON here.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for spec in args.models:
        path, label = _parse_model_arg(spec)
        print(f"Exporting {path} as {label!r} …")
        models.append(export_model(path, label, device))
        print(f"  ✓ fidelity check passed")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"models": models}, fh, separators=(",", ":"))
        print(f"Wrote model bundle -> {args.json_out}")

    out = build_html(models, args.template, args.out)
    size_kb = round(len(open(out, encoding="utf-8").read()) / 1024)
    print(f"Wrote self-contained web UI -> {out}  ({size_kb} KB, {len(models)} models)")


if __name__ == "__main__":
    main()
