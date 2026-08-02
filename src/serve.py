"""A tiny local web server that wires the mobile UI to the *live* PyTorch model.

Two ways to run the front-end exist, on purpose:

* ``src/export_web.py`` bakes the weights into a static HTML file that runs the net
  in the browser — great for sharing, works with no server at all.
* **this module** serves the same UI but generates names by calling the real
  checkpoints in this process. It is the honest "connected to the inner workings"
  path: every name you see came out of the actual trained model, and swapping in a
  freshly trained checkpoint changes the output immediately with no re-export.

It deliberately uses only the Python standard library (``http.server``) so there is
nothing extra to install — see the guardrails in HANDOFF §9. It binds ``0.0.0.0`` so a
phone on the same network can open it at ``http://<your-computer-ip>:8000``.

This is also the front end that scales: the static export can only bake in as many
models as fit in a phone-sized file, but here the weights never leave the process, so
the page stays ~40 KB whether you load 3 checkpoints or all 29. If the gallery shows a
dataset marked "server", this is where to run it.

Usage:
    python -m src.serve                                   # auto-loads checkpoints/*.pt
    python -m src.serve --checkpoint checkpoints/car_models_ft.pt:"Car models"
    python -m src.serve --port 8000 --host 0.0.0.0

API:
    GET /api/health    -> {"status","models","checkpoints":[…]}
    GET /api/models    -> {"models":[{id,label,domain,verified,provenance,count}]}
    GET /api/generate  -> {"names":[{name,novel,exact}]}
        engine=<dataset id|index>  count  temperature  prefix  novel_only
        top_k  top_p  repetition_penalty  min_length
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Set
from urllib.parse import parse_qs, urlparse

import torch

from .export_web import (
    BUNDLE_FORMAT, ENGINE_MARKER, TEMPLATE_MARKER, dataset_catalog, resolve_dataset)
from .sample import generate_one, load_checkpoint

TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "app_template.html")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


class Engine:
    """One loaded checkpoint plus the training-name set used to flag novelty."""

    def __init__(self, path: str, label: str = "", device: str = "cpu"):
        self.path = path
        self.device = device
        self.model, self.vocab, self.cfg, training = load_checkpoint(path, device)
        self.training: Set[str] = set(training)
        # Case-insensitive too: "Rolls-royce" is the same real name as "Rolls-Royce",
        # and calling it an invention because one letter changed case would be dishonest.
        # The exported app applies the identical rule (see noveltyOf in app_template.html).
        self.training_lower: Set[str] = {n.lower() for n in self.training}
        info = resolve_dataset(path, self.cfg)
        self.id = info["id"]
        self.label = label or info["label"]
        self.domain = info["domain"]
        self.verified = info["verified"]
        self.provenance = info["provenance"]

    def novelty(self, name: str) -> Dict:
        if name in self.training:
            return {"novel": False, "exact": True}
        if name.lower() in self.training_lower:
            return {"novel": False, "exact": False}
        return {"novel": True, "exact": False}

    def generate(self, count: int, temperature: float, prefix: str, novel_only: bool,
                 *, top_k: int = 0, top_p: float = 1.0,
                 repetition_penalty: float = 1.0, min_length: int = 2) -> List[Dict]:
        """Sample distinct names from the real model, each tagged novel vs. training.

        The decoding controls are passed straight through to ``src.sample.generate_one``
        so the served UI drives exactly the same code path as ``python -m src.sample``.
        """
        results, seen = [], set()
        attempts, cap = 0, count * 60
        floor = max(2, min_length)
        while len(results) < count and attempts < cap:
            attempts += 1
            name = generate_one(
                self.model, self.vocab, temperature, self.cfg.max_length,
                prefix=prefix, device=self.device,
                top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty, min_length=min_length)
            if len(name) < floor or name in seen:
                continue
            seen.add(name)
            item = dict(self.novelty(name), name=name)
            if novel_only and not item["novel"]:
                continue
            results.append(item)
        return results

    def meta(self) -> Dict:
        """What the gallery needs to list this engine (no weights: the model stays here)."""
        return {
            "id": self.id, "label": self.label, "domain": self.domain,
            "verified": self.verified, "provenance": self.provenance,
            "count": len(self.training),
        }


def discover_checkpoints(checkpoint_dir: str = "checkpoints") -> List[str]:
    """Prefer fine-tuned checkpoints; fall back to whatever *.pt exists."""
    fts = sorted(glob.glob(os.path.join(checkpoint_dir, "*_ft.pt")))
    if fts:
        return fts
    return sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt")))


# --- build the served page from the shared template ---------------------------------

def _live_engine() -> str:
    """The only JS the server build changes: where the names come from.

    Everything else — the gallery, the decoding controls, favorites, novelty badges —
    is the shared template's code, so the two front ends cannot drift apart.
    """
    return """
engineGenerate = async (model, opts) => {
  const params = new URLSearchParams({
    engine: model.id,
    count: opts.count,
    temperature: opts.temperature,
    prefix: (opts.prefix || "").trim(),
    novel_only: opts.novelOnly ? "1" : "0",
    top_k: opts.topK,
    top_p: opts.topP,
    repetition_penalty: opts.repetitionPenalty,
    min_length: opts.minLength,
  });
  let res;
  try {
    res = await fetch("/api/generate?" + params.toString());
  } catch (e) {
    throw new Error("Could not reach the model server. Is `python -m src.serve` still running?");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || ("The server responded " + res.status + "."));
  return data.names || [];
};
"""


def build_page(engines: List[Engine], data_dir: str = DATA_DIR) -> str:
    """Build the served page from the *whole* shared template.

    Before WS-12 this took only the template's ``<style>`` + markup and appended a
    separately-maintained script, so every UI feature had to be written twice. Now both
    builds use the same file and the same script; the server build differs in exactly two
    spliced values — model metadata with no weights, and a fetch-based ``engineGenerate``.
    """
    with open(TEMPLATE, "r", encoding="utf-8") as fh:
        template = fh.read()
    for marker in (TEMPLATE_MARKER, ENGINE_MARKER):
        if marker not in template:
            raise ValueError(f"Template {TEMPLATE!r} is missing marker {marker}")

    # The catalog lists every dataset that exists; a checkpoint is what makes one usable.
    loaded = {e.id for e in engines}
    catalog = [dict(c, bundled=c["id"] in loaded) for c in dataset_catalog(data_dir)]
    for e in engines:
        if e.id not in {c["id"] for c in catalog}:
            catalog.append(dict(e.meta(), bundled=True))
    catalog.sort(key=lambda c: (c["domain"].lower(), c["label"].lower()))

    bundle = json.dumps({
        "format": BUNDLE_FORMAT,
        # No weights: on this path the real PyTorch model does the generating, so the
        # browser needs nothing but labels. That is why the served page stays ~40 KB
        # however many checkpoints are loaded.
        "models": [e.meta() for e in engines],
        "catalog": catalog,
    }, ensure_ascii=True, separators=(",", ":"))

    page = template.replace(TEMPLATE_MARKER, bundle).replace(ENGINE_MARKER, _live_engine())
    return page + "\n<script>document.title = 'Burple-Fink \\u2014 live model';</script>\n"


def make_handler(engines: List[Engine], page: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, content_type: str):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, code: int, payload: dict):
            self._send(code, json.dumps(payload).encode(), "application/json")

        def _send_error_json(self, code: int, message: str):
            self._send_json(code, {"error": message})

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/health":
                self._health()
            elif parsed.path == "/api/models":
                self._send_json(200, {"models": [e.meta() for e in engines]})
            elif parsed.path == "/api/generate":
                self._generate(parse_qs(parsed.query))
            elif parsed.path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")  # browsers auto-request this
            else:
                self._send_error_json(404, f"No such endpoint: {parsed.path!r}")

        def _health(self):
            checkpoints = [
                {"label": e.label, "id": e.id, "domain": e.domain, "path": e.path,
                 "training_names": len(e.training), "verified": e.verified}
                for e in engines
            ]
            self._send_json(200, {"status": "ok", "models": len(engines),
                                  "checkpoints": checkpoints})

        def _generate(self, q: Dict[str, List[str]]):
            def one(key, default):
                return q.get(key, [default])[0]

            if not engines:
                self._send_error_json(503, "No checkpoints are loaded on the server.")
                return

            # 'engine' accepts a dataset id ("dog_breeds") or a legacy positional index.
            # The gallery sends ids, because an index silently means a different model as
            # soon as the set of loaded checkpoints changes.
            raw_engine = one("engine", "0")
            by_id = {e.id: i for i, e in enumerate(engines)}
            if raw_engine in by_id:
                engine_idx = by_id[raw_engine]
            else:
                try:
                    engine_idx = int(raw_engine)
                except ValueError:
                    self._send_error_json(
                        400, f"'engine' {raw_engine!r} is not a loaded model. "
                             f"Known: {', '.join(sorted(by_id))}.")
                    return
                if not (0 <= engine_idx < len(engines)):
                    self._send_error_json(
                        400,
                        f"'engine' {engine_idx} is out of range; there "
                        f"{'is' if len(engines) == 1 else 'are'} {len(engines)} "
                        f"loaded checkpoint(s) (0-{len(engines) - 1}).")
                    return

            def number(key, default, cast, lo, hi, what):
                """Parse one bounded numeric query param, or send a 400 and return None."""
                raw = one(key, default)
                try:
                    val = cast(raw)
                except ValueError:
                    kind = "a whole number" if cast is int else "a number"
                    self._send_error_json(400, f"{key!r} must be {kind}, got {raw!r}.")
                    return None
                return max(lo, min(hi, val))

            count = number("count", "12", int, 1, 60, "count")
            if count is None:
                return
            temperature = number("temperature", "0.8", float, 0.05, 2.0, "temperature")
            if temperature is None:
                return
            # WS-7 decoding controls, all defaulting to off so an old client that does not
            # send them gets exactly the pre-wave-3 plain-temperature behavior.
            top_k = number("top_k", "0", int, 0, 1000, "top_k")
            if top_k is None:
                return
            top_p = number("top_p", "1.0", float, 0.01, 1.0, "top_p")
            if top_p is None:
                return
            repetition_penalty = number("repetition_penalty", "1.0", float, 1.0, 3.0,
                                        "repetition_penalty")
            if repetition_penalty is None:
                return
            min_length = number("min_length", "2", int, 0, 40, "min_length")
            if min_length is None:
                return

            prefix = one("prefix", "")
            novel_only = one("novel_only", "0") in ("1", "true", "True")

            try:
                names = engines[engine_idx].generate(
                    count, temperature, prefix, novel_only,
                    top_k=top_k, top_p=top_p,
                    repetition_penalty=repetition_penalty, min_length=min_length)
            except Exception as exc:  # keep the server alive; report cleanly to the UI
                self._send_error_json(
                    500, f"Generation failed on the server: {exc}")
                return
            self._send_json(200, {"names": names})

        def log_message(self, fmt, *args):  # quieter console
            if "/api/generate" in (args[0] if args else ""):
                return
            super().log_message(fmt, *args)

    return Handler


def serve(checkpoints: List[str], host: str = "0.0.0.0", port: int = 8000,
          device: str = "cpu") -> None:
    engines = []
    for spec in checkpoints:
        path, label = (spec.split(":", 1) + [None])[:2]
        engines.append(Engine(path, label or "", device))
        e = engines[-1]
        print(f"  loaded {e.label!r:24} [{e.domain}] <- {path} "
              f"({len(e.training)} training names)")
    if not engines:
        raise SystemExit("No checkpoints to serve. Train one first (see README).")

    page = build_page(engines)
    httpd = ThreadingHTTPServer((host, port), make_handler(engines, page))
    shown = "localhost" if host in ("0.0.0.0", "") else host
    print(f"\nBurple-Fink is live → http://{shown}:{port}")
    if host in ("0.0.0.0", ""):
        print(f"On your phone (same Wi-Fi): http://<this-computer-ip>:{port}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        httpd.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the Burple-Fink UI backed by the live PyTorch model.")
    parser.add_argument("--checkpoint", action="append", dest="checkpoints",
                        metavar="PATH[:LABEL]",
                        help="Checkpoint to serve (repeatable). Default: checkpoints/*.pt")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    checkpoints = args.checkpoints or discover_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        raise SystemExit(
            f"No checkpoints found in {args.checkpoint_dir!r}. Train one first, e.g.\n"
            f"  python -m src.pretrain --epochs 300\n"
            f"  python -m src.finetune --data data/car_models.txt --name car_models")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    serve(checkpoints, host=args.host, port=args.port, device=device)


if __name__ == "__main__":
    main()
