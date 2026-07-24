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

Usage:
    python -m src.serve                                   # auto-loads checkpoints/*.pt
    python -m src.serve --checkpoint checkpoints/car_models_ft.pt:"Car models"
    python -m src.serve --port 8000 --host 0.0.0.0
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

from .sample import generate_one, load_checkpoint

TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "app_template.html")


class Engine:
    """One loaded checkpoint plus the training-name set used to flag novelty."""

    def __init__(self, path: str, label: str, device: str = "cpu"):
        self.path = path
        self.label = label
        self.device = device
        self.model, self.vocab, self.cfg, training = load_checkpoint(path, device)
        self.training: Set[str] = set(training)

    def generate(self, count: int, temperature: float, prefix: str,
                 novel_only: bool) -> List[Dict]:
        """Sample distinct names from the real model, each tagged novel vs. training."""
        results, seen = [], set()
        attempts, cap = 0, count * 60
        while len(results) < count and attempts < cap:
            attempts += 1
            name = generate_one(self.model, self.vocab, temperature,
                                 self.cfg.max_length, prefix=prefix, device=self.device)
            if len(name) < 2 or name in seen:
                continue
            seen.add(name)
            novel = name not in self.training
            if novel_only and not novel:
                continue
            results.append({"name": name, "novel": novel})
        return results


def _pretty_label(path: str) -> str:
    """Turn 'checkpoints/car_models_ft.pt' into 'Car models'."""
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = stem[:-3] if stem.endswith("_ft") else stem
    return stem.replace("_", " ").replace("-", " ").strip().capitalize()


def discover_checkpoints(checkpoint_dir: str = "checkpoints") -> List[str]:
    """Prefer fine-tuned checkpoints; fall back to whatever *.pt exists."""
    fts = sorted(glob.glob(os.path.join(checkpoint_dir, "*_ft.pt")))
    if fts:
        return fts
    return sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt")))


# --- build the served page from the shared template ---------------------------------

def _live_script(labels: List[str]) -> str:
    """The browser-side script for the *server* build: same UI, but fetches the API."""
    labels_json = json.dumps(labels)
    return """
const LABELS = __LABELS__;
const $ = (id) => document.getElementById(id);
const root = document.documentElement;
const state = { engine: 0, temp: 1.0, count: 12, prefix: "", novelOnly: false };

function hexToRgb(h){return [1,3,5].map(i=>parseInt(h.slice(i,i+2),16));}
function lerp(a,b,t){return a.map((v,i)=>Math.round(v+(b[i]-v)*t));}
const COLD=hexToRgb("37b6ff"),MID=hexToRgb("ffc23a"),HOT=hexToRgb("ff4326");
function accentFor(t){const x=(t-0.4)/1.2;return x<0.5?lerp(COLD,MID,x/0.5):lerp(MID,HOT,(x-0.5)/0.5);}
function luminance([r,g,b]){return (0.299*r+0.587*g+0.114*b)/255;}
function applyAccent(){
  const rgb=accentFor(state.temp);
  const hex="#"+rgb.map(v=>v.toString(16).padStart(2,"0")).join("");
  root.style.setProperty("--accent",hex);
  root.style.setProperty("--accent-soft",`rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.18)`);
  root.style.setProperty("--accent-ink",luminance(rgb)>0.6?"#14161c":"#fff");
}
function toast(msg){const t=$("toast");t.textContent=msg;t.classList.add("show");
  clearTimeout(toast._t);toast._t=setTimeout(()=>t.classList.remove("show"),1500);}
function copy(text){
  const done=()=>toast('Copied "'+text+'"');
  if(navigator.clipboard&&navigator.clipboard.writeText){navigator.clipboard.writeText(text).then(done).catch(()=>fb(text,done));}
  else fb(text,done);
}
function fb(text,done){const ta=document.createElement("textarea");ta.value=text;ta.style.position="fixed";ta.style.opacity="0";
  document.body.appendChild(ta);ta.select();try{document.execCommand("copy");done();}catch(e){}document.body.removeChild(ta);}

function buildSeg(host,options,current,onPick){
  host.innerHTML="";
  options.forEach((opt,i)=>{
    const b=document.createElement("button");b.type="button";b.textContent=opt.label;
    b.setAttribute("aria-pressed",String(i===current));
    b.addEventListener("click",()=>{[...host.children].forEach((c,j)=>c.setAttribute("aria-pressed",String(j===i)));onPick(i,opt);});
    host.appendChild(b);
  });
}
function buildSelect(host,options,current,onPick){
  host.innerHTML="";
  options.forEach((opt,i)=>{
    const o=document.createElement("option");o.value=String(i);o.textContent=opt.label;
    if(i===current)o.selected=true;
    host.appendChild(o);
  });
  host.addEventListener("change",()=>{const i=host.selectedIndex;onPick(i,options[i]);});
}
function render(batch){
  const box=$("results");box.innerHTML="";
  if(!batch.length){box.innerHTML='<p class="empty">The net drew a blank at this setting &mdash; try nudging the dial or clearing the prefix.</p>';$("stat").hidden=true;return;}
  batch.forEach((item,i)=>{
    const el=document.createElement("button");el.type="button";
    el.className="name "+(item.novel?"novel":"known");el.style.animationDelay=(i*45)+"ms";
    el.innerHTML='<span class="word"></span><span class="tag">'+(item.novel?"new":"real")+'</span><span class="copy">copy</span>';
    el.querySelector(".word").textContent=item.name;
    el.addEventListener("click",()=>copy(item.name));
    box.appendChild(el);
  });
  const novelCount=batch.filter(b=>b.novel).length;
  const stat=$("stat");stat.innerHTML="<b>"+novelCount+"</b> of "+batch.length+" never existed before";stat.hidden=false;
}
async function run(){
  const go=$("go");go.disabled=true;go.textContent="Inventing\\u2026";
  try{
    const params=new URLSearchParams({engine:state.engine,count:state.count,temperature:state.temp,prefix:(state.prefix||"").trim(),novel_only:state.novelOnly?"1":"0"});
    const res=await fetch("/api/generate?"+params.toString());
    if(!res.ok) throw new Error("server "+res.status);
    const data=await res.json();
    render(data.names||[]);
  }catch(e){
    $("results").innerHTML='<p class="empty">Could not reach the model server. Is <code>python -m src.serve</code> still running?</p>';
  }finally{
    go.disabled=false;go.textContent="Invent names";
  }
}

buildSelect($("engine"),LABELS.map(l=>({label:l})),state.engine,(i)=>{state.engine=i;});
buildSeg($("count"),[6,12,24].map(n=>({label:String(n),value:n})),1,(i,opt)=>{state.count=opt.value;});
$("temp").addEventListener("input",e=>{state.temp=parseFloat(e.target.value);$("tempVal").textContent=state.temp.toFixed(2);applyAccent();});
$("prefix").addEventListener("input",e=>{state.prefix=e.target.value;});
$("novelOnly").addEventListener("change",e=>{state.novelOnly=e.target.checked;});
$("go").addEventListener("click",run);
document.title="Burple-Fink \\u2014 live model";
applyAccent();
""".replace("__LABELS__", labels_json)


def build_page(labels: List[str]) -> str:
    """Reuse the shared UI's CSS + markup, but swap in the fetch-based script."""
    with open(TEMPLATE, "r", encoding="utf-8") as fh:
        template = fh.read()
    # Everything up to the <script> tag (the <style> block and the markup) is shared
    # verbatim; only the "brain" differs between the static export and the live server.
    head = template.split("<script>", 1)[0]
    return head + "<script>\n" + _live_script(labels) + "\n</script>\n"


def make_handler(engines: List[Engine], page: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, content_type: str):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/models":
                body = json.dumps({"models": [e.label for e in engines]}).encode()
                self._send(200, body, "application/json")
            elif parsed.path == "/api/generate":
                self._generate(parse_qs(parsed.query))
            elif parsed.path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")  # browsers auto-request this
            else:
                self._send(404, b"not found", "text/plain")

        def _generate(self, q: Dict[str, List[str]]):
            def one(key, default):
                return q.get(key, [default])[0]
            try:
                idx = max(0, min(len(engines) - 1, int(one("engine", "0"))))
                count = max(1, min(60, int(one("count", "12"))))
                temperature = max(0.05, min(2.0, float(one("temperature", "0.8"))))
                prefix = one("prefix", "")
                novel_only = one("novel_only", "0") in ("1", "true", "True")
                names = engines[idx].generate(count, temperature, prefix, novel_only)
                self._send(200, json.dumps({"names": names}).encode(), "application/json")
            except Exception as exc:  # keep the server alive; report cleanly to the UI
                self._send(400, json.dumps({"error": str(exc)}).encode(),
                           "application/json")

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
        engines.append(Engine(path, label or _pretty_label(path), device))
        print(f"  loaded {engines[-1].label!r:22} <- {path} "
              f"({len(engines[-1].training)} training names)")
    if not engines:
        raise SystemExit("No checkpoints to serve. Train one first (see README).")

    page = build_page([e.label for e in engines])
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
