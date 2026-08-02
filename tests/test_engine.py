"""End-to-end tests for the model, training loop, sampling, and the web export.

These are deliberately tiny (a handful of synthetic names, a few epochs, a small net)
so the whole suite still runs in seconds on a CPU — the same "small and CPU-friendly"
guardrail the models themselves follow. They assert the pieces compose: a model trains,
loss goes *down*, sampling yields decodable strings, novelty filtering works, and the
browser export reproduces the trained model's logits closely enough to trust.
"""

import os
import tempfile
import unittest

import torch

from src.config import Config
from src.data import Vocab, make_pairs, make_batches
from src.model import CharRNN
from src.sample import generate_many, load_checkpoint
from src.train import fit, save_checkpoint
from src.export_web import (
    export_model, build_html, TEMPLATE_MARKER, weight_count, unpack_floats)


# A tiny synthetic "domain": short, patterned pseudo-names the net can learn fast.
NAMES = [
    "Vroomio", "Zaxon", "Turbex", "Velcar", "Roadix", "Motoza", "Carvo",
    "Zoomer", "Draxel", "Vexor", "Torro", "Zephex", "Racton", "Vantek",
]


def _tiny_cfg(epochs=12):
    return Config(embedding_dim=8, hidden_dim=24, num_layers=1, dropout=0.0,
                  epochs=epochs, batch_size=8, learning_rate=5e-3, sample_every=1000,
                  seed=7, max_length=20)


class ModelTests(unittest.TestCase):
    def test_forward_shapes(self):
        cfg = _tiny_cfg()
        vocab = Vocab(NAMES)
        model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id)
        x = torch.zeros((3, 5), dtype=torch.long)
        logits, hidden = model(x)
        self.assertEqual(tuple(logits.shape), (3, 5, len(vocab)))
        self.assertIsNotNone(hidden)


class TrainingTests(unittest.TestCase):
    def test_training_reduces_loss_and_samples_are_valid(self):
        cfg = _tiny_cfg(epochs=15)
        vocab = Vocab(NAMES)
        model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id)

        first = _epoch_loss(model, vocab, cfg)
        fit(model, vocab, NAMES, cfg)          # trains in place
        last = _epoch_loss(model, vocab, cfg)
        self.assertLess(last, first, "loss should fall after training")

        names = generate_many(model, vocab, cfg, num=5, temperature=0.8,
                              training_names=set(NAMES), only_novel=False)
        self.assertEqual(len(names), 5)
        for n in names:
            self.assertIsInstance(n, str)
            self.assertGreaterEqual(len(n), 2)

    def test_only_novel_excludes_training_names(self):
        cfg = _tiny_cfg(epochs=15)
        vocab = Vocab(NAMES)
        model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id)
        fit(model, vocab, NAMES, cfg)
        novel = generate_many(model, vocab, cfg, num=8, temperature=1.0,
                              training_names=set(NAMES), only_novel=True)
        for n in novel:
            self.assertNotIn(n, set(NAMES))

    def test_checkpoint_roundtrip_preserves_contract_keys(self):
        cfg = _tiny_cfg(epochs=3)
        vocab = Vocab(NAMES)
        model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id)
        fit(model, vocab, NAMES, cfg)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tiny.pt")
            save_checkpoint(path, model, cfg, vocab, NAMES)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            for key in ("model_state", "config", "vocab", "training_names"):
                self.assertIn(key, ckpt, f"checkpoint is missing contract key {key!r}")
            # reload and confirm it samples
            m2, v2, c2, train2 = load_checkpoint(path)
            self.assertEqual(train2, set(NAMES))
            out = generate_many(m2, v2, c2, num=3, temperature=0.7, only_novel=False)
            self.assertEqual(len(out), 3)


class WebExportTests(unittest.TestCase):
    def test_export_matches_torch_and_builds_html(self):
        cfg = _tiny_cfg(epochs=5)
        vocab = Vocab(NAMES)
        model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id)
        fit(model, vocab, NAMES, cfg)
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "tiny.pt")
            save_checkpoint(ckpt, model, cfg, vocab, NAMES)
            # export_model runs the fidelity check internally and raises if it fails,
            # so a clean return *is* the assertion that JS-math == torch-math.
            m = export_model(ckpt, "Tiny")
            self.assertEqual(len(m["itos"]), len(vocab))
            self.assertEqual(m["num_layers"], cfg.num_layers)
            # WS-12 changed the wire format: weights are one base64 blob (float16, or
            # float32 if float16 misses the tolerance) instead of nested JSON arrays,
            # and training_names is newline-joined rather than a list. The blob must
            # hold exactly the number of floats the declared dimensions imply.
            self.assertIn(m["dtype"], ("f16", "f32"))
            n = weight_count(len(vocab), m["embedding_dim"], cfg.hidden_dim, cfg.num_layers)
            self.assertEqual(len(unpack_floats(m["weights"], n, m["dtype"])), n)
            # order is irrelevant — the browser only tests set membership for novelty
            self.assertEqual(set(m["training_names"].split("\n")), set(NAMES))
            # export_model only returns once the fidelity check passed, and it records
            # the margin it passed by.
            self.assertLess(m["max_logit_error"], 5e-3)

            # a minimal template with the marker should splice cleanly
            template = os.path.join(d, "tpl.html")
            with open(template, "w", encoding="utf-8") as fh:
                fh.write("<div>x</div><script>const M=" + TEMPLATE_MARKER + ";</script>")
            out = os.path.join(d, "out.html")
            build_html([m], template, out)
            html = open(out, encoding="utf-8").read()
            self.assertNotIn(TEMPLATE_MARKER, html)
            self.assertIn('"Tiny"', html)


def _epoch_loss(model, vocab, cfg):
    """Mean loss over one pass without updating weights (for the before/after check)."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    pairs = make_pairs(NAMES, vocab)
    total, n = 0.0, 0
    with torch.no_grad():
        for inp, tgt, _ in make_batches(pairs, cfg.batch_size, vocab.pad_id, shuffle=False):
            logits, _ = model(inp)
            total += criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)).item()
            n += 1
    return total / max(n, 1)


if __name__ == "__main__":
    unittest.main()
