"""Tests for the web front ends: the static export and the live server (WS-12).

Two things make this suite worth having.

**The exporter's safety net is tested by breaking it.** ``src/export_web.py`` refuses to
write an HTML file unless a pure-Python re-implementation of the forward pass — the same
algorithm, over the same flat weight layout, that the browser runs — reproduces the real
PyTorch model's logits. A test that only asserts "export succeeds" would still pass if
that check were deleted. So the tests below corrupt exported weights in several different
ways and assert the check *fails*, which is the only way to know the net is real.

**Almost nothing here needs a trained checkpoint.** Fidelity is a property of the maths,
not of the loss: an untrained ``CharRNN`` with random weights is a *harder* test than a
trained one, because trained weights are smoother and more forgiving of rounding. So the
checkpoint fixture is built directly with ``torch.save`` and never runs a training loop,
which also keeps this suite independent of the lanes editing ``src/train.py``.
"""

import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import torch

from src.config import Config
from src.data import Vocab
from src.model import CharRNN
from src import export_web
from src.export_web import (
    BUNDLE_FORMAT, ENGINE_MARKER, TEMPLATE_MARKER, Unpacked, build_bundle, build_html,
    dataset_catalog, export_model, forward_logits, model_bytes, pack_floats, plan_bundle,
    pretty_label, read_meta, resolve_dataset, size_report, unpack_floats, verify,
    weight_count,
)

torch.set_num_threads(1)

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "web", "app_template.html")

NAMES = ["Vroomio", "Zaxon", "Turbex", "Velcar", "Roadix", "Motoza", "Carvo", "Zoomer"]


def tiny_cfg(**kw) -> Config:
    """A deliberately small net: big enough to exercise multi-layer LSTM index maths."""
    base = dict(embedding_dim=6, hidden_dim=8, num_layers=2, max_length=12, temperature=0.9)
    base.update(kw)
    return Config(**base)


def write_checkpoint(path: str, cfg: Config, names=NAMES, seed: int = 0):
    """Save an *untrained* checkpoint straight to disk.

    Deliberately does not import ``src/train.py``: this suite tests the export/serve
    plumbing, and other lanes are editing the training module.

    Returns the model in ``eval()`` mode. This matters: ``cfg.dropout`` defaults to 0.2,
    so a model left in training mode returns *different logits every call* and any test
    comparing against it is silently comparing against noise (measured: two consecutive
    calls differed by 0.018, well over the exporter's 5e-3 tolerance).
    """
    torch.manual_seed(seed)
    vocab = Vocab(names)
    model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id)
    model.eval()
    torch.save({
        "config": cfg.to_dict(),
        "vocab": vocab.to_dict(),
        "model_state": model.state_dict(),
        "training_names": list(names),
    }, path)
    return model, vocab


class WeightCodecTests(unittest.TestCase):
    """base64 float16/float32 packing — the wire format the browser decodes."""

    def test_roundtrip_f32_is_exact(self):
        vals = [0.0, 1.0, -1.0, 0.5, -0.001953125, 1234.5]
        got = unpack_floats(pack_floats(vals, "f32"), len(vals), "f32")
        self.assertEqual(got, vals)

    def test_roundtrip_f16_is_close(self):
        vals = [0.0, 1.0, -1.0, 0.0625, -0.03125, 0.1, -2.5]
        got = unpack_floats(pack_floats(vals, "f16"), len(vals), "f16")
        for a, b in zip(vals, got):
            self.assertAlmostEqual(a, b, delta=1e-3)

    def test_f16_is_half_the_bytes_of_f32(self):
        vals = [0.01 * i for i in range(200)]
        self.assertAlmostEqual(
            len(pack_floats(vals, "f16")) / len(pack_floats(vals, "f32")), 0.5, delta=0.02)

    def test_weight_count_matches_the_real_state_dict(self):
        """The browser rebuilds tensor shapes from the declared dims alone; if this
        formula and the actual model ever disagree, every weight after the first
        mismatched tensor would be read from the wrong offset."""
        cfg = tiny_cfg()
        vocab = Vocab(NAMES)
        model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id)
        actual = sum(t.numel() for t in
                     export_web.tensor_order(model.state_dict(), cfg.num_layers))
        self.assertEqual(
            weight_count(len(vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.num_layers),
            actual)


class FidelityCheckTests(unittest.TestCase):
    """The point of the exporter: refuse to ship a net that is not the trained net.

    Each test here corrupts something and asserts the check *fails*. If someone deletes
    ``verify()``'s comparison, these go red.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ckpt = os.path.join(cls.tmp.name, "tiny.pt")
        cls.cfg = tiny_cfg()
        cls.model, cls.vocab = write_checkpoint(cls.ckpt, cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_clean_export_passes_and_records_its_margin(self):
        m = export_model(self.ckpt, "Tiny", data_dir=self.tmp.name)
        self.assertIn(m["dtype"], ("f16", "f32"))
        self.assertLess(m["max_logit_error"], 5e-3)
        # sanity: the blob really holds what the declared dimensions imply
        n = weight_count(len(m["itos"]), m["embedding_dim"], m["hidden_dim"],
                         m["num_layers"])
        self.assertEqual(len(unpack_floats(m["weights"], n, m["dtype"])), n)

    def _corrupt(self, m, index, delta):
        """Rebuild a model dict with one weight shifted by `delta`."""
        n = weight_count(len(m["itos"]), m["embedding_dim"], m["hidden_dim"],
                         m["num_layers"])
        w = unpack_floats(m["weights"], n, m["dtype"])
        w[index] += delta
        return dict(m, weights=pack_floats(w, m["dtype"]))

    def test_corrupted_head_weight_is_caught(self):
        """A tampered output-layer weight must not slip through."""
        m = export_model(self.ckpt, "Tiny", precision="f32", data_dir=self.tmp.name)
        u = Unpacked(m)
        bad = self._corrupt(m, u.head_w, 1.0)
        with self.assertRaises(RuntimeError) as ctx:
            verify(bad, self.model, self.vocab)
        self.assertIn("fidelity check", str(ctx.exception))

    def test_corrupted_lstm_weight_is_caught(self):
        """Corruption inside the recurrence, which only shows up after a few steps."""
        m = export_model(self.ckpt, "Tiny", precision="f32", data_dir=self.tmp.name)
        u = Unpacked(m)
        bad = self._corrupt(m, u.layers[0][0], 2.0)   # w_ih of layer 0
        with self.assertRaises(RuntimeError):
            verify(bad, self.model, self.vocab)

    def test_corrupted_embedding_is_caught(self):
        """Corrupt the START token's embedding row — the one vector every single
        generation begins from. (Corrupting row 0 would prove nothing: that is PAD, whose
        embedding `padding_idx` pins to zero and which is never fed to the net.)"""
        m = export_model(self.ckpt, "Tiny", precision="f32", data_dir=self.tmp.name)
        u = Unpacked(m)
        bad = self._corrupt(m, u.emb + m["start_id"] * m["embedding_dim"], 5.0)
        with self.assertRaises(RuntimeError):
            verify(bad, self.model, self.vocab)

    def test_corrupting_unused_pad_embedding_is_genuinely_harmless(self):
        """The mirror image, pinned deliberately: PAD's embedding is never read, so
        changing it cannot change any logit. This documents why the test above targets
        START, instead of leaving a passing-for-the-wrong-reason hole."""
        m = export_model(self.ckpt, "Tiny", precision="f32", data_dir=self.tmp.name)
        u = Unpacked(m)
        bad = self._corrupt(m, u.emb + m["pad_id"] * m["embedding_dim"], 5.0)
        verify(bad, self.model, self.vocab)      # must NOT raise

    def test_wrong_weight_order_is_caught(self):
        """Reversing the blob keeps every number but destroys the layout — exactly the
        failure a shape-free wire format risks, and exactly what must not ship."""
        m = export_model(self.ckpt, "Tiny", precision="f32", data_dir=self.tmp.name)
        n = weight_count(len(m["itos"]), m["embedding_dim"], m["hidden_dim"],
                         m["num_layers"])
        w = unpack_floats(m["weights"], n, m["dtype"])
        bad = dict(m, weights=pack_floats(list(reversed(w)), m["dtype"]))
        with self.assertRaises(RuntimeError):
            verify(bad, self.model, self.vocab)

    def test_truncated_blob_is_caught(self):
        """A short blob must raise rather than silently decode garbage."""
        m = export_model(self.ckpt, "Tiny", precision="f32", data_dir=self.tmp.name)
        n = weight_count(len(m["itos"]), m["embedding_dim"], m["hidden_dim"],
                         m["num_layers"])
        w = unpack_floats(m["weights"], n, m["dtype"])
        bad = dict(m, weights=pack_floats(w[: n // 2], m["dtype"]))
        with self.assertRaises(Exception):
            verify(bad, self.model, self.vocab)

    def test_export_of_a_tampered_checkpoint_refuses(self):
        """End-to-end: mutate the saved weights so the model no longer matches its own
        state, and confirm export_model refuses instead of writing a bundle."""
        path = os.path.join(self.tmp.name, "tampered.pt")
        cfg = tiny_cfg()
        write_checkpoint(path, cfg, seed=1)
        real = export_model(path, "T", precision="f32", data_dir=self.tmp.name)

        # Patch pack_floats so the *shipped* bytes differ from the verified model — the
        # exact silent-divergence scenario the fidelity check exists to prevent.
        original = export_web.pack_floats

        def sabotage(values, dtype):
            values = list(values)
            values[-1] += 3.0        # the output-layer bias: read on every single step
            return original(values, dtype)

        export_web.pack_floats = sabotage
        try:
            with self.assertRaises(RuntimeError):
                export_model(path, "T", precision="f32", data_dir=self.tmp.name)
        finally:
            export_web.pack_floats = original
        self.assertLess(real["max_logit_error"], 5e-3)   # the clean one was fine

    def test_verify_tolerance_is_actually_enforced(self):
        """Shrinking the tolerance to zero must reject even a perfect export, proving the
        comparison is live rather than short-circuited."""
        m = export_model(self.ckpt, "Tiny", precision="f16", data_dir=self.tmp.name)
        with self.assertRaises(RuntimeError):
            verify(m, self.model, self.vocab, tol=0.0)

    def test_reference_forward_matches_torch_step_by_step(self):
        """The Python reference pass mirrors the JS engine; pin it against torch over a
        multi-character prime so recurrence errors cannot hide."""
        m = export_model(self.ckpt, "Tiny", precision="f32", data_dir=self.tmp.name)
        u = Unpacked(m)
        chars = self.vocab.itos[3:6]
        ids = [self.vocab.start_id] + [self.vocab.stoi[c] for c in chars]
        with torch.no_grad():
            expected = self.model(torch.tensor([ids]))[0][0, -1, :].tolist()
        for a, b in zip(forward_logits(u, ids), expected):
            self.assertAlmostEqual(a, b, delta=1e-4)

    def test_f16_and_f32_agree_on_the_same_checkpoint(self):
        a = export_model(self.ckpt, "T", precision="f16", data_dir=self.tmp.name)
        b = export_model(self.ckpt, "T", precision="f32", data_dir=self.tmp.name)
        self.assertEqual(a["dtype"], "f16")
        self.assertEqual(b["dtype"], "f32")
        self.assertLess(len(a["weights"]), len(b["weights"]))
        ua, ub = Unpacked(a), Unpacked(b)
        ids = [self.vocab.start_id, self.vocab.stoi[self.vocab.itos[3]]]
        for x, y in zip(forward_logits(ua, ids), forward_logits(ub, ids)):
            self.assertAlmostEqual(x, y, delta=5e-3)


class SizeAccountingTests(unittest.TestCase):
    """'Do not silently ship a 40 MB page' has to be a number, not an intention."""

    def _fake(self, mid, weight_floats=100, names=("A", "B")):
        return {
            "id": mid, "label": mid.title(), "domain": "Nature", "dtype": "f16",
            "hidden_dim": 4, "num_layers": 1, "embedding_dim": 2, "max_length": 8,
            "default_temperature": 0.8, "itos": list("abcde"), "pad_id": 0,
            "start_id": 1, "end_id": 2, "verified": False, "provenance": "",
            "training_names": "\n".join(names),
            "weights": pack_floats([0.1] * weight_floats, "f16"),
        }

    def test_report_totals_add_up(self):
        models = [self._fake("a"), self._fake("b", 300)]
        rep = size_report(models, template_bytes=1000)
        self.assertEqual(rep["model_count"], 2)
        self.assertEqual(rep["models_bytes"], sum(model_bytes(m) for m in models))
        self.assertEqual(rep["total_bytes"], rep["models_bytes"] + 1000)
        for row, m in zip(rep["models"], models):
            # every byte of a model is attributed to weights, names, or scaffolding
            self.assertEqual(
                row["weight_bytes"] + row["training_name_bytes"] + row["other_bytes"],
                row["bytes"])
            self.assertGreater(row["weight_bytes"], 0)

    def test_bigger_model_reports_more_weight_bytes(self):
        small = size_report([self._fake("s", 100)])["models"][0]
        big = size_report([self._fake("b", 1000)])["models"][0]
        self.assertGreater(big["weight_bytes"], small["weight_bytes"] * 5)

    def test_model_bytes_is_the_exact_serialized_size(self):
        """Not an estimate: the budget is only meaningful if this equals reality."""
        m = self._fake("x")
        self.assertEqual(
            model_bytes(m),
            len(json.dumps(m, ensure_ascii=True, separators=(",", ":")).encode()))

    def test_exported_models_carry_no_self_referential_size_field(self):
        """A stored `bytes` field would itself add bytes to the JSON, so the recorded
        size could never equal the real one. Nothing may reintroduce it."""
        m = self._fake("x")
        self.assertNotIn("bytes", m)

    def test_budget_keeps_prefix_and_skips_the_rest(self):
        models = [self._fake(c, 400) for c in "abcde"]
        each = model_bytes(models[0])
        kept, skipped = plan_bundle(models, budget_mb=(each * 2.5) / 1048576)
        self.assertEqual([m["id"] for m in kept], ["a", "b"])
        self.assertEqual([m["id"] for m in skipped], ["c", "d", "e"])
        self.assertTrue(all("budget" in m["skip_reason"] for m in skipped))
        self.assertLessEqual(sum(model_bytes(m) for m in kept), each * 2.5)

    def test_budget_zero_means_unlimited(self):
        models = [self._fake(c) for c in "abcde"]
        kept, skipped = plan_bundle(models, budget_mb=0)
        self.assertEqual(len(kept), 5)
        self.assertEqual(skipped, [])

    def test_max_models_caps_the_count(self):
        models = [self._fake(c) for c in "abcde"]
        kept, skipped = plan_bundle(models, budget_mb=0, max_models=2)
        self.assertEqual(len(kept), 2)
        self.assertTrue(all("max-models" in m["skip_reason"] for m in skipped))

    def test_first_model_is_never_dropped(self):
        """A budget smaller than one model still yields a usable page; main() reports
        the overshoot rather than writing an empty gallery."""
        models = [self._fake("solo", 5000)]
        kept, skipped = plan_bundle(models, budget_mb=0.000001)
        self.assertEqual(len(kept), 1)
        self.assertEqual(skipped, [])

    def test_template_overhead_counts_against_the_budget(self):
        models = [self._fake(c, 400) for c in "abc"]
        each = model_bytes(models[0])
        no_overhead, _ = plan_bundle(models, budget_mb=(each * 2.5) / 1048576)
        with_overhead, _ = plan_bundle(models, budget_mb=(each * 2.5) / 1048576,
                                       overhead=each * 2)
        self.assertEqual(len(no_overhead), 2)
        self.assertEqual(len(with_overhead), 1)


class ManifestTests(unittest.TestCase):
    """The gallery must show every dataset that exists, not only the bundled ones."""

    def _fake(self, mid, domain="Nature"):
        return {"id": mid, "label": mid.replace("_", " ").title(), "domain": domain,
                "dtype": "f16", "hidden_dim": 4, "num_layers": 1, "embedding_dim": 2,
                "max_length": 8, "default_temperature": 0.8, "itos": list("abc"),
                "pad_id": 0, "start_id": 1, "end_id": 2, "verified": False,
                "provenance": "recalled", "training_names": "A\nB",
                "weights": pack_floats([0.1] * 10, "f16")}

    def test_bundle_has_format_and_both_sections(self):
        b = build_bundle([self._fake("a")], [], [])
        self.assertEqual(b["format"], BUNDLE_FORMAT)
        self.assertIn("models", b)
        self.assertIn("catalog", b)

    def test_catalog_marks_bundled_and_unbundled(self):
        catalog = [
            {"id": "a", "label": "A", "domain": "Nature", "count": 10},
            {"id": "b", "label": "B", "domain": "Nature", "count": 20},
        ]
        b = build_bundle([self._fake("a")], [self._fake("b")], catalog)
        by_id = {c["id"]: c for c in b["catalog"]}
        self.assertTrue(by_id["a"]["bundled"])
        self.assertFalse(by_id["b"]["bundled"])
        self.assertIn("skip_reason", by_id["b"])

    def test_dataset_with_no_checkpoint_is_still_listed(self):
        """29 datasets and 3 checkpoints must still show 29 rows."""
        catalog = [{"id": x, "label": x, "domain": "Nature", "count": 1}
                   for x in ("a", "b", "c")]
        b = build_bundle([self._fake("a")], [], catalog)
        self.assertEqual(len(b["catalog"]), 3)
        self.assertEqual(sum(1 for c in b["catalog"] if c["bundled"]), 1)

    def test_bundled_model_missing_from_catalog_is_appended(self):
        """A model must never be unreachable in the UI just because --no-catalog ran."""
        b = build_bundle([self._fake("orphan")], [], [])
        self.assertEqual([c["id"] for c in b["catalog"]], ["orphan"])
        self.assertTrue(b["catalog"][0]["bundled"])

    def test_catalog_is_grouped_by_domain(self):
        catalog = [
            {"id": "z", "label": "Z", "domain": "Vehicles & Transport", "count": 1},
            {"id": "a", "label": "A", "domain": "Nature", "count": 1},
            {"id": "m", "label": "M", "domain": "Nature", "count": 1},
        ]
        b = build_bundle([], [], catalog)
        self.assertEqual([c["id"] for c in b["catalog"]], ["a", "m", "z"])

    def test_provenance_and_verified_survive_into_the_bundle(self):
        b = build_bundle([self._fake("a")], [], [])
        self.assertIn("provenance", b["catalog"][0])
        self.assertFalse(b["catalog"][0]["verified"])


class MetadataTests(unittest.TestCase):
    """Sidecar handling: use data/<stem>.meta.json where present, degrade where absent."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _dataset(self, stem, names=("Alpha", "Beta"), meta=None):
        with open(os.path.join(self.dir, f"{stem}.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(names) + "\n")
        if meta is not None:
            with open(os.path.join(self.dir, f"{stem}.meta.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(meta, fh)

    def test_reads_label_and_domain_from_sidecar(self):
        self._dataset("dog_breeds", meta={"label": "Dog breeds", "domain": "Nature",
                                          "verified": False, "provenance": "recalled"})
        info = resolve_dataset("checkpoints/ws12_dog_breeds.pt", None, self.dir)
        self.assertEqual(info["id"], "dog_breeds")
        self.assertEqual(info["label"], "Dog breeds")
        self.assertEqual(info["domain"], "Nature")
        self.assertEqual(info["provenance"], "recalled")
        self.assertFalse(info["verified"])

    def test_falls_back_gracefully_without_a_sidecar(self):
        self._dataset("car_models")
        info = resolve_dataset("checkpoints/car_models_ft.pt", None, self.dir)
        self.assertEqual(info["id"], "car_models")
        self.assertEqual(info["label"], "Car models")
        self.assertEqual(info["domain"], export_web.DOMAIN_FALLBACK)

    def test_lane_prefix_does_not_defeat_the_lookup(self):
        self._dataset("greek_myth", meta={"label": "Greek myth", "domain": "Words"})
        info = resolve_dataset("checkpoints/ws12_greek_myth.pt", None, self.dir)
        self.assertEqual(info["id"], "greek_myth")
        self.assertEqual(info["domain"], "Words")

    def test_longest_dataset_match_wins(self):
        """'car_models' must not shadow 'car_models_electric'."""
        self._dataset("car_models", meta={"label": "Short", "domain": "A"})
        self._dataset("car_models_electric", meta={"label": "Long", "domain": "B"})
        info = resolve_dataset("checkpoints/ws12_car_models_electric.pt", None, self.dir)
        self.assertEqual(info["id"], "car_models_electric")
        self.assertEqual(info["label"], "Long")

    def test_corrupt_sidecar_does_not_break_the_export(self):
        with open(os.path.join(self.dir, "broken.meta.json"), "w", encoding="utf-8") as fh:
            fh.write("{not json")
        self.assertEqual(read_meta("broken", self.dir), {})

    def test_missing_sidecar_returns_empty(self):
        self.assertEqual(read_meta("nope", self.dir), {})

    def test_config_fields_win_over_the_filename(self):
        self._dataset("dog_breeds", meta={"label": "Sidecar label", "domain": "Nature"})
        cfg = Config(dataset_path="data/dog_breeds.txt", dataset_label="Config label")
        info = resolve_dataset("checkpoints/whatever.pt", cfg, self.dir)
        self.assertEqual(info["id"], "dog_breeds")
        self.assertEqual(info["label"], "Config label")
        self.assertEqual(info["domain"], "Nature")

    def test_catalog_counts_deduplicated_names(self):
        self._dataset("dupes", names=("A", "B", "A", "", "C"))
        entry = [e for e in dataset_catalog(self.dir) if e["id"] == "dupes"][0]
        self.assertEqual(entry["count"], 3)

    def test_pretty_label(self):
        self.assertEqual(pretty_label("car_manufacturers"), "Car manufacturers")
        self.assertEqual(pretty_label("greek-myth"), "Greek myth")

    def test_real_data_dir_sidecars_are_usable(self):
        """Guard against the live data/ directory drifting out of the shape we read."""
        entries = dataset_catalog("data")
        if not entries:
            self.skipTest("no datasets present")
        for e in entries:
            self.assertTrue(e["label"])
            self.assertTrue(e["domain"])
            self.assertGreaterEqual(e["count"], 0)


class TemplateTests(unittest.TestCase):
    """The exported page must stay a single offline file."""

    @classmethod
    def setUpClass(cls):
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as fh:
            cls.template = fh.read()

    def test_has_both_splice_markers(self):
        self.assertIn(TEMPLATE_MARKER, self.template)
        self.assertIn(ENGINE_MARKER, self.template)

    def test_no_external_resources(self):
        """No CDN, font or script host — the whole point is that it works offline."""
        for needle in ("http://", "https://", "src=\"//", "@import", "cdn."):
            self.assertNotIn(needle, self.template,
                             f"template references something external: {needle!r}")

    def test_engine_and_template_agree_on_bundle_format(self):
        """A stale template plus a fresh bundle would mis-decode weights into
        plausible-looking garbage, so both sides pin the same version number."""
        self.assertIn(f"const BUNDLE_FORMAT = {BUNDLE_FORMAT};", self.template)

    def test_build_html_splices_and_leaves_no_marker(self):
        m = {"id": "a", "label": "Tiny", "domain": "Nature", "dtype": "f16",
             "hidden_dim": 2, "num_layers": 1, "embedding_dim": 2, "max_length": 4,
             "default_temperature": 0.8, "itos": list("ab"), "pad_id": 0, "start_id": 1,
             "end_id": 1, "verified": False, "provenance": "", "training_names": "A",
             "weights": pack_floats([0.5] * 4, "f16")}
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out.html")
            build_html([m], TEMPLATE_PATH, out, [], [])
            with open(out, encoding="utf-8") as fh:
                html = fh.read()
        self.assertNotIn(TEMPLATE_MARKER, html)
        # both build markers are consumed, so a shipped artifact never looks half-spliced
        self.assertNotIn(ENGINE_MARKER, html)
        self.assertIn('"Tiny"', html)
        self.assertIn(f'"format":{BUNDLE_FORMAT}', html)

    def test_build_html_rejects_a_template_without_the_marker(self):
        with tempfile.TemporaryDirectory() as d:
            tpl = os.path.join(d, "tpl.html")
            with open(tpl, "w", encoding="utf-8") as fh:
                fh.write("<div>no marker</div>")
            with self.assertRaises(ValueError):
                build_html([], tpl, os.path.join(d, "out.html"))

    def test_ui_exposes_every_decoding_control(self):
        """WS-7 added four knobs; the UI is supposed to surface all of them."""
        for control in ("temp", "rep", "topk", "topp", "minlen"):
            self.assertIn(f'id="{control}"', self.template)

    def test_truncation_is_presented_as_measured_worse(self):
        """The wave-2 sweep found top-k/nucleus lost to plain temperature here. The UI
        must not quietly present them as the better setting."""
        self.assertIn("measured worse", self.template)
        self.assertIn('value="0"', self.template)      # top-k defaults to off

    def test_novelty_badge_names_the_training_data(self):
        self.assertIn("in training data", self.template)

    def test_favorites_use_localstorage(self):
        self.assertIn("localStorage", self.template)

    def test_every_element_the_script_looks_up_exists(self):
        """A typo in a getElementById id is invisible until someone taps the control on a
        phone, which is the one place this app is meant to work. Cheap to pin here."""
        import re
        markup = self.template.split("<script>")[0]
        script = self.template.split("<script>", 1)[1].rsplit("</script>", 1)[0]
        declared = set(re.findall(r'\bid="([^"]+)"', markup))
        referenced = set(re.findall(r'\$\("([^"]+)"\)', script))
        self.assertTrue(referenced, "no element lookups found — did the script move?")
        self.assertEqual(referenced - declared, set())

    def test_provenance_disclosure_is_present(self):
        """Every wave-3 dataset is verified:false; the page must not imply otherwise."""
        self.assertIn('id="prov"', self.template)
        self.assertIn("Unverified training data", self.template)


# --------------------------------------------------------------------------------------
# live server
# --------------------------------------------------------------------------------------

def _node() -> str:
    """Path to a node binary, or '' if there isn't one."""
    return shutil.which("node") or shutil.which("nodejs") or ""


def _engine_js() -> str:
    """The template's engine functions, sliced out so they can run without a DOM.

    Everything from ``f16()`` down to the pluggable-brain marker is pure computation:
    weight decoding, the LSTM forward pass and sampling. The UI below that needs a
    browser, so it is deliberately excluded.
    """
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as fh:
        js = fh.read().split("<script>", 1)[1].rsplit("</script>", 1)[0]
    start = js.index("function f16(")
    end = js.index('/* ---- the pluggable "brain"')
    return js[start:end]


@unittest.skipUnless(_node(), "node not available — JS/Python parity not checked")
class JsEngineParityTests(unittest.TestCase):
    """Run the *actual shipped JavaScript* and compare it to PyTorch and to src/sample.py.

    The Python-side fidelity check in ``export_web.verify`` guarantees a *reference*
    implementation matches torch. These tests close the remaining gap: they prove the
    reference and the browser's real code are the same algorithm, so that guarantee
    actually reaches the phone. Skipped rather than failed where node is absent, since
    node is a convenience for testing and never a runtime dependency of the export.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = cls.tmp.name
        cls.ckpt = os.path.join(cls.dir, "tiny.pt")
        cls.model, cls.vocab = write_checkpoint(cls.ckpt, tiny_cfg(hidden_dim=12))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run_node(self, harness: str, job: dict):
        with open(os.path.join(self.dir, "job.json"), "w", encoding="utf-8") as fh:
            json.dump(job, fh)
        path = os.path.join(self.dir, "harness.mjs")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_engine_js() + harness)
        proc = subprocess.run([_node(), path], capture_output=True, text=True,
                              cwd=self.dir, timeout=120)
        if proc.returncode != 0:
            self.fail(f"node harness failed:\n{proc.stderr[:2000]}")
        return json.loads(proc.stdout)

    def test_js_engine_reproduces_torch_logits(self):
        """The browser's forward pass vs. the real PyTorch model, over a multi-character
        prime so errors in the recurrence cannot cancel out."""
        m = export_model(self.ckpt, "Tiny", data_dir=self.dir)
        chars = self.vocab.itos[3:9]
        ids = [self.vocab.start_id] + [self.vocab.stoi[c] for c in chars]
        with torch.no_grad():
            expected = self.model(torch.tensor([ids]))[0][0, -1, :].tolist()
        reference = forward_logits(Unpacked(m), ids)

        got = self._run_node("""
import fs from 'fs';
const job = JSON.parse(fs.readFileSync('job.json','utf8'));
const model = prepare(job.model);
const state = zeroState(model);
let logits;
for (const id of job.ids) logits = step(model, state, id);
console.log(JSON.stringify(Array.from(logits)));
""", {"model": m, "ids": ids})

        self.assertEqual(len(got), len(expected))
        for a, b in zip(got, expected):
            self.assertAlmostEqual(a, b, delta=5e-3)      # same bar the exporter enforces
        # And the shipped JS must agree with the reference pass to machine precision —
        # that equivalence is what makes verify()'s guarantee meaningful for the browser.
        for a, b in zip(got, reference):
            self.assertAlmostEqual(a, b, delta=1e-9)

    def test_js_decoding_filters_match_sample_py(self):
        """Every WS-7 control, hand-ported to JS, must produce the same distribution as
        src/sample.py — including the order they are applied in.

        Recovers the browser's categorical distribution exactly by stubbing Math.random
        with a uniform sweep, then compares it to the real Python filters.
        """
        import torch.nn.functional as F
        from src.sample import (_apply_repetition_penalty, _top_k_filter, _top_p_filter)

        torch.manual_seed(7)
        pad_id, start_id, end_id = 0, 1, 2
        logits = (torch.randn(12) * 2).tolist()
        grid = 40000

        def python_probs(temp, top_k, top_p, rep, used, min_len, emitted):
            t = torch.tensor([logits])
            t = _apply_repetition_penalty(t, list(used), rep)
            t = t / max(temp, 1e-6)
            t = _top_k_filter(t, top_k)
            t = _top_p_filter(t, top_p)
            if min_len > 0 and emitted < min_len:
                t = t.clone()
                for special in (end_id, pad_id, start_id):
                    t[..., special] = float("-inf")
            return F.softmax(t, dim=-1)[0].tolist()

        cases = {
            "plain temperature": (1.15, 0, 1.0, 1.0, [], 0, 0),
            "repetition penalty": (1.0, 0, 1.0, 1.4, [3, 5, 7], 0, 3),
            "top_k": (1.0, 4, 1.0, 1.0, [], 0, 0),
            "top_p": (1.0, 0, 0.85, 1.0, [], 0, 0),
            "min_length masking": (1.0, 0, 1.0, 1.0, [], 5, 2),
            "all four combined": (1.25, 6, 0.9, 1.3, [4, 9], 4, 1),
        }
        for name, (temp, top_k, top_p, rep, used, min_len, emitted) in cases.items():
            with self.subTest(control=name):
                expected = python_probs(temp, top_k, top_p, rep, used, min_len, emitted)
                got = self._run_node("""
import fs from 'fs';
const job = JSON.parse(fs.readFileSync('job.json','utf8'));
const model = {end_id: job.end_id, pad_id: job.pad_id, start_id: job.start_id};
const n = job.logits.length, N = job.grid;
const counts = new Array(n).fill(0);
for (let i = 0; i < N; i++) {
  Math.random = () => (i + 0.5) / N;
  counts[sampleId(model, Float64Array.from(job.logits), job.opts,
                  new Set(job.used), job.emitted)]++;
}
console.log(JSON.stringify(counts.map((c) => c / N)));
""", {"logits": logits, "grid": grid, "used": used, "emitted": emitted,
      "pad_id": pad_id, "start_id": start_id, "end_id": end_id,
      "opts": {"temperature": temp, "topK": top_k, "topP": top_p,
               "repetitionPenalty": rep, "minLength": min_len}})
                for a, b in zip(expected, got):
                    self.assertAlmostEqual(a, b, delta=2e-3)
                # masked characters must be genuinely unreachable, not merely unlikely
                if min_len > emitted:
                    for special in (end_id, pad_id, start_id):
                        self.assertEqual(got[special], 0.0)


class ServerTestCase(unittest.TestCase):
    """Spin the real ThreadingHTTPServer on an ephemeral port and talk HTTP to it."""

    engines_factory = None

    @classmethod
    def setUpClass(cls):
        from src import serve as serve_mod
        cls.serve_mod = serve_mod
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ckpt = os.path.join(cls.tmp.name, "tiny.pt")
        write_checkpoint(cls.ckpt, tiny_cfg())
        cls.engines = cls.build_engines()
        page = serve_mod.build_page(cls.engines, data_dir=cls.tmp.name)
        cls.httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0), serve_mod.make_handler(cls.engines, page))
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def build_engines(cls):
        from src.serve import Engine
        return [Engine(cls.ckpt, "Tiny", "cpu")]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)
        cls.tmp.cleanup()

    def get(self, path):
        """Return (status, parsed-json-or-raw-bytes) without raising on 4xx/5xx."""
        url = f"http://127.0.0.1:{self.port}{path}"
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                body, status = r.read(), r.status
        except urllib.error.HTTPError as e:
            body, status = e.read(), e.code
        try:
            return status, json.loads(body.decode())
        except ValueError:
            return status, body


class ServerHappyPathTests(ServerTestCase):
    def test_index_serves_the_gallery_page(self):
        status, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"Burple", body)
        # the served build must have both markers resolved
        self.assertNotIn(TEMPLATE_MARKER.encode(), body)
        self.assertNotIn(ENGINE_MARKER.encode(), body)
        self.assertIn(b"engineGenerate = async", body)

    def test_served_page_carries_no_weights(self):
        """The live build's size must not scale with the number of checkpoints."""
        _, body = self.get("/")
        self.assertNotIn(b'"weights"', body)

    def test_health_reports_loaded_checkpoints(self):
        status, data = self.get("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["models"], 1)
        entry = data["checkpoints"][0]
        for key in ("label", "id", "domain", "path", "training_names", "verified"):
            self.assertIn(key, entry)
        self.assertEqual(entry["training_names"], len(NAMES))

    def test_models_endpoint_returns_gallery_metadata(self):
        status, data = self.get("/api/models")
        self.assertEqual(status, 200)
        m = data["models"][0]
        for key in ("id", "label", "domain", "verified", "provenance", "count"):
            self.assertIn(key, m)

    def test_generate_returns_tagged_names(self):
        status, data = self.get("/api/generate?count=4&temperature=1.0")
        self.assertEqual(status, 200)
        self.assertLessEqual(len(data["names"]), 4)
        for item in data["names"]:
            self.assertIn("name", item)
            self.assertIsInstance(item["novel"], bool)
            self.assertIsInstance(item["exact"], bool)

    def test_generate_accepts_the_decoding_controls(self):
        status, data = self.get(
            "/api/generate?count=3&temperature=1.2&top_k=5&top_p=0.9"
            "&repetition_penalty=1.3&min_length=4")
        self.assertEqual(status, 200)
        for item in data["names"]:
            self.assertGreaterEqual(len(item["name"]), 4)

    def test_generate_accepts_a_dataset_id_as_engine(self):
        engine_id = self.engines[0].id
        status, _ = self.get(f"/api/generate?engine={engine_id}&count=2")
        self.assertEqual(status, 200)

    def test_novelty_is_case_insensitive(self):
        """'Zaxon' and 'zaxon' are the same real name; calling the second an invention
        because one letter changed case would be dishonest."""
        e = self.engines[0]
        self.assertFalse(e.novelty("Zaxon")["novel"])
        self.assertTrue(e.novelty("Zaxon")["exact"])
        self.assertFalse(e.novelty("zaxon")["novel"])
        self.assertFalse(e.novelty("zaxon")["exact"])
        self.assertTrue(e.novelty("Qqqqzzz")["novel"])

    def test_favicon_is_answered_not_404(self):
        status, _ = self.get("/favicon.ico")
        self.assertEqual(status, 204)


class ServerErrorPathTests(ServerTestCase):
    """Every failure must be JSON with an 'error' key — the UI shows it verbatim."""

    def test_unknown_endpoint_is_json_404(self):
        status, data = self.get("/api/nope")
        self.assertEqual(status, 404)
        self.assertIn("error", data)
        self.assertIn("/api/nope", data["error"])

    def test_bad_temperature_is_json_400(self):
        status, data = self.get("/api/generate?temperature=hot")
        self.assertEqual(status, 400)
        self.assertIn("temperature", data["error"])

    def test_bad_count_is_json_400(self):
        status, data = self.get("/api/generate?count=lots")
        self.assertEqual(status, 400)
        self.assertIn("count", data["error"])

    def test_bad_top_k_is_json_400(self):
        status, data = self.get("/api/generate?top_k=many")
        self.assertEqual(status, 400)
        self.assertIn("top_k", data["error"])

    def test_bad_repetition_penalty_is_json_400(self):
        status, data = self.get("/api/generate?repetition_penalty=high")
        self.assertEqual(status, 400)
        self.assertIn("repetition_penalty", data["error"])

    def test_unknown_engine_is_json_400_and_lists_options(self):
        status, data = self.get("/api/generate?engine=not_a_dataset")
        self.assertEqual(status, 400)
        self.assertIn("not_a_dataset", data["error"])

    def test_engine_index_out_of_range_is_json_400(self):
        status, data = self.get("/api/generate?engine=99")
        self.assertEqual(status, 400)
        self.assertIn("out of range", data["error"])

    def test_out_of_band_values_are_clamped_not_rejected(self):
        """A slider that overshoots should still generate, not error."""
        status, _ = self.get("/api/generate?count=100000&temperature=99")
        self.assertEqual(status, 200)

    def test_generation_failure_is_json_500(self):
        """If the model raises mid-request the server must stay up and say so."""
        engine = self.engines[0]
        original = engine.generate

        def boom(*a, **kw):
            raise RuntimeError("synthetic model explosion")

        engine.generate = boom
        try:
            status, data = self.get("/api/generate?count=2")
        finally:
            engine.generate = original
        self.assertEqual(status, 500)
        self.assertIn("synthetic model explosion", data["error"])
        # and the server is still alive afterwards
        self.assertEqual(self.get("/api/health")[0], 200)


class ServerNoModelsTests(unittest.TestCase):
    """A server with nothing loaded must explain itself rather than crash."""

    def test_generate_without_engines_is_503(self):
        from src import serve as serve_mod
        with tempfile.TemporaryDirectory() as d:
            page = serve_mod.build_page([], data_dir=d)
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve_mod.make_handler([], page))
            port = httpd.server_address[1]
            t = threading.Thread(target=httpd.serve_forever, daemon=True)
            t.start()
            try:
                try:
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/api/generate", timeout=20) as r:
                        status, body = r.status, r.read()
                except urllib.error.HTTPError as e:
                    status, body = e.code, e.read()
                self.assertEqual(status, 503)
                self.assertIn("error", json.loads(body.decode()))
            finally:
                httpd.shutdown()
                httpd.server_close()
                t.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
