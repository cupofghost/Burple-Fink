"""Tests for the evaluation harness (WS-11).

Two things are being pinned here.

*The metric functions* are checked against values worked out by hand, not against
whatever the code happens to return today. An evaluation harness is the instrument
the rest of the project trusts; if `novelty` or `near_duplicate_rate` silently drifts,
every number in `reports/BENCHMARK.md` and every "X beat Y" claim built on it becomes
fiction. Hand-computed expectations are the only kind that can catch that.

*The held-out path* is checked both ways: a checkpoint that carries `val_names` must
produce a real held-out NLL and a real train→held-out gap, and a checkpoint written
before `--val-fraction` existed must degrade to an explicit "n/a" rather than crashing
or — far worse — reporting a number that isn't held out.

Checkpoints here are built by writing the checkpoint dict directly rather than by
calling `src.train.save_checkpoint`. The dict layout is the documented format contract
(HANDOFF §2); depending on `src/train.py` would couple these tests to a file this lane
does not own and that other lanes are actively rewriting.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import unittest

import torch

from src.config import Config
from src.data import Vocab
from src.evaluate import (
    _bigram_logprobs,
    _edit_distance,
    _gap_verdict,
    _score_name,
    clear_checkpoint_cache,
    evaluate,
    generalization_gap,
    held_out_nll,
    mean_bigram_ll,
    mean_pairwise_edit_distance,
    near_duplicate_rate,
    sweep,
    write_report,
)
from src.model import CharRNN

torch.set_num_threads(1)


TRAIN_NAMES = ["alfa", "beta", "gamma", "delta", "sigma", "omega", "kappa", "lambda"]
VAL_NAMES = ["zeta", "theta"]


def _write_checkpoint(path: str, training_names, val_names, *, include_val_key=True,
                      seed: int = 7) -> None:
    """Save a tiny but genuine checkpoint at ``path``.

    Deliberately small (16-wide, 1 layer) — these tests are about the harness's
    arithmetic and file output, not about whether a model is any good, and a small
    model keeps the suite fast.
    """
    cfg = Config()
    cfg.hidden_dim = 16
    cfg.embedding_dim = 8
    cfg.num_layers = 1
    cfg.dropout = 0.0
    cfg.max_length = 12
    cfg.dataset_label = "greek"
    cfg.dataset_path = "data/greek.txt"

    vocab = Vocab(list(training_names) + list(val_names))
    torch.manual_seed(seed)
    model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id)

    payload = {
        "model_state": model.state_dict(),
        "config": cfg.to_dict(),
        "vocab": vocab.to_dict(),
        "training_names": list(training_names),
    }
    if include_val_key:
        payload["val_names"] = list(val_names)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)


class EditDistanceTest(unittest.TestCase):
    """Levenshtein distance, worked out by hand."""

    def test_known_pairs(self):
        cases = [
            ("", "", 0),
            ("abc", "abc", 0),
            ("abc", "abd", 1),            # one substitution
            ("abc", "ab", 1),             # one deletion
            ("ab", "abc", 1),             # one insertion
            ("kitten", "sitting", 3),     # the textbook example
            ("flaw", "lawn", 2),
            ("", "abc", 3),               # three insertions
            ("Volvo", "Volvos", 1),
        ]
        for a, b, expected in cases:
            with self.subTest(a=a, b=b):
                self.assertEqual(_edit_distance(a, b), expected)

    def test_symmetric(self):
        self.assertEqual(_edit_distance("gamma", "kappa"), _edit_distance("kappa", "gamma"))


class MeanPairwiseEditDistanceTest(unittest.TestCase):
    def test_hand_computed_mean(self):
        # Three names -> three pairs: ab/ac = 1, ab/abc = 1, ac/abc = 1. Mean = 1.0.
        self.assertAlmostEqual(mean_pairwise_edit_distance(["ab", "ac", "abc"]), 1.0)

    def test_hand_computed_mean_uneven(self):
        # cat/car = 1, cat/dog = 3, car/dog = 3  ->  7 / 3.
        self.assertAlmostEqual(mean_pairwise_edit_distance(["cat", "car", "dog"]), 7 / 3)

    def test_fewer_than_two_names_is_nan(self):
        # Not 0.0: "there were no pairs to compare" is not "the names were identical".
        self.assertTrue(math.isnan(mean_pairwise_edit_distance(["only"])))
        self.assertTrue(math.isnan(mean_pairwise_edit_distance([])))

    def test_sampling_is_deterministic_for_a_given_rng(self):
        import random
        names = [f"name{i:03d}" for i in range(80)]  # 3160 pairs > the 2000 cap
        a = mean_pairwise_edit_distance(names, max_pairs=100, rng=random.Random(5))
        b = mean_pairwise_edit_distance(names, max_pairs=100, rng=random.Random(5))
        self.assertEqual(a, b)


class NearDuplicateRateTest(unittest.TestCase):
    """The number that catches memorization plain novelty misses."""

    TRAIN = ["Volvo", "Toyota", "Honda"]

    def test_exact_copies_count_as_near_duplicates(self):
        self.assertEqual(near_duplicate_rate(["Volvo", "Honda"], self.TRAIN, max_dist=1), 1.0)

    def test_one_edit_away(self):
        # Volvos = insertion, Vulvo = substitution, Vlvo = deletion: all distance 1.
        names = ["Volvos", "Vulvo", "Vlvo"]
        self.assertEqual(near_duplicate_rate(names, self.TRAIN, max_dist=1), 1.0)

    def test_distant_names_are_not_near_duplicates(self):
        self.assertEqual(near_duplicate_rate(["Zqxjk", "Wrmpf"], self.TRAIN, max_dist=1), 0.0)

    def test_hand_computed_fraction(self):
        # Volvo (0), Volvox (1), Zqxjk (far), Hondaa (1)  ->  3 of 4.
        names = ["Volvo", "Volvox", "Zqxjk", "Hondaa"]
        self.assertEqual(near_duplicate_rate(names, self.TRAIN, max_dist=1), 0.75)

    def test_max_dist_2_is_a_superset_of_max_dist_1(self):
        # Toyot_a_ -> "Toyta" is 1 deletion; "Toya" is 2. So d<=1 catches 1 of 2,
        # d<=2 catches both.
        names = ["Toyta", "Toya"]
        self.assertEqual(near_duplicate_rate(names, self.TRAIN, max_dist=1), 0.5)
        self.assertEqual(near_duplicate_rate(names, self.TRAIN, max_dist=2), 1.0)

    def test_empty_input_is_nan(self):
        self.assertTrue(math.isnan(near_duplicate_rate([], self.TRAIN)))

    def test_length_pruning_does_not_lose_matches(self):
        # A brute-force scan must agree with the length-bucketed fast path.
        train = ["alpha", "beta", "gammaray", "x"]
        probe = ["alph", "alphaa", "beta", "gammaray!", "y", "zzzzzzzz"]
        for dist in (1, 2):
            brute = sum(
                any(_edit_distance(p, t) <= dist for t in train) for p in probe
            ) / len(probe)
            self.assertAlmostEqual(near_duplicate_rate(probe, train, max_dist=dist), brute)


class BigramPlausibilityTest(unittest.TestCase):
    def test_score_matches_hand_computation(self):
        # Corpus "ab" -> bracketed \x02 a b \x03. Alphabet = {START, a, b, END}, V = 4.
        # Row for 'a': counts {b: 1}; total = 1 + 4 = 5. So P(b|a) = (1+1)/5 = 2/5.
        table = _bigram_logprobs(["ab"])
        self.assertAlmostEqual(table["a"]["b"], math.log(2 / 5))
        # Unseen transition a->a still gets add-1 mass: (0+1)/5.
        self.assertAlmostEqual(table["a"]["a"], math.log(1 / 5))

    def test_mean_bigram_ll_of_empty_list_is_nan(self):
        self.assertTrue(math.isnan(mean_bigram_ll([], _bigram_logprobs(["ab"]))))

    def test_typical_names_score_above_gibberish(self):
        table = _bigram_logprobs(TRAIN_NAMES)
        self.assertGreater(_score_name("alfa", table), _score_name("qxzjv", table))


class GeneralizationGapTest(unittest.TestCase):
    def test_gap_is_val_minus_train(self):
        self.assertAlmostEqual(generalization_gap(1.25, 7.40), 6.15)

    def test_gap_is_none_without_held_out_names(self):
        # None, not 0.0: "not measured" must never render as "measured, and fine".
        self.assertIsNone(generalization_gap(1.25, None))

    def test_verdicts_are_ordered(self):
        self.assertIn("generalizes", _gap_verdict(0.1))
        self.assertIn("some overfitting", _gap_verdict(1.0))
        self.assertIn("memorized", _gap_verdict(6.15))


class HeldOutNllTest(unittest.TestCase):
    """The WS-11 headline: a real number when val_names exist, an honest n/a when not."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ws11_eval_")
        cls.with_val = os.path.join(cls.tmp, "with_val.pt")
        cls.without_val = os.path.join(cls.tmp, "without_val.pt")
        _write_checkpoint(cls.with_val, TRAIN_NAMES, VAL_NAMES)
        _write_checkpoint(cls.without_val, TRAIN_NAMES, VAL_NAMES, include_val_key=False)

    @classmethod
    def tearDownClass(cls):
        clear_checkpoint_cache()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_val_names_path_reports_a_real_number(self):
        m = evaluate(self.with_val, num=12, temperature=0.9, seed=3, verbose=False)
        self.assertIsNotNone(m["held_out_nll"])
        self.assertGreater(m["held_out_nll"], 0.0)
        self.assertEqual(m["num_val_names"], len(VAL_NAMES))
        self.assertEqual(m["num_training_names"], len(TRAIN_NAMES))

    def test_gap_is_reported_next_to_the_training_nll(self):
        m = evaluate(self.with_val, num=12, temperature=0.9, seed=3, verbose=False)
        self.assertAlmostEqual(m["nll_gap"], m["held_out_nll"] - m["training_nll"], places=9)

    def test_held_out_nll_matches_the_standalone_function(self):
        from src.evaluate import _load
        ckpt = _load(self.with_val)
        direct = held_out_nll(ckpt.model, ckpt.vocab, VAL_NAMES, ckpt.cfg)
        m = evaluate(self.with_val, num=12, temperature=0.9, seed=3, verbose=False)
        self.assertAlmostEqual(m["held_out_nll"], direct, places=9)

    def test_missing_val_names_degrades_gracefully(self):
        m = evaluate(self.without_val, num=12, temperature=0.9, seed=3, verbose=False)
        self.assertIsNone(m["held_out_nll"])
        self.assertIsNone(m["nll_gap"])
        self.assertEqual(m["num_val_names"], 0)
        # The rest of the harness must still work — no val_names is not an error.
        self.assertGreater(m["training_nll"], 0.0)
        self.assertGreaterEqual(m["novelty"], 0.0)

    def test_empty_val_names_list_is_treated_as_absent(self):
        path = os.path.join(self.tmp, "empty_val.pt")
        _write_checkpoint(path, TRAIN_NAMES, [])
        self.assertIsNone(evaluate(path, num=8, seed=1, verbose=False)["held_out_nll"])

    def test_printed_report_shows_both_nlls_and_the_gap(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            evaluate(self.with_val, num=8, seed=3, verbose=True)
        out = buf.getvalue()
        self.assertIn("train", out)
        self.assertIn("held-out", out)
        self.assertIn("gap", out)

    def test_printed_report_says_n_a_without_val_names(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            evaluate(self.without_val, num=8, seed=3, verbose=True)
        self.assertIn("n/a", buf.getvalue())


class NoveltyAndUniquenessTest(unittest.TestCase):
    """Novelty and uniqueness are computed inline in evaluate(); pin their definitions."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ws11_nov_")
        cls.ckpt = os.path.join(cls.tmp, "m.pt")
        _write_checkpoint(cls.ckpt, TRAIN_NAMES, VAL_NAMES)

    @classmethod
    def tearDownClass(cls):
        clear_checkpoint_cache()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_definitions_hold_against_the_returned_samples(self):
        m = evaluate(self.ckpt, num=40, temperature=1.0, seed=11, verbose=False)
        samples = m["samples"]
        distinct = set(samples)
        self.assertEqual(m["generated"], len(samples))
        self.assertEqual(m["distinct"], len(distinct))
        self.assertAlmostEqual(m["uniqueness"], len(distinct) / len(samples))
        novel = [n for n in distinct if n not in set(TRAIN_NAMES)]
        self.assertAlmostEqual(m["novelty"], len(novel) / len(distinct))
        self.assertAlmostEqual(m["avg_length"],
                               sum(len(n) for n in samples) / len(samples))

    def test_novelty_is_bounded_by_near_duplicate_rate(self):
        # Every exact copy is also a near-duplicate, so (1 - novelty) <= near_dup_1.
        m = evaluate(self.ckpt, num=40, temperature=1.0, seed=11, verbose=False)
        self.assertLessEqual(1.0 - m["novelty"] - 1e-9, m["near_dup_rate_1"])
        self.assertLessEqual(m["near_dup_rate_1"] - 1e-9, m["near_dup_rate_2"])

    def test_same_seed_gives_identical_metrics(self):
        a = evaluate(self.ckpt, num=25, temperature=0.9, seed=42, verbose=False)
        b = evaluate(self.ckpt, num=25, temperature=0.9, seed=42, verbose=False)
        self.assertEqual(a["samples"], b["samples"])
        self.assertEqual(a["novelty"], b["novelty"])
        self.assertEqual(a["training_nll"], b["training_nll"])


class ReportTest(unittest.TestCase):
    """--report must be self-contained and byte-identical between runs."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ws11_report_")
        cls.ckpt = os.path.join(cls.tmp, "m.pt")
        cls.bare = os.path.join(cls.tmp, "bare.pt")
        _write_checkpoint(cls.ckpt, TRAIN_NAMES, VAL_NAMES)
        _write_checkpoint(cls.bare, TRAIN_NAMES, VAL_NAMES, include_val_key=False)

    @classmethod
    def tearDownClass(cls):
        clear_checkpoint_cache()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _render(self, name, checkpoint=None, **kwargs):
        checkpoint = checkpoint or self.ckpt
        out = os.path.join(self.tmp, name)
        metrics = evaluate(checkpoint, num=kwargs.pop("num", 40), temperature=0.9,
                           seed=17, verbose=False)
        write_report(out, checkpoint, metrics, temperature=0.9, num=40, seed=17, **kwargs)
        with open(out, encoding="utf-8") as fh:
            return fh.read()

    def test_report_is_deterministic_under_a_fixed_seed(self):
        first = self._render("a.md")
        clear_checkpoint_cache()          # force a full reload, as a second process would
        second = self._render("b.md")
        self.assertEqual(first, second)

    def test_report_is_self_contained(self):
        text = self._render("c.md")
        for expected in ("# Evaluation report", "## Checkpoint", "sha256", "## Dataset",
                         "## Generalization", "## Sampling", "## Metrics",
                         "## Generated names", "## Recommended decoding",
                         "novelty", "near-duplicate", "plausibility ratio",
                         "greek", "data/greek.txt"):
            with self.subTest(expected=expected):
                self.assertIn(expected, text)

    def test_report_shows_train_val_and_gap_together(self):
        text = self._render("d.md")
        self.assertIn("model NLL on training names", text)
        self.assertIn("model NLL on held-out names", text)
        self.assertIn("train → held-out gap", text)
        # The gap row must carry a signed number, not "n/a".
        gap_row = [ln for ln in text.splitlines() if "train → held-out gap" in ln][0]
        self.assertNotIn("n/a", gap_row)

    def test_report_says_n_a_without_val_names(self):
        text = self._render("e.md", checkpoint=self.bare)
        self.assertIn("no `val_names`", text)
        gap_row = [ln for ln in text.splitlines() if "train → held-out gap" in ln][0]
        self.assertIn("n/a", gap_row)

    def test_report_contains_real_generated_names(self):
        metrics = evaluate(self.ckpt, num=60, temperature=0.9, seed=17, verbose=False)
        out = os.path.join(self.tmp, "f.md")
        write_report(out, self.ckpt, metrics, temperature=0.9, num=60, seed=17,
                     num_samples=30)
        with open(out, encoding="utf-8") as fh:
            text = fh.read()
        block = text.split("## Generated names")[1].split("```")[1]
        listed = [n for n in (x.strip() for x in block.replace("\n", ",").split(",")) if n]
        self.assertGreater(len(listed), 0)
        self.assertLessEqual(len(listed), 30)
        # Every name in the block must be one the model actually produced.
        for name in listed:
            self.assertIn(name, metrics["samples"])

    def test_report_with_sweep_rows_recommends_from_the_grid(self):
        rows = sweep(self.ckpt, num=12, temperatures=[0.8, 1.2], top_ks=[5], top_ps=[0.9],
                     seed=17, verbose=False)
        metrics = evaluate(self.ckpt, num=20, temperature=0.9, seed=17, verbose=False)
        out = os.path.join(self.tmp, "g.md")
        write_report(out, self.ckpt, metrics, temperature=0.9, num=20, seed=17,
                     sweep_rows=rows)
        with open(out, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn(f"over the {len(rows)} settings below", text)
        self.assertIn("| temp | decoding |", text)
        self.assertNotIn("No sweep was run", text)

    def test_report_without_sweep_rows_admits_it(self):
        self.assertIn("No sweep was run", self._render("h.md", sweep_rows=None))

    def test_report_creates_missing_directories(self):
        nested = os.path.join(self.tmp, "deep", "deeper", "r.md")
        metrics = evaluate(self.ckpt, num=10, temperature=0.9, seed=17, verbose=False)
        write_report(nested, self.ckpt, metrics, temperature=0.9, num=10, seed=17)
        self.assertTrue(os.path.exists(nested))


class CheckpointCacheTest(unittest.TestCase):
    def test_cache_reuses_one_load_but_notices_a_rewrite(self):
        from src.evaluate import _load
        tmp = tempfile.mkdtemp(prefix="ws11_cache_")
        try:
            path = os.path.join(tmp, "m.pt")
            _write_checkpoint(path, TRAIN_NAMES, VAL_NAMES, seed=1)
            first = _load(path)
            self.assertIs(_load(path), first)
            # Rewriting the file must invalidate the entry, or a retrained checkpoint
            # would be silently evaluated with the old weights.
            _write_checkpoint(path, TRAIN_NAMES, VAL_NAMES, seed=99)
            os.utime(path, (0, 0))  # force a different mtime regardless of clock granularity
            self.assertIsNot(_load(path), first)
        finally:
            clear_checkpoint_cache()
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
