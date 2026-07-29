"""Tests for WS-7: decoding controls in src/sample.py and the near-duplicate
metric / --sweep in src/evaluate.py.

Same "seconds on a CPU" convention as tests/test_engine.py: a tiny synthetic
domain, a tiny net, a handful of epochs (or none at all -- several of these
tests only need a randomly-initialized model to exercise the sampling loop).
"""

import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

from src.config import Config
from src.data import Vocab
from src.model import CharRNN
from src.sample import (
    generate_one,
    generate_many,
    _apply_repetition_penalty,
    _top_k_filter,
    _top_p_filter,
)
from src.evaluate import evaluate, sweep, near_duplicate_rate
from src.train import fit, save_checkpoint


NAMES = [
    "Vroomio", "Zaxon", "Turbex", "Velcar", "Roadix", "Motoza", "Carvo",
    "Zoomer", "Draxel", "Vexor", "Torro", "Zephex", "Racton", "Vantek",
]


def _tiny_cfg(**overrides):
    base = dict(embedding_dim=8, hidden_dim=24, num_layers=1, dropout=0.0,
                epochs=40, batch_size=8, learning_rate=5e-3, sample_every=1000,
                seed=7, max_length=20)
    base.update(overrides)
    return Config(**base)


def _tiny_model(vocab: Vocab, cfg: Config) -> CharRNN:
    """A briefly-trained (not just randomly-initialized) tiny model.

    A few dozen epochs are enough that PAD/START -- which the model is
    trained never to predict -- stay effectively unreachable, so tests that
    exercise many generation steps aren't flaky over which random-init logits
    happened to favor a special token.
    """
    torch.manual_seed(cfg.seed)
    model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id)
    fit(model, vocab, NAMES, cfg)
    model.eval()
    return model


class TopKFilterTests(unittest.TestCase):
    def test_filter_keeps_only_k_highest_logits(self):
        logits = torch.tensor([[5.0, 1.0, 4.0, 0.5, 3.0]])
        filtered = _top_k_filter(logits, 2)
        kept = (filtered > float("-inf")).sum().item()
        self.assertEqual(kept, 2)
        self.assertEqual(filtered[0, 0].item(), 5.0)
        self.assertEqual(filtered[0, 2].item(), 4.0)

    def test_zero_is_off(self):
        logits = torch.tensor([[5.0, 1.0, 4.0]])
        self.assertTrue(torch.equal(_top_k_filter(logits, 0), logits))

    def test_top_k_one_makes_generation_deterministic(self):
        """assert on the support, not on a lucky draw: with top_k=1 only one
        character is ever reachable at each step, so the whole name is fixed
        regardless of the RNG seed."""
        vocab = Vocab(NAMES)
        cfg = _tiny_cfg()
        model = _tiny_model(vocab, cfg)

        torch.manual_seed(1)
        first = generate_one(model, vocab, temperature=1.5, max_length=15, top_k=1)
        for seed in (2, 3, 4, 99):
            torch.manual_seed(seed)
            again = generate_one(model, vocab, temperature=1.5, max_length=15, top_k=1)
            self.assertEqual(again, first)


class TopPFilterTests(unittest.TestCase):
    def test_keeps_smallest_sufficient_set(self):
        # softmax([3, 2, 1, 0]) puts ~64% of the mass on the first entry alone.
        logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
        filtered = _top_p_filter(logits, 0.5)
        kept = (filtered > float("-inf")).sum().item()
        self.assertEqual(kept, 1)
        self.assertEqual(filtered[0, 0].item(), 3.0)

    def test_one_point_zero_is_off(self):
        logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
        self.assertTrue(torch.equal(_top_p_filter(logits, 1.0), logits))


class RepetitionPenaltyTests(unittest.TestCase):
    def test_positive_logit_divided_negative_logit_multiplied(self):
        logits = torch.tensor([[2.0, -2.0, 0.5]])
        out = _apply_repetition_penalty(logits, [0, 1], penalty=2.0)
        self.assertAlmostEqual(out[0, 0].item(), 1.0)
        self.assertAlmostEqual(out[0, 1].item(), -4.0)
        self.assertAlmostEqual(out[0, 2].item(), 0.5)  # never emitted -> untouched

    def test_one_point_zero_is_noop(self):
        logits = torch.tensor([[2.0, -2.0, 0.5]])
        out = _apply_repetition_penalty(logits, [0, 1], penalty=1.0)
        self.assertTrue(torch.equal(out, logits))

    def test_penalty_reduces_repeated_characters(self):
        vocab = Vocab(NAMES)
        cfg = _tiny_cfg()
        model = _tiny_model(vocab, cfg)

        def repeated_char_count(penalty, n=80):
            torch.manual_seed(0)
            total = 0
            for _ in range(n):
                name = generate_one(model, vocab, temperature=1.4, max_length=25,
                                     repetition_penalty=penalty)
                total += sum(1 for a, b in zip(name, name[1:]) if a == b)
            return total

        baseline = repeated_char_count(1.0)
        penalized = repeated_char_count(2.5)
        self.assertLess(penalized, baseline)


class MinLengthTests(unittest.TestCase):
    def test_min_length_is_honored_during_generation(self):
        vocab = Vocab(NAMES)
        cfg = _tiny_cfg()
        model = _tiny_model(vocab, cfg)

        for seed in range(25):
            torch.manual_seed(seed)
            name = generate_one(model, vocab, temperature=1.5, max_length=15, min_length=5)
            self.assertGreaterEqual(len(name), 5)

    def test_zero_is_off(self):
        vocab = Vocab(NAMES)
        cfg = _tiny_cfg()
        model = _tiny_model(vocab, cfg)
        torch.manual_seed(3)
        with_off = generate_one(model, vocab, temperature=0.9, max_length=15, min_length=0)
        torch.manual_seed(3)
        without_arg = generate_one(model, vocab, temperature=0.9, max_length=15)
        self.assertEqual(with_off, without_arg)


class BackwardCompatibilityTests(unittest.TestCase):
    """The core proof: top_k=0, top_p=1.0, repetition_penalty=1.0 (all defaults)
    must reproduce the pre-WS-7 plain-temperature sampling loop exactly."""

    def _old_generate_one(self, model, vocab, temperature, max_length):
        model.eval()
        tokens = [vocab.start_id]
        inp = torch.tensor([tokens], dtype=torch.long, device="cpu")
        logits, hidden = model(inp)
        out_ids = []
        next_logits = logits[:, -1, :]
        for _ in range(max_length):
            probs = F.softmax(next_logits / max(temperature, 1e-6), dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            nxt_id = nxt.item()
            if nxt_id == vocab.end_id or nxt_id in (vocab.pad_id, vocab.start_id):
                break
            out_ids.append(nxt_id)
            logits, hidden = model(nxt, hidden)
            next_logits = logits[:, -1, :]
        return vocab.decode(out_ids)

    def test_defaults_match_the_old_loop_for_a_fixed_seed(self):
        vocab = Vocab(NAMES)
        cfg = _tiny_cfg()
        model = _tiny_model(vocab, cfg)

        for seed in range(10):
            torch.manual_seed(seed)
            expected = self._old_generate_one(model, vocab, temperature=0.9, max_length=20)
            torch.manual_seed(seed)
            actual = generate_one(model, vocab, temperature=0.9, max_length=20)
            self.assertEqual(actual, expected)


class NearDuplicateRateTests(unittest.TestCase):
    def test_exact_and_one_edit_away_are_caught(self):
        training = ["Vroomio", "Zaxon"]
        generated = ["Vroomio", "Vroomia", "Something Else Entirely"]
        rate = near_duplicate_rate(generated, training, max_dist=1)
        self.assertAlmostEqual(rate, 2 / 3)

    def test_max_dist_2_is_looser_than_1(self):
        training = ["Vroomio"]
        generated = ["Vroomxy"]  # two substitutions away (i->x, o->y)
        rate1 = near_duplicate_rate(generated, training, max_dist=1)
        rate2 = near_duplicate_rate(generated, training, max_dist=2)
        self.assertEqual(rate1, 0.0)
        self.assertEqual(rate2, 1.0)


class HeldOutNLLTests(unittest.TestCase):
    def _make_checkpoint(self, val_names=None):
        tmpdir = tempfile.mkdtemp()
        vocab = Vocab(NAMES)
        cfg = _tiny_cfg()
        model = _tiny_model(vocab, cfg)
        path = save_checkpoint(os.path.join(tmpdir, "test.pt"), model, cfg, vocab, NAMES)
        if val_names is not None:
            ckpt = torch.load(path)
            ckpt["val_names"] = val_names
            torch.save(ckpt, path)
        return path

    def test_missing_val_names_degrades_gracefully(self):
        path = self._make_checkpoint(val_names=None)
        metrics = evaluate(path, num=10, seed=0, verbose=False)
        self.assertIsNone(metrics["held_out_nll"])

    def test_present_val_names_reports_a_number(self):
        path = self._make_checkpoint(val_names=NAMES[:3])
        metrics = evaluate(path, num=10, seed=0, verbose=False)
        self.assertIsInstance(metrics["held_out_nll"], float)


class SweepTests(unittest.TestCase):
    def setUp(self):
        tmpdir = tempfile.mkdtemp()
        vocab = Vocab(NAMES)
        cfg = _tiny_cfg()
        model = _tiny_model(vocab, cfg)
        self.ckpt_path = save_checkpoint(os.path.join(tmpdir, "test.pt"), model, cfg, vocab, NAMES)

    def test_sweep_returns_one_row_per_grid_point_with_metrics_populated(self):
        temperatures = [0.8, 1.2]
        rows = sweep(self.ckpt_path, num=20, temperatures=temperatures,
                     top_ks=[3], top_ps=[0.9], seed=0)
        # decoding settings = plain + top_k=3 + top_p=0.9 = 3, x 2 temperatures
        self.assertEqual(len(rows), 3 * len(temperatures))
        required = {
            "temperature", "decoding", "novelty", "near_dup_rate_1", "near_dup_rate_2",
            "uniqueness", "avg_edit_distance", "bigram_ll_generated",
            "bigram_ll_training", "training_nll", "held_out_nll",
        }
        for row in rows:
            self.assertTrue(required.issubset(row.keys()))


if __name__ == "__main__":
    unittest.main()
