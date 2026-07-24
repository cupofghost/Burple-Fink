"""Tests for the WS-4 dual-output head: name + numeric attribute regression.

Tiny by the same "seconds on a CPU" convention as tests/test_engine.py. Confirms
the value head is additive (a plain Config/CharRNN is unaffected), that a dual
model actually learns the attribute (MSE falls), and that sample.py's
``return_value`` path denormalizes and reports it.
"""

import os
import tempfile
import unittest

import torch

from src.config import Config
from src.data import Vocab, load_name_value_pairs, make_pairs
from src.model import CharRNN
from src.sample import generate_many
from src.train_dual import fit_dual, train_dual


PAIRS = [
    ("Vroomio", 1990), ("Zaxon", 1985), ("Turbex", 2001), ("Velcar", 1978),
    ("Roadix", 1995), ("Motoza", 2005), ("Carvo", 1988), ("Zoomer", 2012),
    ("Draxel", 1999), ("Vexor", 2008), ("Torro", 1980), ("Zephex", 2015),
]
NAMES = [n for n, _ in PAIRS]


def _tiny_cfg(epochs=40):
    return Config(embedding_dim=8, hidden_dim=24, num_layers=1, dropout=0.0,
                  epochs=epochs, batch_size=6, learning_rate=8e-3, sample_every=1000,
                  seed=7, max_length=20)


class NonDualUnaffectedTests(unittest.TestCase):
    def test_plain_config_has_no_value_head(self):
        cfg = _tiny_cfg()
        self.assertFalse(cfg.dual_output)
        vocab = Vocab(NAMES)
        model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id)
        self.assertIsNone(model.value_head)
        with self.assertRaises(RuntimeError):
            model.predict_value(torch.zeros(1, cfg.hidden_dim))


class LoadNameValuePairsTests(unittest.TestCase):
    def test_parses_tsv_and_dedupes(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pairs.tsv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("# comment\nFord\t1903\nFord\t1903\n\nTesla\t2003\n")
            pairs = load_name_value_pairs(path)
            self.assertEqual(pairs, [("Ford", 1903.0), ("Tesla", 2003.0)])


class DualTrainingTests(unittest.TestCase):
    def test_fit_dual_reduces_value_mse_and_samples_have_values(self):
        cfg = _tiny_cfg()
        cfg.dual_output = True
        vocab = Vocab(NAMES)
        model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id)

        values = torch.tensor([v for _, v in PAIRS], dtype=torch.float32)
        mean, std = values.mean().item(), values.std().item()
        cfg.value_mean, cfg.value_std = mean, std
        values_z = ((values - mean) / std).tolist()

        first = _value_mse(model, vocab, NAMES, values_z)
        fit_dual(model, vocab, NAMES, values_z, cfg)
        last = _value_mse(model, vocab, NAMES, values_z)
        self.assertLess(last, first, "value regression loss should fall after training")

        results = generate_many(model, vocab, cfg, num=4, temperature=0.8,
                                 training_names=set(NAMES), only_novel=False,
                                 return_value=True)
        self.assertEqual(len(results), 4)
        for name, value in results:
            self.assertIsInstance(name, str)
            self.assertIsInstance(value, float)

    def test_train_dual_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            data_path = os.path.join(d, "data.tsv")
            with open(data_path, "w", encoding="utf-8") as fh:
                for name, value in PAIRS:
                    fh.write(f"{name}\t{value}\n")

            cfg = _tiny_cfg(epochs=5)
            out_path = train_dual(
                data_path, "tiny_dual", cfg,
                value_label="test year", checkpoint_dir=d,
            )
            ckpt = torch.load(out_path, map_location="cpu", weights_only=False)
            for key in ("model_state", "config", "vocab", "training_names"):
                self.assertIn(key, ckpt, f"checkpoint is missing contract key {key!r}")
            self.assertTrue(ckpt["config"]["dual_output"])
            self.assertEqual(ckpt["config"]["value_label"], "test year")


def _value_mse(model, vocab, names, values_z):
    """Mean value-regression MSE over one pass without updating weights."""
    model.eval()
    pairs = make_pairs(names, vocab)
    total, n = 0.0, 0
    with torch.no_grad():
        for i, (inp, _tgt) in enumerate(pairs):
            x = torch.tensor([inp], dtype=torch.long)
            out, _hidden = model.encode(x)
            state = out[:, -1, :]
            pred = model.predict_value(state).item()
            total += (pred - values_z[i]) ** 2
            n += 1
    return total / max(n, 1)


if __name__ == "__main__":
    unittest.main()
