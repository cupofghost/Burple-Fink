"""Tests for the WS-4 dual-output path: name/value data loading, the value head on
CharRNN, and the joint char+value training loop in src.train_dual.

Deliberately tiny (a handful of synthetic name/value pairs, a few epochs) so this
runs in seconds on CPU, matching the rest of the suite.
"""

import os
import tempfile
import unittest

import torch

from src.config import Config
from src.data import Vocab, load_name_value_pairs
from src.model import CharRNN
from src.sample_dual import generate_many_with_value
from src.train_dual import fit_dual


# A tiny synthetic name/value "domain": short names, value = 0.1 * name length,
# so the value head has an easy, learnable signal to check the loss actually falls.
NAMES = ["Vroomio", "Zaxon", "Turbex", "Velcar", "Roadix", "Motoza", "Carvo", "Zoomer"]
VALUES = [round(0.1 * len(n), 2) for n in NAMES]


def _tiny_cfg(epochs=20):
    return Config(embedding_dim=8, hidden_dim=24, num_layers=1, dropout=0.0,
                  epochs=epochs, batch_size=8, learning_rate=5e-3, sample_every=1000,
                  seed=7, max_length=20)


class NameValueDataTests(unittest.TestCase):
    def test_load_name_value_pairs_parses_and_dedupes(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nv.tsv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("Alpha\t0.5\nBeta\t0.25\nAlpha\t0.9\n\n")
            pairs = load_name_value_pairs(path)
            self.assertEqual(pairs, [("Alpha", 0.5), ("Beta", 0.25)])


class DualModelTests(unittest.TestCase):
    def test_forward_signature_unchanged_with_value_head(self):
        cfg = _tiny_cfg()
        vocab = Vocab(NAMES)
        model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id, predict_value=True)
        x = torch.zeros((3, 5), dtype=torch.long)
        logits, hidden = model(x)  # still exactly a (logits, hidden) pair
        self.assertEqual(tuple(logits.shape), (3, 5, len(vocab)))
        self.assertIsNotNone(hidden)

    def test_regress_value_shape_and_guard(self):
        cfg = _tiny_cfg()
        vocab = Vocab(NAMES)
        plain = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id)
        with self.assertRaises(RuntimeError):
            plain.regress_value(torch.zeros((2, 4), dtype=torch.long),
                                 torch.tensor([4, 3]))

        dual = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id, predict_value=True)
        x = torch.zeros((2, 4), dtype=torch.long)
        lengths = torch.tensor([4, 2])
        values = dual.regress_value(x, lengths)
        self.assertEqual(tuple(values.shape), (2,))


class DualTrainingTests(unittest.TestCase):
    def test_fit_dual_reduces_value_mse(self):
        cfg = _tiny_cfg(epochs=25)
        vocab = Vocab(NAMES)
        model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id, predict_value=True)

        fit_dual(model, vocab, NAMES, VALUES, cfg)

        # After training, predicted values should track the (easy, length-based)
        # target reasonably well rather than being arbitrary.
        model.eval()
        with torch.no_grad():
            ids = [[vocab.start_id] + [vocab.stoi[c] for c in n] for n in NAMES]
            max_len = max(len(seq) for seq in ids)
            x = torch.zeros((len(NAMES), max_len), dtype=torch.long)
            lengths = torch.tensor([len(seq) for seq in ids])
            for row, seq in enumerate(ids):
                x[row, :len(seq)] = torch.tensor(seq)
            preds = model.regress_value(x, lengths)
            targets = torch.tensor(VALUES)
            mse = torch.mean((preds - targets) ** 2).item()
        self.assertLess(mse, 0.05, "value head should fit an easy length-based signal")

        results = generate_many_with_value(
            model, vocab, cfg, num=4, temperature=0.8,
            training_names=set(NAMES), only_novel=False,
        )
        self.assertEqual(len(results), 4)
        for name, value in results:
            self.assertIsInstance(name, str)
            self.assertIsInstance(value, float)


if __name__ == "__main__":
    unittest.main()
