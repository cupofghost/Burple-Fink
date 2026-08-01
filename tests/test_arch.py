"""WS-9: the architecture switch (`cfg.arch`) and the contract all three must keep.

The interesting test in here is `test_stepwise_matches_full_sequence`. `src/sample.py`
generates one character at a time and carries `hidden` forward; an LSTM and a GRU do
that natively, but a transformer has no recurrent state and has to reconstruct the
prefix every step. If that reconstruction is off by one position, or the causal mask
is built for the wrong length, generation still produces plausible-looking names while
conditioning on the wrong thing -- a silent failure. So every architecture is required
to prove that decoding a sequence one token at a time gives the same logits as one
full-sequence forward pass.

The rest pin the things it is easy to break while making that work: causal masking,
PAD's non-contribution, the `max_position` ceiling failing loudly instead of silently,
the dual-output value head, and checkpoint round-tripping for a non-default arch.
"""

import os
import tempfile
import unittest

import torch

from src.arch import ARCHITECTURES, PrefixState
from src.config import Config
from src.data import Vocab
from src.model import CharRNN
from src.sample import generate_one, load_checkpoint
from src.train import save_checkpoint

torch.set_num_threads(1)

NAMES = [
    "Vroomio", "Zaxon", "Turbex", "Velcar", "Roadix", "Motoza", "Carvo",
    "Zoomer", "Draxel", "Vexor", "Torro", "Zephex", "Racton", "Vantek",
]


def _cfg(arch: str, **over) -> Config:
    """A tiny but *structurally faithful* config: >1 layer, >1 head, dropout off.

    num_layers=2 matters -- a 1-layer model would hide a "which layer is the top
    layer" bug in the state contract. dropout=0.0 matters because incremental
    decoding is only deterministic in eval mode, and these tests compare numbers.
    """
    base = dict(
        arch=arch, embedding_dim=8, hidden_dim=16, num_layers=2, dropout=0.0,
        num_heads=4, ff_dim=32, max_position=24, seed=7,
    )
    base.update(over)
    return Config(**base)


def _model(arch: str, vocab: Vocab, **over) -> CharRNN:
    torch.manual_seed(11)
    model = CharRNN(len(vocab), _cfg(arch, **over), pad_id=vocab.pad_id)
    model.eval()
    return model


class ShapeContractTests(unittest.TestCase):
    """Every arch must satisfy the API the rest of the repo already calls."""

    def setUp(self):
        self.vocab = Vocab(NAMES)

    def test_forward_and_encode_shapes(self):
        for arch in ARCHITECTURES:
            with self.subTest(arch=arch):
                model = _model(arch, self.vocab)
                x = torch.randint(1, len(self.vocab), (3, 5))
                logits, hidden = model(x)
                self.assertEqual(tuple(logits.shape), (3, 5, len(self.vocab)))
                self.assertIsNotNone(hidden)

                out, hidden2 = model.encode(x)
                self.assertEqual(tuple(out.shape), (3, 5, model.cfg.hidden_dim))
                self.assertIsNotNone(hidden2)

    def test_state_exposes_last_timestep_as_batch_by_hidden(self):
        """`src/sample.py` reads `hidden[0][-1]`; it must be (batch, hidden_dim)."""
        for arch in ARCHITECTURES:
            with self.subTest(arch=arch):
                model = _model(arch, self.vocab)
                x = torch.randint(1, len(self.vocab), (3, 5))
                _, hidden = model(x)
                last = hidden[0][-1]
                self.assertEqual(tuple(last.shape), (3, model.cfg.hidden_dim))

    def test_attributes_and_class_name_survive(self):
        for arch in ARCHITECTURES:
            with self.subTest(arch=arch):
                model = _model(arch, self.vocab)
                self.assertEqual(type(model).__name__, "CharRNN")
                self.assertEqual(model.vocab_size, len(self.vocab))
                self.assertEqual(model.cfg.arch, arch)

    def test_default_config_is_lstm_with_unchanged_state_dict_keys(self):
        """A default-config checkpoint must still load: same keys, same names.

        `src/export_web.py` indexes `lstm.weight_ih_l0` etc. by hand, so these key
        names are a real external contract, not an implementation detail.
        """
        cfg = Config(embedding_dim=8, hidden_dim=16, num_layers=2, dropout=0.0)
        self.assertEqual(cfg.arch, "lstm")
        model = CharRNN(len(self.vocab), cfg, pad_id=self.vocab.pad_id)
        keys = set(model.state_dict().keys())
        expected = {"embedding.weight", "head.weight", "head.bias"}
        for layer in range(cfg.num_layers):
            for part in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                expected.add(f"lstm.{part}_l{layer}")
        self.assertEqual(keys, expected)

    def test_unknown_arch_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            CharRNN(len(self.vocab), _cfg("rnn"), pad_id=self.vocab.pad_id)
        self.assertIn("rnn", str(ctx.exception))
        self.assertIn("transformer", str(ctx.exception))

    def test_heads_must_divide_hidden_dim(self):
        with self.assertRaises(ValueError) as ctx:
            CharRNN(len(self.vocab), _cfg("transformer", hidden_dim=18, num_heads=4),
                    pad_id=self.vocab.pad_id)
        self.assertIn("num_heads", str(ctx.exception))


class IncrementalDecodingTests(unittest.TestCase):
    """The whole correctness argument for the transformer's fake hidden state."""

    def setUp(self):
        self.vocab = Vocab(NAMES)
        torch.manual_seed(3)
        self.x = torch.randint(1, len(self.vocab), (2, 9))

    def test_stepwise_matches_full_sequence(self):
        for arch in ARCHITECTURES:
            with self.subTest(arch=arch):
                model = _model(arch, self.vocab)
                with torch.no_grad():
                    full, _ = model(self.x)

                    hidden = None
                    steps = []
                    for t in range(self.x.size(1)):
                        step, hidden = model(self.x[:, t:t + 1], hidden)
                        steps.append(step)
                    stepwise = torch.cat(steps, dim=1)

                self.assertEqual(tuple(stepwise.shape), tuple(full.shape))
                max_diff = (stepwise - full).abs().max().item()
                self.assertLess(
                    max_diff, 1e-4,
                    f"{arch}: one-token-at-a-time decoding diverged from a single "
                    f"forward pass by {max_diff:.3g}",
                )

    def test_chunked_decoding_matches_full_sequence(self):
        """Not just 1 token at a time: a prefix prompt then single steps.

        This is literally what `generate_one(prefix=...)` does -- prime with
        START+prefix in one call, then step -- so it deserves its own check.
        """
        for arch in ARCHITECTURES:
            with self.subTest(arch=arch):
                model = _model(arch, self.vocab)
                with torch.no_grad():
                    full, _ = model(self.x)
                    head, hidden = model(self.x[:, :4])
                    mid, hidden = model(self.x[:, 4:7], hidden)
                    tail, _ = model(self.x[:, 7:], hidden)
                    chunked = torch.cat([head, mid, tail], dim=1)
                max_diff = (chunked - full).abs().max().item()
                self.assertLess(max_diff, 1e-4, f"{arch}: chunked decoding diverged")

    def test_transformer_state_carries_the_prefix(self):
        """A shape assertion on the design, so a future refactor states its intent."""
        model = _model("transformer", self.vocab)
        with torch.no_grad():
            _, hidden = model(self.x[:, :4])
            self.assertIsInstance(hidden, PrefixState)
            self.assertEqual(hidden.prefix.size(1), 4)
            _, hidden = model(self.x[:, 4:5], hidden)
            self.assertEqual(hidden.prefix.size(1), 5)

    def test_generate_one_works_for_every_arch(self):
        """End-to-end through the *unmodified* sampler, including a prefix."""
        for arch in ARCHITECTURES:
            with self.subTest(arch=arch):
                model = _model(arch, self.vocab)
                torch.manual_seed(5)
                name = generate_one(model, self.vocab, temperature=0.8, max_length=12)
                self.assertIsInstance(name, str)
                primed = generate_one(model, self.vocab, temperature=0.8,
                                      max_length=12, prefix="Va")
                self.assertTrue(primed.startswith("Va"))


class CausalMaskingTests(unittest.TestCase):
    """A name generator must never see the future."""

    def setUp(self):
        self.vocab = Vocab(NAMES)

    def test_changing_a_later_token_leaves_earlier_logits_alone(self):
        torch.manual_seed(4)
        x = torch.randint(1, len(self.vocab), (1, 8))
        edited = x.clone()
        edited[0, 5] = (int(x[0, 5]) % (len(self.vocab) - 1)) + 1
        self.assertNotEqual(int(edited[0, 5]), int(x[0, 5]))

        for arch in ARCHITECTURES:
            with self.subTest(arch=arch):
                model = _model(arch, self.vocab)
                with torch.no_grad():
                    a, _ = model(x)
                    b, _ = model(edited)
                # Positions 0..4 precede the edit and must be untouched.
                self.assertTrue(
                    torch.equal(a[:, :5, :], b[:, :5, :]),
                    f"{arch}: editing position 5 changed the logits at positions 0-4",
                )
                # Sanity: the edit did reach *some* later position, so this test
                # is not vacuously passing on a model that ignores its input.
                self.assertFalse(torch.equal(a[:, 5:, :], b[:, 5:, :]))

    def test_trailing_pad_contributes_nothing(self):
        """Right-padding a batch must not change the real positions' logits."""
        for arch in ARCHITECTURES:
            with self.subTest(arch=arch):
                model = _model(arch, self.vocab)
                self.assertTrue(
                    torch.equal(model.embedding.weight[self.vocab.pad_id],
                                torch.zeros(model.cfg.embedding_dim)),
                    f"{arch}: PAD's embedding row is not zero",
                )
                torch.manual_seed(9)
                x = torch.randint(1, len(self.vocab), (1, 6))
                padded = torch.cat(
                    [x, torch.full((1, 3), self.vocab.pad_id, dtype=torch.long)], dim=1
                )
                with torch.no_grad():
                    a, _ = model(x)
                    b, _ = model(padded)
                self.assertTrue(
                    torch.equal(a, b[:, :6, :]),
                    f"{arch}: trailing PAD changed the real positions' logits",
                )


class MaxPositionTests(unittest.TestCase):
    """Overflowing the positional table must fail loudly, not answer wrongly."""

    def setUp(self):
        self.vocab = Vocab(NAMES)

    def test_full_sequence_overflow_raises(self):
        model = _model("transformer", self.vocab, max_position=8)
        x = torch.randint(1, len(self.vocab), (1, 9))
        with self.assertRaises(ValueError) as ctx:
            model(x)
        msg = str(ctx.exception)
        self.assertIn("max_position", msg)
        self.assertIn("9", msg)

    def test_incremental_overflow_raises_at_the_step_that_crosses(self):
        model = _model("transformer", self.vocab, max_position=4)
        x = torch.randint(1, len(self.vocab), (1, 6))
        with torch.no_grad():
            _, hidden = model(x[:, :4])          # exactly at the ceiling: fine
            with self.assertRaises(ValueError):
                model(x[:, 4:5], hidden)         # one over: must raise

    def test_recurrent_arches_have_no_length_ceiling(self):
        """The RNNs genuinely have no such limit; don't invent one for them."""
        for arch in ("lstm", "gru"):
            with self.subTest(arch=arch):
                model = _model(arch, self.vocab, max_position=4)
                x = torch.randint(1, len(self.vocab), (1, 30))
                with torch.no_grad():
                    logits, _ = model(x)
                self.assertEqual(logits.size(1), 30)


class DualOutputTests(unittest.TestCase):
    """WS-4's value head must work through every core."""

    def setUp(self):
        self.vocab = Vocab(NAMES)

    def test_value_head_from_generation_state(self):
        for arch in ARCHITECTURES:
            with self.subTest(arch=arch):
                model = _model(arch, self.vocab, dual_output=True,
                               value_mean=1950.0, value_std=20.0)
                x = torch.randint(1, len(self.vocab), (3, 5))
                with torch.no_grad():
                    _, hidden = model(x)
                    value = model.predict_value(hidden[0][-1])
                self.assertEqual(tuple(value.shape), (3,))

    def test_value_head_from_encode_output(self):
        """The training path (`src/train_dual.py`) reads encode()'s last timestep."""
        for arch in ARCHITECTURES:
            with self.subTest(arch=arch):
                model = _model(arch, self.vocab, dual_output=True)
                x = torch.randint(1, len(self.vocab), (2, 5))
                with torch.no_grad():
                    out, _ = model.encode(x)
                    value = model.predict_value(out[:, -1, :])
                self.assertEqual(tuple(value.shape), (2,))

    def test_generation_state_and_encode_output_agree(self):
        """`hidden[0][-1]` and `encode()[:, -1]` must be the same vector.

        `src/sample.py` uses the first and `src/train_dual.py` the second, for the
        same quantity. If they ever disagreed, a dual-output model would train on
        one representation and report another.
        """
        for arch in ARCHITECTURES:
            with self.subTest(arch=arch):
                model = _model(arch, self.vocab, dual_output=True)
                x = torch.randint(1, len(self.vocab), (2, 5))
                with torch.no_grad():
                    out, hidden = model.encode(x)
                max_diff = (hidden[0][-1] - out[:, -1, :]).abs().max().item()
                self.assertLess(max_diff, 1e-6, f"{arch}: {max_diff:.3g}")

    def test_predict_value_still_refuses_without_dual_output(self):
        for arch in ARCHITECTURES:
            with self.subTest(arch=arch):
                model = _model(arch, self.vocab)
                with self.assertRaises(RuntimeError):
                    model.predict_value(torch.zeros(1, model.cfg.hidden_dim))


class CheckpointRoundTripTests(unittest.TestCase):
    """A non-default arch must survive save -> load -> identical logits."""

    def test_round_trip_for_every_arch(self):
        vocab = Vocab(NAMES)
        x = torch.randint(1, len(vocab), (2, 6))
        for arch in ARCHITECTURES:
            with self.subTest(arch=arch):
                model = _model(arch, vocab)
                with torch.no_grad():
                    before, _ = model(x)
                with tempfile.TemporaryDirectory() as d:
                    path = os.path.join(d, f"ws9_{arch}.pt")
                    save_checkpoint(path, model, model.cfg, vocab, NAMES, val_names=[])
                    loaded, loaded_vocab, loaded_cfg, _ = load_checkpoint(path)
                self.assertEqual(loaded_cfg.arch, arch)
                self.assertEqual(len(loaded_vocab), len(vocab))
                with torch.no_grad():
                    after, _ = loaded(x)
                self.assertTrue(torch.equal(before, after),
                                f"{arch}: reloaded model produced different logits")

    def test_checkpoint_without_arch_key_loads_as_lstm(self):
        """Pre-wave-3 checkpoints have no `arch` field; they must default to LSTM."""
        vocab = Vocab(NAMES)
        model = _model("lstm", vocab)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ws9_legacy.pt")
            save_checkpoint(path, model, model.cfg, vocab, NAMES, val_names=[])
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            del ckpt["config"]["arch"]          # simulate an old checkpoint
            torch.save(ckpt, path)
            loaded, _, loaded_cfg, _ = load_checkpoint(path)
        self.assertEqual(loaded_cfg.arch, "lstm")
        x = torch.randint(1, len(vocab), (1, 5))
        with torch.no_grad():
            a, _ = model(x)
            b, _ = loaded(x)
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
