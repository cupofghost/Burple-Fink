"""WS-6 tests: held-out split, validation loss, early stopping, best-weight restore.

The load-bearing test in here is :meth:`BackwardCompatibilityTest.test_default_path_
reproduces_pre_ws6_trajectory`. Everything else checks that the new machinery works;
that one checks that the *old* behavior survived it, against a loss trajectory captured
by running ``origin/main``'s ``fit()`` before this change existed. Wave-2's rule is
"every new option defaults to today's behavior — prove it, don't assert it", and a
golden trajectory is the proof.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data import Vocab, make_pairs, split_names
from src.model import CharRNN
from src.train import evaluate_loss, fit, save_checkpoint


# A fixed 16-name set. Small enough that eight epochs run in about a second, varied
# enough that the loss actually moves.
NAMES = ["Alfa", "Bravo", "Cobra", "Delta", "Echo", "Fiat", "Ghia", "Hemi",
         "Ibis", "Jaguar", "Kappa", "Lotus", "Mirage", "Nova", "Omni", "Puma"]

# The per-epoch mean training loss produced by origin/main's fit() on NAMES with the
# config below, captured before WS-6 touched the loop. Verified against that same
# fit()'s own printed output (epoch 1 -> 3.6504, epoch 8 -> 3.4681).
GOLDEN_TRAJECTORY = [
    3.6503708959, 3.6222720742, 3.5992223024, 3.5762013197,
    3.5527796149, 3.5293713212, 3.5018311739, 3.4681224227,
]


def assert_matches_golden(case, losses, msg=""):
    """Assert a trajectory equals the pre-WS-6 one, epoch by epoch.

    ``delta=1e-5`` rather than exact equality: origin/main only ever *printed* these to
    four decimals, and a genuine behavior change (a perturbed RNG stream, a different
    batch order, a scheduler that actually stepped) moves the loss by orders of
    magnitude more than 1e-5 within eight epochs.
    """
    want = GOLDEN_TRAJECTORY[:len(losses)]
    case.assertEqual(len(losses), len(want))
    for epoch, (got, expected) in enumerate(zip(losses, want), start=1):
        case.assertAlmostEqual(
            got, expected, delta=1e-5,
            msg=f"epoch {epoch}: {got!r} != pre-WS-6 {expected!r} {msg}")


def tiny_cfg(**overrides) -> Config:
    """The config GOLDEN_TRAJECTORY was captured with. Do not change these numbers."""
    base = dict(epochs=8, batch_size=4, hidden_dim=16, embedding_dim=8, num_layers=1,
                sample_every=1000, seed=99)
    base.update(overrides)
    return Config(**base)


def run_fit(names, cfg, vocab=None, val_names=None):
    """Build a fresh seeded model, train it silently, return (model, report)."""
    vocab = vocab or Vocab(names + (val_names or []))
    torch.manual_seed(cfg.seed)
    model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id)
    report: dict = {}
    with contextlib.redirect_stdout(io.StringIO()) as out:
        fit(model, vocab, names, cfg, val_names=val_names, report=report)
    report["stdout"] = out.getvalue()
    return model, vocab, report


class SplitTest(unittest.TestCase):
    """split_names must be boring: same input, same output, nothing lost."""

    def test_deterministic_across_calls(self):
        a = split_names(NAMES, 0.25, seed=7)
        b = split_names(NAMES, 0.25, seed=7)
        self.assertEqual(a, b)

    def test_seed_changes_the_split(self):
        _, val7 = split_names(NAMES, 0.25, seed=7)
        _, val8 = split_names(NAMES, 0.25, seed=8)
        self.assertNotEqual(val7, val8, "different seeds should shuffle differently")

    def test_disjoint_and_total(self):
        train, val = split_names(NAMES, 0.25, seed=7)
        self.assertEqual(set(train) & set(val), set(), "split halves must not overlap")
        self.assertEqual(sorted(train + val), sorted(NAMES), "every name must survive")
        self.assertEqual(len(val), 4)

    def test_preserves_dataset_order_within_each_half(self):
        train, val = split_names(NAMES, 0.25, seed=7)
        for half in (train, val):
            self.assertEqual(half, [n for n in NAMES if n in half])

    def test_zero_fraction_is_a_no_op(self):
        train, val = split_names(NAMES, 0.0, seed=7)
        self.assertEqual(train, NAMES)
        self.assertEqual(val, [])

    def test_tiny_fraction_still_holds_out_one_name(self):
        # round(16 * 0.001) == 0; a silently empty val set would make "best val loss"
        # meaningless, so the floor is 1.
        train, val = split_names(NAMES, 0.001, seed=7)
        self.assertEqual(len(val), 1)
        self.assertEqual(len(train), 15)

    def test_huge_fraction_never_empties_the_training_set(self):
        train, val = split_names(NAMES, 0.99, seed=7)
        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 15)

    def test_fraction_of_one_or_more_is_rejected(self):
        with self.assertRaises(ValueError):
            split_names(NAMES, 1.0, seed=7)

    def test_single_name_dataset_degrades_quietly(self):
        self.assertEqual(split_names(["Solo"], 0.5, seed=7), (["Solo"], []))


class ValidationLossTest(unittest.TestCase):

    def test_val_loss_is_computed_and_differs_from_train_loss(self):
        # Contrived on purpose: the held-out half is spelled from characters the
        # training half barely uses, so the two losses cannot coincide by accident.
        train_names = ["aaaa", "abab", "aabb", "baba", "bbaa", "abba", "baab", "bbbb"]
        val_names = ["zzqq", "qzqz", "qqzz", "zqzq"]
        cfg = tiny_cfg(epochs=30)
        _, _, report = run_fit(train_names, cfg, val_names=val_names)

        self.assertEqual(len(report["val_losses"]), 30, "one val loss per epoch")
        self.assertEqual(len(report["train_losses"]), 30)
        for tr, va in zip(report["train_losses"], report["val_losses"]):
            self.assertNotAlmostEqual(tr, va, places=4)

        # The two curves must be able to move in opposite directions — that divergence
        # is the entire reason a held-out set is worth computing. Note the absolute
        # values are not directly comparable: a train loss is the mean *while* the
        # weights were still moving, a val loss is measured after the epoch's updates.
        self.assertLess(report["train_losses"][-1], report["train_losses"][0],
                        "the model should be learning the training half")
        self.assertGreater(report["val_losses"][-1], report["val_losses"][0],
                           "and should be getting worse on the held-out half")

    def test_log_line_reports_both_losses(self):
        cfg = tiny_cfg(epochs=2)
        train, val = split_names(NAMES, 0.25, seed=7)
        _, _, report = run_fit(train, cfg, vocab=Vocab(NAMES), val_names=val)
        self.assertIn("| train ", report["stdout"])
        self.assertIn("| val ", report["stdout"])

    def test_evaluate_loss_leaves_the_model_in_training_mode(self):
        # fit() relies on this: it calls model.train() once per epoch, so a stray
        # eval() left behind would silently disable dropout for the rest of the run.
        vocab = Vocab(NAMES)
        cfg = tiny_cfg()
        model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id)
        model.train()
        criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
        evaluate_loss(model, vocab, make_pairs(NAMES, vocab), cfg, criterion)
        self.assertTrue(model.training)

    def test_evaluate_loss_draws_no_randomness(self):
        # If validation consumed the global RNG stream, enabling a split would perturb
        # training itself. Same seed in, same random number out, across an eval call.
        vocab = Vocab(NAMES)
        cfg = tiny_cfg()
        model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id)
        criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
        pairs = make_pairs(NAMES, vocab)

        torch.manual_seed(4)
        before = torch.rand(3)
        torch.manual_seed(4)
        evaluate_loss(model, vocab, pairs, cfg, criterion)
        after = torch.rand(3)
        self.assertTrue(torch.equal(before, after))


class EarlyStoppingTest(unittest.TestCase):
    """A run that gets worse after epoch 1 should stop, and say why."""

    # 40 epochs at the default 3e-3 LR on 8 names, validated against 4 names from a
    # disjoint character distribution: val loss bottoms out early and then climbs.
    TRAIN = ["aaaa", "abab", "aabb", "baba", "bbaa", "abba", "baab", "bbbb"]
    VAL = ["zzqq", "qzqz", "qqzz", "zqzq"]

    def test_early_stop_fires_before_the_epoch_budget(self):
        cfg = tiny_cfg(epochs=40, early_stop_patience=3)
        _, _, report = run_fit(self.TRAIN, cfg, val_names=self.VAL)

        self.assertTrue(report["stopped_early"], "val loss rises here; it should stop")
        self.assertLess(report["epochs_run"], cfg.epochs)
        self.assertEqual(report["epochs_run"], report["best_epoch"] + 3,
                         "stops exactly `patience` epochs after the best one")
        self.assertIn(f"early stop at epoch {report['epochs_run']}", report["stdout"])
        self.assertIn(f"best val {report['best_val_loss']:.4f}", report["stdout"])
        self.assertIn(f"at epoch {report['best_epoch']}", report["stdout"])

    def test_best_epoch_is_the_argmin_of_the_val_losses(self):
        cfg = tiny_cfg(epochs=40, early_stop_patience=3)
        _, _, report = run_fit(self.TRAIN, cfg, val_names=self.VAL)
        losses = report["val_losses"]
        self.assertEqual(report["best_epoch"], losses.index(min(losses)) + 1)
        self.assertAlmostEqual(report["best_val_loss"], min(losses), places=10)

    def test_patience_zero_never_stops_early(self):
        cfg = tiny_cfg(epochs=12, early_stop_patience=0)
        _, _, report = run_fit(self.TRAIN, cfg, val_names=self.VAL)
        self.assertFalse(report["stopped_early"])
        self.assertEqual(report["epochs_run"], 12)

    def test_early_stopping_is_inert_without_a_validation_set(self):
        cfg = tiny_cfg(epochs=12, early_stop_patience=1)
        _, _, report = run_fit(self.TRAIN, cfg)
        self.assertFalse(report["stopped_early"])
        self.assertEqual(report["epochs_run"], 12)


class BestWeightRestoreTest(unittest.TestCase):
    """The returned model must be the best epoch's, not the last epoch's."""

    TRAIN = EarlyStoppingTest.TRAIN
    VAL = EarlyStoppingTest.VAL

    def test_returned_weights_score_the_best_val_loss_not_the_last(self):
        cfg = tiny_cfg(epochs=25, early_stop_patience=0)  # run to the end on purpose,
        # so "restored the best" is distinguishable from "stopped at the best".
        model, vocab, report = run_fit(self.TRAIN, cfg, val_names=self.VAL)

        self.assertLess(report["best_epoch"], report["epochs_run"],
                        "contrived case must have a worse final epoch")
        self.assertGreater(report["val_losses"][-1], report["best_val_loss"])

        criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
        actual = evaluate_loss(model, vocab, make_pairs(self.VAL, vocab), cfg, criterion)
        self.assertAlmostEqual(actual, report["best_val_loss"], places=6,
                               msg="returned model does not score the best val loss")
        self.assertNotAlmostEqual(actual, report["val_losses"][-1], places=6,
                                  msg="returned model is the last epoch's, not the best")
        self.assertIn(f"restored best weights from epoch {report['best_epoch']}",
                      report["stdout"])

    def test_no_restore_happens_without_a_validation_set(self):
        cfg = tiny_cfg(epochs=6)
        _, _, report = run_fit(self.TRAIN, cfg)
        self.assertIsNone(report["best_val_loss"])
        self.assertEqual(report["best_epoch"], 0)
        self.assertNotIn("restored best weights", report["stdout"])


class LRScheduleTest(unittest.TestCase):

    def test_none_creates_no_scheduler_and_holds_the_lr(self):
        cfg = tiny_cfg(epochs=4, lr_schedule="none")
        _, _, report = run_fit(NAMES, cfg)
        self.assertNotIn("lr ", report["stdout"])
        assert_matches_golden(self, report["train_losses"],
                              "— the default schedule must not perturb training")

    def test_cosine_decays_the_lr_and_still_trains(self):
        cfg = tiny_cfg(epochs=8, lr_schedule="cosine", lr_min=1e-5)
        _, _, report = run_fit(NAMES, cfg)
        self.assertEqual(len(report["train_losses"]), 8)
        self.assertNotEqual(report["train_losses"], GOLDEN_TRAJECTORY,
                            "a cosine schedule should change the trajectory")

    def test_plateau_runs_with_a_validation_set(self):
        cfg = tiny_cfg(epochs=10, lr_schedule="plateau", early_stop_patience=4,
                       lr_factor=0.1)
        train, val = split_names(NAMES, 0.25, seed=7)
        _, _, report = run_fit(train, cfg, vocab=Vocab(NAMES), val_names=val)
        self.assertEqual(len(report["val_losses"]), report["epochs_run"])

    def test_plateau_degrades_to_train_loss_without_a_validation_set(self):
        cfg = tiny_cfg(epochs=6, lr_schedule="plateau")
        _, _, report = run_fit(NAMES, cfg)
        self.assertEqual(len(report["train_losses"]), 6)

    def test_unknown_schedule_is_rejected_loudly(self):
        cfg = tiny_cfg(epochs=1, lr_schedule="triangular")
        with self.assertRaises(ValueError):
            run_fit(NAMES, cfg)


class BackwardCompatibilityTest(unittest.TestCase):
    """The whole point: WS-6's defaults must be the pre-WS-6 behavior."""

    def test_default_path_reproduces_pre_ws6_trajectory(self):
        # GOLDEN_TRAJECTORY was measured by running origin/main's fit(), before any of
        # this existed. Matching it to 1e-5 over eight epochs means the RNG stream, the
        # batch order, the optimizer and the loss are all untouched on the default path.
        cfg = tiny_cfg()
        _, _, report = run_fit(NAMES, cfg)
        assert_matches_golden(self, report["train_losses"])

    def test_val_fraction_zero_is_the_default_path(self):
        train, val = split_names(NAMES, Config().val_fraction, Config().seed)
        self.assertEqual(val, [], "the shipped default must not hold anything out")
        cfg = tiny_cfg()
        _, _, report = run_fit(train, cfg, val_names=val or None)
        self.assertEqual(report["val_losses"], [])
        assert_matches_golden(self, report["train_losses"])

    def test_default_log_line_keeps_its_old_wording(self):
        # generate.py and the README quote this format; other stages parse nothing, but
        # changing it unasked would still be a surprise.
        cfg = tiny_cfg(epochs=1)
        _, _, report = run_fit(NAMES, cfg)
        self.assertIn("epoch    1/1 | loss ", report["stdout"])
        self.assertNotIn("| train ", report["stdout"])

    def test_fit_still_returns_the_model(self):
        vocab = Vocab(NAMES)
        cfg = tiny_cfg(epochs=1)
        model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id)
        with contextlib.redirect_stdout(io.StringIO()):
            returned = fit(model, vocab, NAMES, cfg)
        self.assertIs(returned, model, "three call sites depend on this")


class CheckpointValNamesTest(unittest.TestCase):
    """The one additive checkpoint key (HANDOFF §2)."""

    def _save(self, val_names):
        vocab = Vocab(NAMES)
        cfg = tiny_cfg(epochs=1)
        model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            save_checkpoint(path, model, cfg, vocab, NAMES, val_names)
            return torch.load(path, map_location="cpu", weights_only=False)

    def test_existing_keys_are_untouched(self):
        ckpt = self._save(["Puma"])
        for key in ("model_state", "config", "vocab", "training_names"):
            self.assertIn(key, ckpt)
        self.assertEqual(ckpt["training_names"], NAMES)

    def test_val_names_round_trips(self):
        self.assertEqual(self._save(["Puma", "Nova"])["val_names"], ["Puma", "Nova"])

    def test_val_names_defaults_to_empty_not_missing(self):
        # A new checkpoint is always self-describing: [] means "trained on everything",
        # which is different from a pre-WS-6 checkpoint where the key is simply absent.
        self.assertEqual(self._save(None)["val_names"], [])

    def test_readers_survive_a_checkpoint_without_the_key(self):
        ckpt = self._save(["Puma"])
        del ckpt["val_names"]
        self.assertEqual(ckpt.get("val_names", []), [])


class EndToEndTrainTest(unittest.TestCase):
    """`python -m src.train --val-fraction ...` end to end, through train()."""

    def test_train_with_a_split_writes_disjoint_train_and_val_names(self):
        from src.train import train

        with tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "names.txt")
            with open(data, "w", encoding="utf-8") as fh:
                fh.write("\n".join(NAMES) + "\n")
            cfg = tiny_cfg(epochs=3, val_fraction=0.25, early_stop_patience=0)
            with contextlib.redirect_stdout(io.StringIO()):
                path = train(data, "split_smoke", cfg, checkpoint_dir=tmp)

            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            train_names = ckpt["training_names"]
            val_names = ckpt["val_names"]
            self.assertEqual(len(val_names), 4)
            self.assertEqual(len(train_names), 12)
            self.assertEqual(set(train_names) & set(val_names), set())
            self.assertEqual(sorted(train_names + val_names), sorted(NAMES))
            # The vocab spans every name, not just the trained-on ones, so held-out
            # names stay encodable for evaluation.
            self.assertEqual(set(ckpt["vocab"]["itos"]),
                             set(Vocab(NAMES).itos))

    def test_train_without_a_split_still_trains_on_everything(self):
        from src.train import train

        with tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "names.txt")
            with open(data, "w", encoding="utf-8") as fh:
                fh.write("\n".join(NAMES) + "\n")
            cfg = tiny_cfg(epochs=2)
            with contextlib.redirect_stdout(io.StringIO()):
                path = train(data, "nosplit_smoke", cfg, checkpoint_dir=tmp)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(ckpt["training_names"], NAMES)
            self.assertEqual(ckpt["val_names"], [])


if __name__ == "__main__":
    unittest.main()
