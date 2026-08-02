"""WS-6/WS-10 tests: held-out split, validation loss, early stopping, best-weight
restore, seeded initialization, weight decay / label smoothing / LR warmup, and the
size-derived epoch budget.

The load-bearing test in here is :meth:`BackwardCompatibilityTest.test_default_path_
reproduces_pre_ws6_trajectory`. Everything else checks that the new machinery works;
that one checks that the *old* behavior survived it, against a loss trajectory captured
by running ``origin/main``'s ``fit()`` before this change existed. Wave-2's rule is
"every new option defaults to today's behavior — prove it, don't assert it", and a
golden trajectory is the proof.

WS-10 note on that pin. WS-10 flipped exactly one default: ``Config.seed_init``, which
seeds the RNG *before* the model is constructed. That is a change in :func:`train` /
:func:`pretrain`, not in :func:`fit` — the loop itself is untouched — so
``GOLDEN_TRAJECTORY`` still holds and is still the right pin. :func:`run_fit` seeds and
then builds the model, which is precisely what ``seed_init=True`` now does for real
callers, so the golden trajectory *is* the post-WS-10 default trajectory of the loop.
:class:`SeededInitTest` covers the other half — that the seeding now actually happens at
the entry points — so together they prove "no flags = the same loop as before, just
seeded earlier".

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import contextlib
import io
import math
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
from src.train import (
    AUTO_EPOCH_MAX,
    AUTO_EPOCH_MIN,
    apply_auto_epochs,
    derive_epochs,
    derive_patience,
    evaluate_loss,
    fit,
    save_checkpoint,
    seed_for_init,
)


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


def train_to_state_dict(tmp, cfg, name):
    """Run the real :func:`train` entry point end to end, return its saved weights."""
    from src.train import train

    data = os.path.join(tmp, "names.txt")
    if not os.path.exists(data):
        with open(data, "w", encoding="utf-8") as fh:
            fh.write("\n".join(NAMES) + "\n")
    with contextlib.redirect_stdout(io.StringIO()):
        path = train(data, name, cfg, checkpoint_dir=tmp)
    return torch.load(path, map_location="cpu", weights_only=False)["model_state"]


def states_equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


class SeededInitTest(unittest.TestCase):
    """WS-10 Task 1: the initialization is now seeded too.

    Before this, ``fit()`` seeded the RNG as its first act — but ``train()`` had already
    called ``CharRNN(...)``, so the *initial weights* came from whatever state the
    process started in. Three identical ``--val-fraction 0.15`` runs on
    ``car_manufacturers`` landed on best epoch 12, 16 and 19. These tests pin the fix
    from both sides: it is on by default, and turning it off restores the old behavior.
    """

    def test_shipped_default_is_on(self):
        # The one WS-10 default that is deliberately NOT the pre-wave behavior.
        self.assertTrue(Config().seed_init)

    def test_train_gives_identical_weights_from_different_process_states(self):
        # Two runs that begin from different global RNG states — the in-process stand-in
        # for two separate `python -m src.train` invocations — must now agree exactly.
        cfg = tiny_cfg(epochs=2)
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(11111)
            first = train_to_state_dict(tmp, cfg, "seeded_a")
            torch.manual_seed(22222)
            second = train_to_state_dict(tmp, cfg, "seeded_b")
        self.assertTrue(states_equal(first, second),
                        "seed_init=True must make the whole run reproducible")

    def test_disabling_it_restores_the_old_unreproducible_initialization(self):
        # The negative control. Without this test "reproducible" could just mean the
        # test harness happened to be deterministic.
        cfg = tiny_cfg(epochs=2, seed_init=False)
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(11111)
            first = train_to_state_dict(tmp, cfg, "unseeded_a")
            torch.manual_seed(22222)
            second = train_to_state_dict(tmp, cfg, "unseeded_b")
        self.assertFalse(states_equal(first, second),
                         "--no-seed-init must reproduce the pre-WS-10 behavior")

    def test_seed_for_init_is_inert_when_off(self):
        torch.manual_seed(3)
        before = torch.rand(3)
        torch.manual_seed(3)
        seed_for_init(Config(seed_init=False))
        self.assertTrue(torch.equal(before, torch.rand(3)),
                        "when off it must not touch the RNG at all, not even differently")

    def test_seed_for_init_resets_the_stream_when_on(self):
        cfg = Config(seed_init=True, seed=99)
        seed_for_init(cfg)
        first = torch.rand(3)
        torch.manual_seed(7)  # scramble it
        torch.rand(50)
        seed_for_init(cfg)
        self.assertTrue(torch.equal(first, torch.rand(3)))

    def test_the_loop_itself_is_unchanged_by_the_flip(self):
        # fit() must not read seed_init at all: the flip moves *when* the seed is set,
        # it does not alter the training loop. Same trajectory either way, and it is
        # still the pre-WS-6 golden one.
        for flag in (True, False):
            with self.subTest(seed_init=flag):
                _, _, report = run_fit(NAMES, tiny_cfg(seed_init=flag))
                assert_matches_golden(self, report["train_losses"])


class WeightDecayTest(unittest.TestCase):
    """WS-10 Task 2: `weight_decay > 0` means AdamW, not Adam's L2."""

    def _optimizers_used(self, cfg):
        import unittest.mock as mock

        with mock.patch.object(torch.optim, "Adam", wraps=torch.optim.Adam) as adam, \
                mock.patch.object(torch.optim, "AdamW", wraps=torch.optim.AdamW) as adamw:
            run_fit(NAMES, cfg)
        return adam.call_count, adamw.call_count, adamw.call_args

    def test_default_still_builds_plain_adam(self):
        adam, adamw, _ = self._optimizers_used(tiny_cfg(epochs=1))
        self.assertEqual((adam, adamw), (1, 0))

    def test_nonzero_weight_decay_builds_adamw_and_passes_the_value(self):
        # Adam's own weight_decay is L2-added-to-the-gradient, which Adam's adaptive
        # scaling then distorts; AdamW's is decoupled. They are different algorithms and
        # the flag promises the decoupled one.
        adam, adamw, call = self._optimizers_used(tiny_cfg(epochs=1, weight_decay=0.05))
        self.assertEqual((adam, adamw), (0, 1))
        self.assertAlmostEqual(call.kwargs["weight_decay"], 0.05)

    def test_zero_weight_decay_reproduces_the_golden_trajectory(self):
        _, _, report = run_fit(NAMES, tiny_cfg(weight_decay=0.0))
        assert_matches_golden(self, report["train_losses"])

    def test_weight_decay_actually_shrinks_the_weights(self):
        # A knob that reports a different loss but leaves the weight norm alone would be
        # decaying nothing; this is the check that it is wired to the optimizer.
        plain, _, _ = run_fit(NAMES, tiny_cfg(epochs=12))
        decayed, _, _ = run_fit(NAMES, tiny_cfg(epochs=12, weight_decay=0.5))
        norm = lambda m: sum(float(p.detach().pow(2).sum()) for p in m.parameters())
        self.assertLess(norm(decayed), norm(plain))


class LabelSmoothingTest(unittest.TestCase):
    """WS-10 Task 2: smoothing applies to training only, so val stays comparable."""

    TRAIN = ["aaaa", "abab", "aabb", "baba", "bbaa", "abba", "baab", "bbbb"]
    VAL = ["abab", "baba", "aabb", "bbaa"]

    def test_zero_smoothing_reproduces_the_golden_trajectory(self):
        _, _, report = run_fit(NAMES, tiny_cfg(label_smoothing=0.0))
        assert_matches_golden(self, report["train_losses"])

    def test_reported_val_loss_is_the_unsmoothed_one(self):
        # The load-bearing test of the decision documented in fit()'s docstring: if the
        # validation criterion were smoothed too, "best val loss 2.94" would mean
        # something different in every run and could not be compared with any number
        # already recorded in STATUS.md or the README.
        cfg = tiny_cfg(epochs=6, label_smoothing=0.3)
        model, vocab, report = run_fit(self.TRAIN, cfg, val_names=self.VAL)
        pairs = make_pairs(self.VAL, vocab)

        unsmoothed = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
        smoothed = nn.CrossEntropyLoss(ignore_index=vocab.pad_id, label_smoothing=0.3)
        self.assertAlmostEqual(
            evaluate_loss(model, vocab, pairs, cfg, unsmoothed),
            report["best_val_loss"], places=6,
            msg="the reported val loss must be on the unsmoothed scale")
        self.assertNotAlmostEqual(
            evaluate_loss(model, vocab, pairs, cfg, smoothed),
            report["best_val_loss"], places=4,
            msg="...and must therefore differ from the smoothed one")

    def test_val_losses_are_comparable_across_smoothing_settings(self):
        # Same data, same seed, two smoothing settings: the val numbers must live on one
        # scale. Smoothing raises the *floor* of a smoothed loss by ~0.5 nat at 0.3 on
        # this vocab, which would swamp the real differences being measured.
        plain = run_fit(self.TRAIN, tiny_cfg(epochs=4), val_names=self.VAL)[2]
        smooth = run_fit(self.TRAIN, tiny_cfg(epochs=4, label_smoothing=0.3),
                         val_names=self.VAL)[2]
        for a, b in zip(plain["val_losses"], smooth["val_losses"]):
            self.assertLess(abs(a - b), 0.5,
                            "val losses drifted onto different scales")

    def test_training_loss_is_on_the_smoothed_scale(self):
        # The flip side, and the reason fit()'s docstring warns about it: the *train*
        # number does move, so a smoothed run's train loss is not comparable to an
        # unsmoothed one's. Documented, not hidden.
        plain = run_fit(self.TRAIN, tiny_cfg(epochs=4), val_names=self.VAL)[2]
        smooth = run_fit(self.TRAIN, tiny_cfg(epochs=4, label_smoothing=0.3),
                         val_names=self.VAL)[2]
        self.assertNotEqual(plain["train_losses"], smooth["train_losses"],
                            "smoothing must reach the training criterion")

    def test_smoothing_raises_the_floor_which_is_why_val_stays_unsmoothed(self):
        # The concrete reason the decision matters. Once a model is actually confident,
        # a smoothed criterion scores it *worse* than an unsmoothed one on the very same
        # data — the smoothed loss cannot reach zero. Score validation that way and a
        # "best val loss" would be a different quantity for every smoothing setting.
        model, vocab, _ = run_fit(self.TRAIN, tiny_cfg(epochs=60))
        pairs = make_pairs(self.TRAIN, vocab)
        cfg = tiny_cfg()
        plain = evaluate_loss(model, vocab, pairs, cfg,
                              nn.CrossEntropyLoss(ignore_index=vocab.pad_id))
        smoothed = evaluate_loss(
            model, vocab, pairs, cfg,
            nn.CrossEntropyLoss(ignore_index=vocab.pad_id, label_smoothing=0.3))
        self.assertGreater(smoothed, plain,
                           "a confident model scores worse under a smoothed criterion")


class WarmupTest(unittest.TestCase):
    """WS-10 Task 2: linear LR warmup, and how it composes with the schedules."""

    def test_zero_warmup_reproduces_the_golden_trajectory(self):
        _, _, report = run_fit(NAMES, tiny_cfg(warmup_epochs=0))
        assert_matches_golden(self, report["train_losses"])

    def test_lr_ramps_linearly_and_lands_exactly_on_the_base_lr(self):
        cfg = tiny_cfg(epochs=8, warmup_epochs=4)
        _, _, report = run_fit(NAMES, cfg)
        base = cfg.learning_rate
        self.assertEqual(len(report["lrs"]), 8)
        for epoch in range(1, 5):
            self.assertAlmostEqual(report["lrs"][epoch - 1], base * epoch / 4, places=12)
        for lr in report["lrs"][4:]:
            self.assertAlmostEqual(lr, base, places=12,
                                   msg="after the ramp the LR must be exactly the base")

    def test_warmup_announces_itself(self):
        _, _, report = run_fit(NAMES, tiny_cfg(epochs=3, warmup_epochs=2))
        self.assertIn("warmup:", report["stdout"])

    def test_warmup_holds_cosine_back_then_hands_over(self):
        # The composition that would otherwise fight: cosine decays from epoch 1, warmup
        # ramps up over epoch 1..N. Warmup wins first, then cosine anneals over what is
        # left, so the LR still reaches lr_min by the last epoch instead of stopping
        # part-way down.
        cfg = tiny_cfg(epochs=12, warmup_epochs=4, lr_schedule="cosine", lr_min=1e-6)
        _, _, report = run_fit(NAMES, cfg)
        lrs = report["lrs"]
        self.assertLess(lrs[0], lrs[3], "the ramp must go up")
        self.assertAlmostEqual(lrs[4], cfg.learning_rate, places=12)
        self.assertTrue(all(b <= a for a, b in zip(lrs[4:], lrs[5:])),
                        "after the hand-off cosine must only decay")
        self.assertLess(lrs[-1], cfg.learning_rate / 10,
                        "cosine should have most of the way down by the last epoch")

    def test_cosine_alone_is_unchanged_by_the_warmup_plumbing(self):
        # T_max is now `epochs - warmup_epochs`; with warmup off that is `epochs`, so a
        # plain cosine run must be exactly what WS-6 shipped.
        cfg = tiny_cfg(epochs=8, lr_schedule="cosine", lr_min=1e-5)
        _, _, report = run_fit(NAMES, cfg)
        expected = [cfg.learning_rate]
        for step in range(1, 8):
            expected.append(
                cfg.lr_min + (cfg.learning_rate - cfg.lr_min)
                * (1 + math.cos(math.pi * step / 8)) / 2)
        for got, want in zip(report["lrs"], expected):
            self.assertAlmostEqual(got, want, places=10)

    def test_plateau_cannot_cut_the_lr_during_the_ramp(self):
        # ReduceLROnPlateau counts stalled epochs; if it were stepped during warmup it
        # would spend its patience on epochs whose loss is high *because* the LR is low.
        cfg = tiny_cfg(epochs=10, warmup_epochs=5, lr_schedule="plateau",
                       early_stop_patience=2, lr_factor=0.1)
        train, val = split_names(NAMES, 0.25, seed=7)
        _, _, report = run_fit(train, cfg, vocab=Vocab(NAMES), val_names=val)
        base = cfg.learning_rate
        for epoch, lr in enumerate(report["lrs"][:5], start=1):
            self.assertAlmostEqual(lr, base * epoch / 5, places=12)


class AutoEpochsTest(unittest.TestCase):
    """WS-10 Task 3: the epoch budget the user no longer has to look up."""

    def test_small_datasets_land_on_the_readme_floor(self):
        # The README said 60 for anything under ~200 names; that is the floor here, and
        # a 159-name set lands just above it (70) — still 5x its measured bottom of 13.
        self.assertEqual(derive_epochs(120), AUTO_EPOCH_MIN)
        self.assertEqual(derive_epochs(63), AUTO_EPOCH_MIN)
        self.assertEqual(derive_epochs(159), 70)

    def test_clears_every_measured_val_bottom_with_headroom(self):
        # The measured epoch at which each dataset's held-out loss actually bottomed
        # (WS-10, 15% holdout, patience 25). The derived ceiling must sit well clear of
        # it — that is the entire safety property the budget has to have.
        measured = {159: 13, 309: 10, 435: 26, 590: 9, 863: 13,
                    1218: 10, 1691: 8, 2223: 10, 8631: 7}
        for names, bottom in measured.items():
            self.assertGreaterEqual(derive_epochs(names), 4 * bottom,
                                    f"budget for {names} names must clear epoch {bottom}")

    def test_monotone_across_every_shipped_dataset_size(self):
        sizes = [63, 309, 355, 435, 590, 863, 1218, 1691, 2223, 8631]
        derived = [derive_epochs(n) for n in sizes]
        self.assertEqual(derived, sorted(derived),
                         "a bigger dataset must never get a smaller budget")

    def test_clamped_at_both_ends(self):
        self.assertEqual(derive_epochs(1), AUTO_EPOCH_MIN)
        self.assertEqual(derive_epochs(0), AUTO_EPOCH_MIN)
        self.assertEqual(derive_epochs(8631), AUTO_EPOCH_MAX)
        self.assertEqual(derive_epochs(10 ** 6), AUTO_EPOCH_MAX)

    def test_never_exceeds_todays_default_budget(self):
        for n in (63, 590, 2223, 8631, 50000):
            self.assertLessEqual(derive_epochs(n), Config().epochs)

    def test_patience_keeps_the_readme_ratio(self):
        self.assertEqual(derive_patience(60), 10)
        self.assertEqual(derive_patience(120), 20)
        self.assertEqual(derive_patience(300), 30)  # clamped

    def test_apply_auto_epochs_sets_the_budget_and_says_why(self):
        cfg = Config(val_fraction=0.15)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            apply_auto_epochs(cfg, 590)
        self.assertEqual(cfg.epochs, derive_epochs(590))
        self.assertEqual(cfg.early_stop_patience, derive_patience(cfg.epochs))
        text = out.getvalue()
        self.assertIn("auto-epochs:", text)
        self.assertIn("590 names", text)  # the reason, not just the number

    def test_an_explicit_patience_is_never_overwritten(self):
        cfg = Config(val_fraction=0.15, early_stop_patience=7)
        with contextlib.redirect_stdout(io.StringIO()):
            apply_auto_epochs(cfg, 590)
        self.assertEqual(cfg.early_stop_patience, 7)

    def test_warns_when_there_is_no_split_to_stop_on(self):
        # The budget is a ceiling, not a promise — but only if --patience can fire.
        cfg = Config(val_fraction=0.0)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            apply_auto_epochs(cfg, 590)
        self.assertIn("--val-fraction", out.getvalue())

    def test_end_to_end_auto_epochs_overrides_the_config_budget(self):
        from src.train import train

        with tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "names.txt")
            with open(data, "w", encoding="utf-8") as fh:
                fh.write("\n".join(NAMES) + "\n")
            cfg = tiny_cfg(epochs=999)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                path = train(data, "auto_smoke", cfg, checkpoint_dir=tmp,
                             auto_epochs=True)
            self.assertIn("auto-epochs:", out.getvalue())
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            # The checkpoint records the budget that actually ran, not the 999.
            self.assertEqual(ckpt["config"]["epochs"], derive_epochs(len(NAMES)))


class RegimenArgsTest(unittest.TestCase):
    """The shared `--arch` / regimen flags, wired identically on all three entry points."""

    def _parse(self, argv, include_seed_init=True):
        import argparse

        from src.train import add_regimen_args, apply_regimen_args

        parser = argparse.ArgumentParser()
        add_regimen_args(parser, include_seed_init=include_seed_init)
        cfg = Config()
        apply_regimen_args(cfg, parser.parse_args(argv))
        return cfg

    def test_no_flags_leaves_every_config_default_alone(self):
        cfg, default = self._parse([]), Config()
        for field in ("arch", "seed_init", "weight_decay", "label_smoothing",
                      "warmup_epochs"):
            self.assertEqual(getattr(cfg, field), getattr(default, field))

    def test_arch_only_sets_the_config_field(self):
        # WS-9 owns src/model.py; this lane owns the CLI and nothing else.
        self.assertEqual(self._parse(["--arch", "gru"]).arch, "gru")
        self.assertEqual(self._parse(["--arch", "transformer"]).arch, "transformer")
        self.assertEqual(self._parse([]).arch, "lstm")

    def test_regimen_flags_reach_the_config(self):
        cfg = self._parse(["--weight-decay", "0.01", "--label-smoothing", "0.1",
                           "--warmup-epochs", "5"])
        self.assertAlmostEqual(cfg.weight_decay, 0.01)
        self.assertAlmostEqual(cfg.label_smoothing, 0.1)
        self.assertEqual(cfg.warmup_epochs, 5)

    def test_no_seed_init_turns_the_new_default_off(self):
        self.assertFalse(self._parse(["--no-seed-init"]).seed_init)
        self.assertTrue(self._parse(["--seed-init"]).seed_init)

    def test_finetune_does_not_offer_a_switch_wired_to_nothing(self):
        # Fine-tuning loads its weights, so there is no random init to seed.
        import argparse

        from src.train import add_regimen_args

        parser = argparse.ArgumentParser()
        add_regimen_args(parser, include_seed_init=False)
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                parser.parse_args(["--no-seed-init"])


class FinetuneArchGuardTest(unittest.TestCase):
    """`--arch` on a fine-tune can only confirm the base's architecture."""

    def test_mismatched_arch_is_rejected_with_a_useful_message(self):
        from src.finetune import finetune

        with tempfile.TemporaryDirectory() as tmp:
            vocab = Vocab(NAMES)
            cfg = tiny_cfg(epochs=1)
            model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id)
            base = save_checkpoint(os.path.join(tmp, "base.pt"), model, cfg, vocab, NAMES)
            data = os.path.join(tmp, "names.txt")
            with open(data, "w", encoding="utf-8") as fh:
                fh.write("\n".join(NAMES) + "\n")

            with self.assertRaises(ValueError) as ctx:
                finetune(base, data, "ft_smoke", epochs=1, checkpoint_dir=tmp,
                         arch="transformer")
            self.assertIn("architecture", str(ctx.exception))

    def test_matching_arch_is_accepted(self):
        from src.finetune import finetune

        with tempfile.TemporaryDirectory() as tmp:
            vocab = Vocab(NAMES)
            cfg = tiny_cfg(epochs=1)
            model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id)
            base = save_checkpoint(os.path.join(tmp, "base.pt"), model, cfg, vocab, NAMES)
            data = os.path.join(tmp, "names.txt")
            with open(data, "w", encoding="utf-8") as fh:
                fh.write("\n".join(NAMES) + "\n")
            with contextlib.redirect_stdout(io.StringIO()):
                out = finetune(base, data, "ft_smoke", epochs=1, checkpoint_dir=tmp,
                               arch="lstm")
            self.assertTrue(os.path.exists(out))


class RegimenBackwardCompatibilityTest(unittest.TestCase):
    """All three WS-10 regimen knobs at their shipped defaults = the pre-WS-6 loop."""

    def test_shipped_regimen_defaults_are_all_off(self):
        default = Config()
        self.assertEqual(default.weight_decay, 0.0)
        self.assertEqual(default.label_smoothing, 0.0)
        self.assertEqual(default.warmup_epochs, 0)

    def test_config_defaults_reproduce_the_golden_trajectory(self):
        default = Config()
        cfg = tiny_cfg(weight_decay=default.weight_decay,
                       label_smoothing=default.label_smoothing,
                       warmup_epochs=default.warmup_epochs)
        _, _, report = run_fit(NAMES, cfg)
        assert_matches_golden(self, report["train_losses"])
        self.assertEqual(report["lrs"], [cfg.learning_rate] * len(report["lrs"]),
                         "the default path must hold the LR constant")


if __name__ == "__main__":
    unittest.main()
