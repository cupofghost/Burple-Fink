"""Tests for the vocabulary and the shared-vocab machinery that fine-tuning relies on.

The shared vocabulary is a *contract* (HANDOFF §6): if a fine-tune's vocab ever drifts
from the base model's, weights stop loading. These tests pin the invariants that keep
that contract honest — special-token positions, encode/decode roundtripping, and the
superset property that lets one base model be fine-tuned on any dataset.
"""

import json
import os
import tempfile
import unittest

from src.config import PAD_TOKEN, START_TOKEN, END_TOKEN
from src.data import (
    Vocab,
    build_shared_vocab,
    filter_to_vocab,
    list_dataset_files,
    load_all_names,
    load_names,
    load_shared_vocab,
    make_pairs,
    save_shared_vocab,
)

DATA_DIR = "data"


class VocabTests(unittest.TestCase):
    def test_special_tokens_at_fixed_indices(self):
        v = Vocab(["abc"])
        # PAD must be 0 so it doubles as the embedding's padding_idx.
        self.assertEqual(v.itos[0], PAD_TOKEN)
        self.assertEqual(v.itos[1], START_TOKEN)
        self.assertEqual(v.itos[2], END_TOKEN)
        self.assertEqual((v.pad_id, v.start_id, v.end_id), (0, 1, 2))

    def test_encode_wraps_and_decode_strips(self):
        v = Vocab(["Go"])
        ids = v.encode("Go")
        self.assertEqual(ids[0], v.start_id)
        self.assertEqual(ids[-1], v.end_id)
        # decode drops the special tokens, recovering the original name.
        self.assertEqual(v.decode(ids), "Go")

    def test_dict_roundtrip_preserves_mapping(self):
        v = Vocab(["Toyota", "Honda"])
        v2 = Vocab.from_dict(v.to_dict())
        self.assertEqual(v.itos, v2.itos)
        self.assertEqual(v.stoi, v2.stoi)

    def test_make_pairs_is_next_char_shift(self):
        v = Vocab(["Go"])
        (inp, tgt), = make_pairs(["Go"], v)
        # target is input shifted left by one: predict the following character.
        self.assertEqual(inp, v.encode("Go")[:-1])
        self.assertEqual(tgt, v.encode("Go")[1:])
        self.assertEqual(len(inp), len(tgt))


class SharedVocabTests(unittest.TestCase):
    def setUp(self):
        self.paths = list_dataset_files(DATA_DIR)
        self.assertTrue(self.paths, "expected bundled datasets under data/")

    def test_shared_vocab_is_superset_of_each_dataset(self):
        shared = build_shared_vocab(self.paths)
        shared_chars = set(shared.itos)
        for path in self.paths:
            per_dataset = set(Vocab(load_names(path)).itos)
            self.assertTrue(
                per_dataset.issubset(shared_chars),
                f"{path} has characters missing from the shared vocab",
            )
        # specials still pinned to the front
        self.assertEqual(shared.itos[:3], [PAD_TOKEN, START_TOKEN, END_TOKEN])

    def test_shared_vocab_is_deterministic(self):
        a = build_shared_vocab(self.paths).itos
        b = build_shared_vocab(list(reversed(self.paths))).itos
        self.assertEqual(a, b, "shared vocab must not depend on file ordering")

    def test_save_load_roundtrip(self):
        shared = build_shared_vocab(self.paths)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "shared_vocab.json")
            save_shared_vocab(shared, path)
            # control-character tokens must survive a JSON round trip
            with open(path, encoding="utf-8") as fh:
                self.assertIn("itos", json.load(fh))
            reloaded = load_shared_vocab(path)
            self.assertEqual(shared.itos, reloaded.itos)

    def test_filter_to_vocab_splits_representable_and_dropped(self):
        v = Vocab(["Toyota"])  # knows T,o,y,a plus specials
        kept, dropped = filter_to_vocab(["Toya", "naïve"], v)
        self.assertEqual(kept, ["Toya"])
        self.assertEqual(dropped, ["naïve"])  # 'ï','n','v','e' not all in vocab

    def test_load_all_names_dedupes_across_files(self):
        names = load_all_names(self.paths)
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
