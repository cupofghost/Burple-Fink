"""Tests for scripts/check_data.py (WS-13).

Stdlib only, no torch. Every test builds its own temp `data/` directory: the real one
is being written to by four parallel data lanes while this suite runs, so asserting
against it would be flaky by construction. The one exception is
`RealDataDirTests`, which asserts only the invariant that must hold no matter what the
lanes land — that the checker itself runs and that nothing is a hard error.
"""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_data import (
    KNOWN_NONCONFORMING,
    REQUIRED_META_KEYS,
    RULE_BLANK,
    RULE_CHARSET,
    RULE_DUPLICATE,
    RULE_SHORT,
    RULE_SIDECAR,
    DatasetReport,
    check_lines,
    check_sidecar,
    read_lines,
    collect,
    format_table,
    run,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _meta(name, count, **overrides):
    body = {
        "name": name,
        "label": f"{name} label",
        "domain": "Test",
        "count": count,
        "provenance": "invented in a test",
        "verified": False,
        "added": "2026-08-01",
        "signature": "Test | Model | low",
    }
    body.update(overrides)
    return body


class _TempData(unittest.TestCase):
    def setUp(self):
        self.data = Path(tempfile.mkdtemp()) / "data"
        self.data.mkdir()

    def write_dataset(self, name, lines, meta=None, suffix=".txt"):
        path = self.data / f"{name}{suffix}"
        path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        if meta is not None:
            (self.data / f"{name}.meta.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
        return path

    def rules(self, findings):
        return sorted(f.rule for f in findings)


class CharacterSetTests(unittest.TestCase):
    def test_plain_names_pass(self):
        self.assertEqual(check_lines("x.txt", ["Alpha", "Beta Gamma", "Jean-Luc"]), [])

    def test_digits_are_allowed(self):
        """Load-bearing in real names: ATR 42, RAV4, MR2, 90 Minute."""
        lines = ["ATR 42", "RAV4", "MR2", "90 Minute", "ATR 42-300", "Mazda3"]
        self.assertEqual(check_lines("x.txt", lines), [])

    def test_apostrophe_is_an_error(self):
        findings = check_lines("x.txt", ["Man O' War"])
        self.assertEqual(self.rules_of(findings), [RULE_CHARSET])
        self.assertEqual(findings[0].level, "error")
        self.assertIn("line 1", findings[0].message)
        self.assertIn("U+0027", findings[0].message)

    def test_period_ampersand_slash_are_errors(self):
        findings = check_lines("x.txt", ["Dr. Fager", "Bells & Whistles", "GOES R"])
        self.assertEqual(len(findings), 2, findings)
        self.assertIn("U+002E", findings[0].message)
        self.assertIn("U+0026", findings[1].message)

    def test_accented_latin_is_an_error(self):
        findings = check_lines("x.txt", ["Afonso Claudio", "Araucania", "Araxá"])
        self.assertEqual(len(findings), 1, findings)
        self.assertIn("line 3", findings[0].message)
        self.assertIn("U+00E1", findings[0].message)

    def test_invisible_soft_hyphen_is_caught_and_named(self):
        """The nastiest real case in world_cities.txt: it looks like a clean name."""
        findings = check_lines("x.txt", ["Beer­sheva"])
        self.assertEqual(len(findings), 1, findings)
        self.assertIn("U+00AD", findings[0].message)

    def test_leading_and_trailing_space_or_hyphen_rejected(self):
        findings = check_lines("x.txt", [" Alpha", "Beta ", "-Gamma", "Delta-"])
        self.assertEqual(len(findings), 4, findings)
        self.assertTrue(all(f.rule == RULE_CHARSET for f in findings))

    def test_message_notes_when_char_is_also_missing_from_vocab(self):
        findings = check_lines("x.txt", ["Cafeé"], vocab=set("abcdefgh"))
        self.assertIn("absent from the current shared_vocab.json", findings[0].message)

    def rules_of(self, findings):
        return sorted(f.rule for f in findings)


class DuplicateBlankShortTests(unittest.TestCase):
    def test_case_insensitive_duplicates_are_flagged_once(self):
        findings = check_lines("x.txt", ["Skoda", "Audi", "skoda", "SKODA"])
        self.assertEqual([f.rule for f in findings], [RULE_DUPLICATE, RULE_DUPLICATE])
        self.assertIn("line 1", findings[0].message)

    def test_blank_line_is_flagged(self):
        findings = check_lines("x.txt", ["Alpha", "", "Beta", "   "])
        self.assertEqual([f.rule for f in findings], [RULE_BLANK, RULE_BLANK])

    def test_single_character_line_is_too_short(self):
        findings = check_lines("x.txt", ["Alpha", "Q"])
        self.assertEqual([f.rule for f in findings], [RULE_SHORT])

    def test_blank_line_is_not_also_reported_as_a_duplicate(self):
        findings = check_lines("x.txt", ["", "", "Alpha"])
        self.assertEqual([f.rule for f in findings], [RULE_BLANK, RULE_BLANK])


class GrandfatherTests(unittest.TestCase):
    def test_grandfathered_rule_becomes_a_warning(self):
        findings = check_lines("legacy.txt", ["Man O' War"], grandfathered={RULE_CHARSET})
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].level, "warning")
        self.assertTrue(findings[0].grandfathered)

    def test_grandfathering_one_rule_does_not_excuse_another(self):
        findings = check_lines(
            "legacy.txt", ["Man O' War", "man o' war"], grandfathered={RULE_CHARSET},
        )
        charset = [f for f in findings if f.rule == RULE_CHARSET]
        dupes = [f for f in findings if f.rule == RULE_DUPLICATE]
        self.assertTrue(all(f.level == "warning" for f in charset))
        self.assertEqual([f.level for f in dupes], ["error"])

    def test_new_dataset_gets_no_grandfathering(self):
        findings = check_lines("brand_new.txt", ["Man O' War"])
        self.assertEqual([f.level for f in findings], ["error"])

    def test_amnesty_list_is_empty(self):
        """Guards against a lane quietly adding its own file to the amnesty list.

        This started life as an allowlist of the seven files that pre-dated the
        validator. All seven were fixed on 2026-08-02 — the data was normalized rather
        than exempted — so the guard is now the stronger statement: nothing is exempt.

        If you are here because you added an entry to make a build pass, that is the
        case this test exists to stop. An exemption means a dataset can carry characters
        `data/shared_vocab.json` has no symbol for, which makes `filter_to_vocab` drop
        those names at fine-tune time without saying so. Fix the data instead.
        """
        self.assertEqual(dict(KNOWN_NONCONFORMING), {})


class SidecarTests(_TempData):
    def test_missing_sidecar_is_a_warning_not_an_error(self):
        findings = check_sidecar("x.txt", self.data / "x.meta.json", 3, "x")
        self.assertEqual([f.level for f in findings], ["warning"])
        self.assertEqual(findings[0].rule, RULE_SIDECAR)

    def test_complete_sidecar_passes(self):
        path = self.data / "x.meta.json"
        path.write_text(json.dumps(_meta("x", 3)), encoding="utf-8")
        self.assertEqual(check_sidecar("x.txt", path, 3, "x"), [])

    def test_count_mismatch_is_an_error(self):
        path = self.data / "x.meta.json"
        path.write_text(json.dumps(_meta("x", 99)), encoding="utf-8")
        findings = check_sidecar("x.txt", path, 3, "x")
        self.assertEqual([f.level for f in findings], ["error"])
        self.assertIn("count=99", findings[0].message)
        self.assertIn("3 line(s)", findings[0].message)

    def test_wrong_name_is_an_error(self):
        path = self.data / "x.meta.json"
        path.write_text(json.dumps(_meta("something_else", 3)), encoding="utf-8")
        findings = check_sidecar("x.txt", path, 3, "x")
        self.assertEqual([f.level for f in findings], ["error"])
        self.assertIn("expected 'x'", findings[0].message)

    def test_each_missing_key_is_reported(self):
        body = _meta("x", 3)
        del body["provenance"]
        del body["signature"]
        path = self.data / "x.meta.json"
        path.write_text(json.dumps(body), encoding="utf-8")
        findings = check_sidecar("x.txt", path, 3, "x")
        self.assertEqual(len(findings), 1)
        self.assertIn("provenance", findings[0].message)
        self.assertIn("signature", findings[0].message)

    def test_malformed_json_is_an_error(self):
        path = self.data / "x.meta.json"
        path.write_text("{not json", encoding="utf-8")
        findings = check_sidecar("x.txt", path, 3, "x")
        self.assertEqual([f.level for f in findings], ["error"])

    def test_non_boolean_verified_is_an_error(self):
        path = self.data / "x.meta.json"
        path.write_text(json.dumps(_meta("x", 3, verified="yes")), encoding="utf-8")
        findings = check_sidecar("x.txt", path, 3, "x")
        self.assertEqual([f.level for f in findings], ["error"])

    def test_count_must_be_an_int_not_a_string(self):
        path = self.data / "x.meta.json"
        path.write_text(json.dumps(_meta("x", "3")), encoding="utf-8")
        findings = check_sidecar("x.txt", path, 3, "x")
        self.assertEqual([f.level for f in findings], ["error"])
        self.assertIn("not an integer", findings[0].message)

    def test_required_keys_are_the_documented_eight(self):
        self.assertEqual(
            set(REQUIRED_META_KEYS),
            {"name", "label", "domain", "count", "provenance", "verified", "added",
             "signature"},
        )


class CollectTests(_TempData):
    def test_clean_dataset_with_sidecar_is_ok(self):
        self.write_dataset("widgets", ["Alpha", "Beta"], _meta("widgets", 2))
        reports = collect(self.data, known_nonconforming={})
        self.assertEqual([r.dataset for r in reports], ["widgets.txt"])
        self.assertEqual(reports[0].status, "ok")
        self.assertEqual(reports[0].count, 2)
        self.assertEqual(reports[0].domain, "Test")

    def test_errors_surface_in_status(self):
        self.write_dataset("widgets", ["Man O' War"], _meta("widgets", 1))
        reports = collect(self.data, known_nonconforming={})
        self.assertEqual(reports[0].status, "ERROR")

    def test_grandfathered_status_is_distinct_from_error(self):
        self.write_dataset("legacy", ["Man O' War"], _meta("legacy", 1))
        reports = collect(self.data, known_nonconforming={"legacy.txt": {RULE_CHARSET}})
        self.assertEqual(reports[0].status, "grandfathered")

    def test_tsv_sidecar_is_validated_via_its_file_key(self):
        """`paint_colors.tsv` cannot own `paint_colors.meta.json` (the .txt does), so
        its sidecar is `paint_colors_tsv.meta.json` and names its target explicitly."""
        (self.data / "colors.tsv").write_text("Red\t1\nBlue\t2\n", encoding="utf-8")
        (self.data / "colors_tsv.meta.json").write_text(
            json.dumps(_meta("colors_tsv", 2, file="colors.tsv", format="name_value",
                             value_label="an example number")),
            encoding="utf-8",
        )
        reports = collect(self.data, known_nonconforming={})
        self.assertEqual([r.dataset for r in reports], ["colors.tsv"])
        self.assertEqual(reports[0].status, "ok")
        self.assertEqual(reports[0].count, 2)

    def test_tsv_sidecar_count_mismatch_is_an_error(self):
        (self.data / "colors.tsv").write_text("Red\t1\nBlue\t2\n", encoding="utf-8")
        (self.data / "colors_tsv.meta.json").write_text(
            json.dumps(_meta("colors_tsv", 7, file="colors.tsv")), encoding="utf-8",
        )
        reports = collect(self.data, known_nonconforming={})
        self.assertEqual(reports[0].status, "ERROR")

    def test_orphan_sidecar_is_a_warning(self):
        (self.data / "ghost.meta.json").write_text(
            json.dumps(_meta("ghost", 0)), encoding="utf-8"
        )
        reports = collect(self.data, known_nonconforming={})
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].status, "WARN")
        self.assertIn("orphan sidecar", reports[0].findings[0].message)

    def test_missing_sidecar_leaves_the_dataset_merely_warned(self):
        self.write_dataset("widgets", ["Alpha", "Beta"])
        reports = collect(self.data, known_nonconforming={})
        self.assertEqual(reports[0].status, "WARN")
        self.assertEqual(reports[0].domain, "—")


class ExitCodeTests(_TempData):
    def _run(self, strict=False):
        import io

        buf = io.StringIO()
        code = run(self.data, strict=strict, stream=buf)
        return code, buf.getvalue()

    def test_clean_tree_exits_zero(self):
        self.write_dataset("widgets", ["Alpha", "Beta"], _meta("widgets", 2))
        code, out = self._run()
        self.assertEqual(code, 0)
        self.assertIn("all clear", out)

    def test_error_exits_one(self):
        self.write_dataset("widgets", ["Man O' War"], _meta("widgets", 1))
        code, _ = self._run()
        self.assertEqual(code, 1)

    def test_warning_alone_exits_zero(self):
        self.write_dataset("widgets", ["Alpha", "Beta"])  # no sidecar
        self.assertEqual(self._run()[0], 0)

    def test_strict_promotes_warning_to_failure(self):
        self.write_dataset("widgets", ["Alpha", "Beta"])  # no sidecar
        code, out = self._run(strict=True)
        self.assertEqual(code, 1)
        self.assertIn("--strict", out)

    def test_empty_data_dir_exits_one(self):
        code, out = self._run()
        self.assertEqual(code, 1)
        self.assertIn("no datasets found", out)


class TableTests(unittest.TestCase):
    def test_table_has_the_four_documented_columns(self):
        table = format_table([
            DatasetReport("widgets.txt", 1234, "Test", []),
        ])
        header, rule, row = table.splitlines()
        self.assertEqual(header.split(), ["dataset", "count", "domain", "status"])
        self.assertIn("1,234", row)
        self.assertIn("ok", row)

    def test_columns_line_up_for_varied_name_lengths(self):
        table = format_table([
            DatasetReport("a.txt", 1, "X", []),
            DatasetReport("a_very_long_dataset_name.txt", 100000, "Y", []),
        ])
        lines = table.splitlines()
        self.assertEqual(len(lines), 4)  # header, rule, two rows
        # The count column starts at the same offset on every line.
        offsets = {line.index("count") for line in lines[:1]}
        offsets |= {len("a_very_long_dataset_name.txt") + 2}
        self.assertEqual(len(offsets), 1, lines)
        self.assertIn("100,000", lines[3])


class TsvCommentCountingTests(unittest.TestCase):
    """Regression: `read_lines` must count a .tsv the way its loader reads it.

    `src/data.py` has two loaders with genuinely different rules. `load_names` (.txt)
    treats a leading `#` as an ordinary character — a "comment" in a name list is a name.
    `load_name_value_pairs` (.tsv) skips blanks and `#` comments.

    Counting raw lines in a .tsv made all six wave-4 dual-output datasets report three
    more entries than the trainer would ever see, because each carries a three-line
    header comment — and `check_data` then reported the *sidecar* as wrong. It wasn't.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_tsv_comments_and_blanks_are_not_counted(self):
        p = self.dir / "x.tsv"
        p.write_text("# header\n# more\n\nAlpha\t1\nBeta\t2\n", encoding="utf-8")
        self.assertEqual(read_lines(p), ["Alpha\t1", "Beta\t2"])

    def test_txt_hash_lines_are_still_counted_as_names(self):
        # Not symmetric with the .tsv case, and deliberately so: load_names would read
        # "#Alpha" as a name, so the validator must too.
        p = self.dir / "x.txt"
        p.write_text("#Alpha\nBeta\n", encoding="utf-8")
        self.assertEqual(read_lines(p), ["#Alpha", "Beta"])

    def test_counts_match_the_real_tsv_loader(self):
        """The invariant that actually matters, checked against the live loader."""
        try:
            from src.data import load_name_value_pairs
        except Exception:  # torch missing — this suite must stay stdlib-only
            self.skipTest("src.data unavailable (torch not installed)")
        for tsv in sorted(Path("data").glob("*.tsv")):
            with self.subTest(dataset=tsv.name):
                self.assertEqual(len(read_lines(tsv)), len(load_name_value_pairs(tsv)))


class RealDataDirTests(unittest.TestCase):
    """The one test that touches the live `data/`. It asserts only what must be true
    however many datasets the other lanes have landed by the time it runs."""

    def test_repo_data_dir_has_no_hard_errors(self):
        reports = collect(REPO_ROOT / "data")
        errors = [
            f for r in reports for f in r.findings if f.level == "error"
        ]
        self.assertEqual(errors, [], f"{len(errors)} dataset error(s): {errors[:3]}")

    def test_every_grandfathered_file_still_exists(self):
        """If a lane normalises one of these, the entry should be deleted, not left to
        rot as a silent amnesty for a filename that may be reused later."""
        for name in KNOWN_NONCONFORMING:
            self.assertTrue(
                (REPO_ROOT / "data" / name).exists(),
                f"KNOWN_NONCONFORMING lists {name}, which no longer exists — remove it",
            )


if __name__ == "__main__":
    unittest.main()
