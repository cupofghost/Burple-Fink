"""Tests for scripts/check_repo.py (WS-8).

Runs without torch: hygiene checks are pure stdlib and this repo's CI runs them in a
separate, torch-free job so bookkeeping errors fail in seconds (see
`.github/workflows/ci.yml` and `docs/upgrade/AGENT-C.md`).
"""

import tempfile
import unittest
from pathlib import Path

from scripts.check_repo import (
    check_no_committed_weights,
    check_registry_drift,
    check_secrets_and_pii,
    list_dataset_files,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class RegistryDriftTests(unittest.TestCase):
    def _fixture(self, data_files, handoff_body, readme_body):
        tmp = Path(tempfile.mkdtemp())
        data_dir = tmp / "data"
        data_dir.mkdir()
        for name in data_files:
            _write(data_dir / name, "one\ntwo\n")
        handoff = _write(tmp / "HANDOFF.md", handoff_body)
        readme = _write(tmp / "README.md", readme_body)
        return data_dir, handoff, readme

    def test_clean_repo_has_no_drift(self):
        handoff = (
            "## 3. Workstreams\nirrelevant\n\n"
            "## 4. Dataset registry\n"
            "| File | Domain |\n|---|---|\n"
            "| `aircraft.txt` | planes |\n"
            "| `paint_colors.tsv` | colors |\n\n"
            "## 5. Adding a new dataset\nirrelevant\n"
        )
        readme = (
            "## Dataset catalog\n"
            "| Dataset file | Domain |\n|---|---|\n"
            "| `aircraft.txt` | planes |\n"
            "| `paint_colors.tsv` | colors |\n\n"
            "## Contributing\nirrelevant\n"
        )
        data_dir, handoff_path, readme_path = self._fixture(
            ["aircraft.txt", "paint_colors.tsv"], handoff, readme
        )
        self.assertEqual(
            check_registry_drift(data_dir, handoff_path, readme_path), []
        )

    def test_dataset_missing_from_both_tables_is_flagged(self):
        handoff = "## 4. Dataset registry\n| File |\n|---|\n\n## 5. Next\nx\n"
        readme = "## Dataset catalog\n| Dataset file |\n|---|\n\n## Contributing\nx\n"
        data_dir, handoff_path, readme_path = self._fixture(
            ["new_dataset.txt"], handoff, readme
        )
        issues = check_registry_drift(data_dir, handoff_path, readme_path)
        self.assertEqual(len(issues), 2)
        self.assertTrue(any("new_dataset.txt" in i and "HANDOFF" in i for i in issues))
        self.assertTrue(any("new_dataset.txt" in i and "README" in i for i in issues))

    def test_stale_table_row_with_no_backing_file_is_flagged(self):
        handoff = (
            "## 4. Dataset registry\n| File |\n|---|\n"
            "| `ghost.txt` |\n\n## 5. Next\nx\n"
        )
        readme = "## Dataset catalog\n| Dataset file |\n|---|\n\n## Contributing\nx\n"
        data_dir, handoff_path, readme_path = self._fixture([], handoff, readme)
        issues = check_registry_drift(data_dir, handoff_path, readme_path)
        self.assertTrue(any("ghost.txt" in i and "does not exist" in i for i in issues))

    def test_narrative_mention_outside_registry_section_is_ignored(self):
        """A filename named in an unrelated section (e.g. branch history) must not be
        treated as a registry row — regression test for a real false positive found
        against this repo's own HANDOFF.md §7 branch-strategy table."""
        handoff = (
            "## 3. Workstreams\nsome text mentioning `old_file.txt` in passing\n\n"
            "## 4. Dataset registry\n| File |\n|---|\n"
            "| `aircraft.txt` |\n\n## 5. Next\nx\n"
        )
        readme = (
            "## Dataset catalog\n| Dataset file |\n|---|\n"
            "| `aircraft.txt` |\n\n## Contributing\nx\n"
        )
        data_dir, handoff_path, readme_path = self._fixture(
            ["aircraft.txt"], handoff, readme
        )
        self.assertEqual(
            check_registry_drift(data_dir, handoff_path, readme_path), []
        )

    def test_list_dataset_files_only_matches_txt_and_tsv(self):
        tmp = Path(tempfile.mkdtemp())
        data_dir = tmp / "data"
        data_dir.mkdir()
        _write(data_dir / "aircraft.txt", "x")
        _write(data_dir / "colors.tsv", "x\tx")
        _write(data_dir / "shared_vocab.json", "{}")
        self.assertEqual(
            list_dataset_files(data_dir), ["aircraft.txt", "colors.tsv"]
        )


class CommittedWeightsTests(unittest.TestCase):
    def test_no_weights_tracked_is_clean(self):
        tracked = ["src/model.py", "data/aircraft.txt", "README.md"]
        self.assertEqual(check_no_committed_weights(tracked), [])

    def test_tracked_pt_file_is_flagged(self):
        tracked = ["checkpoints/car_models.pt", "src/model.py"]
        issues = check_no_committed_weights(tracked)
        self.assertEqual(len(issues), 1)
        self.assertIn("car_models.pt", issues[0])

    def test_tracked_pth_file_is_flagged(self):
        issues = check_no_committed_weights(["weights/base.pth"])
        self.assertEqual(len(issues), 1)
        self.assertIn("base.pth", issues[0])


class SecretsAndPiiTests(unittest.TestCase):
    def _scan(self, content):
        tmp = Path(tempfile.mkdtemp())
        f = _write(tmp / "sample.md", content)
        return check_secrets_and_pii([f])

    def test_clean_file_has_no_findings(self):
        self.assertEqual(self._scan("Just some ordinary documentation text.\n"), [])

    def test_email_address_is_flagged(self):
        issues = self._scan("Contact: someone@example.com for details.\n")
        self.assertEqual(len(issues), 1)
        self.assertIn("email", issues[0])

    def test_openai_style_key_is_flagged(self):
        issues = self._scan("token = 'sk-ABCDEFGHIJKLMNOPQRSTUVWX'\n")  # check_repo: allow (fake fixture)
        self.assertTrue(any("key" in i for i in issues))

    def test_github_token_is_flagged(self):
        issues = self._scan("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ01234\n")  # check_repo: allow (fake fixture)
        self.assertTrue(any("GitHub token" in i for i in issues))

    def test_aws_key_is_flagged(self):
        issues = self._scan("AKIAABCDEFGHIJKLMNOP\n")  # check_repo: allow (fake fixture)
        self.assertTrue(any("AWS" in i for i in issues))

    def test_private_key_block_is_flagged(self):
        issues = self._scan("-----BEGIN RSA PRIVATE KEY-----\nMIIB...\n")  # check_repo: allow (fake fixture)
        self.assertTrue(any("private key" in i for i in issues))

    def test_placeholder_ellipsis_examples_are_not_flagged(self):
        """This repo's own docs use literal `sk-…`/`ghp_…`/`AKIA…` placeholders (an
        ellipsis character, not real chars) as examples of what to look for — they must
        not trip the scanner themselves."""
        issues = self._scan(
            "obvious key patterns (`sk-…`, `ghp_…`, `AKIA…`, "
            "`-----BEGIN … PRIVATE KEY-----`)\n"
        )
        self.assertEqual(issues, [])


class FixtureSuppressionTests(unittest.TestCase):
    """WS-13: the scanner used to flag its own fixtures and the STATUS.md paragraph
    describing that bug, which made CI's `hygiene` job red on every push.

    These tests pin down that the fix is *narrow*. There is no file-level or
    directory-level exemption: every test below feeds the scanner a secret from a file
    that has not opted out, and expects it to be caught.
    """

    def _scan_file(self, name, content):
        tmp = Path(tempfile.mkdtemp())
        return check_secrets_and_pii([_write(tmp / name, content)])

    def test_new_secret_in_a_non_fixture_file_is_still_caught(self):
        """The load-bearing test: a genuinely new credential, in an ordinary source
        file, with no pragma anywhere, still fails the build."""
        content = (
            "import os\n"
            "\n"
            "DEPLOY_TOKEN = 'ghp_ZZZZYYYYXXXXWWWWVVVVUUUUTTTT'\n"  # check_repo: allow (fake)
            "\n"
            "def main():\n"
            "    return DEPLOY_TOKEN\n"
        )
        issues = self._scan_file("deploy.py", content)
        self.assertEqual(len(issues), 1, issues)
        self.assertIn("GitHub token", issues[0])
        self.assertIn(":3:", issues[0])

    def test_secret_under_tests_dir_is_not_blanket_skipped(self):
        """Guards against the lazy fix (`if 'tests/' in path: continue`)."""
        tmp = Path(tempfile.mkdtemp())
        path = _write(  # check_repo: allow (fake)
            tmp / "tests" / "test_thing.py", "KEY = 'AKIAQQQQWWWWEEEERRRR'\n",
        )
        issues = check_secrets_and_pii([path])
        self.assertTrue(any("AWS" in i for i in issues), issues)

    def test_allow_pragma_exempts_only_its_own_line(self):
        """One un-pragma'd secret three lines below a pragma'd one is still reported."""
        content = (
            "fixture = 'AKIAAAAABBBBCCCCDDDD'  # check_repo: allow (fake)\n"
            "harmless = 1\n"
            "\n"
            "real = 'AKIAZZZZYYYYXXXXWWWW'\n"  # check_repo: allow (fake)
        )
        issues = self._scan_file("mixed.py", content)
        self.assertEqual(len(issues), 1, issues)
        self.assertIn(":4:", issues[0])

    def test_pragma_does_not_exempt_a_different_pattern_elsewhere(self):
        content = (
            "aws = 'AKIAAAAABBBBCCCCDDDD'  # check_repo: allow\n"
            "openai = 'sk-QQQQWWWWEEEERRRRTTTTYYYY'\n"  # check_repo: allow (fake)
        )
        issues = self._scan_file("two_kinds.py", content)
        self.assertEqual(len(issues), 1, issues)
        self.assertIn("OpenAI", issues[0])

    def test_real_email_domain_is_still_pii(self):
        issues = self._scan_file(  # check_repo: allow (fake)
            "notes.md", "ping owner@gmail.com about it\n",
        )
        self.assertEqual(len(issues), 1, issues)
        self.assertIn("owner@gmail.com", issues[0])  # check_repo: allow (fake)

    def test_reserved_example_domains_are_not_pii(self):
        """RFC 2606 / RFC 6761 names cannot be registered, so an address at one is
        documentation. This is what lets STATUS.md keep quoting `someone@example.com`
        while describing the very bug this class fixes, unedited."""
        content = (
            "someone@example.com\n"
            "a@example.org\n"
            "b@mail.example.com\n"
            "c@my-service.test\n"
            "d@thing.invalid\n"
        )
        self.assertEqual(self._scan_file("doc.md", content), [])

    def test_findings_carry_a_line_number(self):
        issues = self._scan_file(  # check_repo: allow (fake)
            "x.md", "clean\nclean\nmail: owner@company.co.uk\n",
        )
        self.assertEqual(len(issues), 1, issues)
        self.assertIn(":3:", issues[0])

    def test_this_repos_own_fixture_files_scan_clean(self):
        """Regression test for the red `hygiene` job: the four files that produced the
        six original false positives must scan clean, on the real tree."""
        targets = [
            REPO_ROOT / "tests" / "test_repo_hygiene.py",
            REPO_ROOT / "tests" / "test_data_hygiene.py",
            REPO_ROOT / "STATUS.md",
            REPO_ROOT / "scripts" / "check_repo.py",
            REPO_ROOT / "scripts" / "check_data.py",
        ]
        self.assertEqual(check_secrets_and_pii([p for p in targets if p.exists()]), [])


class DriftMessageTests(unittest.TestCase):
    """Wave 3 lands ~22 datasets, so drift is expected until the orchestrator merges the
    catalog tables. The message has to make that merge mechanical."""

    def test_missing_rows_are_grouped_and_quote_sidecar_metadata(self):
        tmp = Path(tempfile.mkdtemp())
        data_dir = tmp / "data"
        data_dir.mkdir()
        _write(data_dir / "widgets.txt", "one\ntwo\n")
        _write(data_dir / "gadgets.txt", "one\ntwo\n")
        _write(
            data_dir / "widgets.meta.json",
            '{"name": "widgets", "label": "Widget names", "domain": "Industry",'
            ' "count": 2, "provenance": "made up", "verified": false,'
            ' "added": "2026-08-01", "signature": "Agent | Model | low"}',
        )
        handoff = _write(tmp / "HANDOFF.md", "## 4. Dataset registry\n| File |\n|---|\n\n## 5. x\n")
        readme = _write(tmp / "README.md", "## Dataset catalog\n| Dataset file |\n|---|\n\n## C\nx\n")

        issues = check_registry_drift(data_dir, handoff, readme)
        self.assertEqual(len(issues), 2, issues)
        readme_issue = next(i for i in issues if "README" in i)
        # Both files named once, in one grouped issue, with paste-ready rows.
        self.assertIn("gadgets.txt", readme_issue)
        self.assertIn("| `widgets.txt` | Widget names | 2 | ✅ added |", readme_issue)
        # No sidecar for gadgets -> explicit TODOs rather than silence.
        self.assertIn("| `gadgets.txt` | TODO | TODO | ✅ added |", readme_issue)

    def test_pipes_in_a_signature_do_not_break_the_pasted_row(self):
        tmp = Path(tempfile.mkdtemp())
        data_dir = tmp / "data"
        data_dir.mkdir()
        _write(data_dir / "widgets.txt", "one\n")
        _write(
            data_dir / "widgets.meta.json",
            '{"name": "widgets", "domain": "Industry", "count": 1,'
            ' "signature": "Claude Code | Opus 5 | high"}',
        )
        handoff = _write(tmp / "HANDOFF.md", "## 4. Dataset registry\n| File |\n|---|\n\n## 5. x\n")
        readme = _write(tmp / "README.md", "## Dataset catalog\n| Dataset file |\n|---|\n\n## C\nx\n")
        handoff_issue = next(
            i for i in check_registry_drift(data_dir, handoff, readme) if "HANDOFF" in i
        )
        self.assertIn(r"Claude Code \| Opus 5 \| high", handoff_issue)


if __name__ == "__main__":
    unittest.main()
