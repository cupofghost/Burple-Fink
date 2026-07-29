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
        issues = self._scan("token = 'sk-ABCDEFGHIJKLMNOPQRSTUVWX'\n")
        self.assertTrue(any("key" in i for i in issues))

    def test_github_token_is_flagged(self):
        issues = self._scan("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ01234\n")
        self.assertTrue(any("GitHub token" in i for i in issues))

    def test_aws_key_is_flagged(self):
        issues = self._scan("AKIAABCDEFGHIJKLMNOP\n")
        self.assertTrue(any("AWS" in i for i in issues))

    def test_private_key_block_is_flagged(self):
        issues = self._scan("-----BEGIN RSA PRIVATE KEY-----\nMIIB...\n")
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


if __name__ == "__main__":
    unittest.main()
