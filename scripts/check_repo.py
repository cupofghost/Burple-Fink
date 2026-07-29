"""Repo hygiene checks — stdlib only, no torch required.

Catches the two failure modes that have already cost this project a session each:
bookkeeping drift (a dataset added to `data/` without updating the registry tables in
HANDOFF.md §4 and the README catalog, or vice versa) and accidentally committing model
weights, secrets, or PII (AGENTS.md §3). Runnable as a CLI (exit non-zero on any finding)
or imported function-by-function from tests/CI.

This script never deletes or modifies anything — it only reports. Per AGENTS.md §3, a
found secret or PII must be flagged to the owner, not auto-removed (proper removal
requires cleaning git history).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
HANDOFF_PATH = REPO_ROOT / "HANDOFF.md"
README_PATH = REPO_ROOT / "README.md"

DATASET_SUFFIXES = (".txt", ".tsv")

# A dataset filename as it appears inside a markdown table cell, e.g. `car_models.txt`.
_TABLE_FILENAME_RE = re.compile(r"`([\w.\-]+\.(?:txt|tsv))`")

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SECRET_PATTERNS = [
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9]{16,}\b")),
    ("GitHub token", re.compile(r"\bghp_[A-Za-z0-9]{20,}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
]

# Extensions worth scanning for secrets/PII; skip binaries and checkpoints outright.
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".tsv", ".json", ".html", ".yml", ".yaml", ".sh", ".cfg", ".ini",
}


def list_dataset_files(data_dir: Path = DATA_DIR) -> List[str]:
    """Every `data/*.txt` and `data/*.tsv` name (not full paths), sorted."""
    data_dir = Path(data_dir)
    return sorted(
        p.name for p in data_dir.iterdir()
        if p.is_file() and p.suffix in DATASET_SUFFIXES
    )


def _table_filenames(text: str) -> set:
    return set(_TABLE_FILENAME_RE.findall(text))


def _section(text: str, heading: str) -> str:
    """Slice out one `## heading` section (up to the next `## `), so we only scan the
    actual dataset registry table and not unrelated tables elsewhere in the doc (e.g.
    HANDOFF's branch-strategy table, which narrates superseded/renamed files by name)."""
    start = text.find(heading)
    if start == -1:
        return ""
    rest = text[start + len(heading):]
    end = rest.find("\n## ")
    return rest if end == -1 else rest[:end]


def check_registry_drift(
    data_dir: Path = DATA_DIR,
    handoff_path: Path = HANDOFF_PATH,
    readme_path: Path = README_PATH,
) -> List[str]:
    """Every dataset file must appear in both registry tables, and vice versa."""
    dataset_files = set(list_dataset_files(data_dir))
    handoff_section = _section(
        Path(handoff_path).read_text(encoding="utf-8"), "## 4. Dataset registry"
    )
    readme_section = _section(
        Path(readme_path).read_text(encoding="utf-8"), "## Dataset catalog"
    )
    handoff_files = _table_filenames(handoff_section)
    readme_files = _table_filenames(readme_section)

    issues: List[str] = []
    for name in sorted(dataset_files - handoff_files):
        issues.append(
            f"data/{name} exists but has no row in the dataset registry table in "
            f"{handoff_path} (§4). Add one — see HANDOFF §5 'Adding a new dataset'."
        )
    for name in sorted(dataset_files - readme_files):
        issues.append(
            f"data/{name} exists but has no row in the dataset catalog table in "
            f"{readme_path}. Add one."
        )
    for name in sorted(handoff_files - dataset_files):
        issues.append(
            f"{handoff_path} lists `{name}` in its registry table, but data/{name} "
            f"does not exist. Fix or remove that row."
        )
    for name in sorted(readme_files - dataset_files):
        issues.append(
            f"{readme_path} lists `{name}` in its dataset catalog, but data/{name} "
            f"does not exist. Fix or remove that row."
        )
    return issues


def _git_tracked_files(repo_root: Path = REPO_ROOT) -> List[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return []
    return [line for line in out.splitlines() if line.strip()]


def check_no_committed_weights(
    tracked_files: Optional[Iterable[str]] = None, repo_root: Path = REPO_ROOT,
) -> List[str]:
    """No `*.pt` / `*.pth` model weight should ever be tracked by git."""
    files = list(tracked_files) if tracked_files is not None else _git_tracked_files(repo_root)
    issues = []
    for f in files:
        if f.endswith(".pt") or f.endswith(".pth"):
            issues.append(
                f"{f} is a tracked model checkpoint (*.pt/*.pth). Checkpoints must never "
                f"be committed — untrack it with `git rm --cached {f}` and confirm it "
                f"matches a `.gitignore` pattern."
            )
    return issues


def check_secrets_and_pii(
    files: Optional[Iterable[Path]] = None, repo_root: Path = REPO_ROOT,
) -> List[str]:
    """Scan tracked text files for email addresses and common secret-key patterns.

    Never deletes anything — per AGENTS.md §3, a hit must be flagged to the owner so
    they can decide how to clean git history, not silently scrubbed.
    """
    if files is None:
        candidates = _git_tracked_files(repo_root)
        paths = [
            Path(repo_root) / f for f in candidates
            if Path(f).suffix in TEXT_SUFFIXES
        ]
    else:
        paths = [Path(f) for f in files]

    issues = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in EMAIL_RE.finditer(text):
            issues.append(
                f"{path}: possible email address found ({m.group(0)}). AGENTS.md §3 "
                f"forbids committing PII — flag this to the owner; do not delete it "
                f"yourself (proper removal requires cleaning git history)."
            )
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                issues.append(
                    f"{path}: possible {label} found. AGENTS.md §3 forbids committing "
                    f"credentials — stop and flag this to the owner; do not delete it "
                    f"yourself (proper removal requires cleaning git history)."
                )
    return issues


def run_all_checks(repo_root: Path = REPO_ROOT) -> List[str]:
    issues: List[str] = []
    issues += check_registry_drift(
        data_dir=Path(repo_root) / "data",
        handoff_path=Path(repo_root) / "HANDOFF.md",
        readme_path=Path(repo_root) / "README.md",
    )
    issues += check_no_committed_weights(repo_root=repo_root)
    issues += check_secrets_and_pii(repo_root=repo_root)
    return issues


def main() -> int:
    issues = run_all_checks()
    if not issues:
        print("check_repo: all clear (registry, checkpoints, secrets/PII).")
        return 0
    print(f"check_repo: {len(issues)} issue(s) found:\n")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
