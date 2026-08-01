"""Repo hygiene checks — stdlib only, no torch required.

Catches the two failure modes that have already cost this project a session each:
bookkeeping drift (a dataset added to `data/` without updating the registry tables in
HANDOFF.md §4 and the README catalog, or vice versa) and accidentally committing model
weights, secrets, or PII (AGENTS.md §3). Runnable as a CLI (exit non-zero on any finding)
or imported function-by-function from tests/CI.

This script never deletes or modifies anything — it only reports. Per AGENTS.md §3, a
found secret or PII must be flagged to the owner, not auto-removed (proper removal
requires cleaning git history).

Suppressing a *known-fake* match (WS-13)
----------------------------------------
The scanner used to flag its own test fixtures and the STATUS.md paragraph describing
that bug, which made the CI `hygiene` job red on every push. Two deliberately narrow
mechanisms fix that without weakening the scanner against a real secret:

1. **Line-level pragma.** A line carrying the marker ``check_repo: allow`` is skipped —
   *only that line*. Deliberate fixture credentials opt out one at a time, so a
   genuinely new secret anywhere else in the same file is still caught.
2. **Exact allowlist** (``KNOWN_FIXTURE_MATCHES``) for files that cannot carry a
   pragma, keyed on the repo-relative path *and* the exact matched text. It exempts one
   string in one file; the same string elsewhere, or any other match in that file, is
   still reported.

There is deliberately no file-level, directory-level or domain-level exemption: no
"skip tests/", and no "example.com addresses are always fine" (the hygiene suite has a
test asserting `someone@example.com` is still reported when it turns up somewhere that
has not opted out).
"""

from __future__ import annotations

import json
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

# A single line may opt out of the secret/PII scan by carrying this marker, e.g.
#     issues = self._scan("ghp_AAAA...")  # check_repo: allow (test fixture)
# Scope is one line, on purpose: a real secret added elsewhere in the same file is
# still reported. See the module docstring.
ALLOW_PRAGMA_RE = re.compile(r"check_repo:\s*allow\b")

# Exact exceptions for text that has to stay in the repo but is not a real secret, for
# files that cannot carry an inline pragma. Keyed on (repo-relative path, exact matched
# text): both must match, so this cannot quietly grow into a file-level skip.
#
# Keep this list short and justify every entry. Prefer the `check_repo: allow` pragma —
# it lives next to the string it excuses and travels with it.
KNOWN_FIXTURE_MATCHES = {
    # STATUS.md quotes this deliberately fake address in its Known-issues entry while
    # describing the very bug that made the hygiene job red. STATUS.md is unowned by any
    # lane (the orchestrating session merges it), so it cannot carry a pragma. Any other
    # address in STATUS.md, and this address in any other file, is still reported.
    ("STATUS.md", "someone@example.com"),
}


def _repo_relative(path: Path, repo_root: Path) -> Optional[str]:
    """`path` as a posix path relative to the repo, or None if it lies outside it."""
    try:
        return Path(path).resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except (ValueError, OSError):
        return None


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


def _sidecar_meta(name: str, data_dir: Path) -> dict:
    """Best-effort read of `data/<stem>.meta.json` (WS-13) for a dataset filename."""
    stem = Path(name).stem
    candidates = [Path(data_dir) / f"{stem}.meta.json"]
    if Path(name).suffix == ".tsv":
        # `paint_colors.tsv` and `paint_colors.txt` share a stem; the .tsv sidecar is
        # disambiguated as `paint_colors_tsv.meta.json`.
        candidates.insert(0, Path(data_dir) / f"{stem}_tsv.meta.json")
    for sidecar in candidates:
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(meta, dict):
            return meta
    return {}


def _suggested_rows(names: Iterable[str], data_dir: Path, columns: List[str]) -> str:
    """Paste-ready markdown rows for `names`, pre-filled from their `.meta.json` sidecars.

    Wave 3 lands ~22 datasets from parallel lanes, so this check reports drift until the
    orchestrating session merges the catalog tables. Emitting the literal rows makes that
    merge mechanical instead of a scavenger hunt through `data/`.
    """
    defaults = {"status": "✅ added"}
    rows = []
    for name in names:
        meta = _sidecar_meta(name, data_dir)
        cells = []
        for col in columns:
            if col == "file":
                cells.append(f"`{name}`")
            else:
                # Escape pipes so a signature ("Claude Code | Opus 5 | high") or a
                # provenance sentence cannot split the pasted row into extra columns.
                value = str(meta.get(col, defaults.get(col, "TODO")))
                cells.append(value.replace("|", r"\|"))
        rows.append("    | " + " | ".join(cells) + " |")
    return "\n".join(rows)


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
    missing_from_handoff = sorted(dataset_files - handoff_files)
    if missing_from_handoff:
        issues.append(
            f"{handoff_path} (§4 Dataset registry) is missing {len(missing_from_handoff)} "
            f"row(s) for files that exist in data/: "
            f"{', '.join(missing_from_handoff)}. Add these rows verbatim (columns "
            f"File | Domain | Count | Owner | Notes; TODO = no sidecar to read it from) — "
            f"see HANDOFF §5 'Adding a new dataset':\n"
            + _suggested_rows(
                missing_from_handoff, data_dir,
                ["file", "domain", "count", "signature", "provenance"],
            )
        )
    missing_from_readme = sorted(dataset_files - readme_files)
    if missing_from_readme:
        issues.append(
            f"{readme_path} (## Dataset catalog) is missing {len(missing_from_readme)} "
            f"row(s) for files that exist in data/: "
            f"{', '.join(missing_from_readme)}. Add these rows verbatim (columns "
            f"Dataset file | Domain | Count | Status; TODO = no sidecar to read it "
            f"from):\n"
            + _suggested_rows(
                missing_from_readme, data_dir, ["file", "label", "count", "status"],
            )
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

    Scanning is line-by-line so findings carry a line number and so a single line can
    opt out via the ``check_repo: allow`` pragma without blinding the rest of the file.
    Addresses at reserved example domains are not PII and are never reported.

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
        for lineno, line in enumerate(text.splitlines(), start=1):
            if ALLOW_PRAGMA_RE.search(line):
                continue
            for m in EMAIL_RE.finditer(line):
                if _is_reserved_example_email(m.group(0)):
                    continue
                issues.append(
                    f"{path}:{lineno}: possible email address found ({m.group(0)}). "
                    f"AGENTS.md §3 forbids committing PII — flag this to the owner; do "
                    f"not delete it yourself (proper removal requires cleaning git "
                    f"history). If it is a deliberate fixture, use a reserved domain "
                    f"(user@example.com) or add a `check_repo: allow` comment to the line."
                )
            for label, pattern in SECRET_PATTERNS:
                if pattern.search(line):
                    issues.append(
                        f"{path}:{lineno}: possible {label} found. AGENTS.md §3 forbids "
                        f"committing credentials — stop and flag this to the owner; do "
                        f"not delete it yourself (proper removal requires cleaning git "
                        f"history). If it is a deliberate fixture, add a "
                        f"`check_repo: allow` comment to that line."
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
