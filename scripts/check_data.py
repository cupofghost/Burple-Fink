#!/usr/bin/env python3
"""Validate every dataset in `data/` — stdlib only, no torch (WS-13).

Wave 3 lands ~22 new datasets from four parallel lanes. Nobody reviews 22 files by eye,
so this is the gate: run it before you hand a dataset off.

Why the character rule matters
------------------------------
The model has a fixed alphabet, and `src/data.py` maps an unknown character to nothing
at all. A stray `é`, apostrophe or soft hyphen therefore doesn't crash anything — it
silently changes what the model can spell and quietly invalidates comparisons against
existing checkpoints. Catching it here costs a second; catching it after a training
sweep costs the sweep.

Digits are *allowed*: they are load-bearing in real names (`ATR 42`, `RAV4`, `MR2`,
`90 Minute`), and excluding them would condemn three otherwise-clean datasets. Anything
beyond letters, digits, spaces and hyphens is an error — apostrophes, periods,
ampersands, slashes and accented Latin all have to be normalised before the data lands.

Note on `data/shared_vocab.json`: it is known to be **stale** — its alphabet omits the
digit `1` as well as accents and apostrophes, so some names are already being dropped
silently at fine-tune time. Regenerating it is the orchestrator's job at consolidation.
This script therefore validates against the rule above, not against that file; it only
reads the vocab to *annotate* a finding with "and this character isn't in the current
vocab either", never to decide whether something is an error.

What it checks
--------------
1. Character set — every line matches ``^[A-Za-z0-9][A-Za-z0-9 -]*[A-Za-z0-9]$``.
2. Duplicates within a file, case-insensitively.
3. Blank lines and lines shorter than two characters.
4. The `data/<stem>.meta.json` sidecar: required keys, `name` == stem, `count` == the
   file's real line count. A *missing* sidecar is a warning (lanes are still landing
   them); a sidecar that *disagrees* with the file is an error.

Usage
-----
    python scripts/check_data.py              # human-readable, warnings tolerated
    python scripts/check_data.py --strict     # warnings become errors (use in review)
    python scripts/check_data.py --verbose    # list every grandfathered offence too
    python scripts/check_data.py --data-dir some/other/dir

Exit code is 0 unless there is an error (or, under `--strict`, any finding at all).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
VOCAB_PATH = "shared_vocab.json"

#: Letters, digits, interior spaces and hyphens; no leading/trailing whitespace; at
#: least two characters. Digits are in because real names use them ("ATR 42", "RAV4").
#: Apostrophes, periods, ampersands, slashes and accented Latin are out: they are the
#: characters that actually go missing at encode time.
LINE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 -]*[A-Za-z0-9]$")

MIN_LINE_LENGTH = 2

REQUIRED_META_KEYS = (
    "name", "label", "domain", "count", "provenance", "verified", "added", "signature",
)

META_SUFFIX = ".meta.json"

# Rule ids, usable as `--only`-style filters by future callers and as the keys of
# KNOWN_NONCONFORMING below.
RULE_CHARSET = "charset"
RULE_DUPLICATE = "duplicate"
RULE_BLANK = "blank"
RULE_SHORT = "short"
RULE_SIDECAR = "sidecar"

# ---------------------------------------------------------------------------
# Grandfather clause.
#
# These datasets pre-date this validator and belong to other lanes (wave-1 WS-1 and
# wave-3 WS-17), so their findings are reported as warnings rather than errors. This is
# a record of known debt, not an amnesty: `--strict` still fails on them, `--verbose`
# still lists every offending line, and a *new* dataset gets no such treatment. Do not
# add an entry here for a file you are creating — fix the data instead.
#
# Verified against the tree on 2026-08-01; counts are offending lines at that time.
# ---------------------------------------------------------------------------
#
# EMPTY, AND THAT IS THE POINT. Every entry this list was created to hold has been
# paid off rather than tolerated (orchestrator, 2026-08-02):
#
#   craft_beers.txt   18 lines — apostrophes + one "ä"      -> normalized
#   racehorses.txt     5 lines — apostrophes + "Dr. Fager"  -> normalized
#   spacecraft.txt     2 lines — "Chang'e", one "/"         -> fixed by WS-17
#   paint_colors.txt   1 line  — "Bells & Whistles"         -> "Bells and Whistles"
#   world_cities.txt  93 lines — accented Latin, periods, apostrophes, and one
#                               U+00AD SOFT HYPHEN invisible in every editor
#                                                            -> normalized
#   car_manufacturers.txt / car_models.txt — duplicate lines -> fixed by WS-17
#
# Normalizing world_cities.txt also collapsed 17 pairs that were the SAME city stored
# twice, once accented and once not ("Belém" and "Belem" as separate lines) — debris
# from the 2026-07-24 merge of two independently-built city lists, which deduplicated
# case-insensitively and so could not see them. No city was lost; 1,691 lines became
# 1,674 rows and one fewer silent trap.
#
# Leave this empty. An entry here means a dataset is exempt from the rule that keeps
# data/shared_vocab.json honest; fix the data instead.
KNOWN_NONCONFORMING: Dict[str, Set[str]] = {}


class Finding(NamedTuple):
    """One problem with one dataset. `level` is "error" or "warning"."""

    level: str
    dataset: str
    rule: str
    message: str
    grandfathered: bool = False


class DatasetReport(NamedTuple):
    dataset: str
    count: int
    domain: str
    findings: List[Finding]

    @property
    def status(self) -> str:
        if any(f.level == "error" for f in self.findings):
            return "ERROR"
        if any(f.grandfathered for f in self.findings):
            return "grandfathered"
        if self.findings:
            return "WARN"
        return "ok"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_vocab(data_dir: Path = DATA_DIR) -> Optional[Set[str]]:
    """The frozen model alphabet from `shared_vocab.json`, or None if unreadable."""
    try:
        blob = json.loads((Path(data_dir) / VOCAB_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    itos = blob.get("itos") if isinstance(blob, dict) else None
    if not isinstance(itos, list):
        return None
    return {c for c in itos if isinstance(c, str)}


def read_lines(path: Path) -> List[str]:
    """The dataset's usable entries, newline stripped.

    For ``.txt`` this is every raw line — ``src/data.py:load_names`` treats a leading
    ``#`` as an ordinary character, so a "comment" in a name list really is a name.

    For ``.tsv`` it is every line ``src/data.py:load_name_value_pairs`` would keep, i.e.
    blank lines and ``#``-prefixed comments removed. The two loaders genuinely differ,
    and this function has to match whichever one will actually read the file.

    Getting this wrong is not cosmetic. Counting raw lines in a ``.tsv`` made every
    wave-4 dual-output dataset report a count three higher than the number of pairs the
    trainer would see, because each carries a three-line header comment — and the
    validator then blamed the *sidecar* for the mismatch. The sidecars were right.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if Path(path).suffix == ".tsv":
        return [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    return lines


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _describe_char(ch: str, vocab: Optional[Set[str]]) -> str:
    shown = repr(ch)
    note = f"U+{ord(ch):04X}"
    if not ch.isprintable():
        note += " (non-printable)"
    if vocab is not None and ch not in vocab:
        # Annotation only — shared_vocab.json is stale, so its absence is corroborating
        # evidence, never the reason a line fails.
        note += "; also absent from the current shared_vocab.json, so it is being "
        note += "dropped silently at encode time"
    return f"{shown} ({note})"


def check_lines(
    dataset: str,
    lines: Sequence[str],
    vocab: Optional[Set[str]] = None,
    grandfathered: Iterable[str] = (),
) -> List[Finding]:
    """Rules 1–3 for one dataset's lines. Line numbers are 1-based."""
    grandfathered = set(grandfathered)
    findings: List[Finding] = []

    def add(rule: str, lineno: int, message: str) -> None:
        soft = rule in grandfathered
        findings.append(
            Finding(
                level="warning" if soft else "error",
                dataset=dataset,
                rule=rule,
                message=f"line {lineno}: {message}",
                grandfathered=soft,
            )
        )

    seen: Dict[str, int] = {}
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            add(RULE_BLANK, lineno, "blank line — delete it")
            continue
        if len(line.strip()) < MIN_LINE_LENGTH:
            add(
                RULE_SHORT, lineno,
                f"{line.strip()!r} is shorter than {MIN_LINE_LENGTH} characters",
            )
            continue

        if not LINE_RE.match(line):
            offenders = [
                (i, ch) for i, ch in enumerate(line, start=1)
                if not (ch.isascii() and ch.isalnum())
                and not (ch in " -" and 1 < i < len(line))
            ]
            if not offenders:
                # Matches the alphabet but not the shape: leading/trailing space or
                # hyphen, or a single character already handled above.
                add(
                    RULE_CHARSET, lineno,
                    f"{line!r} must start and end with a letter (no leading/trailing "
                    f"space or hyphen)",
                )
            else:
                shown = ", ".join(
                    f"col {i}: {_describe_char(ch, vocab)}" for i, ch in offenders[:5]
                )
                more = "" if len(offenders) <= 5 else f" (+{len(offenders) - 5} more)"
                add(
                    RULE_CHARSET, lineno,
                    f"{line!r} does not match ^[A-Za-z0-9][A-Za-z0-9 -]*[A-Za-z0-9]$ — "
                    f"{shown}{more}",
                )

        key = line.strip().lower()
        if key in seen:
            add(
                RULE_DUPLICATE, lineno,
                f"{line.strip()!r} duplicates line {seen[key]} (case-insensitive)",
            )
        else:
            seen[key] = lineno

    return findings


def check_sidecar(
    dataset: str,
    meta_path: Path,
    actual_count: int,
    expected_name: str,
) -> List[Finding]:
    """Rule 4. A missing sidecar warns; a sidecar that contradicts the file errors."""
    meta_path = Path(meta_path)
    rel = meta_path.name

    if not meta_path.exists():
        return [Finding(
            "warning", dataset, RULE_SIDECAR,
            f"no sidecar {rel} — add one with keys "
            f"{', '.join(REQUIRED_META_KEYS)} (count={actual_count})",
        )]

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [Finding("error", dataset, RULE_SIDECAR, f"{rel} is not readable JSON: {exc}")]

    if not isinstance(meta, dict):
        return [Finding("error", dataset, RULE_SIDECAR, f"{rel} must contain a JSON object")]

    findings: List[Finding] = []
    missing = [k for k in REQUIRED_META_KEYS if k not in meta]
    if missing:
        findings.append(Finding(
            "error", dataset, RULE_SIDECAR,
            f"{rel} is missing required key(s): {', '.join(missing)}",
        ))

    if "name" in meta and meta["name"] != expected_name:
        findings.append(Finding(
            "error", dataset, RULE_SIDECAR,
            f"{rel} has name={meta['name']!r}, expected {expected_name!r}",
        ))

    if "count" in meta:
        if not isinstance(meta["count"], int) or isinstance(meta["count"], bool):
            findings.append(Finding(
                "error", dataset, RULE_SIDECAR,
                f"{rel} has count={meta['count']!r}, which is not an integer",
            ))
        elif meta["count"] != actual_count:
            findings.append(Finding(
                "error", dataset, RULE_SIDECAR,
                f"{rel} says count={meta['count']} but the file has {actual_count} "
                f"line(s). Fix whichever is wrong — a stale count silently misreports "
                f"dataset size in the README catalog and in evaluation runs.",
            ))

    if "verified" in meta and not isinstance(meta["verified"], bool):
        findings.append(Finding(
            "error", dataset, RULE_SIDECAR,
            f"{rel} has verified={meta['verified']!r}, expected true or false",
        ))

    return findings


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _meta_stem(path: Path) -> str:
    return path.name[: -len(META_SUFFIX)]


def collect(
    data_dir: Path = DATA_DIR,
    known_nonconforming: Optional[Dict[str, Set[str]]] = None,
) -> List[DatasetReport]:
    """Validate every `data/*.txt`, plus any sidecar that points at another file."""
    data_dir = Path(data_dir)
    known = KNOWN_NONCONFORMING if known_nonconforming is None else known_nonconforming
    vocab = load_vocab(data_dir)

    reports: List[DatasetReport] = []
    claimed: Set[Path] = set()

    for path in sorted(data_dir.glob("*.txt")):
        stem = path.stem
        lines = read_lines(path)
        meta_path = data_dir / f"{stem}{META_SUFFIX}"
        claimed.add(meta_path)

        findings = check_lines(
            path.name, lines, vocab=vocab, grandfathered=known.get(path.name, set()),
        )
        findings += check_sidecar(path.name, meta_path, len(lines), stem)
        reports.append(DatasetReport(path.name, len(lines), _domain(meta_path), findings))

    # Sidecars that describe a non-.txt dataset (the WS-4 dual-output .tsv demos) or
    # nothing at all. `paint_colors.tsv` cannot use `paint_colors.meta.json` — the .txt
    # dataset owns that name — so its sidecar is `paint_colors_tsv.meta.json` and names
    # its target explicitly via a "file" key.
    for meta_path in sorted(data_dir.glob("*" + META_SUFFIX)):
        if meta_path in claimed:
            continue
        stem = _meta_stem(meta_path)
        target = _declared_file(meta_path)
        if target is None or not (data_dir / target).exists():
            reports.append(DatasetReport(meta_path.name, 0, _domain(meta_path), [Finding(
                "warning", meta_path.name, RULE_SIDECAR,
                f"orphan sidecar: no data/{stem}.txt, and its \"file\" key is "
                f"{target!r}. Point it at a real dataset or delete it.",
            )]))
            continue
        count = len(read_lines(data_dir / target))
        findings = check_sidecar(target, meta_path, count, stem)
        reports.append(DatasetReport(target, count, _domain(meta_path), findings))

    return sorted(reports, key=lambda r: r.dataset)


def _load_meta(meta_path: Path) -> dict:
    try:
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _domain(meta_path: Path) -> str:
    return str(_load_meta(meta_path).get("domain", "—"))


def _declared_file(meta_path: Path) -> Optional[str]:
    value = _load_meta(meta_path).get("file")
    return value if isinstance(value, str) else None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_table(reports: Sequence[DatasetReport]) -> str:
    headers = ("dataset", "count", "domain", "status")
    rows = [(r.dataset, f"{r.count:,}", r.domain, r.status) for r in reports]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(4)
    ]
    out = [
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip(),
        "  ".join("-" * widths[i] for i in range(4)),
    ]
    for row in rows:
        out.append("  ".join(row[i].ljust(widths[i]) for i in range(4)).rstrip())
    return "\n".join(out)


def _print_findings(
    reports: Sequence[DatasetReport], verbose: bool, stream, limit: int = 10,
) -> None:
    for report in reports:
        hard = [f for f in report.findings if not f.grandfathered]
        soft = [f for f in report.findings if f.grandfathered]
        if not hard and not soft:
            continue
        print(f"\n{report.dataset}", file=stream)
        shown = hard if verbose else hard[:limit]
        for finding in shown:
            print(
                f"  {finding.level.upper():7} [{finding.rule}] {finding.message}",
                file=stream,
            )
        if len(hard) > len(shown):
            print(
                f"  ... and {len(hard) - len(shown)} more (use --verbose to see all)",
                file=stream,
            )
        if soft:
            by_rule: Dict[str, int] = defaultdict(int)
            for finding in soft:
                by_rule[finding.rule] += 1
            summary = ", ".join(f"{n} {rule}" for rule, n in sorted(by_rule.items()))
            print(
                f"  WARNING [grandfathered] {summary} — pre-existing, see "
                f"KNOWN_NONCONFORMING in scripts/check_data.py",
                file=stream,
            )
            if verbose:
                for finding in soft:
                    print(f"      [{finding.rule}] {finding.message}", file=stream)


def run(
    data_dir: Path = DATA_DIR,
    strict: bool = False,
    verbose: bool = False,
    stream=None,
) -> int:
    """Validate `data_dir`, print a report, return the process exit code."""
    stream = stream or sys.stdout
    reports = collect(data_dir)
    if not reports:
        print(f"check_data: no datasets found in {data_dir}", file=stream)
        return 1

    findings = [f for r in reports for f in r.findings]
    errors = [f for f in findings if f.level == "error"]
    warnings = [f for f in findings if f.level == "warning"]

    print(format_table(reports), file=stream)
    _print_findings(reports, verbose=verbose, stream=stream)

    total = sum(r.count for r in reports)
    print(
        f"\ncheck_data: {len(reports)} dataset(s), {total:,} names, "
        f"{len(errors)} error(s), {len(warnings)} warning(s)"
        f"{' — --strict: warnings count as errors' if strict else ''}",
        file=stream,
    )
    if errors:
        return 1
    if warnings and strict:
        return 1
    print("check_data: all clear.", file=stream)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_data.py",
        description="Validate every dataset in data/ (character set, duplicates, "
                    "blank/short lines, .meta.json sidecars).",
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="defaults to data/")
    parser.add_argument(
        "--strict", action="store_true",
        help="treat warnings (missing sidecars, grandfathered files) as errors",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="list every grandfathered offence instead of a per-file count",
    )
    args = parser.parse_args(argv)
    return run(Path(args.data_dir), strict=args.strict, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
