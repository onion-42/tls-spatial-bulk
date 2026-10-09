"""Everything that would be committed: no internal paths, no non-public sample ids, no private cohort files."""

import re
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from tls_model import io_bulk

ROOT = Path(__file__).resolve().parents[1]
INTERNAL_PATTERNS = re.compile(r"/uftp|/home/jovyan|/bigdata/|s3://")
TEXT_SUFFIXES = {".py", ".md", ".tsv", ".csv", ".yaml", ".yml", ".toml", ".ipynb", ".txt", ".json", ".gmt"}


def _committable_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [ROOT / line for line in out.splitlines() if line and (ROOT / line).is_file()]


def _private_ids() -> set[str]:
    labels = ROOT / "data" / "private" / io_bulk.LABELS_PRIMARY
    if not labels.exists():
        return set()
    table = pd.read_csv(labels, sep="\t", dtype=str)
    return set(table["sample_id"].dropna()) | set(table["expression_id"].dropna())


def test_no_private_cohort_files_are_committable():
    stems = {io_bulk.cohort_stem(c) for c in io_bulk.PRIVATE_COHORTS}
    files = _committable_files()

    offenders = [f for f in files if any(stem in f.name for stem in stems)]

    assert not offenders


def test_committable_text_has_no_internal_paths_or_private_ids():
    private_ids = {i for i in _private_ids() if len(i) >= 4}
    this_file = Path(__file__).resolve()
    offenders = []
    for path in _committable_files():
        if path.suffix not in TEXT_SUFFIXES or path.resolve() == this_file:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if INTERNAL_PATTERNS.search(text):
            offenders.append((path.name, "internal path"))
        hits = [i for i in private_ids if i in text]
        if hits:
            offenders.append((path.name, f"{len(hits)} private sample ids"))

    assert not offenders


def test_public_label_tables_have_no_private_rows_or_internal_columns():
    for name in (io_bulk.LABELS_PRIMARY, io_bulk.LABELS_LONG):
        path = ROOT / "data" / "bulk" / name
        if not path.exists():
            pytest.skip("bulk data not imported")
        table = pd.read_csv(path, sep="\t", dtype=str)

        assert not set(table["cohort"]) & set(io_bulk.PRIVATE_COHORTS)
        assert set(table["cohort"]) <= set(io_bulk.PUBLIC_COHORTS)
        assert set(table["method"]) <= set(io_bulk.PUBLIC_METHODS)
        assert not set(io_bulk.DROP_COLUMNS) & set(table.columns)
