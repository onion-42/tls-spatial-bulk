"""Expression matrices of the HookNet-TLS holdout cohorts (TCGA-BLCA, TCGA-LUSC) from the UCSC Xena GDC hub,
in the archive layout, plus the HookNet-TLS label rows. Logged, idempotent (existing outputs are kept).

    uv run python scripts/fetch_tcga_xena.py [--zenodo data/private/hooknet_tls/data/data]

Downloads (data/private/xena, gitignored): TCGA-<cohort>.star_tpm.tsv.gz (log2(TPM + 1), Ensembl ids) and the
GENCODE v36 probemap (Ensembl -> HUGO). Outputs: data/bulk/expressions/TCGA-<cohort>__log2TPM.tsv.gz reindexed to
the archive gene list (header of TCGA-LUAD__log2TPM.tsv.gz; absent genes = 0), and the HookNet rows appended to
data/bulk/labels_primary.tsv (BLCA, LUSC) and labels_long.tsv (BLCA, LUSC, KIRC).
"""

from __future__ import annotations

import argparse
import gzip
import logging
import sys
import traceback
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tls_model import io_bulk, io_bulk_hooknet  # noqa: E402

HUB = "https://gdc-hub.s3.us-east-1.amazonaws.com/download"
PROBEMAP = "gencode.v36.annotation.gtf.gene.probemap"
COHORTS = ("TCGA-BLCA", "TCGA-LUSC")
LONG_COHORTS = ("TCGA-BLCA", "TCGA-LUSC", "TCGA-KIRC/KIRP/KICH")
ARCHIVE_REFERENCE = "TCGA-LUAD__log2TPM.tsv.gz"
DOWNLOAD_DIR = ROOT / "data" / "private" / "xena"
BULK_DIR = ROOT / "data" / "bulk"
ZENODO_DEFAULT = ROOT / "data" / "private" / "hooknet_tls" / "data" / "data"
log = logging.getLogger(__name__)


def download(name: str) -> Path:
    dest = DOWNLOAD_DIR / name
    if dest.exists():
        log.info("SKIP download %s (exists)", name)
        return dest
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    log.info("GET %s/%s", HUB, name)
    with urllib.request.urlopen(f"{HUB}/{name}", timeout=600) as response, dest.open("wb") as handle:  # noqa: S310
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    return dest


def archive_genes() -> list[str]:
    with gzip.open(BULK_DIR / "expressions" / ARCHIVE_REFERENCE, "rt") as handle:
        return handle.readline().rstrip("\n").split("\t")[1:]


def write_matrix(cohort: str, genes: list[str]) -> Path:
    dest = BULK_DIR / "expressions" / f"{cohort}__log2TPM.tsv.gz"
    if dest.exists():
        log.info("SKIP matrix %s (exists)", dest.name)
        return dest
    probemap = pd.read_csv(download(PROBEMAP), sep="\t", usecols=["id", "gene"], dtype=str)
    xena = pd.read_csv(download(f"{cohort}.star_tpm.tsv.gz"), sep="\t", index_col=0)
    matrix = io_bulk_hooknet.xena_to_archive(xena, probemap, genes)
    with gzip.open(dest, "wt") as handle:
        matrix.to_csv(handle, sep="\t", float_format="%.4f")
    log.info("wrote %s: %s", dest.name, matrix.shape)
    return dest


def matrix_samples(cohort: str) -> set[str]:
    with gzip.open(BULK_DIR / "expressions" / f"{io_bulk.cohort_stem(cohort)}__log2TPM.tsv.gz", "rt") as handle:
        return {line.split("\t", 1)[0] for line in handle.readlines()[1:]}


def append_labels(table: pd.DataFrame, name: str, cohorts: tuple[str, ...]) -> None:
    path = BULK_DIR / name
    # Read as text so the existing rows are written back byte-for-byte.
    existing = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if existing["method"].eq(io_bulk_hooknet.METHOD).any():
        log.info("SKIP labels %s (HookNet rows present)", name)
        return
    # Archive columns only: the strict (GC-anchored) label is TLS_mature_count >= 1, n_slides is logged above.
    rows = table[table["cohort"].isin(cohorts)].reindex(columns=existing.columns).fillna("").astype(str)
    pd.concat([existing, rows], ignore_index=True).to_csv(path, sep="\t", index=False)
    log.info("appended %d rows to %s", len(rows), name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zenodo", type=Path, default=ZENODO_DEFAULT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "results" / "fetch_tcga_xena.log"),
                                  logging.StreamHandler(sys.stdout)])
    genes = archive_genes()
    failed = []
    for i, cohort in enumerate(COHORTS, 1):
        log.info("[%d/%d] START %s", i, len(COHORTS), cohort)
        try:
            write_matrix(cohort, genes)
            log.info("[%d/%d] OK %s", i, len(COHORTS), cohort)
        except Exception:  # noqa: BLE001 - one cohort must not stop the other
            failed.append(cohort)
            log.error("[%d/%d] FAIL %s\n%s", i, len(COHORTS), cohort, traceback.format_exc())
    if failed:
        log.error("FAILED: %s", failed)
        return 1
    table = io_bulk_hooknet.hooknet_labels(args.zenodo)
    with_expression = {c: matrix_samples(c) for c in LONG_COHORTS}
    table["has_expression"] = [s in with_expression.get(c, set()) for c, s in zip(table["cohort"], table["sample_id"])]
    log.info("samples with expression: %s", table.groupby("cohort")["has_expression"].sum().to_dict())
    append_labels(table, io_bulk.LABELS_PRIMARY, COHORTS)
    append_labels(table, io_bulk.LABELS_LONG, LONG_COHORTS)
    log.info("SUMMARY done=%d failed=0", len(COHORTS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
