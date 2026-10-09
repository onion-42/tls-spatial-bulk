"""Bulk RNA-seq cohorts: one-off sanitised import of the aggregated archive, and loading to AnnData.

Layout after import:
    <public_dir>/expressions/<cohort>__log2TPM.tsv.gz   public cohorts (in git)
    <public_dir>/labels_primary.tsv, labels_long.tsv     public label rows (in git)
    <public_dir>/genesets/signatures.tsv                 paper / CHESS signatures only
    <private_dir>/...                                    same layout: non-public cohorts, internal signatures and
                                                         all other archive gene-set files (gitignored)
Matrices: rows = samples (expression_id), columns = HUGO genes, values = log2(TPM + 1).
"""

from __future__ import annotations

import csv
import gzip
import logging
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from tls_model.labels import bulk_labels

log = logging.getLogger(__name__)

PRIVATE_COHORTS = ("Internal_lung", "MGH_LUAD_DagogoJack")
# Allowlist: only these cohorts and label methods may reach the public (git) tables; any other name stays private.
PUBLIC_COHORTS = ("CPTAC-PAAD", "CPTAC-ccRCC", "TCGA-BRCA", "TCGA-KIRC/KIRP/KICH", "TCGA-LUAD", "TCGA-PAAD")
PUBLIC_METHODS = ("pathologist_HE", "pathologist_IHC", "HookNet-TLS_a", "HookNet-TLS_b")
# Internal file paths and pathologist initials never leave the archive.
DROP_COLUMNS = ("source", "notes", "expression_file")
LABELS_PRIMARY = "labels_primary.tsv"
LABELS_LONG = "labels_long.tsv"
SKIPPED_GENESET_FILES = ("_index.tsv",)
SIGNATURE_SHEET = "TLS_signatures - TLS signatures.tsv"
SIGNATURES = "signatures.tsv"
PUBLIC_PROVENANCE = ("paper", "CHESS")


def cohort_stem(cohort: str) -> str:
    return cohort.replace("/", "_")


def is_public_cohort(cohort: str) -> bool:
    """Cohort name or file stem on the public allowlist."""
    return cohort in PUBLIC_COHORTS or cohort in {cohort_stem(c) for c in PUBLIC_COHORTS}


def sanitize_labels(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add `excluded`, drop internal columns, split rows into (public, private): public = allowlisted cohort
    AND allowlisted label method; everything else stays private."""
    excluded = table["notes"].fillna("").str.contains("EXCLUDED")
    clean = table.assign(excluded=excluded).drop(columns=[c for c in DROP_COLUMNS if c in table.columns])
    is_public = clean["cohort"].isin(PUBLIC_COHORTS) & clean["method"].isin(PUBLIC_METHODS)
    return clean[is_public].reset_index(drop=True), clean[~is_public].reset_index(drop=True)


def split_signatures(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Signature sheet (wide, ragged gene columns) -> long (name, provenance, priority, paper, genes).

    Public = provenance 'paper' or 'CHESS'. Internal and unlabelled signatures stay private.
    """
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    header, body = rows[0], rows[1:]
    gene_start = header.index("GMT")
    records = [
        {
            "name": r[0].strip(),
            "provenance": r[2].strip(),
            "priority": r[1].strip(),
            "paper": r[3].strip(),
            "genes": " ".join(g.strip() for g in r[gene_start:] if g.strip()),
        }
        for r in body
        if r and r[0].strip()
    ]
    table = pd.DataFrame(records)
    is_public = table["provenance"].isin(PUBLIC_PROVENANCE)
    return table[is_public].reset_index(drop=True), table[~is_public].reset_index(drop=True)


def load_signatures(*dirs: Path) -> dict[str, list[str]]:
    """{name: genes} from every signatures.tsv found in `dirs`."""
    out = {}
    for d in dirs:
        path = d / "genesets" / SIGNATURES
        if path.exists():
            table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
            out.update({row.name: row.genes.split() for row in table.itertuples()})
    return out


def import_archive(archive_dir: Path, public_dir: Path, private_dir: Path) -> None:
    """Copy the archive into the repo layout. Idempotent: existing outputs are kept."""
    for name, out_name in (("TLS_annotation_primary.tsv", LABELS_PRIMARY), ("TLS_annotation_long.tsv", LABELS_LONG)):
        if (public_dir / out_name).exists() and (private_dir / out_name).exists():
            continue
        public, private = sanitize_labels(pd.read_csv(archive_dir / name, sep="\t"))
        for frame, out_dir in ((public, public_dir), (private, private_dir)):
            out_dir.mkdir(parents=True, exist_ok=True)
            frame.to_csv(out_dir / out_name, sep="\t", index=False)
        log.info("labels %s: public=%d private=%d", out_name, len(public), len(private))

    for src in sorted((archive_dir / "expressions").glob("*__log2TPM.tsv.gz")):
        cohort = src.name.split("__")[0]
        dest = (public_dir if is_public_cohort(cohort) else private_dir) / "expressions" / src.name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            log.info("copied %s -> %s", src.name, dest)

    # Gene sets: only paper/CHESS signatures are public; every other file may hold internal sets.
    sheet = archive_dir / "genesets" / SIGNATURE_SHEET
    public, private = split_signatures(sheet)
    for frame, out_dir in ((public, public_dir), (private, private_dir)):
        dest = out_dir / "genesets" / SIGNATURES
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(dest, sep="\t", index=False)
    for src in sorted((archive_dir / "genesets").iterdir()):
        dest = private_dir / "genesets" / src.name
        if src.name not in SKIPPED_GENESET_FILES and not dest.exists():
            shutil.copy2(src, dest)


def _read_matrix(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt") as handle:
        return pd.read_csv(handle, sep="\t", index_col=0)


def _read_labels(dirs: list[Path], name: str) -> pd.DataFrame:
    frames = [pd.read_csv(d / name, sep="\t") for d in dirs if (d / name).exists()]
    if not frames:
        raise FileNotFoundError(f"{name} not found in {dirs}; run io_bulk.import_archive first")
    return pd.concat(frames, ignore_index=True)


def load_labels(public_dir: Path, private_dir: Path | None = None, name: str = LABELS_PRIMARY) -> pd.DataFrame:
    """Label rows with expression and not EXCLUDED, plus harmonised labels and a global patient key."""
    dirs = [public_dir] + ([private_dir] if private_dir is not None else [])
    table = _read_labels(dirs, name)
    table = table[table["has_expression"].astype(bool) & ~table["excluded"].astype(bool)]
    table = table.assign(patient=table["cohort"] + ":" + table["patient_id"].astype(str))
    return pd.concat([table, bulk_labels(table)], axis=1)


def load_bulk(public_dir: Path, private_dir: Path | None = None) -> ad.AnnData:
    """All cohorts that are present on disk -> one AnnData (obs: cohort, patient, labels)."""
    table = load_labels(public_dir, private_dir)
    dirs = [public_dir] + ([private_dir] if private_dir is not None else [])
    blocks = []
    for cohort, rows in table.groupby("cohort", sort=True):
        paths = [d / "expressions" / f"{cohort_stem(cohort)}__log2TPM.tsv.gz" for d in dirs]
        path = next((p for p in paths if p.exists()), None)
        if path is None:
            log.warning("no expression matrix for %s; skipped", cohort)
            continue
        matrix = _read_matrix(path)
        missing = set(rows["expression_id"]) - set(matrix.index)
        if missing:
            raise ValueError(f"{cohort}: {len(missing)} labelled samples missing from {path.name}")
        blocks.append((rows.set_index("expression_id"), matrix.loc[rows["expression_id"]]))

    genes = sorted(set.intersection(*(set(m.columns) for _, m in blocks)))
    obs = pd.concat([o for o, _ in blocks])
    X = np.vstack([m[genes].to_numpy(dtype=np.float32) for _, m in blocks])
    obs.index = obs.index.astype(str)
    obs.index.name = None
    obs = obs.assign(unit="bulk", platform="bulk")
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
