"""Public Visium data: idempotent download (GEO + Zenodo) and loading to AnnData.

Cohorts: GSE175540 (ccRCC, Meylan 2022; train / test), USZ (Zenodo 14620362; train / test), GSE273952 (ccRCC,
BIONIKK; the 8 sections that are not GSE175540 re-annotations; OOD) and GSE351843 (breast CytAssist FFPE, labels
from Zenodo 18917100; OOD). One `Cohort` record per GEO
series describes its sections, file names and annotation; loading is the same for all of them.

Run the download driver once (resumable, logged):
    uv run python -m tls_model.io_visium --out data/visium/raw
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import shutil
import sys
import traceback
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from tls_model.labels import spot_labels

log = logging.getLogger(__name__)

GEO_SAMPLE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm_dir}/{gsm}/suppl/{gsm}_{name}_{file}"
GSE_BULK_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE175nnn/GSE175540/suppl/"
    "GSE175540_batch_corrected_bulk_RNA_seq_TPM_25_02_22.csv.gz"
)
USZ_ZIP_URL = "https://zenodo.org/records/14620362/files/TLS_VISIUM_USZ.zip"
BREAST_ZIP_URL = "https://zenodo.org/records/18917100/files/pathology_annotations.zip"
# (url, zip path under raw_dir, extraction dir under raw_dir)
ARCHIVES = (
    (USZ_ZIP_URL, "usz/TLS_VISIUM_USZ.zip", "usz/extracted"),
    (BREAST_ZIP_URL, "gse351843/zenodo/pathology_annotations.zip", "gse351843/zenodo"),
)
BREAST_ANNOTATION_DIR = Path("gse351843/zenodo/pathology_annotations/visium")

# GSM order on GEO: GSM5924030 + index in this list.
GSE_SECTION_ORDER = (
    "ffpe_c_2", "ffpe_c_3", "ffpe_c_4", "ffpe_c_7", "ffpe_c_10", "ffpe_c_20", "ffpe_c_21",
    "ffpe_c_34", "ffpe_c_36", "ffpe_c_39", "ffpe_c_45", "ffpe_c_51",
    "frozen_a_1", "frozen_a_3", "frozen_a_15", "frozen_a_17",
    "frozen_b_1", "frozen_b_7", "frozen_b_13", "frozen_b_18",
    "frozen_c_2", "frozen_c_5", "frozen_c_23", "frozen_c_57",
)
GSE_UNANNOTATED = frozenset({"ffpe_c_10"})
# Frozen sections with a paired bulk sample in the GSE175540 bulk matrix.
GSE_BULK_PAIRS = frozenset({"a_1", "a_3", "a_15", "a_17", "b_1", "b_7", "b_13", "b_18"})
# GSE273952 (Fridman lab, BIONIKK): 11 of the 19 sections are the GSE175540 sections re-annotated (identical barcodes
# and UMI counts, checked 2026-11-08) -> train / test tissue, excluded. The 8 others are new FFPE sections.
GSE273952_DUPLICATES = {
    "P_1": "frozen_c_2", "P_2": "ffpe_c_3", "P_3": "ffpe_c_4", "P_4": "frozen_c_5", "P_5": "ffpe_c_7",
    "P_6": "ffpe_c_20", "P_7": "frozen_c_23", "P_8": "ffpe_c_36", "P_9": "frozen_c_57", "P_14": "ffpe_c_34",
    "P_18": "ffpe_c_51",
}
# GSE351843 tumour sections in GSM order (GSM10144987 + index); the 7 lymph-node sections that follow are skipped.
BREAST_SECTION_ORDER = tuple(f"P{i:02d}_Vis" for i in range(1, 29)) + (
    "P29_Vis_Rep1", "P29_Vis_Rep2", "P30_Vis", "P31_Vis", "P32_Vis",
)

GEO_FILES = {
    "matrix": "filtered_feature_bc_matrix.h5",
    "positions": "tissue_positions_list.csv.gz",
    "scalefactors": "scalefactors_json.json.gz",
    "image": "tissue_lowres_image.png.gz",
    "annotation": "TLS_annotation.csv.gz",
}
CYTASSIST_FILES = {
    "matrix": "filtered_feature_bc_matrix.h5",
    "positions": "spatial_tissue_positions.csv.gz",
    "scalefactors": "spatial_scalefactors_json.json.gz",
    "image": "spatial_tissue_lowres_image.png.gz",
}


@dataclass(frozen=True)
class Cohort:
    source: str  # GEO series; obs["source"]
    folder: str  # sub-folder of the raw dir
    first_gsm: int  # GSM of names[0]; the others follow consecutively
    names: tuple[str, ...]  # GEO sample names in GSM order
    protocol: str  # "ffpe" (probe-based) or "frozen" (poly-A); GSE175540 reads it from the name
    files: dict[str, str]  # role -> GEO file suffix (also the local file name)
    annotation_column: str  # label column of a Zenodo table; "" = second column of the two-column GEO file
    section_prefix: str = ""  # "" keeps the historical GSE175540 section ids / h5ad names
    skip: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Section:
    name: str  # GEO sample name (file names, raw sub-folder)
    gsm: str
    protocol: str
    patient: str
    source: str
    section: str  # unique section id: obs["section"], h5ad name
    bulk_id: str | None = None


COHORTS = (
    Cohort("GSE175540", "gse175540", 5924030, GSE_SECTION_ORDER, "", GEO_FILES, "", skip=GSE_UNANNOTATED),
    Cohort("GSE273952", "gse273952", 8440294, tuple(f"P_{i}" for i in range(1, 20)), "ffpe", GEO_FILES, "",
           section_prefix="GSE273952_", skip=frozenset(GSE273952_DUPLICATES)),
    Cohort("GSE351843", "gse351843", 10144987, BREAST_SECTION_ORDER, "ffpe", CYTASSIST_FILES, "immune",
           section_prefix="GSE351843_"),
)
COHORT_BY_SOURCE = {c.source: c for c in COHORTS}


def _parse_gse_name(name: str) -> tuple[str, str]:
    """'frozen_a_15' -> ('frozen', 'a_15')."""
    protocol, series, number = name.split("_")
    return protocol, f"{series}_{number}"


def _section(cohort: Cohort, index: int, name: str) -> Section:
    gsm = f"GSM{cohort.first_gsm + index}"
    if cohort.source == "GSE175540":
        protocol, sample = _parse_gse_name(name)
        bulk_id = sample if protocol == "frozen" and sample in GSE_BULK_PAIRS else None
        return Section(name, gsm, protocol, f"GSE175540_{sample}", cohort.source, name, bulk_id)
    # 'P_1' -> P_1; 'P29_Vis_Rep1' -> P29 (two replicate sections of one patient)
    patient = name.split("_Vis")[0]
    return Section(name, gsm, cohort.protocol, f"{cohort.source}_{patient}", cohort.source,
                   f"{cohort.section_prefix}{name}")


def cohort_sections(source: str) -> list[Section]:
    cohort = COHORT_BY_SOURCE[source]
    return [_section(cohort, i, name) for i, name in enumerate(cohort.names) if name not in cohort.skip]


def gse_sections() -> list[Section]:
    """All annotated GSE175540 Visium sections."""
    return cohort_sections("GSE175540")


def all_sections() -> list[Section]:
    return [s for cohort in COHORTS for s in cohort_sections(cohort.source)]


def section_file_urls(section: Section) -> dict[str, str]:
    """Local file name -> GEO URL for one section."""
    gsm_dir = section.gsm[:-3] + "nnn"
    return {
        f: GEO_SAMPLE_URL.format(gsm_dir=gsm_dir, gsm=section.gsm, name=section.name, file=f)
        for f in COHORT_BY_SOURCE[section.source].files.values()
    }


def section_dir(section: Section, raw_dir: Path) -> Path:
    return raw_dir / COHORT_BY_SOURCE[section.source].folder / section.name


def _remote_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        length = response.headers.get("Content-Length")
    return int(length) if length else None


def download(url: str, dest: Path) -> str:
    """Download `url` to `dest` unless it exists. Returns 'skip' or 'ok'.

    `dest` only appears via an atomic rename of a size-checked `.part` file, so an existing `dest` is complete and
    needs no request (repeated HEAD calls over hundreds of files get rate-limited by NCBI with HTTP 403).
    """
    if dest.exists():
        return "skip"
    expected = _remote_size(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    if expected is not None and part.stat().st_size != expected:
        raise OSError(f"size mismatch for {url}: {part.stat().st_size} != {expected}")
    part.replace(dest)
    return "ok"


def download_jobs(raw_dir: Path) -> list[tuple[str, Path]]:
    jobs = [(GSE_BULK_URL, raw_dir / "gse175540" / "bulk_TPM.csv.gz")]
    for section in all_sections():
        for file, url in section_file_urls(section).items():
            jobs.append((url, section_dir(section, raw_dir) / file))
    jobs += [(url, raw_dir / zip_path) for url, zip_path, _ in ARCHIVES]
    return jobs


def unzip(zip_path: Path, dest: Path) -> None:
    """Extract once; a `.unzipped` flag next to the archive marks completion."""
    done_flag = zip_path.parent / ".unzipped"
    if done_flag.exists():
        return
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest)
    done_flag.touch()


def download_all(raw_dir: Path) -> list[str]:
    """Logged, resumable driver over all downloads. Returns the failed URLs."""
    jobs = download_jobs(raw_dir)
    failed = []
    for i, (url, dest) in enumerate(jobs, 1):
        try:
            status = download(url, dest)
            log.info("[%d/%d] %s %s", i, len(jobs), status.upper(), dest.relative_to(raw_dir))
        except Exception:
            failed.append(url)
            log.error("[%d/%d] FAIL %s\n%s", i, len(jobs), url, traceback.format_exc())
    for url, zip_path, extract_dir in ARCHIVES:
        if url in failed:
            continue
        try:
            unzip(raw_dir / zip_path, raw_dir / extract_dir)
            log.info("extracted %s", zip_path)
        except Exception:
            failed.append("unzip:" + url)
            log.error("FAIL unzip %s\n%s", zip_path, traceback.format_exc())
    log.info("SUMMARY jobs=%d failed=%d", len(jobs), len(failed))
    if failed:
        log.error("FAILED: %s", failed)
    return failed


POSITION_COLUMNS = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
USZ_SECTIONS = ("KC1", "KC2", "KC3", "LC1", "LC2", "LC3", "LC4", "LC5")
USZ_H5AD_DIR = Path("usz/extracted/TLS_VISIUM_USZ/h5ad_preprocessed")
POSITION_RENAMES = {"pxl_row_in_fullres": "pxl_row", "pxl_col_in_fullres": "pxl_col"}


def read_positions(path: Path) -> pd.DataFrame:
    """Space Ranger tissue positions, with (>= 2.0: `barcode,...` header) or without a header row."""
    first_field = str(pd.read_csv(path, header=None, nrows=1).iloc[0, 0])
    if first_field == "barcode":
        return pd.read_csv(path).rename(columns=POSITION_RENAMES)[POSITION_COLUMNS]
    return pd.read_csv(path, header=None, names=POSITION_COLUMNS)


def annotation_table(section: Section, raw_dir: Path) -> pd.DataFrame:
    """Two columns (barcode, raw label) for one section: the GEO TLS_annotation file, or the Zenodo
    pathology annotation of GSE351843 (column `immune`: '' / lymphocytes_present / aggregate / tls)."""
    cohort = COHORT_BY_SOURCE[section.source]
    if "annotation" in cohort.files:
        path = section_dir(section, raw_dir) / cohort.files["annotation"]
    else:
        path = raw_dir / BREAST_ANNOTATION_DIR / f"{section.name}_pathology_annotations.csv"
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    # GEO label columns are named inconsistently (TLS_2_cat / TLS / TLS_annotation) -> take the second column.
    return table[[table.columns[0], cohort.annotation_column or table.columns[1]]]


# An annotation whose barcodes mostly miss the matrix (e.g. a missing "-1" suffix) would silently label every spot
# negative; below this share of annotation barcodes found in the matrix the join is refused.
MIN_ANNOTATION_MATCH = 0.5


def spot_obs(
    barcodes: pd.Index, positions: pd.DataFrame, annotation: pd.DataFrame, aggregate_is_tls: bool = True
) -> tuple[pd.DataFrame, np.ndarray]:
    """Join positions and the annotation (barcode, <label column>) onto the matrix barcodes."""
    labels = annotation.set_index(annotation.columns[0])[annotation.columns[1]]
    matched = labels.index.isin(barcodes).mean() if len(labels) else 1.0
    if matched < MIN_ANNOTATION_MATCH:
        raise ValueError(f"only {matched:.0%} of annotation barcodes are in the matrix barcodes; check the barcode format")
    pos = positions.set_index("barcode").loc[barcodes]
    raw = labels.reindex(barcodes).fillna("")
    obs = pd.DataFrame(
        {"array_row": pos["array_row"].to_numpy(), "array_col": pos["array_col"].to_numpy(), "label_raw": raw.to_numpy()},
        index=barcodes,
    )
    obs = pd.concat([obs, spot_labels(obs["label_raw"], aggregate_is_tls)], axis=1)
    spatial = pos[["pxl_col", "pxl_row"]].to_numpy()
    return obs, spatial


def _finish(adata: ad.AnnData, section: str, patient: str, protocol: str, source: str) -> ad.AnnData:
    adata.var_names_make_unique()
    adata.obs = adata.obs.assign(
        section=section, patient=patient, protocol=protocol, source=source, unit="spot",
        platform=f"visium_{protocol}",
    )
    adata.obs_names = [f"{section}:{b}" for b in adata.obs_names]
    adata.X = sp.csr_matrix(adata.X, dtype=np.float32)
    return adata


def load_geo_section(section: Section, raw_dir: Path, aggregate_is_tls: bool = True) -> ad.AnnData:
    import scanpy as sc
    from matplotlib.image import imread

    files = COHORT_BY_SOURCE[section.source].files
    folder = section_dir(section, raw_dir)
    adata = sc.read_10x_h5(folder / files["matrix"])
    positions = read_positions(folder / files["positions"])
    obs, spatial = spot_obs(adata.obs_names, positions, annotation_table(section, raw_dir), aggregate_is_tls)
    adata.obs = obs
    adata.obsm["spatial"] = spatial
    with gzip.open(folder / files["scalefactors"], "rt") as handle:
        scalefactors = json.load(handle)
    with gzip.open(folder / files["image"], "rb") as handle:
        image = imread(io.BytesIO(handle.read()), format="png")
    adata.uns["spatial"] = {section.section: {"images": {"lowres": image}, "scalefactors": scalefactors}}
    adata.uns["aggregate_is_tls"] = aggregate_is_tls
    adata = _finish(adata, section.section, section.patient, section.protocol, section.source)
    adata.obs["bulk_id"] = section.bulk_id or ""
    return adata


def load_usz_section(name: str, raw_dir: Path) -> ad.AnnData:
    """USZ (Zenodo 14620362): raw counts in X, labels in obs['manual_anno_tls']."""
    source = ad.read_h5ad(raw_dir / USZ_H5AD_DIR / f"{name}.h5ad")
    obs = pd.DataFrame(
        {
            # USZ x_array / y_array = Visium array_row / array_col
            "array_row": source.obs["x_array"].to_numpy(),
            "array_col": source.obs["y_array"].to_numpy(),
            "label_raw": source.obs["manual_anno_tls"].astype(object).fillna("").astype(str).to_numpy(),
        },
        index=source.obs_names,
    )
    obs = pd.concat([obs, spot_labels(obs["label_raw"])], axis=1)
    adata = ad.AnnData(X=source.X, obs=obs, var=pd.DataFrame(index=source.var_names))
    adata.obsm["spatial"] = np.asarray(source.obsm["spatial"])
    adata = _finish(adata, f"USZ_{name}", f"USZ_{name}", "ffpe", "USZ")
    adata.obs["bulk_id"] = ""
    return adata


AGGREGATE_SOURCE = "GSE351843"  # the only cohort whose labels depend on `aggregate_is_tls`


def needs_build(dest: Path, source: str, aggregate_is_tls: bool) -> bool:
    """Missing, or a breast section built with the other `aggregate` mapping (files without the flag were built
    with the default, True)."""
    if not dest.exists():
        return True
    if source != AGGREGATE_SOURCE:
        return False
    built = ad.read_h5ad(dest, backed="r")
    flag = bool(built.uns.get("aggregate_is_tls", True))
    built.file.close()
    return flag != aggregate_is_tls


def build_h5ad(raw_dir: Path, out_dir: Path, aggregate_is_tls: bool = True) -> list[str]:
    """One h5ad per section (idempotent, logged). Returns failed section ids."""
    out_dir.mkdir(parents=True, exist_ok=True)
    loaders = [(s.section, s.source, lambda s=s: load_geo_section(s, raw_dir, aggregate_is_tls)) for s in all_sections()]
    loaders += [(f"USZ_{n}", "USZ", lambda n=n: load_usz_section(n, raw_dir)) for n in USZ_SECTIONS]
    failed = []
    for i, (name, source, load) in enumerate(loaders, 1):
        dest = out_dir / f"{name}.h5ad"
        if not needs_build(dest, source, aggregate_is_tls):
            log.info("[%d/%d] SKIP %s", i, len(loaders), name)
            continue
        try:
            adata = load()
            adata.write_h5ad(dest, compression="gzip")
            log.info("[%d/%d] OK %s spots=%d tls=%d", i, len(loaders), name, adata.n_obs, adata.obs["tls"].sum())
        except Exception:
            failed.append(name)
            log.error("[%d/%d] FAIL %s\n%s", i, len(loaders), name, traceback.format_exc())
    return failed


def load_gse_bulk(raw_dir: Path) -> pd.DataFrame:
    """GSE175540 bulk (linear TPM, genes × samples) -> samples × genes log2(TPM + 1).

    Some symbols occur on several rows (different gene ids); their TPMs are summed before the log.
    """
    tpm = pd.read_csv(raw_dir / "gse175540" / "bulk_TPM.csv.gz", sep=";", index_col=0)
    tpm = tpm.groupby(level=0, sort=False).sum()
    return np.log2(tpm.T + 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/visium/raw"))
    parser.add_argument("--h5ad", type=Path, default=Path("data/visium/h5ad"))
    parser.add_argument("--aggregate-is-tls", type=lambda v: v.lower() in ("1", "true", "yes"), default=True,
                        help="GSE351843 'aggregate' spots = immature TLS (default) or excluded")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.out / "download.log"), logging.StreamHandler(sys.stdout)],
    )
    failed = download_all(args.out)
    failed += build_h5ad(args.out, args.h5ad, args.aggregate_is_tls)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
