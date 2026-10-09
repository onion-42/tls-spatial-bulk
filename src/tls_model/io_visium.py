"""Public Visium data: idempotent download (GEO GSE175540 + Zenodo USZ) and loading to AnnData.

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
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from tls_model.labels import spot_labels

log = logging.getLogger(__name__)

GEO_SAMPLE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5924nnn/{gsm}/suppl/{gsm}_{name}_{file}"
GSE_BULK_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE175nnn/GSE175540/suppl/"
    "GSE175540_batch_corrected_bulk_RNA_seq_TPM_25_02_22.csv.gz"
)
USZ_ZIP_URL = "https://zenodo.org/records/14620362/files/TLS_VISIUM_USZ.zip"

GSE_FIRST_GSM = 5924030
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
GSE_FILES = (
    "filtered_feature_bc_matrix.h5",
    "tissue_positions_list.csv.gz",
    "scalefactors_json.json.gz",
    "tissue_lowres_image.png.gz",
    "TLS_annotation.csv.gz",
)


@dataclass(frozen=True)
class Section:
    name: str
    gsm: str
    protocol: str  # "ffpe" (probe-based) or "frozen" (poly-A)
    patient: str
    bulk_id: str | None


def _parse_gse_name(name: str) -> tuple[str, str]:
    """'frozen_a_15' -> ('frozen', 'a_15')."""
    protocol, series, number = name.split("_")
    return protocol, f"{series}_{number}"


def gse_sections() -> list[Section]:
    """All annotated GSE175540 Visium sections."""
    sections = []
    for index, name in enumerate(GSE_SECTION_ORDER):
        if name in GSE_UNANNOTATED:
            continue
        protocol, sample = _parse_gse_name(name)
        sections.append(
            Section(
                name=name,
                gsm=f"GSM{GSE_FIRST_GSM + index}",
                protocol=protocol,
                patient=f"GSE175540_{sample}",
                bulk_id=sample if protocol == "frozen" and sample in GSE_BULK_PAIRS else None,
            )
        )
    return sections


def gse_file_urls(section: Section) -> dict[str, str]:
    return {f: GEO_SAMPLE_URL.format(gsm=section.gsm, name=section.name, file=f) for f in GSE_FILES}


def _remote_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        length = response.headers.get("Content-Length")
    return int(length) if length else None


def download(url: str, dest: Path) -> str:
    """Download `url` to `dest` unless a complete copy exists. Returns 'skip' or 'ok'."""
    expected = _remote_size(url)
    if dest.exists() and (expected is None or dest.stat().st_size == expected):
        return "skip"
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
    for section in gse_sections():
        for file, url in gse_file_urls(section).items():
            jobs.append((url, raw_dir / "gse175540" / section.name / file))
    jobs.append((USZ_ZIP_URL, raw_dir / "usz" / "TLS_VISIUM_USZ.zip"))
    return jobs


def unzip_usz(raw_dir: Path) -> None:
    import zipfile

    zip_path = raw_dir / "usz" / "TLS_VISIUM_USZ.zip"
    done_flag = raw_dir / "usz" / ".unzipped"
    if done_flag.exists():
        return
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(raw_dir / "usz" / "extracted")
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
    if USZ_ZIP_URL not in failed:
        try:
            unzip_usz(raw_dir)
            log.info("USZ zip extracted")
        except Exception:
            failed.append("unzip:" + USZ_ZIP_URL)
            log.error("FAIL unzip USZ\n%s", traceback.format_exc())
    log.info("SUMMARY jobs=%d failed=%d", len(jobs), len(failed))
    if failed:
        log.error("FAILED: %s", failed)
    return failed


POSITION_COLUMNS = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
USZ_SECTIONS = ("KC1", "KC2", "KC3", "LC1", "LC2", "LC3", "LC4", "LC5")
USZ_H5AD_DIR = Path("usz/extracted/TLS_VISIUM_USZ/h5ad_preprocessed")


def spot_obs(
    barcodes: pd.Index, positions: pd.DataFrame, annotation: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """Join positions and the GSE annotation (Barcode, <label column>) onto the matrix barcodes."""
    pos = positions.set_index("barcode").loc[barcodes]
    raw = annotation.set_index(annotation.columns[0])[annotation.columns[1]].reindex(barcodes).fillna("")
    obs = pd.DataFrame(
        {"array_row": pos["array_row"].to_numpy(), "array_col": pos["array_col"].to_numpy(), "label_raw": raw.to_numpy()},
        index=barcodes,
    )
    obs = pd.concat([obs, spot_labels(obs["label_raw"])], axis=1)
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


def load_gse_section(section: Section, raw_dir: Path) -> ad.AnnData:
    import scanpy as sc
    from matplotlib.image import imread

    folder = raw_dir / "gse175540" / section.name
    adata = sc.read_10x_h5(folder / "filtered_feature_bc_matrix.h5")
    positions = pd.read_csv(folder / "tissue_positions_list.csv.gz", header=None, names=POSITION_COLUMNS)
    annotation = pd.read_csv(folder / "TLS_annotation.csv.gz", dtype=str, keep_default_na=False)
    obs, spatial = spot_obs(adata.obs_names, positions, annotation)
    adata.obs = obs
    adata.obsm["spatial"] = spatial
    with gzip.open(folder / "scalefactors_json.json.gz", "rt") as handle:
        scalefactors = json.load(handle)
    with gzip.open(folder / "tissue_lowres_image.png.gz", "rb") as handle:
        image = imread(io.BytesIO(handle.read()), format="png")
    adata.uns["spatial"] = {section.name: {"images": {"lowres": image}, "scalefactors": scalefactors}}
    adata = _finish(adata, section.name, section.patient, section.protocol, "GSE175540")
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


def build_h5ad(raw_dir: Path, out_dir: Path) -> list[str]:
    """One h5ad per section (idempotent, logged). Returns failed section names."""
    out_dir.mkdir(parents=True, exist_ok=True)
    loaders = [(s.name, lambda s=s: load_gse_section(s, raw_dir)) for s in gse_sections()]
    loaders += [(f"USZ_{n}", lambda n=n: load_usz_section(n, raw_dir)) for n in USZ_SECTIONS]
    failed = []
    for i, (name, load) in enumerate(loaders, 1):
        dest = out_dir / f"{name}.h5ad"
        if dest.exists():
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
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.out / "download.log"), logging.StreamHandler(sys.stdout)],
    )
    failed = download_all(args.out)
    failed += build_h5ad(args.out, args.h5ad)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
