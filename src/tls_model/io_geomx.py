"""GeoMx DSP region-of-interest (ROI / AOI) datasets as `roi` units: labelled mini-bulks between a spot and a sample.

GSE271255 (glioma, 81 ROIs, CTA-like 7,157-gene panel): `Sample_type` TLS vs control (tumour region without a TLS).
GSE276935 (HGSOC, WTA, 566 AOIs on 25 slides): no explicit TLS label; the segment of each ROI gives the mapping
CD20 (B-cell follicle) = TLS+, Immune / Stroma / Tumor = negative (Immune = hard negative: T cells without a
follicle), Other = excluded, Full ROI = unlabelled (scored, never evaluated). The lymph-node slide is excluded like
LN spots in Visium. Patient = the leading digits of the slide name (one patient has up to three sites). Neither
series gives a germinal-centre call, so maturity is unknown everywhere.

Expression: raw counts (GSE271255) or the authors' Q3-normalised target counts (GSE276935); both are turned into
log2-CPM on the model genes, genes absent from the panel = 0 (not detected), the coverage share is reported.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from tls_model.pseudobulk import log2_cpm

log = logging.getLogger(__name__)

SOURCES = ("GSE271255", "GSE276935")
FILES = {
    "GSE271255": ("GSE271255_counts.csv.gz", "GSE271255_metadata.csv.gz"),
    "GSE276935": ("GSE276935_Processed_Data_File_All_Experiments.xlsx",),
}
# GSE276935 segment label -> TLS label (NaN = unlabelled); labels not listed here are excluded.
SEGMENT_TLS = {"CD20": 1.0, "Immune": 0.0, "Stroma": 0.0, "Tumor": 0.0, "Full ROI": np.nan}
LYMPH_NODE_SLIDE = "LNLymph Node"
SEGMENT_NAME = re.compile(r"^(?P<slide>.+?)(?P<roi>\d{3})(?P<segment>CD20|Immune|Stroma|Tumor|Full ROI|Other)$")
OBS_COLUMNS = ["roi", "source", "group", "patient", "annotation", "tls", "tls_mature", "excluded", "qc_flagged"]


def _adata(source: str, X: np.ndarray, obs: pd.DataFrame, genes: pd.Index) -> ad.AnnData:
    obs = obs.assign(source=source, tls_mature=np.nan)[OBS_COLUMNS]
    obs.index = [f"{source}:roi:{roi}" for roi in obs["roi"]]
    return ad.AnnData(X=np.asarray(X, dtype=np.float32), obs=obs, var=pd.DataFrame(index=genes.astype(str)))


def gse271255_units(counts: pd.DataFrame, metadata: pd.DataFrame) -> ad.AnnData:
    """counts: genes × ROIs (raw), metadata: one row per ROI (Sample_name, Patient, Sample_type)."""
    meta = metadata.set_index("Sample_name").loc[counts.columns]
    obs = pd.DataFrame({"roi": counts.columns, "group": "GSE271255", "patient": meta["Patient"].to_numpy(),
                        "annotation": meta["Sample_type"].to_numpy(),
                        "tls": (meta["Sample_type"] == "TLS").astype(float).to_numpy(),
                        "excluded": False, "qc_flagged": False})
    return _adata("GSE271255", counts.to_numpy().T, obs, counts.index)


def gse276935_units(counts: pd.DataFrame, segments: pd.DataFrame) -> ad.AnnData:
    """counts: genes × AOIs (Q3-normalised), segments: SegmentProperties sheet (TargetName, SegmentLabel, QCFlags)."""
    seg = segments.set_index("TargetName").loc[counts.columns]
    parts = seg.index.to_series().str.extract(SEGMENT_NAME)
    if parts["slide"].isna().any():
        raise ValueError(f"unparsable AOI names: {list(parts.index[parts['slide'].isna()])[:5]}")
    label = seg["SegmentLabel"].astype(str)
    obs = pd.DataFrame({"roi": counts.columns, "group": parts["slide"].to_numpy(),
                        "patient": parts["slide"].str.extract(r"^(\d+)")[0].fillna(parts["slide"]).to_numpy(),
                        "annotation": label.to_numpy(), "tls": label.map(SEGMENT_TLS).astype(float).to_numpy(),
                        "excluded": (~label.isin(SEGMENT_TLS) | (parts["slide"] == LYMPH_NODE_SLIDE)).to_numpy(),
                        "qc_flagged": seg["QCFlags"].fillna("").astype(str).str.strip().ne("").to_numpy()})
    return _adata("GSE276935", counts.to_numpy().T, obs, counts.index)


def load_gse271255(geomx_dir: Path) -> ad.AnnData:
    counts_file, meta_file = FILES["GSE271255"]
    counts = pd.read_csv(geomx_dir / counts_file, index_col=0)
    return gse271255_units(counts, pd.read_csv(geomx_dir / meta_file))


def load_gse276935(geomx_dir: Path) -> ad.AnnData:
    (xlsx,) = FILES["GSE276935"]
    segments = pd.read_excel(geomx_dir / xlsx, sheet_name="SegmentProperties")
    counts = pd.read_excel(geomx_dir / xlsx, sheet_name="TargetCountMatrix", index_col=0)
    return gse276935_units(counts, segments)


LOADERS = {"GSE271255": load_gse271255, "GSE276935": load_gse276935}


def load_geomx(geomx_dir: Path, sources: list[str]) -> dict[str, ad.AnnData]:
    """{source: ROIs × genes} for the requested series whose files are present (missing ones are skipped)."""
    out = {}
    for source in sources:
        if not all((Path(geomx_dir) / name).exists() for name in FILES[source]):
            log.warning("GeoMx %s not found in %s (scripts/fetch_geomx.py); skipped", source, geomx_dir)
            continue
        out[source] = LOADERS[source](Path(geomx_dir))
        obs = out[source].obs
        log.info("GeoMx %s: %d ROIs, %d patients, TLS+ %d, TLS- %d, unlabelled %d, excluded %d, QC-flagged %d",
                 source, len(obs), obs["patient"].nunique(), int((obs["tls"] == 1).sum()), int((obs["tls"] == 0).sum()),
                 int(obs["tls"].isna().sum()), int(obs["excluded"].sum()), int(obs["qc_flagged"].sum()))
    return out


def roi_log2cpm(adata: ad.AnnData, genes: list[str]) -> tuple[np.ndarray, float]:
    """log2-CPM of the ROIs on `genes` (CPM over the panel's total counts); genes absent from the panel = 0.
    Returns (ROIs × genes, share of `genes` present in the panel)."""
    idx = adata.var_names.get_indexer(genes)
    present = idx >= 0
    values = np.zeros((adata.n_obs, len(genes)), dtype=np.float32)
    values[:, present] = log2_cpm(np.asarray(adata.X))[:, idx[present]]
    return values, float(present.mean())


def qc_table(adatas: dict[str, ad.AnnData], coverage: dict[str, float]) -> pd.DataFrame:
    """One row per series: ROIs, patients, label counts and the model-gene coverage share."""
    rows = []
    for source, adata in adatas.items():
        obs = adata.obs
        kept = obs[~obs["excluded"]]
        rows.append({"source": source, "rois": len(kept), "patients": kept["patient"].nunique(),
                     "slides": kept["group"].nunique(), "tls_pos": int((kept["tls"] == 1).sum()),
                     "tls_neg": int((kept["tls"] == 0).sum()), "unlabelled": int(kept["tls"].isna().sum()),
                     "excluded": int(obs["excluded"].sum()), "qc_flagged": int(kept["qc_flagged"].sum()),
                     "panel_genes": adata.n_vars, "model_gene_coverage": coverage.get(source, np.nan)})
    return pd.DataFrame(rows)
