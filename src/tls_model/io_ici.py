"""Treated (immunotherapy) cohorts: load, harmonise clinical annotation, score TLS model + baseline signatures.

Input: one h5ad per study in <ici_dir> (X = TPM, obs = clinical annotation, var = HUGO genes), local only.
Output of `load_cohorts`: log2(TPM + 1) samples × genes and one harmonised clinical row per sample.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from tls_model.features import FeatureSpace, ssgsea

log = logging.getLogger(__name__)

DIAGNOSIS = {
    "Skin Cutaneous Melanoma": "SKCM",
    "Melanoma": "SKCM",
    "Non-Small Cell Lung Cancer": "NSCLC",
    "Non Small Cell Lung Carcinoma": "NSCLC",
    "Lung Adenocarcinoma": "NSCLC",
    "Lung Squamous Cell Carcinoma": "NSCLC",
    "Clear Cell Renal Cell Carcinoma": "ccRCC",
    "Urinary Bladder Urothelial Carcinoma": "BLCA",
    "Colorectal Adenocarcinoma": "COREAD",
    "Esophageal Adenocarcinoma": "ESGA",
}  # anything else (non-clear-cell RCC, PEComa, ...) -> None = excluded
RECIST_CONTROL = {"CR": 1.0, "PR": 1.0, "SD": 1.0, "PD": 0.0}
RECIST_RESPONSE = {"CR": 1.0, "PR": 1.0, "SD": 0.0, "PD": 0.0}  # objective response (ORR)
RESPONSE = {"R": 1.0, "NR": 0.0}  # study-annotated; definitions differ between studies (e.g. SD counted as R)
DCR_COLUMN = {"DCR": 1.0, "PD": 0.0}
PDL1_TC = {"TC0": 0.0, "TC1": 1.0, "TC2+": 2.0}  # tumour-cell PD-L1 IHC category, as ordinal
SITE_ALIASES = {"Metastatic": "Metastasis", "melanoma metastases": "Metastasis"}
TMB_HIGH_CUTOFF = 10.0  # mutations / Mb (clinical cut-off)

# Baseline TLS signatures (gene lists as published).
TLS_IMPRINT = [  # Meylan et al., Immunity 2022 (29 genes, as listed in Nat Commun 2025, PMC12059173)
    "IGHA1", "IGHG1", "IGHG2", "IGHG3", "IGHG4", "IGHGP", "IGHM", "IGKC", "IGLC1", "IGLC2", "IGLC3", "JCHAIN",
    "CD79A", "FCRL5", "MZB1", "SSR4", "XBP1", "TRBC2", "IL7R", "CXCL12", "LUM", "C1QA", "C7", "CD52", "APOE",
    "PTLP", "PTGDS", "PIM2", "DERL3",
]
BASELINE_FROM_REPO = {"Cabrita (TLS_SKCM)": "TLS_SKCM", "12-chemokine (TLS_Chem)": "TLS_Chem"}


def _column(obs: pd.DataFrame, name: str) -> pd.Series:
    if name in obs.columns:
        return obs[name].astype(object).where(obs[name].notna(), None)
    return pd.Series([None] * len(obs), index=obs.index, dtype=object)


def _numeric(obs: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(_column(obs, name), errors="coerce")


def icb_class(regimen: str | None) -> str | None:
    """'anti-PD1 -> anti-CTLA4' etc. -> a coarse checkpoint class; None if the regimen has no checkpoint inhibitor."""
    if not regimen:
        return None
    has_pd, has_ctla = "anti-PD" in regimen, "anti-CTLA4" in regimen
    if not (has_pd or has_ctla):
        return None
    if "->" in regimen:
        return "sequential anti-CTLA4/anti-PD(L)1"
    base = "anti-CTLA4+anti-PD(L)1" if has_pd and has_ctla else ("anti-PD(L)1" if has_pd else "anti-CTLA4")
    if "Chemo" in regimen:
        base += "+chemo"
    if "PanTK" in regimen or "VEGFR" in regimen:
        base += "+antiangiogenic"
    return base


def _stage(value) -> str | None:
    match = re.match(r"^\s*(IV|III|II|I)", str(value)) if value is not None else None
    return match.group(1) if match else None


def harmonise_clinical(obs: pd.DataFrame, dataset: str, tmb_per_mb: bool = True) -> pd.DataFrame:
    """One row per sample with harmonised endpoints and covariates; NaN / None where a study has no data.

    `tmb_per_mb=False`: the study reports TMB as a mutation count -> no TMB-high call (cut-off is per Mb)."""
    tmb = _numeric(obs, "TMB")
    tmb_high = (tmb >= TMB_HIGH_CUTOFF).astype(float).where(tmb.notna() & tmb_per_mb)
    sample = _column(obs, "Sample_ID").fillna(pd.Series(obs.index.astype(str), index=obs.index))
    regimen = _column(obs, "F_Therapy_Eval")
    classes = regimen.map(icb_class)
    recist_raw = _column(obs, "Recist")
    dcr = recist_raw.map(RECIST_CONTROL).fillna(_column(obs, "DCR").map(DCR_COLUMN)).astype(float)
    orr = recist_raw.map(RECIST_RESPONSE).astype(float)
    annotated = _column(obs, "Response").map(RESPONSE).astype(float)
    response_source = pd.Series(np.where(orr.notna(), "RECIST", np.where(annotated.notna(), "annotated R/NR", None)),
                                index=obs.index, dtype=object)
    out = pd.DataFrame(
        {
            "sample": sample.astype(str),
            "patient": _column(obs, "Patient").fillna(sample).astype(str),
            "dataset": dataset,
            "cohort": _column(obs, "Cohort").fillna(dataset).astype(str),
            "histology": _column(obs, "Diagnosis_long"),
            "diagnosis": _column(obs, "Diagnosis_long").map(DIAGNOSIS),
            "regimen": regimen,
            "icb_class": classes,
            "is_ici": classes.notna(),
            "rnd_use": _column(obs, "RND_USE"),
            "os": _numeric(obs, "OS"),
            "os_event": _numeric(obs, "OS_FLAG"),
            "pfs": _numeric(obs, "PFS"),
            "pfs_event": _numeric(obs, "PFS_FLAG"),
            "response": orr.fillna(annotated),  # ORR from RECIST; study-annotated R/NR only without RECIST
            "response_source": response_source,
            "dcr": dcr,
            "age": _numeric(obs, "Age"),
            "sex": _column(obs, "Gender"),
            "stage": _column(obs, "Stage").map(_stage),
            "site": _column(obs, "Site").replace(SITE_ALIASES),
            "location": _column(obs, "Location"),
            "tmb": tmb,
            "tmb_high": tmb_high,
            "pdl1_tc": _column(obs, "PDL1_TC_IHC_cat").map(PDL1_TC).astype(float),
        }
    )
    out = out.astype({"diagnosis": object, "icb_class": object})
    out["diagnosis"] = out["diagnosis"].where(out["diagnosis"].notna(), None)
    out.index = dataset + ":" + out["sample"]
    return out


def dedup_samples(clin: pd.DataFrame) -> pd.DataFrame:
    """The same sample can sit in several study files: keep its first occurrence."""
    return clin[~clin["sample"].duplicated(keep="first")]


def load_cohorts(ici_dir: Path, tmb_count_scale=()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All <ici_dir>/*.h5ad -> (log2(TPM + 1) samples × shared genes, harmonised clinical table), deduplicated.
    `tmb_count_scale`: file stems whose TMB is a mutation count, not per Mb (from config)."""
    expr, clin = [], []
    for path in sorted(Path(ici_dir).glob("*.h5ad")):
        adata = ad.read_h5ad(path)
        table = harmonise_clinical(adata.obs, dataset=path.stem, tmb_per_mb=path.stem not in set(tmb_count_scale))
        values = np.log2(np.asarray(adata.X, dtype=np.float32) + 1.0)
        expr.append(pd.DataFrame(values, index=table.index, columns=adata.var_names.astype(str)))
        clin.append(table)
        log.info("%s: %d samples, %d with ICI", path.stem, len(table), int(table["is_ici"].sum()))
    if not clin:
        raise FileNotFoundError(f"no *.h5ad in {ici_dir}")
    genes = sorted(set.intersection(*(set(e.columns) for e in expr)))
    clin_all = dedup_samples(pd.concat(clin))
    expr_all = pd.concat([e[genes] for e in expr]).loc[clin_all.index]
    log.info("total %d samples (%d dropped as duplicates), %d genes", len(clin_all), sum(map(len, clin)) - len(clin_all), len(genes))
    return expr_all, clin_all


def score_signatures(log_tpm: pd.DataFrame, signatures: dict[str, list[str]]) -> pd.DataFrame:
    """ssGSEA (same implementation as the model features) over all genes of the matrix."""
    values = ssgsea(log_tpm.to_numpy(dtype=np.float32), list(log_tpm.columns), signatures)
    return pd.DataFrame(values, index=log_tpm.index, columns=list(signatures))


def baseline_signatures(repo_signatures: dict[str, list[str]]) -> dict[str, list[str]]:
    out = {label: repo_signatures[name] for label, name in BASELINE_FROM_REPO.items()}
    out["TLS imprint (Meylan 2022)"] = TLS_IMPRINT
    return out


def score_tls(log_tpm: pd.DataFrame, meta: dict, saved: dict) -> pd.Series:
    """Calibrated P(TLS) of the frozen final model (head A) for bulk samples."""
    genes = list(meta["genes"])
    missing = [g for g in genes if g not in log_tpm.columns]
    if missing:
        log.warning("%d / %d model genes missing in the ICI matrix -> set to 0: %s", len(missing), len(genes), missing[:10])
    X = log_tpm.reindex(columns=genes).fillna(0.0).to_numpy(dtype=np.float32)
    space = FeatureSpace(
        genes=tuple(genes),
        signatures=tuple((name, tuple(members)) for name, members in meta["signatures"].items()),
        representation=saved["variant"]["representation"],
    )
    B, names = space.base(X)
    p = saved["models"]["any"].predict_proba(B, names)
    p = saved["calibrators"]["any"].transform(p, np.full(len(p), "bulk"))
    return pd.Series(p, index=log_tpm.index, name="p_tls")
