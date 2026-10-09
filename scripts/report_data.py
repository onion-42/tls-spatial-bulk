"""Per-sample inputs of the interactive report figures -> data/private/report/ (gitignored).

roc.json  : ROC curves (thinned) of the final model and the baseline signatures on pathologist TLS labels,
            per held-out / OOD cohort and for Visium test spots. No sample ids.
umap.tsv  : UMAP of ICI-treated samples (log2 TPM+1, top-variance genes -> PCA -> UMAP) with diagnosis, cohort,
            response and scores. No sample ids.
Usage: uv run python scripts/report_data.py   (from the repo root)
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve

from tls_model import ici_report, io_bulk, io_ici, pipeline, prepare

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
log = logging.getLogger("report_data")

ROOT = Path.cwd()
OUT = ROOT / "data" / "private" / "report"
MAX_POINTS = 300
N_GENES_UMAP, N_PCS = 2000, 50
CHEM = ici_report.CHEM


def thin_roc(y: np.ndarray, s: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=True)
    if len(fpr) > MAX_POINTS:
        keep = np.unique(np.r_[np.linspace(0, len(fpr) - 1, MAX_POINTS).astype(int), len(fpr) - 1])
        fpr, tpr = fpr[keep], tpr[keep]
    return {"fpr": np.round(fpr, 4).tolist(), "tpr": np.round(tpr, 4).tolist(), "auc": float(roc_auc_score(y, s))}


def roc_data(cfg: dict, paths: dict) -> list[dict]:
    units, meta = prepare.load_units(cfg, ROOT)
    obs = units.obs
    with open(ROOT / "models" / "tls_model.pkl", "rb") as handle:
        saved = pickle.load(handle)
    variant = saved["variant"]
    B, names = pipeline.feature_matrix(units, meta, variant["representation"], bool(variant["offset"]))
    p_tls = pipeline.predict(B, names, obs, saved["models"]["any"], saved["calibrators"]["any"])
    del B
    log.info("final model scored on %d units", len(obs))

    sigs = io_ici.baseline_signatures(io_bulk.load_signatures(paths["bulk_public"]))
    bulk = ad.read_h5ad(ROOT / "data" / "processed" / "bulk.h5ad")
    bulk_expr = pd.DataFrame(np.asarray(bulk.X, dtype=np.float32), index=bulk.obs_names, columns=bulk.var_names.astype(str))
    log.info("bulk expression max %.1f (log2 scale expected)", float(bulk_expr.to_numpy().max()))

    curves = []
    for split in ("test", "ood"):
        scored = pipeline.scored_rows(obs, "any", split)
        is_bulk = (obs["kind"] == "bulk").to_numpy()
        blocks = [(c, scored & is_bulk & (obs["cohort"] == c).to_numpy()) for c in sorted(obs.loc[scored & is_bulk, "cohort"].unique())]
        blocks.append(("all bulk", scored & is_bulk))
        for cohort, mask in blocks:
            y = obs["tls_any"].to_numpy(dtype=float)[mask]
            if mask.sum() < 5 or len(np.unique(y)) < 2:
                log.info("skip %s %s (n=%d)", split, cohort, mask.sum())
                continue
            ids = obs.index[mask]
            scores = {ici_report.OURS: p_tls[mask]}
            scores.update(io_ici.score_signatures(bulk_expr.loc[ids], sigs).to_dict("series"))
            for name, s in scores.items():
                curves.append({"split": split, "block": f"bulk · {cohort}", "score": name, "n": int(mask.sum()),
                               "n_pos": int(y.sum()), **thin_roc(y, np.asarray(s, dtype=float))})
            log.info("ROC %s %s n=%d", split, cohort, mask.sum())

    spots = pipeline.scored_rows(obs, "any", "test") & (obs["unit"] == "spot").to_numpy()
    y = obs["tls_any"].to_numpy(dtype=float)[spots]
    spot_expr = pd.DataFrame(np.asarray(units.X[spots], dtype=np.float32), columns=units.var_names.astype(str))
    scores = {ici_report.OURS: p_tls[spots]}
    scores.update(io_ici.score_signatures(spot_expr, sigs).to_dict("series"))  # panel genes only on spots
    for name, s in scores.items():
        curves.append({"split": "test", "block": "Visium · spots", "score": name, "n": int(spots.sum()),
                       "n_pos": int(y.sum()), **thin_roc(y, np.asarray(s, dtype=float))})
    log.info("ROC Visium spots n=%d", spots.sum())
    return curves


def umap_data(cfg: dict) -> pd.DataFrame:
    import umap

    ici = cfg["ici"]
    expr, clin = io_ici.load_cohorts(ROOT / ici["dir"], ici["tmb_count_scale"])
    scores = pd.read_csv(ROOT / ici["dir"] / "scores.tsv", sep="\t", index_col=0)
    keep = clin["is_ici"].astype(bool) & clin["diagnosis"].notna()
    clin, expr = clin[keep], expr.loc[keep[keep].index]
    top = expr.var().nlargest(N_GENES_UMAP).index
    X = expr[top].to_numpy()
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    pcs = PCA(N_PCS, random_state=0).fit_transform(X)
    xy = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(pcs)
    response = clin["response"].map({1.0: "CR/PR", 0.0: "SD/PD"}).fillna("not available")
    out = pd.DataFrame({"x": xy[:, 0].round(3), "y": xy[:, 1].round(3), "diagnosis": clin["diagnosis"].to_numpy(),
                        "cohort": clin["cohort"].to_numpy(), "rnd_use": clin["rnd_use"].fillna("unknown").to_numpy(),
                        "response": response.to_numpy(),
                        "p_tls": scores.loc[clin.index, "p_tls"].round(4).to_numpy(),
                        "chem": scores.loc[clin.index, CHEM].round(4).to_numpy()})
    log.info("UMAP on %d ICI samples, %d genes", len(out), len(top))
    return out.sample(frac=1.0, random_state=0).reset_index(drop=True)  # shuffle: row order carries no identity


def main() -> None:
    cfg = pipeline.load_config()
    paths = {k: ROOT / v for k, v in cfg["paths"].items()}
    OUT.mkdir(parents=True, exist_ok=True)
    roc_path, umap_path = OUT / "roc.json", OUT / "umap.tsv"
    if roc_path.exists():
        log.info("SKIP roc (cached)")
    else:
        roc_path.write_text(json.dumps(roc_data(cfg, paths)), encoding="utf-8")
    if umap_path.exists():
        log.info("SKIP umap (cached)")
    else:
        umap_data(cfg).to_csv(umap_path, sep="\t", index=False)
    log.info("done -> %s", OUT)


if __name__ == "__main__":
    main()
