"""Build the unit table (bulk + Visium) once: splits, label-free gene filter, platform offset, aggregates.

Output: <processed>/units.h5ad (X = log2 values on kept genes, obs = unit metadata) and <processed>/meta.json.
Everything that is fitted here (gene filter, offset) uses TRAIN data only and no labels.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from tls_model import io_bulk, io_visium
from tls_model.features import gene_filter, platform_offset
from tls_model.pseudobulk import build_units, log2_cpm, spot_prevalence
from tls_model.splits import split_bulk, split_visium

log = logging.getLogger(__name__)

UNIT_COLUMNS = [
    "kind", "unit", "k", "group", "cohort", "section", "patient", "split", "platform", "source",
    "label_source", "tls_any", "tls_mature", "tls_fraction",
]
PREV_PANEL_FILE = "Model_genes.tsv"


def _paths(cfg: dict, root: Path) -> dict[str, Path]:
    return {k: root / v for k, v in cfg["paths"].items()}


def _private_dir(paths: dict[str, Path]) -> Path | None:
    private = paths["bulk_private"]
    return private if (private / io_bulk.LABELS_PRIMARY).exists() else None


def section_table(h5ad_dir: Path) -> pd.DataFrame:
    """One row per section from obs only (backed mode, X is not loaded)."""
    rows = []
    for path in sorted(h5ad_dir.glob("*.h5ad")):
        adata = ad.read_h5ad(path, backed="r")
        obs = adata.obs
        rows.append({"section": obs["section"].iloc[0], "patient": obs["patient"].iloc[0],
                     "source": obs["source"].iloc[0], "protocol": obs["protocol"].iloc[0],
                     "has_tls": int(obs["tls"].max()), "path": path, "genes": set(adata.var_names)})
        adata.file.close()
    return pd.DataFrame(rows)


def _bulk_units(bulk: ad.AnnData, genes: list[str]) -> ad.AnnData:
    obs = bulk.obs
    unit_obs = pd.DataFrame(
        {
            "kind": "bulk", "unit": "bulk", "k": 0, "group": obs["cohort"], "cohort": obs["cohort"], "section": "",
            "patient": obs["patient"], "split": obs["split"], "platform": "bulk", "source": obs["cohort"],
            "label_source": obs["label_source"], "tls_any": obs["tls_any"].astype(float),
            "tls_mature": obs["tls_mature"].astype(float), "tls_fraction": np.nan,
        },
        index=obs.index,
    )
    return ad.AnnData(X=np.asarray(bulk[:, genes].X, dtype=np.float32), obs=unit_obs)


def _visium_units(section: ad.AnnData, split: str, genes: list[str], cfg: dict, rng) -> ad.AnnData:
    if split == "train":
        units = build_units(section, ks=tuple(cfg["units"]["ks"]), cap=cfg["units"]["cap_per_scale"],
                            max_positive=cfg["units"]["max_positive_per_scale"], rng=rng)
    else:
        units = build_units(section, ks=(1,), cap=None, max_positive=None, rng=rng)
    gene_idx = section.var_names.get_indexer(genes)
    values = log2_cpm(units.X)[:, gene_idx]
    obs = units.obs
    unit_obs = pd.DataFrame(
        {
            "kind": "visium", "unit": obs["unit"], "k": obs["k"], "group": obs["section"], "cohort": "",
            "section": obs["section"], "patient": obs["patient"], "split": split, "platform": obs["platform"],
            "source": obs["source"], "label_source": "pathologist", "tls_any": obs["tls"].astype(float),
            "tls_mature": obs["tls_mature"].astype(float), "tls_fraction": obs["tls_fraction"],
        },
        index=obs.index,
    )
    return ad.AnnData(X=values.astype(np.float32), obs=unit_obs)


def _spot_detection(sections: pd.DataFrame, genes: list[str]) -> np.ndarray:
    """Share of train spots with count > 0, per gene."""
    detected, n_spots = np.zeros(len(genes)), 0
    for row in sections.itertuples():
        adata = ad.read_h5ad(row.path)
        sub = adata[:, adata.var_names.get_indexer(genes)].X
        detected += np.asarray((sub > 0).sum(axis=0)).ravel()
        n_spots += adata.n_obs
    return detected / max(n_spots, 1)


def _offset(sections: pd.DataFrame, raw_dir: Path, genes: list[str]) -> np.ndarray:
    """Per-gene bulk − pseudo-bulk from the frozen GSE175540 pairs whose patients are in train."""
    pairs = sections[(sections["split"] == "train") & (sections["protocol"] == "frozen")]
    gse_bulk = io_visium.load_gse_bulk(raw_dir)
    pseudo = {}
    for row in pairs.itertuples():
        adata = ad.read_h5ad(row.path)
        bulk_id = adata.obs["bulk_id"].iloc[0]
        if bulk_id and bulk_id in gse_bulk.index:
            total = np.asarray(adata.X.sum(axis=0))
            pseudo[bulk_id] = pd.Series(log2_cpm(total)[0], index=adata.var_names)
    if not pseudo:
        log.warning("no train bulk/Visium pairs -> zero offset")
        return np.zeros(len(genes), dtype=np.float32)
    log.info("platform offset from %d train pairs: %s", len(pseudo), sorted(pseudo))
    offset = platform_offset(gse_bulk, pd.DataFrame(pseudo).T, genes)
    return offset.to_numpy(dtype=np.float32)


def prev_panel_genes(paths: dict[str, Path]) -> list[str]:
    path = paths["bulk_private"] / "genesets" / PREV_PANEL_FILE
    if not path.exists():
        return []
    return pd.read_csv(path, sep="\t", index_col=0).iloc[:, 0].astype(str).tolist()


def prepare_units(cfg: dict, root: Path = Path(".")) -> tuple[ad.AnnData, dict]:
    paths = _paths(cfg, root)
    seed = cfg["split"]["seed"]
    bulk = io_bulk.load_bulk(paths["bulk_public"], _private_dir(paths))
    bulk.obs["split"] = split_bulk(bulk.obs, cfg["cohorts"]["ood"], cfg["split"]["bulk_test_frac"], seed)

    sections = section_table(paths["visium_h5ad"])
    patients = sections.groupby("patient").agg(source=("source", "first"), has_tls=("has_tls", "max")).reset_index()
    patients["split"] = split_visium(patients, cfg["split"]["visium_test_frac"], seed).to_numpy()
    sections = sections.merge(patients[["patient", "split"]], on="patient")

    shared = sorted(set(bulk.var_names).intersection(*sections["genes"]))
    signatures = io_bulk.load_signatures(paths["bulk_public"])
    protected = {g for genes in signatures.values() for g in genes}
    train_bulk = np.asarray(bulk[bulk.obs["split"] == "train", shared].X)
    detection = _spot_detection(sections[sections["split"] == "train"], shared)
    fcfg = cfg["features"]
    genes = gene_filter(train_bulk, detection, shared, fcfg["n_top_variable"], protected,
                        fcfg["min_bulk_median"], fcfg["min_spot_detection"])
    log.info("genes: shared=%d kept=%d (protected present=%d)", len(shared), len(genes), len(protected & set(shared)))

    offset = _offset(sections, paths["visium_raw"], genes)
    rng = np.random.default_rng(seed)
    blocks = [_bulk_units(bulk, genes)]
    natural = {"n_spots": 0, "n_tls": 0, "n_mature_known": 0, "n_mature": 0}
    for row in sections.itertuples():
        section = ad.read_h5ad(row.path)
        if row.split == "train":
            natural = {k: natural[k] + v for k, v in spot_prevalence(section).items()}
        blocks.append(_visium_units(section, row.split, genes, cfg, rng))
        log.info("units %s (%s): %d", row.section, row.split, blocks[-1].n_obs)
    spot_prior = {"any": natural["n_tls"] / natural["n_spots"],
                  "mature": natural["n_mature"] / max(natural["n_mature_known"], 1)}
    log.info("natural train-spot prevalence: %s (%s)", spot_prior, natural)
    units = ad.concat(blocks, join="outer")
    units.var_names = genes
    units.obs = units.obs[UNIT_COLUMNS]

    meta = {
        "genes": genes,
        "signatures": {name: [g for g in members if g in set(genes)] for name, members in signatures.items()},
        "offset": offset.tolist(),
        "prev_panel_genes": prev_panel_genes(paths),
        "spot_prior": spot_prior,
        "n_shared_genes": len(shared),
        "sections": sections[["section", "patient", "source", "protocol", "has_tls", "split"]].to_dict("records"),
    }
    out = paths["processed"]
    out.mkdir(parents=True, exist_ok=True)
    units.write_h5ad(out / "units.h5ad")
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    bulk.write_h5ad(out / "bulk.h5ad", compression="gzip")
    return units, meta


def load_units(cfg: dict, root: Path = Path(".")) -> tuple[ad.AnnData, dict]:
    out = _paths(cfg, root)["processed"]
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    meta["offset"] = np.asarray(meta["offset"], dtype=np.float32)
    return ad.read_h5ad(out / "units.h5ad"), meta
