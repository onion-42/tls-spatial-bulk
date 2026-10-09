"""Training / evaluation orchestration on the unit table built by `prepare.py`.

Units = bulk samples + Visium spots / k-aggregates / whole sections, one row each, X = log2 values on the
kept genes. obs columns used here: kind, unit, group, cohort, section, patient, split, platform,
label_source, tls_any, tls_mature, tls_fraction.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tls_model.evaluate import binary_metrics, section_spearman, three_class_balanced_accuracy
from tls_model.features import FeatureSpace
from tls_model.models import HeadModel, PlattCalibrator, auroc, three_class
from tls_model.splits import balanced_weights, cv_folds

log = logging.getLogger(__name__)

HEAD_LABEL = {"any": "tls_any", "mature": "tls_mature"}
SHARED_SELECTION = ("enet", "lgbm")  # identical in-fold selection -> computed once per fold


def load_config(path: Path = Path("config.yaml")) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def candidate_name(candidate: dict) -> str:
    return f"{candidate['family']}|{candidate['selector'] or '-'}"


def labelled_rows(obs: pd.DataFrame, head: str, split: str) -> np.ndarray:
    """Rows of `split` with a label for `head`. Maturity labels come from pathologists / experts only."""
    mask = (obs["split"] == split) & obs[HEAD_LABEL[head]].notna()
    if head == "mature":
        mask &= obs["label_source"] == "pathologist"
    return mask.to_numpy()


def train_rows(obs: pd.DataFrame, head: str) -> np.ndarray:
    return labelled_rows(obs, head, "train")


def scored_rows(obs: pd.DataFrame, head: str, split: str) -> np.ndarray:
    """Rows used for metrics: pathologist labels only (CNN labels are train-only)."""
    return labelled_rows(obs, head, split) & (obs["label_source"] == "pathologist").to_numpy()


def feature_matrix(units, meta: dict, representation: str, use_offset: bool) -> tuple[np.ndarray, list[str]]:
    log_values = np.asarray(units.X, dtype=np.float32).copy()
    if use_offset:
        frozen = (units.obs["platform"] == "visium_frozen").to_numpy()
        log_values[frozen] += np.asarray(meta["offset"], dtype=np.float32)
    space = FeatureSpace(
        genes=tuple(meta["genes"]),
        signatures=tuple((name, tuple(genes)) for name, genes in meta["signatures"].items()),
        representation=representation,
    )
    return space.base(log_values)


def unit_weights(obs: pd.DataFrame) -> np.ndarray:
    return balanced_weights(obs["kind"].to_numpy(), obs["group"].to_numpy())


def make_head(candidate: dict, meta: dict, n_panel: int, n_pair_genes: int, seed: int, preselected=None) -> HeadModel:
    signature_genes = {g for genes in meta["signatures"].values() for g in genes}
    return HeadModel(
        candidate["family"], candidate["selector"], n_panel=n_panel, n_pair_genes=n_pair_genes,
        signature_genes=signature_genes, panel_genes=meta.get("prev_panel_genes"), seed=seed,
        preselected=preselected,
    )


def cv_strata(obs: pd.DataFrame, head: str) -> np.ndarray:
    """Stratify folds by block × label (bulk cohort × y, Visium × y). Stratifying on y alone lets the
    thousands of spot units dominate and can put all bulk positives of a head into one or two folds."""
    y = obs[HEAD_LABEL[head]].astype(float).astype(str)
    block = np.where(obs["kind"] == "bulk", obs["cohort"].astype(str), "visium")
    return pd.factorize(pd.Series(block, index=obs.index) + ":" + y)[0]


def _within_cohort_auroc(cohorts: np.ndarray, y: np.ndarray, p: np.ndarray) -> float:
    """Mean AUROC inside cohorts (a pooled AUROC also rewards telling cancer types with different TLS rates apart)."""
    values = [auroc(y[cohorts == c], p[cohorts == c]) for c in np.unique(cohorts)]
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else np.nan


def oof_scores(obs: pd.DataFrame, head: str, p: np.ndarray, mask: np.ndarray, prev_excluded=()) -> dict:
    """CV selection score = mean of bulk within-cohort AUROC (pathologist rows) and Visium spot AUROC (k = 1)."""
    y = obs[HEAD_LABEL[head]].to_numpy(dtype=float)
    cohorts = obs["cohort"].astype(str).to_numpy()
    is_path = (obs["label_source"] == "pathologist").to_numpy()
    bulk = mask & is_path & (obs["kind"] == "bulk").to_numpy()
    spots = mask & (obs["unit"] == "spot").to_numpy()
    clean = bulk & ~np.isin(cohorts, list(prev_excluded))
    out = {
        "auroc_bulk": _within_cohort_auroc(cohorts[bulk], y[bulk], p[bulk]) if bulk.any() else np.nan,
        "auroc_spot": auroc(y[spots], p[spots]) if spots.any() else np.nan,
        "auroc_bulk_pooled": auroc(y[bulk], p[bulk]) if bulk.any() else np.nan,
        "auroc_bulk_not_prev_cohorts": _within_cohort_auroc(cohorts[clean], y[clean], p[clean]) if clean.any() else np.nan,
    }
    out["score"] = float(np.nanmean([out["auroc_bulk"], out["auroc_spot"]]))
    return out


def run_cv(B, names, obs, head, candidates, meta, n_folds, seed, n_panel, n_pair_genes, prev_excluded=()):
    """Grouped CV over train patients. Returns (table sorted by score, {candidate: out-of-fold p})."""
    rows = train_rows(obs, head)
    idx = np.flatnonzero(rows)
    y = obs[HEAD_LABEL[head]].to_numpy(dtype=float)
    w = np.zeros(len(obs))
    w[idx] = unit_weights(obs.iloc[idx])
    folds = cv_folds(obs["patient"].to_numpy()[idx], cv_strata(obs.iloc[idx], head), n_folds, seed)
    selections: dict[tuple, tuple] = {}
    table, oof = [], {}
    for candidate in candidates:
        name = candidate_name(candidate)
        if candidate["family"] == "prev_panel" and not meta.get("prev_panel_genes"):
            log.info("skip %s: no previous-model panel available", name)
            continue
        start, p = time.time(), np.full(len(obs), np.nan)
        for fold in range(n_folds):
            tr, te = idx[folds != fold], idx[folds == fold]
            key = (fold, candidate["selector"]) if candidate["family"] in SHARED_SELECTION else None
            model = make_head(candidate, meta, n_panel, n_pair_genes, seed, preselected=selections.get(key))
            model.fit(B[tr], names, y[tr], w[tr])
            if key is not None:
                selections[key] = (model.panel, model.pair_genes)
            p[te] = model.predict_proba(B[te], names)
        oof[name] = p
        row = {"candidate": name, **candidate, **oof_scores(obs, head, p, rows, prev_excluded)}
        row["seconds"] = round(time.time() - start, 1)
        log.info("CV %s %s: %s", head, name, {k: round(v, 3) for k, v in row.items() if isinstance(v, float)})
        table.append(row)
    return pd.DataFrame(table).sort_values("score", ascending=False).reset_index(drop=True), oof


def best_candidate(table: pd.DataFrame) -> dict:
    """Best by CV score; the previous-model panel is a comparison row only."""
    row = table[table["family"] != "prev_panel"].iloc[0]
    return {"family": row["family"], "selector": None if row["selector"] in (None, "-") or pd.isna(row["selector"]) else row["selector"]}


def calibration_kind(obs: pd.DataFrame) -> np.ndarray:
    """'spot' for spots and spot aggregates; 'bulk' for bulk samples and whole-section pseudo-bulks."""
    return np.where(obs["unit"].isin(["spot", "agg7", "agg19"]), "spot", "bulk")


def calibration_rows(obs: pd.DataFrame, head: str) -> np.ndarray:
    """Pathologist-labelled train bulk samples and train spots (k = 1): the unit types the model is scored on."""
    rows = train_rows(obs, head) & (obs["label_source"] == "pathologist").to_numpy()
    return rows & obs["unit"].isin(["bulk", "spot"]).to_numpy()


def fit_final(B, names, obs, head, candidate, oof_p, meta, n_panel, n_pair_genes, seed):
    """Calibrator from out-of-fold predictions (+ prior shift for class-enriched spot sampling);
    model refit on all train rows."""
    rows = train_rows(obs, head)
    cal = calibration_rows(obs, head)
    y = obs[HEAD_LABEL[head]].to_numpy(dtype=float)
    prior = meta.get("spot_prior", {}).get(head)
    calibrator = PlattCalibrator().fit(oof_p[cal], y[cal].astype(int), calibration_kind(obs)[cal],
                                       prior={"spot": prior} if prior else None)
    model = make_head(candidate, meta, n_panel, n_pair_genes, seed)
    model.fit(B[rows], names, y[rows], unit_weights(obs[rows]))
    return model, calibrator


def predict(B, names, obs, model, calibrator) -> np.ndarray:
    return calibrator.transform(model.predict_proba(B, names), calibration_kind(obs))


def evaluate_split(obs: pd.DataFrame, p_any: np.ndarray, p_mature: np.ndarray | None, split: str,
                   n_boot: int = 1000) -> pd.DataFrame:
    """Metrics per cohort / data block for one split; spot metrics bootstrap whole sections.
    p_mature = None -> head A only (no maturity, no 3-class)."""
    rows = []
    blocks = [(f"bulk:{c}", (obs["kind"] == "bulk") & (obs["cohort"] == c)) for c in sorted(obs.loc[obs["kind"] == "bulk", "cohort"].unique())]
    blocks += [("bulk:all", obs["kind"] == "bulk"), ("visium:spots", obs["unit"] == "spot"), ("visium:sections", obs["unit"] == "section")]
    heads = [("any", p_any)] + ([("mature", p_mature)] if p_mature is not None else [])
    for head, p in heads:
        scored = scored_rows(obs, head, split)
        for block, in_block in blocks:
            mask = scored & in_block.to_numpy()
            if mask.sum() < 2:
                continue
            groups = obs["section"].to_numpy()[mask] if block == "visium:spots" else None
            y = obs[HEAD_LABEL[head]].to_numpy(dtype=float)[mask]
            rows.append({"split": split, "head": head, "block": block, **binary_metrics(y, p[mask], groups, n_boot)})
    probs = three_class(p_any, p_mature) if p_mature is not None else None
    for block, in_block in blocks:
        mask = (obs["split"] == split).to_numpy() & in_block.to_numpy() & (obs["label_source"] == "pathologist").to_numpy()
        if probs is not None and block.startswith("bulk") and mask.any():
            y_any, y_mat = obs["tls_any"].to_numpy()[mask], obs["tls_mature"].to_numpy(dtype=float)[mask]
            known = (y_any == 0) | ~np.isnan(y_mat)
            if known.sum() and len(np.unique(y_any[known])) > 1:
                rows.append({"split": split, "head": "3-class", "block": block,
                             "balanced_accuracy": three_class_balanced_accuracy(y_any, y_mat, probs[mask])})
    sections = (obs["split"] == split).to_numpy() & (obs["unit"] == "spot").to_numpy()
    if sections.any() and obs.loc[sections, "section"].nunique() > 2:
        rows.append({"split": split, "head": "any", "block": "visium:section_spearman",
                     "spearman": section_spearman(obs["section"].to_numpy()[sections], p_any[sections],
                                                  obs["tls_any"].to_numpy(dtype=float)[sections])})
    return pd.DataFrame(rows)


def leave_one_cohort_out(B, names, obs, candidate, meta, n_panel, n_pair_genes, seed) -> pd.DataFrame:
    """Robustness only (not used for selection): train without one bulk cohort, score that cohort."""
    rows_all = train_rows(obs, "any")
    y = obs["tls_any"].to_numpy(dtype=float)
    out = []
    cohorts = sorted(obs.loc[rows_all & (obs["kind"] == "bulk").to_numpy(), "cohort"].unique())
    for cohort in cohorts:
        is_cohort = (obs["cohort"] == cohort).to_numpy()
        fit_rows = rows_all & ~is_cohort
        held = is_cohort & np.isin(obs["split"], ["train", "test"]) & (obs["label_source"] == "pathologist").to_numpy()
        if len(np.unique(y[held])) < 2:
            continue
        model = make_head(candidate, meta, n_panel, n_pair_genes, seed)
        model.fit(B[fit_rows], names, y[fit_rows], unit_weights(obs[fit_rows]))
        p = model.predict_proba(B[held], names)
        out.append({"left_out": cohort, "n": int(held.sum()), "n_pos": int(y[held].sum()), "auroc": auroc(y[held], p)})
        log.info("LOCO %s auroc=%.3f", cohort, out[-1]["auroc"])
    return pd.DataFrame(out)


def label_ceiling(long_table: pd.DataFrame, cohort: str = "TCGA-LUAD") -> pd.DataFrame:
    """Agreement (Cohen's kappa) of TLS +/- between label sources on the same samples."""
    from sklearn.metrics import cohen_kappa_score

    table = long_table[(long_table["cohort"] == cohort) & long_table["TLS_status"].isin(["positive", "negative"])]
    table = table.assign(label_source=table["method"] + "#" + table["priority"].astype(str))
    wide = table.pivot_table(index="sample_id", columns="label_source", values="TLS_status", aggfunc="first")
    out = []
    sources = list(wide.columns)
    for i, a in enumerate(sources):
        for b in sources[i + 1 :]:
            both = wide[[a, b]].dropna()
            if len(both) >= 10:
                out.append({"source_a": a, "source_b": b, "n": len(both),
                            "agreement": float((both[a] == both[b]).mean()),
                            "kappa": float(cohen_kappa_score(both[a], both[b]))})
    return pd.DataFrame(out)
