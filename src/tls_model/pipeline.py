"""Training / evaluation orchestration on the unit table built by `prepare.py`.

Units = bulk samples + Visium spots / k-aggregates / whole sections, one row each, X = log2 values on the
kept genes. obs columns used here: kind, unit, group, cohort, section, patient, split, role, platform,
label_source, tls_any, tls_mature, tls_three, tls_fraction.

Roles: `train` (Visium train sections) carries the gold labels the teacher learns from; `distill` (train split of
the TCGA cohorts) contributes expression only — the teacher's soft predictions become the student's targets;
`test` and `holdout` are never fitted on. Heads: `any` = P(TLS), `mature` = P(mature | TLS), `three` = the
direct no TLS / immature / mature classifier compared with the product of the two heads.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tls_model.evaluate import binary_metrics, macro_ovr_auroc, multiclass_metrics, section_spearman
from tls_model.features import FeatureSpace
from tls_model.labels import three_class_label
from tls_model.models import EPS, HeadModel, PlattCalibrator, auroc, three_class
from tls_model.splits import ROLES, balanced_weights, cv_folds, roles

log = logging.getLogger(__name__)

HEAD_LABEL = {"any": "tls_any", "mature": "tls_mature", "three": "tls_three"}
# Target classes of a head, in the order the soft pseudo-label columns are laid out.
HEAD_CLASSES = {"any": (1, 0), "mature": (1, 0), "three": (0, 1, 2)}
SHARED_SELECTION = ("enet", "lgbm")  # identical in-fold selection -> computed once per fold
FORMS = ("hierarchy", "direct")

__all__ = ["roles", "ROLES"]


def load_config(path: Path = Path("config.yaml")) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def candidate_name(candidate: dict) -> str:
    return f"{candidate['family']}|{candidate['selector'] or '-'}"


def three_class_column(obs: pd.DataFrame) -> np.ndarray:
    """0 / 1 / 2 = no TLS / immature / mature, NaN where unlabelled or maturity unknown."""
    return three_class_label(obs["tls_any"].to_numpy(dtype=float), obs["tls_mature"].to_numpy(dtype=float))


# ----------------------------------------------------------------------------------------------------- rows


def labelled_rows(obs: pd.DataFrame, head: str, split: str) -> np.ndarray:
    """Rows of `split` with a label for `head`. Maturity labels come from pathologists / experts only."""
    mask = (obs["split"] == split) & obs[HEAD_LABEL[head]].notna()
    if head != "any":
        mask &= obs["label_source"] == "pathologist"
    return mask.to_numpy()


def train_rows(obs: pd.DataFrame, head: str) -> np.ndarray:
    """Gold training rows: train-role (Visium) units with a label for `head`."""
    mask = (obs["role"] == "train") & obs[HEAD_LABEL[head]].notna()
    if head != "any":
        mask &= obs["label_source"] == "pathologist"
    return mask.to_numpy()


def distill_rows(obs: pd.DataFrame) -> np.ndarray:
    """Units the teacher pseudo-labels: train split of the distilled bulk cohorts (their gold labels are not used)."""
    return (obs["role"] == "distill").to_numpy()


def scored_rows(obs: pd.DataFrame, head: str, split: str) -> np.ndarray:
    """Rows used for metrics: pathologist labels only (CNN labels are never scored)."""
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


# -------------------------------------------------------------------------------------- self-distillation


@dataclass(frozen=True)
class PseudoLabels:
    """Teacher predictions on the distill rows.

    by_fold[head][f] = predictions of the teacher fitted without CV fold f (used when the student is itself
    cross-validated, so the gold labels of fold f never reach its distill rows); final[head] = the full teacher.
    Arrays run over all units (NaN outside `rows`); `three` holds (n, 3) matrices."""

    rows: np.ndarray
    by_fold: dict[str, np.ndarray]
    final: dict[str, np.ndarray]

    def at(self, head: str, fold: int | None) -> np.ndarray:
        source = self.final[head] if fold is None else self.by_fold[head][fold]
        return source[self.rows]


def soft_targets(head: str, pseudo: PseudoLabels, fold: int | None) -> tuple[tuple[int, ...], np.ndarray]:
    """Classes of `head` and the weight of every class per distill row (rows × classes).

    any: (P, 1 − P); mature: P(TLS) · (P(mature | TLS), 1 − P(mature | TLS)) — a bulk sample contributes to the
    maturity head in proportion to the teacher's belief that it carries a TLS; three: the 3-class matrix."""
    classes = HEAD_CLASSES[head]
    if head == "three":
        return classes, pseudo.at("three", fold)
    p = pseudo.at(head, fold)
    weights = np.column_stack([p, 1.0 - p])
    if head == "mature":
        weights = weights * pseudo.at("any", fold)[:, None]
    return classes, weights


def fit_set(B: np.ndarray, obs: pd.DataFrame, head: str, pseudo: PseudoLabels | None, fold: int | None = None,
            exclude: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, y, w) a head is fitted on: gold train rows, plus — with pseudo-labels — every distill row repeated once
    per class with weight = base weight × soft target (exact for the log-loss). Visium and distilled bulk get
    equal total weight. `exclude` masks out the CV test fold."""
    rows = train_rows(obs, head)
    if exclude is not None:
        rows &= ~exclude
    y_train = obs[HEAD_LABEL[head]].to_numpy(dtype=float)[rows]
    if pseudo is None:
        return B[rows], y_train, unit_weights(obs[rows])
    distill = pseudo.rows
    classes, soft = soft_targets(head, pseudo, fold)
    both = np.concatenate([np.flatnonzero(rows), np.flatnonzero(distill)])
    w = balanced_weights(obs["kind"].to_numpy()[both], obs["group"].to_numpy()[both])
    w_train, w_distill = w[: rows.sum()], w[rows.sum():]
    X = np.vstack([B[rows]] + [B[distill]] * len(classes))
    y = np.concatenate([y_train] + [np.full(distill.sum(), c, dtype=float) for c in classes])
    w_all = np.concatenate([w_train] + [w_distill * soft[:, k] for k in range(len(classes))])
    return X, y, w_all


# ------------------------------------------------------------------------------------------- cross-validation


def cv_strata(obs: pd.DataFrame, head: str) -> np.ndarray:
    """Stratify folds by block × label (bulk cohort × y, Visium × y). Stratifying on y alone lets the
    thousands of spot units dominate and can put all bulk positives of a head into one or two folds."""
    y = obs[HEAD_LABEL[head]].astype(float).astype(str)
    block = np.where(obs["kind"] == "bulk", obs["cohort"].astype(str), "visium")
    return pd.factorize(pd.Series(block, index=obs.index) + ":" + y)[0]


def patient_folds(obs: pd.DataFrame, n_folds: int, seed: int) -> np.ndarray:
    """One patient partition shared by every head (fold id per unit, −1 outside the gold train rows), so a
    fold-f teacher of one head never saw the patients a fold-f student of another head is scored on."""
    rows = train_rows(obs, "any")
    idx = np.flatnonzero(rows)
    patients = obs["patient"].to_numpy()
    n_patients = len(np.unique(patients[idx]))
    if n_patients < 2:
        raise ValueError(f"grouped CV needs at least 2 train patients, found {n_patients}")
    n_folds = min(n_folds, n_patients)
    folds = cv_folds(patients[idx], cv_strata(obs.iloc[idx], "any"), n_folds, seed)
    by_patient = dict(zip(patients[idx], folds))
    return np.array([by_patient.get(p, -1) for p in patients])


def _within_cohort_auroc(cohorts: np.ndarray, y: np.ndarray, p: np.ndarray) -> float:
    """Mean AUROC inside cohorts (a pooled AUROC also rewards telling cancer types with different TLS rates apart)."""
    values = [auroc(y[cohorts == c], p[cohorts == c]) for c in np.unique(cohorts)]
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else np.nan


def oof_scores(obs: pd.DataFrame, head: str, p: np.ndarray, mask: np.ndarray, prev_excluded=()) -> dict:
    """CV selection score = mean of bulk within-cohort AUROC (pathologist rows) and Visium spot AUROC (k = 1).
    Bulk rows are absent from the gold train set since the distillation design; the columns stay for the tables."""
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


def oof_scores_three(obs: pd.DataFrame, probs: np.ndarray, mask: np.ndarray) -> dict:
    """CV selection score of the direct 3-class form: macro one-vs-rest AUROC on train spots (k = 1)."""
    y3 = obs["tls_three"].to_numpy(dtype=float)
    spots = mask & (obs["unit"] == "spot").to_numpy() & ~np.isnan(y3)
    metrics = multiclass_metrics(y3[spots], probs[spots], n_boot=0)
    out = {"macro_auroc": metrics.get("macro_auroc", np.nan), "balanced_accuracy": metrics.get("balanced_accuracy", np.nan),
           "auroc_mature_vs_rest": metrics.get("auroc_mature_vs_rest", np.nan)}
    out["score"] = out["macro_auroc"]
    return out


def run_cv(B, names, obs, head, candidates, meta, n_folds, seed, n_panel, n_pair_genes, prev_excluded=(),
           predict_rows: np.ndarray | None = None, pseudo: PseudoLabels | None = None):
    """Grouped CV over train patients.

    Returns (table sorted by score, {candidate: out-of-fold p}, {candidate: fold predictions}). `predict_rows`
    (e.g. the distill rows) are scored by every fold model -> fold predictions of shape (n_folds, n[, 3]), NaN
    elsewhere; None when not requested. `pseudo` adds the distill rows to every fold's fit set (student CV)."""
    rows = train_rows(obs, head)
    idx = np.flatnonzero(rows)
    folds_all = patient_folds(obs, n_folds, seed)
    folds = folds_all[idx]
    n_folds = int(folds.max()) + 1
    n_classes = len(HEAD_CLASSES[head])
    shape = (len(obs),) if n_classes == 2 else (len(obs), n_classes)
    selections: dict[tuple, tuple] = {}
    table, oof, fold_pred = [], {}, {}
    for candidate in candidates:
        name = candidate_name(candidate)
        if candidate["family"] == "prev_panel" and not meta.get("prev_panel_genes"):
            log.info("skip %s: no previous-model panel available", name)
            continue
        start, p = time.time(), np.full(shape, np.nan)
        fp = np.full((n_folds, *shape), np.nan) if predict_rows is not None else None
        for fold in range(n_folds):
            if not (folds == fold).any():
                continue
            # Out-of-fold predictions for every train unit of the fold, labelled for this head or not, so the heads
            # can be combined into 3-class probabilities on the same units.
            held = folds_all == fold
            X, y, w = fit_set(B, obs, head, pseudo, fold, exclude=held)
            key = (fold, candidate["selector"]) if candidate["family"] in SHARED_SELECTION else None
            model = make_head(candidate, meta, n_panel, n_pair_genes, seed, preselected=selections.get(key))
            model.fit(X, names, y, w)
            if key is not None:
                selections[key] = (model.panel, model.pair_genes)
            p[held] = model.predict_proba(B[held], names)
            if fp is not None:
                fp[fold][predict_rows] = model.predict_proba(B[predict_rows], names)
        oof[name], fold_pred[name] = p, fp
        scores = oof_scores_three(obs, p, rows) if head == "three" else oof_scores(obs, head, p, rows, prev_excluded)
        row = {"candidate": name, **candidate, **scores, "seconds": round(time.time() - start, 1)}
        log.info("CV %s %s: %s", head, name, {k: round(v, 3) for k, v in row.items() if isinstance(v, float)})
        table.append(row)
    return pd.DataFrame(table).sort_values("score", ascending=False).reset_index(drop=True), oof, fold_pred


def best_candidate(table: pd.DataFrame) -> dict:
    """Best by CV score; the previous-model panel is a comparison row only."""
    row = table[table["family"] != "prev_panel"].iloc[0]
    return {"family": row["family"], "selector": None if row["selector"] in (None, "-") or pd.isna(row["selector"]) else row["selector"]}


def compare_forms(obs: pd.DataFrame, forms: dict[str, np.ndarray], n_boot: int = 200) -> pd.DataFrame:
    """Out-of-fold comparison of the forms (hierarchy = product of the two heads, direct = 3-class model).

    3-class metrics on the gold train units with a known 3-class label, plus `auroc_tls_all_spots` = AUROC of
    P(TLS) = 1 − P(no TLS) on every train spot, maturity known or not (the units with a known 3-class label leave
    out the GSE175540 TLS spots, which only the hierarchy learns from)."""
    y3 = obs["tls_three"].to_numpy(dtype=float)
    y_any = obs["tls_any"].to_numpy(dtype=float)
    mask = train_rows(obs, "three")
    spots = train_rows(obs, "any") & (obs["unit"] == "spot").to_numpy()
    for probs in forms.values():
        finite = np.isfinite(probs).all(axis=1)
        mask &= finite
        spots &= finite
    groups = obs["patient"].to_numpy()[mask]
    rows = [{"form": name, **multiclass_metrics(y3[mask], probs[mask], groups, n_boot),
             "auroc_tls_all_spots": auroc(y_any[spots], 1.0 - probs[spots, 0]), "n_all_spots": int(spots.sum())}
            for name, probs in forms.items()]
    return pd.DataFrame(rows)


def choose_form(forms: pd.DataFrame) -> str:
    """Deployed form: the hierarchy (calibrated heads, learns from every TLS spot) unless the direct 3-class model
    beats it on both the 3-class macro AUROC and P(TLS) on all train spots."""
    by_form = forms.set_index("form")
    if "direct" not in by_form.index or "hierarchy" not in by_form.index:
        return "hierarchy"
    direct, hierarchy = by_form.loc["direct"], by_form.loc["hierarchy"]
    wins_both = (direct["macro_auroc"] > hierarchy["macro_auroc"]) and (direct["auroc_tls_all_spots"] > hierarchy["auroc_tls_all_spots"])
    return "direct" if wins_both else "hierarchy"


# --------------------------------------------------------------------------------------- final fit / predict


def calibration_kind(obs: pd.DataFrame) -> np.ndarray:
    """'spot' for spots and spot aggregates; 'bulk' for bulk samples and whole-section pseudo-bulks."""
    return np.where(obs["unit"].isin(["spot", "agg7", "agg19"]), "spot", "bulk")


def calibration_rows(obs: pd.DataFrame, head: str) -> np.ndarray:
    """Gold-labelled rows of the unit types the model is scored on: train spots (k = 1) and the pathologist-labelled
    bulk samples of the distill cohorts (two Platt parameters per kind; the student never fits on those labels)."""
    spots = train_rows(obs, head) & (obs["unit"] == "spot").to_numpy()
    bulk = distill_rows(obs) & (obs["label_source"] == "pathologist").to_numpy() & obs[HEAD_LABEL[head]].notna().to_numpy()
    return spots | bulk


def fit_final(B, names, obs, head, candidate, oof_p, meta, n_panel, n_pair_genes, seed, pseudo: PseudoLabels | None = None):
    """Model refit on all train rows (+ distill rows when pseudo-labels are given); Platt calibrator from
    out-of-fold predictions on train spots and in-sample predictions on the gold distill bulk rows (+ prior shift
    for class-enriched spot sampling). The 3-class form has no calibrator."""
    model = make_head(candidate, meta, n_panel, n_pair_genes, seed)
    X, y, w = fit_set(B, obs, head, pseudo)
    model.fit(X, names, y, w)
    if head == "three":
        return model, None
    cal = calibration_rows(obs, head)
    p_cal = np.asarray(oof_p, dtype=float).copy()
    in_sample = cal & np.isnan(p_cal)
    if in_sample.any():
        p_cal[in_sample] = model.predict_proba(B[in_sample], names)
    labels = obs[HEAD_LABEL[head]].to_numpy(dtype=float)
    prior = meta.get("spot_prior", {}).get(head)
    calibrator = PlattCalibrator().fit(p_cal[cal], labels[cal].astype(int), calibration_kind(obs)[cal],
                                       prior={"spot": prior} if prior else None)
    return model, calibrator


def predict(B, names, obs, model, calibrator) -> np.ndarray:
    return calibrator.transform(model.predict_proba(B, names), calibration_kind(obs))


def predict_three(B, names, obs, saved: dict) -> np.ndarray:
    """(n, 3) probabilities [no TLS, immature, mature] of the saved model (`form` = hierarchy | direct)."""
    if saved["form"] == "direct":
        return saved["models"]["three"].predict_proba(B, names)
    p_any = predict(B, names, obs, saved["models"]["any"], saved["calibrators"]["any"])
    p_mature = predict(B, names, obs, saved["models"]["mature"], saved["calibrators"]["mature"])
    return three_class(p_any, p_mature)


def heads_from_three(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P(TLS) and P(mature | TLS) implied by a 3-class matrix."""
    p_any = 1.0 - probs[:, 0]
    p_mature = np.clip(probs[:, 2] / np.clip(p_any, EPS, None), 0.0, 1.0)
    return p_any, p_mature


PREDICT_CHUNK = 50_000


def _in_chunks(units, meta: dict, variant: dict, score, width: int, chunk: int) -> np.ndarray:
    """Features built `chunk` rows at a time. Every feature is single-sample, so the result equals one pass;
    memory stays bounded for hundreds of thousands of spots (ssGSEA ranks are float64)."""
    out = np.empty((units.n_obs, width) if width > 1 else units.n_obs, dtype=np.float64)
    for start in range(0, units.n_obs, chunk):
        part = units[start : start + chunk]
        B, names = feature_matrix(part, meta, variant["representation"], bool(variant["offset"]))
        out[start : start + chunk] = score(B, names, part.obs)
    return out


def predict_in_chunks(units, meta: dict, variant: dict, model, calibrator, chunk: int = PREDICT_CHUNK) -> np.ndarray:
    """Calibrated P of one binary head for all units."""
    return _in_chunks(units, meta, variant, lambda B, names, obs: predict(B, names, obs, model, calibrator), 1, chunk)


def predict_three_in_chunks(units, meta: dict, variant: dict, saved: dict, chunk: int = PREDICT_CHUNK) -> np.ndarray:
    """(n, 3) class probabilities for all units."""
    return _in_chunks(units, meta, variant, lambda B, names, obs: predict_three(B, names, obs, saved), 3, chunk)


# ------------------------------------------------------------------------------------------------ evaluation


def _blocks(obs: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    bulk = (obs["kind"] == "bulk").to_numpy()
    cohorts = sorted(obs.loc[bulk, "cohort"].unique())
    blocks = [(f"bulk:{c}", bulk & (obs["cohort"] == c).to_numpy()) for c in cohorts]
    return blocks + [("bulk:all", bulk), ("visium:spots", (obs["unit"] == "spot").to_numpy()),
                     ("visium:sections", (obs["unit"] == "section").to_numpy())]


def evaluate_split(obs: pd.DataFrame, probs: np.ndarray, split: str, n_boot: int = 1000) -> pd.DataFrame:
    """Metrics per cohort / data block for one split; spot metrics bootstrap whole sections.
    probs: (n, 3) class probabilities -> heads any + mature and 3-class rows; a 1-D P(TLS) -> head any only."""
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 1:
        heads, three = [("any", probs)], None
    else:
        p_any, p_mature = heads_from_three(probs)
        heads, three = [("any", p_any), ("mature", p_mature)], probs
    rows = []
    blocks = _blocks(obs)
    sections = obs["section"].to_numpy()
    for head, p in heads:
        scored = scored_rows(obs, head, split)
        for block, in_block in blocks:
            mask = scored & in_block
            if mask.sum() < 2:
                continue
            groups = sections[mask] if block == "visium:spots" else None
            y = obs[HEAD_LABEL[head]].to_numpy(dtype=float)[mask]
            rows.append({"split": split, "head": head, "block": block, **binary_metrics(y, p[mask], groups, n_boot)})
    if three is not None:
        y3 = obs["tls_three"].to_numpy(dtype=float)
        known = (obs["split"] == split).to_numpy() & (obs["label_source"] == "pathologist").to_numpy() & ~np.isnan(y3)
        for block, in_block in blocks:
            mask = known & in_block
            if mask.sum() < 2 or len(np.unique(y3[mask])) < 2:
                continue
            groups = sections[mask] if block == "visium:spots" else None
            rows.append({"split": split, "head": "3-class", "block": block,
                         **multiclass_metrics(y3[mask], three[mask], groups, n_boot)})
    spots = (obs["split"] == split).to_numpy() & (obs["unit"] == "spot").to_numpy()
    if spots.any() and obs.loc[spots, "section"].nunique() > 2:
        rows.append({"split": split, "head": "any", "block": "visium:section_spearman",
                     "spearman": section_spearman(sections[spots], heads[0][1][spots],
                                                  obs["tls_any"].to_numpy(dtype=float)[spots])})
    return pd.DataFrame(rows)


def leakage_report(obs: pd.DataFrame, fit_rows: dict[str, np.ndarray], calibration: dict[str, np.ndarray]) -> pd.DataFrame:
    """Patients / units / gold labels per role, and hard checks: no patient in both training (train ∪ distill) and
    test or holdout, none in both test and holdout; every fitted row is a train or distill unit; every calibration
    row is a pathologist-labelled train or distill unit. Raises ValueError on the first violation."""
    role = obs["role"].astype(str)
    patients = {r: set(obs.loc[(role == r).to_numpy(), "patient"]) for r in ROLES}
    fitted = patients["train"] | patients["distill"]
    for held in ("test", "holdout"):
        shared = fitted & patients[held]
        if shared:
            raise ValueError(f"{len(shared)} patient(s) in both training and {held}: {sorted(shared)[:5]}")
    shared = patients["test"] & patients["holdout"]
    if shared:
        raise ValueError(f"{len(shared)} patient(s) in both test and holdout: {sorted(shared)[:5]}")
    may_fit = role.isin(["train", "distill"]).to_numpy()
    gold = may_fit & (obs["label_source"] == "pathologist").to_numpy()
    for head, rows in fit_rows.items():
        bad = np.asarray(rows, dtype=bool) & ~may_fit
        if bad.any():
            raise ValueError(f"head {head}: {bad.sum()} fitted unit(s) from role(s) {sorted(set(role[bad]))}")
    for head, rows in calibration.items():
        bad = np.asarray(rows, dtype=bool) & ~gold
        if bad.any():
            raise ValueError(f"head {head}: {bad.sum()} calibration unit(s) without a gold label or from role(s) "
                             f"{sorted(set(role[bad]))}")
    is_path = (obs["label_source"] == "pathologist").to_numpy()
    table = pd.DataFrame({"role": role, "patient": obs["patient"].to_numpy(), "bulk": (obs["kind"] == "bulk").to_numpy(),
                          "spot": (obs["unit"] == "spot").to_numpy(),
                          "gold_any": is_path & obs["tls_any"].notna().to_numpy(),
                          "gold_maturity": is_path & obs["tls_mature"].notna().to_numpy()})
    summary = table.groupby("role").agg(patients=("patient", "nunique"), units=("patient", "size"), bulk=("bulk", "sum"),
                                        spots=("spot", "sum"), gold_any=("gold_any", "sum"), gold_maturity=("gold_maturity", "sum"))
    return summary.reindex(ROLES).fillna(0).astype(int)


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
