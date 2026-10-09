"""Association of a score with ICI outcome, adjusted for confounders.

Cox (OS / PFS): score + covariates, stratified by cohort; C-index = Harrell's C within strata.
Logistic (response / DCR): score + covariates + cohort fixed effect; discrimination = AUC of the linear predictor.
Scores are z-scored within cohort -> HR / OR per 1 SD. Comparisons: paired bootstrap (patients resampled
within cohort, all models refit on the same replicate) of ΔC-index / ΔAUC; nested models also get a
likelihood-ratio test.

A model is None (covariates only), one score column, or a tuple of score columns (terms); the reported
HR / OR is that of the LAST term, i.e. the one the model adds over its nested base.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)

MIN_EVENTS = 10
NUMERIC_COVARIATES = ("age", "tmb", "tmb_high", "pdl1_tc")
COVARIATES_ONLY = "covariates only"
RIDGE_FALLBACK = 0.01  # used only when the unpenalised Cox fit does not converge (noted per model)


def zscore_within(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for g in pd.unique(groups):
        m = groups == g
        sd = np.nanstd(values[m])
        out[m] = (values[m] - np.nanmean(values[m])) / sd if sd > 0 else 0.0
    return out


def covariate_matrix(clin: pd.DataFrame, covariates: list[str], min_level: int = 5) -> pd.DataFrame:
    """Numeric: median-imputed + missing indicator. Categorical: missing -> 'unknown', levels rarer than
    `min_level` -> 'other', one-hot with the most frequent level as reference. Constant columns dropped."""
    parts = []
    for c in covariates:
        if c in NUMERIC_COVARIATES:
            x = pd.to_numeric(clin[c], errors="coerce")
            parts.append(x.fillna(x.median()).rename(c))
            if x.isna().any():
                parts.append(x.isna().astype(int).rename(f"{c}_missing"))
            continue
        x = clin[c].astype(object).where(clin[c].notna(), "unknown").astype(str)
        counts = x.value_counts()
        x = x.where(x.map(counts) >= min_level, "other")
        levels = x.value_counts().index.tolist()  # most frequent first = reference
        for level in levels[1:]:
            parts.append((x == level).astype(int).rename(f"{c}_{level}"))
    if not parts:
        return pd.DataFrame(index=clin.index)
    X = pd.concat(parts, axis=1).astype(float)
    return X.loc[:, X.nunique() > 1]


def stratified_cindex(time, event, risk, strata) -> float:
    """Harrell's C over pairs inside the same stratum (a pair across cohorts says nothing about the score)."""
    time, event, risk, strata = map(np.asarray, (time, event, risk, strata))
    concordant, comparable = 0.0, 0.0
    for s in pd.unique(strata):
        m = strata == s
        t, e, r = time[m], event[m], risk[m]
        earlier = (t[:, None] < t[None, :]) & (e[:, None] == 1)
        concordant += (earlier * ((r[:, None] > r[None, :]) + 0.5 * (r[:, None] == r[None, :]))).sum()
        comparable += earlier.sum()
    return float(concordant / comparable) if comparable else np.nan


def _terms(score) -> list[str]:
    if score is None:
        return []
    return [score] if isinstance(score, str) else list(score)


def model_name(score) -> str:
    return " + ".join(_terms(score)) if score is not None else COVARIATES_ONLY


def _term_rows(df, score) -> pd.Series:
    return df[_terms(score)].notna().all(axis=1)


def _nan_result(note: str, **extra) -> dict:
    return {"estimate": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "metric": np.nan, "note": note,
            "loglik": np.nan, "k": np.nan, **extra}


def _clean_design(covars: pd.DataFrame, rows: np.ndarray, groups) -> pd.DataFrame:
    """Rows of the design; drop columns without variation inside any group (cohort strata / fixed effects
    already absorb them, keeping them makes the fit singular)."""
    X = covars.loc[rows]
    g = np.asarray(groups)[rows]
    within = (X - X.groupby(g).transform("mean")).to_numpy(dtype=float)
    keep, rank = [], 0
    for k in range(within.shape[1]):  # greedy: keep a column only if it adds within-group rank
        r = np.linalg.matrix_rank(within[:, keep + [k]], tol=1e-8) if len(within) else 0
        if r > rank:
            keep, rank = keep + [k], r
    return X.iloc[:, keep]


def _cox(df, duration, event, score, covars, strata) -> dict:
    from lifelines import CoxPHFitter

    terms = _terms(score)
    rows = (df[[duration, event]].notna().all(axis=1) & _term_rows(df, score)).to_numpy()
    data = _clean_design(covars, rows, strata).copy()
    for t in terms:
        data[f"t:{t}"] = df.loc[rows, t].to_numpy()
    data["_t"], data["_e"] = df.loc[rows, duration].to_numpy(), df.loc[rows, event].to_numpy()
    data["_s"] = np.asarray(strata)[rows]
    n_events = int(data["_e"].sum())
    base = {"n": int(rows.sum()), "events": n_events}
    if n_events < MIN_EVENTS:
        return _nan_result(f"too few events ({n_events})", **base)
    if data.shape[1] == 3:  # nothing varies inside strata: the model is the stratified null
        return {**_nan_result(""), **base, "metric": stratified_cindex(data["_t"], data["_e"], np.zeros(len(data)), data["_s"])}
    fitter, note = None, ""
    for penalizer in (0.0, RIDGE_FALLBACK):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitter = CoxPHFitter(penalizer=penalizer).fit(data, "_t", "_e", strata=["_s"])
            note = f"ridge {penalizer}" if penalizer else ""
            break
        except Exception as exc:  # lifelines convergence / singular design
            note = f"fit failed: {type(exc).__name__}"
    if fitter is None:
        return _nan_result(note, **base)
    risk = fitter.predict_partial_hazard(data).to_numpy()
    out = {"estimate": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "note": note, **base,
           "metric": stratified_cindex(data["_t"], data["_e"], risk, data["_s"]),
           "loglik": float(fitter.log_likelihood_), "k": len(fitter.params_)}
    if terms:
        s = fitter.summary.loc[f"t:{terms[-1]}"]
        out.update(estimate=float(s["exp(coef)"]), lo=float(s["exp(coef) lower 95%"]),
                   hi=float(s["exp(coef) upper 95%"]), p=float(s["p"]))
    return out


def _logit(df, outcome, score, covars, groups) -> dict:
    import statsmodels.api as sm

    terms = _terms(score)
    rows = df[outcome].notna() & _term_rows(df, score)
    y_all = df[outcome]
    informative = y_all[rows].groupby(np.asarray(groups)[rows.to_numpy()]).transform("nunique") > 1
    rows.loc[informative.index] &= informative  # cohorts with one outcome class carry no within-cohort info
    rows = rows.to_numpy()
    y = y_all.to_numpy()[rows]
    base = {"n": int(rows.sum()), "events": int(np.nansum(y))}
    if base["n"] == 0 or len(np.unique(y)) < 2 or min(base["events"], base["n"] - base["events"]) < MIN_EVENTS:
        return _nan_result("too few outcomes of one class", **base)
    X = _clean_design(covars, rows, groups).copy()
    fe = pd.get_dummies(pd.Series(np.asarray(groups)[rows], index=X.index), prefix="cohort", drop_first=True, dtype=float)
    X = pd.concat([X, fe], axis=1)
    for t in terms:
        X[f"t:{t}"] = df.loc[rows, t].to_numpy()
    X = sm.add_constant(X, has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.Logit(y, X).fit(disp=0, maxiter=200)
    except Exception as exc:  # perfect separation / singular design
        return _nan_result(f"fit failed: {type(exc).__name__}", **base)
    out = {"estimate": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "note": "", **base,
           "metric": float(roc_auc_score(y, fit.fittedvalues)), "loglik": float(fit.llf), "k": len(fit.params)}
    if terms:
        last = f"t:{terms[-1]}"
        lo, hi = fit.conf_int().loc[last]
        out.update(estimate=float(np.exp(fit.params[last])), lo=float(np.exp(lo)), hi=float(np.exp(hi)),
                   p=float(fit.pvalues[last]))
    return out


def fit_model(df, kind: str, outcome, score, covars, groups) -> dict:
    """kind 'cox': outcome = (duration, event); kind 'logit': outcome = binary column.
    score: None = covariates only, a column, or a tuple of columns (estimate = last term)."""
    if kind == "cox":
        return _cox(df, outcome[0], outcome[1], score, covars, groups)
    return _logit(df, outcome, score, covars, groups)


def fit_cox(df, duration, event, score, covars, strata) -> dict:
    r = _cox(df, duration, event, score, covars, strata)
    return {"hr": r["estimate"], "hr_lo": r["lo"], "hr_hi": r["hi"], "p": r["p"], "c_index": r["metric"],
            "n": r["n"], "events": r["events"], "note": r["note"]}


def fit_logit(df, outcome, score, covars, groups) -> dict:
    r = _logit(df, outcome, score, covars, groups)
    return {"or": r["estimate"], "or_lo": r["lo"], "or_hi": r["hi"], "p": r["p"], "auc": r["metric"],
            "n": r["n"], "events": r["events"], "note": r["note"]}


def nested_lrt(big: dict, small: dict) -> dict:
    """Likelihood-ratio test of two fits of nested models on the same rows; NaN if not comparable
    (different rows, failed or ridge-penalised fit)."""
    comparable = (big["n"] == small["n"] and big["k"] > small["k"] and not big["note"] and not small["note"]
                  and np.isfinite(big["loglik"]) and np.isfinite(small["loglik"]))
    if not comparable:
        return {"lr_stat": np.nan, "lr_df": np.nan, "lr_p": np.nan}
    stat, dof = max(2 * (big["loglik"] - small["loglik"]), 0.0), int(big["k"] - small["k"])
    return {"lr_stat": float(stat), "lr_df": dof, "lr_p": float(chi2.sf(stat, dof))}


def _resample_within(groups: np.ndarray, rng) -> np.ndarray:
    idx = [rng.choice(np.flatnonzero(groups == g), size=int((groups == g).sum()), replace=True) for g in pd.unique(groups)]
    return np.concatenate(idx)


def bootstrap_metrics(df, kind, outcome, scores, covars, groups, n_boot: int = 500, seed: int = 0):
    """Point metric of every model (one per score; None = covariates only) and an (n_boot × models) matrix of
    the same metrics on paired bootstrap replicates (patients resampled within cohort, every model refit)."""
    groups = np.asarray(groups)

    def metrics(frame, X, g):
        return [fit_model(frame, kind, outcome, s, X, g)["metric"] for s in scores]

    point = np.asarray(metrics(df, covars, groups), dtype=float)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        idx = _resample_within(groups, rng)
        frame, X = df.iloc[idx].reset_index(drop=True), covars.iloc[idx].reset_index(drop=True)
        reps.append(metrics(frame, X, groups[idx]))
    return point, np.asarray(reps, dtype=float).reshape(n_boot, len(scores))


def delta_table(scores, point, reps, pairs) -> pd.DataFrame:
    """Δmetric = metric(a) − metric(b) for every (a, b) in `pairs`, percentile 95% CI over replicates."""
    position = {s: i for i, s in enumerate(scores)}
    rows = []
    for a, b in pairs:
        d = reps[:, position[a]] - reps[:, position[b]]
        d = d[np.isfinite(d)]
        lo, hi = np.percentile(d, [2.5, 97.5]) if len(d) else (np.nan, np.nan)
        rows.append({"reference": model_name(a), "other": model_name(b),
                     "metric_reference": point[position[a]], "metric_other": point[position[b]],
                     "delta": point[position[a]] - point[position[b]], "delta_lo": float(lo), "delta_hi": float(hi),
                     "n_boot_ok": int(len(d))})
    return pd.DataFrame(rows)


def bootstrap_compare(df, kind, outcome, reference, others, covars, groups, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Δmetric of the reference score's model vs each other model (None = covariates only), paired bootstrap CI."""
    scores = [reference] + list(others)
    point, reps = bootstrap_metrics(df, kind, outcome, scores, covars, groups, n_boot, seed)
    return delta_table(scores, point, reps, [(reference, o) for o in others])
