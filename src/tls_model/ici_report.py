"""ICI validation: per diagnosis × endpoint × score models, paired ΔC-index / ΔAUC, forest plots.

Every model: score (z-scored within cohort) + confounders; Cox stratified by cohort, logistic with cohort
fixed effects. 'All' = all diagnoses pooled (cohort strata / fixed effects nest diagnosis).
Parts are cached as TSV per (analysis, diagnosis, endpoint): re-running computes only what is missing.
All models of one part are fitted on the same complete-case rows (every score of every model present), so
nested pairs also get a likelihood-ratio test.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tls_model import ici_stats

log = logging.getLogger(__name__)

OURS = "TLS model (ours)"
ENDPOINTS = {
    "OS": ("cox", ("os", "os_event")),
    "PFS": ("cox", ("pfs", "pfs_event")),
    "Response": ("logit", "response"),
    "DCR": ("logit", "dcr"),
}
POOLED = "All"
CHEM = "12-chemokine (TLS_Chem)"
INTERACTION = "12-chemokine × ours"
# Added value of our score over the 12-chemokine signature: nested models, label -> (bigger, smaller).
_CHEM, _CHEM_ML, _CHEM_ML_X = (CHEM,), (CHEM, OURS), (CHEM, OURS, INTERACTION)
ADDED_MODELS = [_CHEM, _CHEM_ML, _CHEM_ML_X]
ADDED_PAIRS = {
    "a: ours over 12-chemokine": (_CHEM_ML, _CHEM),
    "b: ours + 12-chem×ours over 12-chemokine": (_CHEM_ML_X, _CHEM),
    "b vs a: 12-chem×ours": (_CHEM_ML_X, _CHEM_ML),
}
# Same on the TMB-high + PD-L1 subset (both are covariates there; no interactions with them).
NESTED_ADDED_MODELS = [None, OURS] + ADDED_MODELS
NESTED_ADDED_PAIRS = {
    "c: ours over TMB + PD-L1": (OURS, None),
    "d: ours over TMB + PD-L1 + 12-chemokine": (_CHEM_ML, _CHEM),
    "e: ours + 12-chem×ours over TMB + PD-L1 + 12-chemokine": (_CHEM_ML_X, _CHEM),
    "e vs d: 12-chem×ours": (_CHEM_ML_X, _CHEM_ML),
}
NESTED_COVARIATES = ("tmb_high", "pdl1_tc")
EPS = 1e-6


def analysis_frame(clin: pd.DataFrame, p_tls: pd.Series, signature_scores: pd.DataFrame) -> pd.DataFrame:
    """ICI-treated patients of included histologies + every score z-scored within cohort (ours on the logit scale)."""
    keep = clin["is_ici"].astype(bool) & clin["diagnosis"].notna()
    df = clin[keep].copy()
    groups = df["cohort"].to_numpy()
    p = np.clip(p_tls.loc[df.index].to_numpy(dtype=float), EPS, 1 - EPS)
    df[OURS] = ici_stats.zscore_within(np.log(p / (1 - p)), groups)
    for name in signature_scores.columns:
        df[name] = ici_stats.zscore_within(signature_scores.loc[df.index, name].to_numpy(dtype=float), groups)
    return df


def _outcome_rows(df: pd.DataFrame, kind: str, outcome) -> pd.DataFrame:
    cols = list(outcome) if kind == "cox" else [outcome]
    return df[df[cols].notna().all(axis=1)]


def complete_rows(df: pd.DataFrame, models: list) -> pd.DataFrame:
    """Rows where every score used by any model is present."""
    cols = sorted({t for m in models for t in ici_stats.score_terms(m)})
    return df[df[cols].notna().all(axis=1)]


def with_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the 12-chemokine × ours product (both already z-scored within cohort)."""
    return df.assign(**{INTERACTION: df[CHEM] * df[OURS]})


def _subsets(df: pd.DataFrame):
    for diagnosis in sorted(df["diagnosis"].unique()):
        yield diagnosis, df[df["diagnosis"] == diagnosis]
    yield POOLED, df


def _part(out_dir: Path, analysis: str, diagnosis: str, endpoint: str) -> tuple[Path, Path]:
    stem = f"{analysis}__{diagnosis}__{endpoint}".replace(" ", "_")
    return out_dir / f"{stem}__assoc.tsv", out_dir / f"{stem}__delta.tsv"


def _lrt(fits: dict, a, b) -> dict:
    """LRT when one model's terms are a strict subset of the other's; NaN otherwise."""
    ta, tb = set(ici_stats.score_terms(a)), set(ici_stats.score_terms(b))
    if tb < ta:
        return ici_stats.nested_lrt(fits[a], fits[b])
    if ta < tb:
        return ici_stats.nested_lrt(fits[b], fits[a])
    return dict(ici_stats.NO_LRT)


def _run_one(sub, kind, outcome, scores, covariates, pairs, n_boot, seed):
    sub = complete_rows(sub, scores)
    X = ici_stats.covariate_matrix(sub, covariates)
    groups = sub["cohort"].to_numpy()
    fits = {score: ici_stats.fit_model(sub, kind, outcome, score, X, groups) for score in scores}
    assoc = pd.DataFrame([{"score": ici_stats.model_name(s), **fit, "covariates": ", ".join(X.columns)}
                          for s, fit in fits.items()])
    if assoc["metric"].isna().all() or n_boot == 0:
        return assoc, pd.DataFrame()
    point, reps = ici_stats.bootstrap_metrics(sub, kind, outcome, scores, X, groups, n_boot, seed)
    lrt = pd.DataFrame([_lrt(fits, a, b) for a, b in pairs])
    return assoc, pd.concat([ici_stats.delta_table(scores, point, reps, pairs), lrt], axis=1)


def run_analysis(df: pd.DataFrame, analysis: str, scores: list, covariates: list[str], pairs: list[tuple],
                 out_dir: Path, n_boot: int, seed: int, endpoints=ENDPOINTS, subsets=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loop diagnosis × endpoint; one cached part per combination; per-item failures are logged, not fatal."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(d, sub, e) for d, sub in (subsets or _subsets(df)) for e in endpoints]
    assoc_all, delta_all, failed = [], [], []
    for i, (diagnosis, sub, endpoint) in enumerate(jobs, 1):
        kind, outcome = endpoints[endpoint]
        assoc_path, delta_path = _part(out_dir, analysis, diagnosis, endpoint)
        tag = f"[{i}/{len(jobs)}] {analysis} {diagnosis} {endpoint}"
        if not assoc_path.exists():
            rows = _outcome_rows(sub, kind, outcome)
            log.info("%s START n=%d", tag, len(rows))
            try:
                assoc, delta = _run_one(rows, kind, outcome, scores, covariates, pairs, n_boot, seed)
            except Exception:
                log.exception("%s FAIL", tag)
                failed.append(tag)
                continue
            for frame, path in ((assoc, assoc_path), (delta, delta_path)):
                frame.assign(analysis=analysis, diagnosis=diagnosis, endpoint=endpoint).to_csv(path, sep="\t", index=False)
            log.info("%s OK", tag)
        else:
            log.info("%s SKIP (cached)", tag)
        assoc_all.append(pd.read_csv(assoc_path, sep="\t"))
        if delta_path.stat().st_size > 1:
            delta_all.append(pd.read_csv(delta_path, sep="\t"))
    if failed:
        log.error("FAILED: %s", failed)
    return pd.concat(assoc_all, ignore_index=True), (pd.concat(delta_all, ignore_index=True) if delta_all else pd.DataFrame())


def main_pairs(signatures: list[str]) -> list[tuple]:
    """Ours vs each signature, and every score vs the covariates-only model."""
    return [(OURS, s) for s in signatures] + [(s, None) for s in [OURS] + signatures]


def nested_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Patients with both TMB-high call and PD-L1 IHC category."""
    return df[df[list(NESTED_COVARIATES)].notna().all(axis=1)]


def nested_subsets(df: pd.DataFrame, min_events: int = 30):
    """Pooled always; a diagnosis on its own only if it has ≥ `min_events` OS events."""
    for diagnosis in sorted(df["diagnosis"].unique()):
        sub = df[df["diagnosis"] == diagnosis]
        if sub["os_event"].sum() >= min_events:
            yield diagnosis, sub
    yield POOLED, df


def forest_plot(assoc: pd.DataFrame, delta: pd.DataFrame, endpoint: str, scores: list[str], title: str = ""):
    """Left: HR / OR per 1 SD [95% CI] per diagnosis × score. Right: ΔC-index / ΔAUC of ours vs each signature."""
    import matplotlib.pyplot as plt

    a = assoc[(assoc["endpoint"] == endpoint) & assoc["score"].isin(scores)]
    diagnoses = [d for d in sorted(a["diagnosis"].unique()) if d != POOLED] + ([POOLED] if POOLED in set(a["diagnosis"]) else [])
    colours = dict(zip(scores, ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b"]))
    rows = [(d, s) for d in diagnoses for s in scores]
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 0.32 * len(rows) + 1.8), sharey=True,
                                     gridspec_kw={"width_ratios": [1.3, 1]})
    is_cox = ENDPOINTS[endpoint][0] == "cox"
    for y, (d, s) in enumerate(rows):
        r = a[(a["diagnosis"] == d) & (a["score"] == s)]
        if r.empty or not np.isfinite(r["estimate"].iloc[0]):
            continue
        r = r.iloc[0]
        ax_l.errorbar(r["estimate"], y, xerr=[[r["estimate"] - r["lo"]], [r["hi"] - r["estimate"]]], fmt="o",
                      color=colours[s], ms=4, capsize=2)
        ax_l.text(1.02, y, f"{r['estimate']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]  n={int(r['n'])}",
                  transform=ax_l.get_yaxis_transform(), fontsize=7, va="center")
        if s != OURS and not delta.empty:
            dr = delta[(delta["endpoint"] == endpoint) & (delta["diagnosis"] == d) & (delta["reference"] == OURS) & (delta["other"] == s)]
            if not dr.empty and np.isfinite(dr["delta"].iloc[0]):
                dr = dr.iloc[0]
                ax_r.errorbar(dr["delta"], y, xerr=[[dr["delta"] - dr["delta_lo"]], [dr["delta_hi"] - dr["delta"]]],
                              fmt="s", color=colours[s], ms=4, capsize=2)
    ax_l.axvline(1, color="grey", lw=1)
    ax_l.set_xscale("log")
    ax_l.set_xlabel(("HR" if is_cox else "OR") + " per 1 SD (95% CI)" + ("  — HR < 1 = longer survival" if is_cox else "  — OR > 1 = more response"))
    ax_r.axvline(0, color="grey", lw=1)
    ax_r.set_xlabel(("ΔC-index" if is_cox else "ΔAUC") + " ours − signature (95% bootstrap CI)")
    ax_l.set_yticks(range(len(rows)), [f"{d} · {s}" for d, s in rows], fontsize=7)
    for k in range(1, len(diagnoses)):
        for ax in (ax_l, ax_r):
            ax.axhline(k * len(scores) - 0.5, color="#dddddd", lw=0.8)
    ax_l.invert_yaxis()
    fig.suptitle(title or f"{endpoint}: adjusted association and added discrimination")
    fig.tight_layout()
    return fig


def label_pairs(delta: pd.DataFrame, labelled_pairs: dict) -> pd.DataFrame:
    """Adds a `comparison` column (label of each (bigger, smaller) model pair); other rows dropped."""
    names = {(ici_stats.model_name(a), ici_stats.model_name(b)): label for label, (a, b) in labelled_pairs.items()}
    key = list(zip(delta["reference"], delta["other"]))
    return delta.assign(comparison=[names.get(k) for k in key]).dropna(subset=["comparison"])


def added_value_plot(delta: pd.DataFrame, comparisons: list[str], title: str):
    """One panel per endpoint: ΔC-index / ΔAUC (bigger − smaller model, 95% bootstrap CI) per diagnosis ×
    comparison; LRT p printed next to each point."""
    import matplotlib.pyplot as plt

    diagnoses = [d for d in sorted(delta["diagnosis"].unique()) if d != POOLED] + [POOLED]
    rows = [(d, c) for d in diagnoses for c in comparisons]
    colours = dict(zip(comparisons, ["#1f77b4", "#d62728", "#7f7f7f", "#2ca02c"]))
    endpoints = [e for e in ENDPOINTS if (delta["endpoint"] == e).any()]
    fig, axes = plt.subplots(1, len(endpoints), figsize=(4.2 * len(endpoints), 0.3 * len(rows) + 1.8), sharey=True)
    for ax, endpoint in zip(np.atleast_1d(axes), endpoints):
        t = delta[delta["endpoint"] == endpoint]
        for y, (d, c) in enumerate(rows):
            r = t[(t["diagnosis"] == d) & (t["comparison"] == c)]
            if r.empty or not np.isfinite(r["delta"].iloc[0]):
                continue
            r = r.iloc[0]
            ax.errorbar(r["delta"], y, xerr=[[max(r["delta"] - r["delta_lo"], 0)], [max(r["delta_hi"] - r["delta"], 0)]],
                        fmt="o", color=colours[c], ms=4, capsize=2)
            if np.isfinite(r["lr_p"]):
                ax.text(1.01, y, f"p={r['lr_p']:.2g}", transform=ax.get_yaxis_transform(), fontsize=6, va="center")
        ax.axvline(0, color="grey", lw=1)
        ax.set_title(endpoint, fontsize=9)
        ax.set_xlabel("ΔC-index" if ENDPOINTS[endpoint][0] == "cox" else "ΔAUC", fontsize=8)
        for k in range(1, len(diagnoses)):
            ax.axhline(k * len(comparisons) - 0.5, color="#dddddd", lw=0.8)
    first = np.atleast_1d(axes)[0]
    first.set_yticks(range(len(rows)), [f"{d} · {c}" for d, c in rows], fontsize=7)
    first.invert_yaxis()
    fig.suptitle(title + "  (p = likelihood-ratio test, not corrected for multiplicity)", fontsize=10)
    fig.tight_layout()
    return fig
