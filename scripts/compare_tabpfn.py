"""Side experiment: TabPFN against the existing families, same rows, same blocks and bootstrap as the model
(`pipeline.evaluate_split`). The deployed model (models/tls_model.pkl) is only read.

Feature selection (`--selection`):
    own      (default) every family selects for itself: lgbm by the deployed selector (its own SHAP), logreg by its
             own elastic-net coefficients (`coef`), tabpfn takes every base column (`all`, no selection)
    shared   one panel per head, chosen by the deployed selector, for all families (first version of the experiment)
Tables and caches of `own` carry the suffix `_own`; `shared` keeps the unsuffixed names.

Models: dummy (constant = gold training prevalence), logreg (family enet), lgbm, tabpfn; the last three as
    teacher   fitted on the gold Visium train rows only
    student   fitted on the Visium train rows + the TCGA distill rows with the soft labels of its own teacher
Heads any and mature are fitted; the 3-class rows are their product (hierarchy form). Raw head probabilities, no
calibration: read AUROC / AUPRC (rank-based), not Brier, across models.

TabPFN and the soft labels: the distill rows enter as weighted duplicates (`pipeline.fit_set`). `--weights resample`
(default) turns the weights into row frequencies by a weighted bootstrap of the fit rows; `--weights native` passes
them to `TabPFNClassifier.fit(sample_weight=...)`, which upstream marks experimental and which raises when the
checkpoint takes no row weights. Either way the student is an approximation of the exact log-loss distillation that
logreg and lgbm get.

    uv sync --group tabpfn                               # needs TABPFN_TOKEN (licence: https://ux.priorlabs.ai) and a GPU
    uv run python scripts/compare_tabpfn.py --pilot      # TabPFN teacher, head any, test split only: time + GPU memory
    uv run python scripts/compare_tabpfn.py              # -> results/tabpfn_comparison_own.tsv, results/tabpfn_deltas_own.tsv

Idempotent: the predictions of a finished family are cached in results/tabpfn/<family>.npz and reused on a re-run;
delete that folder after changing the data, config.yaml or models/tls_model.pkl. With `--weights resample` the TabPFN
fit rows are one weighted bootstrap draw (fixed seed): that noise is not part of the reported intervals.
Log: results/compare_tabpfn.log. Needs notebooks 01-03 (data/processed, models/tls_model.pkl).
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tls_model import pipeline, prepare  # noqa: E402
from tls_model.models import TABPFN_WEIGHTS, three_class  # noqa: E402

RES = ROOT / "results"
CACHE = RES / "tabpfn"
FAMILIES = {"logreg": "enet", "lgbm": "lgbm", "tabpfn": "tabpfn"}  # label in the tables -> HeadModel family
OWN_SELECTOR = {"enet": "coef", "tabpfn": "all"}  # family -> its own selector; lgbm keeps the deployed one
ROLES = ("teacher", "student")
HEADS = ("any", "mature")
SPLITS = ("test", "ood")
CHUNK = 10_000  # units per feature chunk: the unit table alone is ~7 GB (see scripts/score_baselines.py)
log = logging.getLogger("compare_tabpfn")


def gpu_peak_gb() -> float:
    """Peak GPU memory of this process in GB; NaN without torch / CUDA."""
    try:
        import torch
    except ImportError:
        return float("nan")
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else float("nan")


def fit_family(family, B, names, obs, meta, cfg, selectors, weights, panels, heads=HEADS, roles=ROLES, own=True) -> dict:
    """{role: {head: fitted HeadModel}}. `panels` caches the feature panel: it is selected once, on the gold teacher
    rows, and reused by the student. With `own` the cache key is (family, head) and the selector is the family's own
    (`OWN_SELECTOR`); without it the key is the head alone, so every family sees the same columns. The main pipeline
    instead lets a student re-select on its own fit set."""
    fc, seed = cfg["features"], cfg["split"]["seed"]

    def fit_head(role: str, head: str, X, y, w):
        selector = OWN_SELECTOR.get(family, selectors[head]) if own else selectors[head]
        key = (family, head) if own else head
        candidate = {"family": family, "selector": selector, "tabpfn_weights": weights}
        start = time.time()
        model = pipeline.make_head(candidate, meta, fc["n_panel"], fc["n_pair_genes"], seed, panels.get(key))
        model.fit(X, names, y, w)
        panels.setdefault(key, (model.panel, model.pair_genes))
        log.info("fit %s %s %s: %d rows x %d features, %.0f s", family, role, head, len(y), len(model.panel),
                 time.time() - start)
        return model

    out = {"teacher": {head: fit_head("teacher", head, *pipeline.fit_set(B, obs, head, None)) for head in heads}}
    if "student" not in roles:
        return out
    distill = pipeline.distill_rows(obs)
    final = {}
    for head in heads:
        final[head] = np.full(len(obs), np.nan)
        final[head][distill] = out["teacher"][head].predict_proba(B[distill], names)
    pseudo = pipeline.PseudoLabels(rows=distill, by_fold={}, final=final)
    out["student"] = {head: fit_head("student", head, *pipeline.fit_set(B, obs, head, pseudo)) for head in heads}
    return out


def score_units(units, rows: np.ndarray, meta: dict, variant: dict, fitted: dict) -> dict[str, np.ndarray]:
    """{role: (len(rows), 3) class probabilities}; with one fitted head only (pilot) {role: P(TLS)}."""
    one_head = any(set(heads) != set(HEADS) for heads in fitted.values())
    out = {role: np.empty(len(rows) if one_head else (len(rows), 3)) for role in fitted}
    for start in range(0, len(rows), CHUNK):
        part = units[rows[start : start + CHUNK]]
        B, names = pipeline.feature_matrix(part, meta, variant["representation"], bool(variant["offset"]))
        for role, heads in fitted.items():
            p_any = heads["any"].predict_proba(B, names)
            out[role][start : start + CHUNK] = p_any if one_head else three_class(p_any, heads["mature"].predict_proba(B, names))
        log.info("scored %d / %d units", min(start + CHUNK, len(rows)), len(rows))
    return out


def paired_delta(y, p_a, p_b, groups=None, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """AUROC(a) - AUROC(b) on the same units with a paired percentile 95 % CI (the resampling of
    `evaluate.bootstrap_ci`: whole groups when given, single units otherwise)."""
    y, p_a, p_b = np.asarray(y).astype(int), np.asarray(p_a), np.asarray(p_b)
    point = float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b))
    rng = np.random.default_rng(seed)
    if groups is None:
        members = [np.array([i]) for i in range(len(y))]
    else:
        groups = np.asarray(groups)
        members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        if len(np.unique(y[idx])) >= 2:
            stats.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    if not stats:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


def delta_table(obs: pd.DataFrame, probs: dict[tuple[str, str], np.ndarray], n_boot: int) -> pd.DataFrame:
    """TabPFN minus logreg / lgbm, per split, head, block and role."""
    rows = []
    for role in ROLES:
        heads = {label: dict(zip(HEADS, pipeline.heads_from_three(probs[label, role]))) for label in FAMILIES}
        for split in SPLITS:
            for head in HEADS:
                y = obs[pipeline.HEAD_LABEL[head]].to_numpy(dtype=float)
                scored = pipeline.scored_rows(obs, head, split)
                for block, in_block in pipeline._blocks(obs):
                    mask = scored & in_block
                    if mask.sum() < 2 or len(np.unique(y[mask])) < 2:
                        continue
                    groups = pipeline._bootstrap_groups(block, obs, mask)
                    for other in ("logreg", "lgbm"):
                        delta, lo, hi = paired_delta(y[mask], heads["tabpfn"][head][mask], heads[other][head][mask],
                                                     groups, n_boot)
                        rows.append({"split": split, "head": head, "block": block, "role": role, "n": int(mask.sum()),
                                     "n_pos": int(y[mask].sum()), "comparison": f"tabpfn - {other}",
                                     "delta_auroc": delta, "delta_lo": lo, "delta_hi": hi,
                                     "verdict": "better" if lo > 0 else "worse" if hi < 0 else "no difference"})
    return pd.DataFrame(rows)


def pilot(units, meta, cfg, variant, selectors, weights, B, names, fit_obs, own) -> int:
    """TabPFN teacher, head any, scored on the test split: wall time and peak GPU memory before the full run."""
    start = time.time()
    fitted = fit_family("tabpfn", B, names, fit_obs, meta, cfg, selectors, weights, {}, heads=("any",), roles=("teacher",),
                        own=own)
    fit_seconds = time.time() - start
    rows = np.flatnonzero((units.obs["split"] == "test").to_numpy())
    start = time.time()
    p = score_units(units, rows, meta, variant, fitted)["teacher"]
    table = pipeline.evaluate_split(units.obs.iloc[rows], p, "test", n_boot=200)
    log.info("\n%s", table[["block", "n", "n_pos", "auroc", "auroc_lo", "auroc_hi"]].round(3).to_string())
    log.info("PILOT fit %.0f s, predict %d units in %.0f s, peak GPU memory %.1f GB", fit_seconds, len(rows),
             time.time() - start, gpu_peak_gb())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="TabPFN vs dummy / logreg / lgbm on the test and holdout splits.")
    parser.add_argument("--weights", choices=TABPFN_WEIGHTS, default="resample", help="how TabPFN gets the sample weights")
    parser.add_argument("--pilot", action="store_true", help="TabPFN teacher, head any, test split only; writes no table")
    parser.add_argument("--selection", choices=("own", "shared"), default="own", help="feature selection per family or one shared panel")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(RES / "compare_tabpfn.log"), logging.StreamHandler(sys.stdout)])
    cfg = pipeline.load_config(ROOT / "config.yaml")
    units, meta = prepare.load_units(cfg, ROOT)
    with open(ROOT / "models" / "tls_model.pkl", "rb") as handle:
        saved = pickle.load(handle)
    variant = saved["variant"]
    selectors = {head: saved["best"][head]["selector"] or "corr" for head in HEADS}
    own = args.selection == "own"
    suffix = "_own" if own else ""
    log.info("units %s, variant %s, selectors %s, selection %s, tabpfn weights %s", units.shape, variant, selectors,
             args.selection, args.weights)

    fit_mask = units.obs["role"].isin(["train", "distill"]).to_numpy()
    fit_obs = units.obs[fit_mask].copy()
    B, names = pipeline.feature_matrix(units[fit_mask], meta, variant["representation"], bool(variant["offset"]))
    if args.pilot:
        return pilot(units, meta, cfg, variant, selectors, args.weights, B, names, fit_obs, own)

    rows = np.flatnonzero(units.obs["split"].isin(SPLITS).to_numpy())
    obs = units.obs.iloc[rows].copy()
    log.info("fit rows %d, scored rows (test + holdout) %d", int(fit_mask.sum()), len(rows))
    probs: dict[tuple[str, str], np.ndarray] = {}
    panels: dict = {}
    done, failed = [], []
    for i, (label, family) in enumerate(FAMILIES.items(), 1):
        cache = CACHE / (f"{label}_{args.weights}{suffix}.npz" if label == "tabpfn" else f"{label}{suffix}.npz")
        log.info("[%d/%d] START %s", i, len(FAMILIES), label)
        try:
            if cache.exists():
                loaded = np.load(cache)
                scores = {role: loaded[role] for role in ROLES}
                log.info("[%d/%d] SKIP (cached %s) %s", i, len(FAMILIES), cache.name, label)
            else:
                fitted = fit_family(family, B, names, fit_obs, meta, cfg, selectors, args.weights, panels, own=own)
                scores = score_units(units, rows, meta, variant, fitted)
                np.savez_compressed(cache, **scores)
            probs.update({(label, role): scores[role] for role in ROLES})
            done.append(label)
            log.info("[%d/%d] OK %s (peak GPU memory %.1f GB)", i, len(FAMILIES), label, gpu_peak_gb())
        except Exception:  # one family failing (e.g. GPU out of memory) must not lose the others
            failed.append(label)
            log.exception("[%d/%d] FAIL %s", i, len(FAMILIES), label)
    dummy = pipeline.dummy_scores(units.obs, meta)[rows]
    del units, B

    tables = [pipeline.evaluate_split(obs, dummy, split, args.n_boot).assign(model="dummy", role="-") for split in SPLITS]
    for (label, role), p in probs.items():
        for split in SPLITS:
            log.info("metrics %s %s on %s", label, role, split)
            tables.append(pipeline.evaluate_split(obs, p, split, args.n_boot).assign(model=label, role=role))
    out = pd.concat(tables, ignore_index=True)
    key = ["split", "head", "block", "model", "role"]
    out = out[[*key, *[c for c in out.columns if c not in key]]]
    out["tabpfn_weights"] = np.where(out["model"] == "tabpfn", args.weights, "")
    out["selection"] = args.selection
    out.to_csv(RES / f"tabpfn_comparison{suffix}.tsv", sep="\t", index=False)
    pivot = out[out["head"].isin(HEADS)].pivot_table(index=["split", "head", "block"], columns=["role", "model"], values="auroc")
    log.info("\n%s", pivot.round(3).to_string())
    if not failed:
        deltas = delta_table(obs, probs, args.n_boot)
        deltas.to_csv(RES / f"tabpfn_deltas{suffix}.tsv", sep="\t", index=False)
        log.info("\n%s", deltas.round(3).to_string())
    log.info("SUMMARY done=%d failed=%d", len(done), len(failed))
    if failed:
        log.error("FAILED: %s (re-run: finished families are cached)", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
