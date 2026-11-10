"""Ablation: does self-distillation on TCGA help? Teacher (Visium train only) vs student (Visium train + TCGA
pseudo-labels), same candidates as the saved model, scored on the TCGA test split, Visium test spots and the bulk
holdout cohorts. Diagnostic only (raw head probabilities; AUROC is rank-based so calibration does not matter).

    uv run python scripts/ablation_distillation.py   -> results/ablation_distillation.tsv
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tls_model import pipeline, prepare  # noqa: E402
from tls_model.evaluate import binary_metrics  # noqa: E402

log = logging.getLogger("ablation")


def blocks(obs: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    """(split, block, mask) pairs: TCGA test per cohort + pooled, Visium test spots, holdout bulk per cohort."""
    bulk = (obs["kind"] == "bulk").to_numpy()
    out = []
    for split in ("test", "ood"):
        in_split = (obs["split"] == split).to_numpy() & bulk
        for cohort in sorted(obs.loc[in_split, "cohort"].unique()):
            out.append((split, f"bulk:{cohort}", in_split & (obs["cohort"] == cohort).to_numpy()))
        out.append((split, "bulk:all", in_split))
    out.append(("test", "visium:spots", (obs["split"] == "test").to_numpy() & (obs["unit"] == "spot").to_numpy()))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "results" / "ablation_distillation.log"), logging.StreamHandler(sys.stdout)])
    cfg = pipeline.load_config(ROOT / "config.yaml")
    units, meta = prepare.load_units(cfg, ROOT)
    keep = (units.obs["role"] != "holdout").to_numpy() | (units.obs["kind"] == "bulk").to_numpy()
    units = units[keep].copy()
    obs = units.obs
    with open(ROOT / "models" / "tls_model.pkl", "rb") as handle:
        saved = pickle.load(handle)
    variant, best = saved["variant"], saved["best"]
    fc, seed = cfg["features"], cfg["split"]["seed"]
    log.info("units %s, variant %s, candidates %s", units.shape, variant, best)
    B, names = pipeline.feature_matrix(units, meta, variant["representation"], bool(variant["offset"]))
    distill = pipeline.distill_rows(obs)

    teacher, student = {}, {}
    for head in ("any", "mature"):
        teacher[head] = pipeline.make_head(best[head], meta, fc["n_panel"], fc["n_pair_genes"], seed)
        teacher[head].fit(*(lambda X, y, w: (X, names, y, w))(*pipeline.fit_set(B, obs, head, None)))
        log.info("teacher %s fitted (%d units)", head, pipeline.train_rows(obs, head).sum())
    final = {}
    for head in ("any", "mature"):
        final[head] = np.full(len(obs), np.nan)
        final[head][distill] = teacher[head].predict_proba(B[distill], names)
    pseudo = pipeline.PseudoLabels(rows=distill, by_fold={}, final=final)
    for head in ("any", "mature"):
        X, y, w = pipeline.fit_set(B, obs, head, pseudo)
        student[head] = pipeline.make_head(best[head], meta, fc["n_panel"], fc["n_pair_genes"], seed).fit(X, names, y, w)
        log.info("student %s fitted (%d rows incl. soft duplicates)", head, len(y))

    rows = []
    for head in ("any", "mature"):
        y = obs[pipeline.HEAD_LABEL[head]].to_numpy(dtype=float)
        p_t, p_s = teacher[head].predict_proba(B, names), student[head].predict_proba(B, names)
        for split, block, in_block in blocks(obs):
            mask = pipeline.scored_rows(obs, head, split) & in_block
            if mask.sum() < 5 or len(np.unique(y[mask])) < 2:
                continue
            groups = obs["section"].to_numpy()[mask] if block == "visium:spots" else None
            t = binary_metrics(y[mask], p_t[mask], groups, n_boot=300)
            s = binary_metrics(y[mask], p_s[mask], groups, n_boot=300)
            rows.append({"head": head, "split": split, "block": block, "n": t["n"], "n_pos": t["n_pos"],
                         "teacher_auroc": t["auroc"], "teacher_lo": t["auroc_lo"], "teacher_hi": t["auroc_hi"],
                         "student_auroc": s["auroc"], "student_lo": s["auroc_lo"], "student_hi": s["auroc_hi"],
                         "delta": s["auroc"] - t["auroc"]})
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "results" / "ablation_distillation.tsv", sep="\t", index=False)
    pd.set_option("display.width", 200)
    log.info("\n%s", table.round(3).to_string())


if __name__ == "__main__":
    main()
