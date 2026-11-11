"""Reference scores on the test and holdout splits: a dummy (constant = gold training prevalence per unit kind) and the
three published TLS signatures (ssGSEA on the model genes), scored with the same blocks and bootstrap as the model
(`pipeline.evaluate_split`). Our model's rows are taken from results/test_metrics.tsv and results/ood_metrics.tsv
(head any), so notebooks 03 / 04 must have run. Output: results/baselines.tsv (split, model, block, metrics; Brier is
left out for signatures, which are rankings, not probabilities). Log: results/score_baselines.log.

    uv run python scripts/score_baselines.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tls_model import pipeline, prepare  # noqa: E402

RES = ROOT / "results"
OURS = "ours (P(TLS))"
SPLIT_FILES = {"test": "test_metrics.tsv", "ood": "ood_metrics.tsv"}
BRIER = ("brier", "brier_lo", "brier_hi")
CHUNK = 10_000
log = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(RES / "score_baselines.log"), logging.StreamHandler(sys.stdout)])
    cfg = pipeline.load_config(ROOT / "config.yaml")
    units, meta = prepare.load_units(cfg, ROOT)
    keep = units.obs["split"].isin(list(SPLIT_FILES)).to_numpy()
    log.info("units on test + holdout: %d of %d", int(keep.sum()), units.n_obs)
    # Score the whole table in small chunks (no subset copy of X) and drop X before the bootstraps: the unit table
    # alone is ~7 GB, a subset copy plus the ssGSEA temporaries exhausted 25 GB.
    scores = {name: score[keep] for name, score in pipeline.baseline_scores(units, meta, chunk=CHUNK).items()}
    obs = units.obs[keep].copy()
    del units
    tables = []
    for split, name in SPLIT_FILES.items():
        ours = pd.read_csv(RES / name, sep="\t")
        tables.append(ours[ours["head"] == "any"].assign(model=OURS))
        for i, (model, score) in enumerate(scores.items(), 1):
            log.info("[%d/%d] %s on %s", i, len(scores), model, split)
            table = pipeline.evaluate_split(obs, score, split).assign(model=model)
            if model != "dummy":
                table[[c for c in BRIER if c in table.columns]] = np.nan
            tables.append(table)
    out = pd.concat(tables, ignore_index=True)
    out = out[["split", "model", "block", *[c for c in out.columns if c not in ("split", "model", "block", "head")]]]
    out.to_csv(RES / "baselines.tsv", sep="\t", index=False)
    log.info("\n%s", out.pivot_table(index=["split", "block"], columns="model", values="auroc").round(3).to_string())
    log.info("SUMMARY done: %d rows", len(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
