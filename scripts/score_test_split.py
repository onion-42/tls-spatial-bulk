"""Re-score the in-distribution test split with the saved model (same computation as the end of notebook 03):
results/test_metrics.tsv and results/figures/03_reliability_test.png. Use after a change of the deployed form that
does not need the 90-minute notebook 03 run.

    uv run python scripts/score_test_split.py
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tls_model import evaluate, pipeline, prepare  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = pipeline.load_config(ROOT / "config.yaml")
    units, meta = prepare.load_units(cfg, ROOT)
    units = units[(units.obs["role"] != "holdout").to_numpy()].copy()
    obs = units.obs
    with open(ROOT / "models" / "tls_model.pkl", "rb") as handle:
        saved = pickle.load(handle)
    logging.info("form %s, variant %s", saved["form"], saved["variant"])
    probs = pipeline.predict_three_in_chunks(units, meta, saved["variant"], saved)
    p = dict(zip(("any", "mature"), pipeline.heads_from_three(probs)))

    test = pipeline.evaluate_split(obs, probs, "test")
    test.to_csv(ROOT / "results" / "test_metrics.tsv", sep="\t", index=False)
    logging.info("\n%s", evaluate.metrics_table(test.to_dict("records")).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, head in zip(axes, ("any", "mature")):
        scored = pipeline.scored_rows(obs, head, "test")
        y = obs[pipeline.HEAD_LABEL[head]].to_numpy(dtype=float)
        for kind, mask in (("bulk", obs["kind"] == "bulk"), ("Visium spots", obs["unit"] == "spot")):
            m = scored & mask.to_numpy()
            if m.sum() > 10 and len(np.unique(y[m])) == 2:
                evaluate.reliability_curve(ax, y[m], p[head][m], kind)
        ax.set_title(f"test reliability — head {head}")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / "03_reliability_test.png", dpi=120)


if __name__ == "__main__":
    main()
