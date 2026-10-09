"""Metrics with bootstrap 95% CIs, 3-class balanced accuracy, reliability curves, SHAP plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score

N_BOOT = 1000
METRICS = {
    "auroc": roc_auc_score,
    "auprc": average_precision_score,
    "brier": brier_score_loss,
}


def bootstrap_ci(metric, y, p, groups=None, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    """Point estimate and percentile 95% CI. With `groups`, whole groups (e.g. Visium sections) are resampled."""
    y, p = np.asarray(y), np.asarray(p)
    point = float(metric(y, p))
    rng = np.random.default_rng(seed)
    if groups is None:
        members = [np.array([i]) for i in range(len(y))]
    else:
        groups = np.asarray(groups)
        members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        if len(np.unique(y[idx])) == 2:
            stats.append(metric(y[idx], p[idx]))
    if not stats:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


def binary_metrics(y, p, groups=None, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    out = {"n": len(y), "n_pos": int(y.sum())}
    if len(np.unique(y)) < 2:
        return out
    for name, metric in METRICS.items():
        point, lo, hi = bootstrap_ci(metric, y, p, groups, n_boot, seed)
        out.update({name: point, f"{name}_lo": lo, f"{name}_hi": hi})
    return out


def three_class_labels(y_any: np.ndarray, y_mature: np.ndarray) -> np.ndarray:
    """0 = no TLS, 1 = immature, 2 = mature, -1 = TLS with unknown maturity (not scored)."""
    y_mature = np.asarray(y_mature, dtype=float)
    labels = np.where(y_mature == 1, 2, 1)
    labels = np.where(np.isnan(y_mature), -1, labels)
    return np.where(np.asarray(y_any) == 0, 0, labels)


def three_class_balanced_accuracy(y_any, y_mature, probs: np.ndarray) -> float:
    labels = three_class_labels(y_any, y_mature)
    known = labels >= 0
    return float(balanced_accuracy_score(labels[known], probs[known].argmax(axis=1)))


def section_spearman(sections, p, true_fraction) -> float:
    frame = pd.DataFrame({"s": sections, "p": p, "f": true_fraction}).groupby("s").mean()
    return float(spearmanr(frame["p"], frame["f"]).statistic)


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    """Rows -> table with 'point [lo, hi]' strings for display."""
    table = pd.DataFrame(rows)
    for name in METRICS:
        if name in table:
            table[name] = [
                f"{v:.3f} [{lo:.3f}, {hi:.3f}]" if pd.notna(v) else ""
                for v, lo, hi in zip(table[name], table[f"{name}_lo"], table[f"{name}_hi"])
            ]
            table = table.drop(columns=[f"{name}_lo", f"{name}_hi"])
    return table


def reliability_curve(ax, y, p, label: str, n_bins: int = 10) -> None:
    bins = np.clip((np.asarray(p) * n_bins).astype(int), 0, n_bins - 1)
    frame = pd.DataFrame({"bin": bins, "y": y, "p": p}).groupby("bin").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.plot(frame["p"], frame["y"], marker="o", label=f"{label} (n={len(y)})")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.legend(fontsize=8)


def shap_summary(values: np.ndarray, columns: list[str], X: np.ndarray, title: str, top: int = 20):
    """Bar (mean |SHAP|) + beeswarm-style dot plot of the top features. Returns the figure and the ranking."""
    import matplotlib.pyplot as plt

    importance = np.abs(values).mean(axis=0)
    order = np.argsort(-importance)[:top][::-1]
    fig, (ax_bar, ax_dot) = plt.subplots(1, 2, figsize=(13, 0.35 * len(order) + 1.5), sharey=True)
    labels = [columns[i] for i in order]
    ax_bar.barh(range(len(order)), importance[order], color="#4C72B0")
    ax_bar.set_yticks(range(len(order)), labels, fontsize=8)
    ax_bar.set_xlabel("mean |SHAP| (log-odds)")
    rng = np.random.default_rng(0)
    for row, i in enumerate(order):
        x = X[:, i]
        span = np.ptp(x) or 1.0
        colour = (x - x.min()) / span
        jitter = rng.uniform(-0.3, 0.3, len(x))
        ax_dot.scatter(values[:, i], row + jitter, c=colour, cmap="coolwarm", s=4, alpha=0.6, rasterized=True)
    ax_dot.axvline(0, c="grey", lw=1)
    ax_dot.set_xlabel("SHAP value (colour = feature value, low → high)")
    fig.suptitle(title)
    fig.tight_layout()
    ranking = pd.DataFrame({"feature": columns, "mean_abs_shap": importance}).sort_values("mean_abs_shap", ascending=False)
    return fig, ranking.reset_index(drop=True)
