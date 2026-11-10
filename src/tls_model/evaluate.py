"""Metrics with bootstrap 95% CIs, 3-class metrics (no TLS / immature / mature), reliability curves, SHAP plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, roc_auc_score,
)

from tls_model.labels import three_class_label

N_BOOT = 1000
METRICS = {
    "auroc": roc_auc_score,
    "auprc": average_precision_score,
    "brier": brier_score_loss,
}
CLASS_NAMES = ("none", "immature", "mature")


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
        if len(np.unique(y[idx])) >= 2:
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
    """0 = no TLS, 1 = immature, 2 = mature, -1 = TLS with unknown maturity or no label (not scored)."""
    labels = three_class_label(np.asarray(y_any, dtype=float), np.asarray(y_mature, dtype=float))
    return np.where(np.isnan(labels), -1, labels).astype(int)


def three_class_balanced_accuracy(y_any, y_mature, probs: np.ndarray) -> float:
    labels = three_class_labels(y_any, y_mature)
    known = labels >= 0
    return float(balanced_accuracy_score(labels[known], probs[known].argmax(axis=1)))


def macro_ovr_auroc(y3: np.ndarray, probs: np.ndarray) -> float:
    """Mean one-vs-rest AUROC over the classes present in y3 (a class absent from y3 is skipped, not an error)."""
    y3 = np.asarray(y3)
    values = []
    for c in range(probs.shape[1]):
        positive = y3 == c
        if positive.any() and (~positive).any():
            values.append(roc_auc_score(positive, probs[:, c]))
    return float(np.mean(values)) if values else float("nan")


def _argmax_balanced_accuracy(y3: np.ndarray, probs: np.ndarray) -> float:
    return float(balanced_accuracy_score(y3, probs.argmax(axis=1)))


MULTICLASS_METRICS = {"balanced_accuracy": _argmax_balanced_accuracy, "macro_auroc": macro_ovr_auroc}


def multiclass_metrics(y3, probs, groups=None, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """3-class metrics (y3 in {0, 1, 2}, probs (n, 3)): balanced accuracy and macro one-vs-rest AUROC with
    bootstrap CIs, per-class one-vs-rest AUROCs, and the confusion matrix as 'none immature mature' rows."""
    y3, probs = np.asarray(y3).astype(int), np.asarray(probs, dtype=float)
    out = {"n": len(y3), **{f"n_{name}": int((y3 == c).sum()) for c, name in enumerate(CLASS_NAMES)}}
    if len(np.unique(y3)) < 2:
        return out
    for name, metric in MULTICLASS_METRICS.items():
        point, lo, hi = bootstrap_ci(metric, y3, probs, groups, n_boot, seed)
        out.update({name: point, f"{name}_lo": lo, f"{name}_hi": hi})
    for c, name in ((2, "mature"), (1, "immature")):
        positive = y3 == c
        out[f"auroc_{name}_vs_rest"] = float(roc_auc_score(positive, probs[:, c])) if 0 < positive.sum() < len(y3) else float("nan")
    tls = y3 > 0
    out["auroc_tls_vs_none"] = float(roc_auc_score(tls, 1 - probs[:, 0])) if 0 < tls.sum() < len(y3) else float("nan")
    matrix = confusion_matrix(y3, probs.argmax(axis=1), labels=[0, 1, 2])
    out["confusion"] = " ".join("/".join(str(v) for v in row) for row in matrix)
    return out


def section_spearman(sections, p, true_fraction) -> float:
    frame = pd.DataFrame({"s": sections, "p": p, "f": true_fraction}).groupby("s").mean()
    return float(spearmanr(frame["p"], frame["f"]).statistic)


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    """Rows -> table with 'point [lo, hi]' strings for display."""
    table = pd.DataFrame(rows)
    for name in (*METRICS, *MULTICLASS_METRICS):
        if name in table and f"{name}_lo" in table:
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
