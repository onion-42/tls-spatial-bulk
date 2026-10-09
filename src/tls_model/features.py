"""Features shared by bulk samples and Visium units. Everything here is single-sample deployable.

Pipeline per unit (log2 TPM/CPM + 1 on the shared genes):
    + platform offset (frozen Visium only) -> gene filter (label-free) -> representation (rank | log)
    -> base features: genes, ssGSEA of public signatures, signature × signature products.
Gene × gene products of the top signature genes are added inside the CV fold (see models.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

SSGSEA_ALPHA = 0.25
CHUNK = 4096
# Low-information or sex-linked genes dropped before anything else.
EXCLUDED_PREFIXES = ("MT-", "RPL", "RPS", "MRPL", "MRPS")
EXCLUDED_HB = ("HBA1", "HBA2", "HBB", "HBD", "HBG1", "HBG2", "HBM", "HBQ1", "HBZ", "HBE1")
CHR_Y_GENES = (
    "RPS4Y1", "DDX3Y", "KDM5D", "UTY", "EIF1AY", "ZFY", "USP9Y", "NLGN4Y", "TXLNGY", "PRKY", "TMSB4Y",
)


def is_uninformative(gene: str) -> bool:
    return gene.startswith(EXCLUDED_PREFIXES) or gene in EXCLUDED_HB or gene in CHR_Y_GENES


def _rank_rows(X: np.ndarray) -> np.ndarray:
    """Ascending average ranks per row (1..N), computed in chunks to bound memory."""
    out = np.empty(X.shape, dtype=np.float32)
    for start in range(0, X.shape[0], CHUNK):
        out[start : start + CHUNK] = rankdata(X[start : start + CHUNK], axis=1)
    return out


def within_sample_rank(X: np.ndarray) -> np.ndarray:
    """Ranks within each sample scaled to [0, 1]; invariant to any monotone per-sample scaling."""
    n_genes = X.shape[1]
    return (_rank_rows(X) - 1.0) / max(n_genes - 1, 1)


def ssgsea(X: np.ndarray, genes: list[str], signatures: dict[str, list[str]], alpha: float = SSGSEA_ALPHA) -> np.ndarray:
    """Single-sample GSEA (Barbie et al., Nature 2009), unnormalised, in closed form.

    With genes sorted by decreasing expression, the running-sum score sum_i (P_hit(i) − P_miss(i)) equals
    sum_S w·r / sum_S w − (N(N+1)/2 − sum_S r) / (N − m), where r = ascending rank, w = r^alpha.
    """
    ranks = _rank_rows(X).astype(np.float64)
    n = X.shape[1]
    position = {g: i for i, g in enumerate(genes)}
    scores = np.zeros((X.shape[0], len(signatures)), dtype=np.float32)
    for k, members in enumerate(signatures.values()):
        idx = [position[g] for g in members if g in position]
        if not idx or len(idx) == n:
            continue
        r = ranks[:, idx]
        w = r**alpha
        hit = (w * r).sum(axis=1) / w.sum(axis=1)
        miss = (n * (n + 1) / 2 - r.sum(axis=1)) / (n - len(idx))
        scores[:, k] = hit - miss
    return scores


def gene_filter(
    bulk_log: np.ndarray,
    spot_detection: np.ndarray,
    genes: list[str],
    n_top: int,
    protected: set[str],
    min_median: float = 1.0,
    min_detection: float = 0.02,
) -> list[str]:
    """Label-free filter: expressed in bulk and detected in spots -> top `n_top` by bulk SD; + protected genes."""
    expressed = (np.median(bulk_log, axis=0) >= min_median) & (spot_detection >= min_detection)
    informative = np.array([not is_uninformative(g) for g in genes])
    candidates = np.flatnonzero(expressed & informative)
    sd = bulk_log[:, candidates].std(axis=0)
    top = set(candidates[np.argsort(-sd)[:n_top]].tolist())
    keep = top | {i for i, g in enumerate(genes) if g in protected}
    return [genes[i] for i in sorted(keep)]


def platform_offset(bulk: pd.DataFrame, pseudobulk: pd.DataFrame, genes: list[str]) -> pd.Series:
    """Per-gene median(bulk − pseudo-bulk) over paired samples (rows = pair ids); 0 where unmeasured."""
    common = bulk.columns.intersection(pseudobulk.columns)
    pairs = bulk.index.intersection(pseudobulk.index)
    diff = bulk.loc[pairs, common] - pseudobulk.loc[pairs, common]
    return diff.median(axis=0).reindex(genes).fillna(0.0)


def weighted_abs_corr(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """|weighted Pearson r| of every column with y; 0 for constant columns."""
    w = w / w.sum()
    xc = X - w @ X
    yc = y - w @ y
    cov = w @ (xc * yc[:, None])
    sx = np.sqrt(w @ xc**2)
    sy = np.sqrt(w @ yc**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(sx > 0, cov / (sx * sy), 0.0)
    return np.abs(np.nan_to_num(r))


def pair_products(X: np.ndarray, names: list[str], prefix: str) -> tuple[np.ndarray, list[str]]:
    pairs = list(combinations(range(X.shape[1]), 2))
    if not pairs:
        return np.zeros((X.shape[0], 0), dtype=np.float32), []
    left, right = zip(*pairs)
    values = (X[:, list(left)] * X[:, list(right)]).astype(np.float32)
    return values, [f"{prefix}:{names[i]}*{names[j]}" for i, j in pairs]


@dataclass(frozen=True)
class FeatureSpace:
    """Fixed (label-free) feature definition: kept genes, signatures, representation."""

    genes: tuple[str, ...]
    signatures: tuple[tuple[str, tuple[str, ...]], ...]
    representation: str  # "rank" or "log"

    @property
    def signature_dict(self) -> dict[str, list[str]]:
        return {name: list(members) for name, members in self.signatures}

    def gene_values(self, log_values: np.ndarray) -> np.ndarray:
        if self.representation == "rank":
            return within_sample_rank(log_values)
        if self.representation == "log":
            return log_values.astype(np.float32)
        raise ValueError(f"unknown representation {self.representation!r}")

    def base(self, log_values: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Base features for units given log2 values on `self.genes` (columns in that order)."""
        genes = list(self.genes)
        gene_part = self.gene_values(log_values)
        sig = ssgsea(log_values, genes, self.signature_dict)
        sig_names = [name for name, _ in self.signatures]
        sig_pairs, sig_pair_names = pair_products(sig, sig_names, prefix="sigpair")
        values = np.hstack([gene_part, sig, sig_pairs]).astype(np.float32)
        names = [f"gene:{g}" for g in genes] + [f"sig:{s}" for s in sig_names] + sig_pair_names
        return values, names
