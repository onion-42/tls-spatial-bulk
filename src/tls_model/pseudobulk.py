"""Multi-scale spot aggregates on the Visium hex grid, and log2-CPM normalisation.

Visium array coordinates: neighbours of (row, col) are (row, col ± 2) and (row ± 1, col ± 1).
Hex distance in these "doubled" coordinates: d = |Δrow| + max(0, (|Δcol| − |Δrow|) / 2).
radius 0 / 1 / 2 -> 1 / 7 / 19 spots in the interior of the tissue.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

K_TO_RADIUS = {1: 0, 7: 1, 19: 2}


def _offsets(radius: int) -> list[tuple[int, int]]:
    out = []
    for dr in range(-radius, radius + 1):
        for dc in range(-2 * radius, 2 * radius + 1):
            if (dr + dc) % 2:
                continue
            if abs(dr) + max(0, (abs(dc) - abs(dr)) // 2) <= radius:
                out.append((dr, dc))
    return out


def neighbourhood(rows: np.ndarray, cols: np.ndarray, radius: int) -> sp.csr_matrix:
    """Binary n×n matrix: entry (i, j) = 1 if spot j is within `radius` hex steps of spot i (incl. i)."""
    index = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
    src, dst = [], []
    for i, (r, c) in enumerate(zip(rows, cols)):
        for dr, dc in _offsets(radius):
            j = index.get((int(r) + dr, int(c) + dc))
            if j is not None:
                src.append(i)
                dst.append(j)
    n = len(rows)
    return sp.csr_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(n, n))


SECTION_COLUMNS = ("section", "patient", "protocol", "platform", "source")


def _aggregate_maturity(adjacency: sp.csr_matrix, mature: np.ndarray) -> np.ndarray:
    """Bulk-like rule: 1 if any mature TLS spot in the unit, 0 if only immature ones, NaN if unknown."""
    n_mature = adjacency @ (mature == 1).astype(float)
    n_immature = adjacency @ (mature == 0).astype(float)
    return np.where(n_mature > 0, 1.0, np.where(n_immature > 0, 0.0, np.nan))


def _pick_centres(eligible: np.ndarray, positive: np.ndarray, cap, max_positive, rng) -> np.ndarray:
    idx = np.flatnonzero(eligible)
    if cap is None:
        return idx
    pos = idx[positive[idx]]
    neg = idx[~positive[idx]]
    pos = rng.permutation(pos)[: max_positive if max_positive is not None else cap]
    neg = rng.permutation(neg)[: max(cap - len(pos), 0)]
    return np.sort(np.concatenate([pos, neg]))


def build_units(section, ks=(1, 7, 19), cap=300, max_positive=150, rng=None):
    """Training/evaluation units of one Visium section: spots, k-neighbourhood aggregates, whole section.

    tls = any TLS spot in the unit; tls_fraction = share of TLS spots; tls_mature as in `_aggregate_maturity`.
    `cap` limits centres per scale (all TLS-containing centres first, up to `max_positive`); None = all spots.
    Returns AnnData with summed counts in X.
    """
    import anndata as ad
    import pandas as pd

    rng = rng if rng is not None else np.random.default_rng(0)
    # Excluded spots (lymph node) are removed entirely: no centres, no neighbour counts, not in the section sum.
    section = section[~section.obs["excluded"].to_numpy(dtype=bool)]
    obs = section.obs
    rows, cols = obs["array_row"].to_numpy(), obs["array_col"].to_numpy()
    tls = obs["tls"].to_numpy()
    mature = obs["tls_mature"].to_numpy(dtype=float)
    eligible = np.ones(len(obs), dtype=bool)
    meta = {c: obs[c].iloc[0] for c in SECTION_COLUMNS}
    counts = sp.csr_matrix(section.X)

    blocks, frames = [], []
    for k in ks:
        adjacency = neighbourhood(rows, cols, K_TO_RADIUS[k])
        fraction = np.asarray(adjacency @ tls.astype(float)).ravel() / np.asarray(adjacency.sum(axis=1)).ravel()
        centres = _pick_centres(eligible, fraction > 0, cap, max_positive, rng)
        sub = adjacency[centres]
        blocks.append(sp.csr_matrix(sub @ counts))
        frames.append(
            pd.DataFrame(
                {
                    "unit": "spot" if k == 1 else f"agg{k}",
                    "k": k,
                    "center": obs.index[centres],
                    "tls": (fraction[centres] > 0).astype(int),
                    "tls_fraction": fraction[centres],
                    "tls_mature": _aggregate_maturity(sub, mature),
                }
            )
        )

    whole = sp.csr_matrix(np.ones((1, len(tls)), dtype=np.float32))
    blocks.append(sp.csr_matrix(counts.sum(axis=0)))
    frames.append(
        pd.DataFrame(
            {
                "unit": "section", "k": len(tls), "center": "", "tls": int(tls.max() > 0),
                "tls_fraction": tls.mean(), "tls_mature": _aggregate_maturity(whole, mature),
            }
        )
    )
    unit_obs = pd.concat(frames, ignore_index=True).assign(**meta)
    unit_obs.index = [f"{meta['section']}:{u}:{c}" for u, c in zip(unit_obs["unit"], unit_obs["center"])]
    return ad.AnnData(X=sp.vstack(blocks).tocsr(), obs=unit_obs, var=section.var[[]].copy())


def spot_prevalence(section) -> dict[str, int]:
    """Natural spot counts of one section (excluded spots dropped), for prior-shift calibration."""
    obs = section.obs[~section.obs["excluded"].to_numpy(dtype=bool)]
    mature = obs["tls_mature"].to_numpy(dtype=float)
    return {
        "n_spots": len(obs),
        "n_tls": int(obs["tls"].sum()),
        "n_mature_known": int((~np.isnan(mature)).sum()),
        "n_mature": int((mature == 1).sum()),
    }


def log2_cpm(counts: sp.spmatrix | np.ndarray) -> np.ndarray:
    """Dense log2(CPM + 1) per row; comparable in scale to bulk log2(TPM + 1)."""
    dense = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
    dense = dense.astype(np.float64)
    library = dense.sum(axis=1, keepdims=True)
    library[library == 0] = 1.0
    return np.log2(dense / library * 1e6 + 1.0).astype(np.float32)
