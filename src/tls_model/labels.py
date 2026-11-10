"""Harmonised TLS labels for bulk samples and Visium spots.

Bulk: TLS = LA1 + LA2 (tumour-associated). LA0-only (lymphoid aggregates without HEV) = negative.
Maturity (head B) = LA2 vs LA1, from pathologists only; HookNet "GC" is a grid proxy, not LA2.
Spots: TLS / ETLS / MTLS / GC (GSE175540, USZ), TLS_* (GSE273952) and tls / aggregate (GSE351843 breast,
pathologist H&E) = positive; NO_TLS, unassigned, T-cell aggregates (T_agg), INFL, TUM, NOR, diffuse
lymphocytes_present = negative; LN (lymph node) and "Folded area" = excluded. Maturity: MTLS + GC + TLS_SFL +
TLS_PFL + tls = mature, ETLS + TLS_immature + aggregate = immature, NaN where not assessable. `aggregate` (lymphoid aggregate without a called follicle) can instead
be excluded (`aggregate_is_tls=False`, config `labels.aggregate_is_tls`) for a sensitivity analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BULK_STATUS = {"positive": 1.0, "negative": 0.0}
MATURE = "mature (LA2 / GC+)"
IMMATURE = "immature only (LA1, no GC)"


def three_class_label(tls_any: np.ndarray, tls_mature: np.ndarray) -> np.ndarray:
    """0 = no TLS, 1 = immature TLS, 2 = mature TLS; NaN = unlabelled or TLS of unknown maturity."""
    tls_any, tls_mature = np.asarray(tls_any, dtype=float), np.asarray(tls_mature, dtype=float)
    label = np.where(tls_mature == 1, 2.0, np.where(tls_mature == 0, 1.0, np.nan))
    label = np.where(tls_any == 0, 0.0, label)
    return np.where(np.isnan(tls_any), np.nan, label)


def label_source(method: pd.Series) -> pd.Series:
    """'pathologist' for pathologist_HE / pathologist_IHC, 'cnn' for every model-made label."""
    return np.where(method.str.startswith("pathologist"), "pathologist", "cnn")


def bulk_labels(table: pd.DataFrame) -> pd.DataFrame:
    """Return tls_any (0/1), tls_mature (0/1/NaN) and label_source for an archive label table."""
    source = label_source(table["method"])
    is_pathologist = source == "pathologist"
    mature = np.where(table["maturity_status"] == MATURE, 1.0, np.nan)
    mature = np.where(table["maturity_status"] == IMMATURE, 0.0, mature)
    return pd.DataFrame(
        {
            # anything other than positive / negative (not evaluable, missing) is unlabelled, not negative
            "tls_any": table["TLS_status"].map(BULK_STATUS).astype(float).to_numpy(),
            "tls_mature": np.where(is_pathologist, mature, np.nan),
            "label_source": source,
        },
        index=table.index,
    )


# value = maturity (NaN = unknown)
SPOT_TLS = {
    "TLS": np.nan, "ETLS": 0.0, "MTLS": 1.0, "GC": 1.0,  # GSE175540, USZ
    # GSE273952: SFL = secondary follicle-like (CD21+ CD23+, germinal centre), PFL = primary follicle-like (CD21+
    # FDC network, CD23-; mature like USZ MTLS), immature = no FDC; CD21 / CD23 not assessable or TLS in adjacent
    # healthy tissue ("sain") = TLS of unknown maturity.
    "TLS_SFL": 1.0, "TLS_PFL": 1.0, "TLS_immature": 0.0,
    "TLS_CD21NA_CD23neg": np.nan, "TLS_CD21NA_CD23NA": np.nan, "TLS_NA": np.nan, "TLS_sain": np.nan,
    "tls": 1.0, "aggregate": 0.0,  # GSE351843 breast (Zenodo pathology annotation, column `immune`)
}
# LN = lymph node (lymphoid but not a TLS), "Folded area" = tissue-fold artefact: neither positive nor negative.
SPOT_EXCLUDED = {"LN", "Folded area"}
SPOT_NEGATIVE = {"", "NO_TLS", "T_agg", "UNASSIGNED", "TUM", "NOR", "INFL", "lymphocytes_present"}
AGGREGATE = "aggregate"


def spot_labels(raw: pd.Series, aggregate_is_tls: bool = True) -> pd.DataFrame:
    """Spot labels for every Visium cohort (see module docstring for the vocabulary).

    tls: 1 for any TLS spot, else 0 (incl. hard negatives T_agg / INFL / lymphocytes_present and unassigned).
    tls_mature: 1 / 0 where the annotation carries maturity, NaN otherwise.
    aggregate_is_tls=False: breast `aggregate` spots are excluded instead of counted as immature TLS.
    """
    tls_map = dict(SPOT_TLS)
    excluded = set(SPOT_EXCLUDED)
    if not aggregate_is_tls:
        del tls_map[AGGREGATE]
        excluded.add(AGGREGATE)
    value = raw.astype(object).fillna("").astype(str).str.strip()
    unknown = set(value) - set(tls_map) - SPOT_NEGATIVE - excluded
    if unknown:
        raise ValueError(f"unknown spot labels {sorted(unknown)}; add them to labels.SPOT_* first")
    return pd.DataFrame(
        {
            "tls": value.isin(tls_map.keys()).astype(int),
            "tls_mature": value.map(tls_map).astype(float),
            "excluded": value.isin(excluded),
        },
        index=raw.index,
    )
