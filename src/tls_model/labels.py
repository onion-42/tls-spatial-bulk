"""Harmonised TLS labels for bulk samples and Visium spots.

Bulk: TLS = LA1 + LA2 (tumour-associated). LA0-only (lymphoid aggregates without HEV) = negative.
Maturity (head B) = LA2 vs LA1, from pathologists only; HookNet "GC" is a grid proxy, not LA2.
Spots: TLS / ETLS / MTLS / GC = positive; NO_TLS, unassigned, T-cell aggregates (T_agg), INFL, TUM, NOR
= negative; LN (lymph node) = excluded. USZ maturity: MTLS + GC = mature, ETLS = immature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BULK_STATUS = {"positive": 1.0, "negative": 0.0}
MATURE = "mature (LA2 / GC+)"
IMMATURE = "immature only (LA1, no GC)"


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


SPOT_TLS = {"TLS": np.nan, "ETLS": 0.0, "MTLS": 1.0, "GC": 1.0}  # value = maturity (NaN = unknown)
SPOT_EXCLUDED = {"LN"}  # lymph node: lymphoid but not a TLS -> neither positive nor negative
SPOT_NEGATIVE = {"", "NO_TLS", "T_agg", "UNASSIGNED", "TUM", "NOR", "INFL"}


def spot_labels(raw: pd.Series) -> pd.DataFrame:
    """Spot labels from GSE175540 (TLS / NO_TLS / T_agg / '') and USZ (ETLS / MTLS / GC / TUM / ...).

    tls: 1 for any TLS spot, else 0 (incl. hard negatives T_agg / INFL and unassigned).
    tls_mature: USZ only, MTLS + GC = 1, ETLS = 0, NaN otherwise.
    """
    value = raw.astype(object).fillna("").astype(str).str.strip()
    unknown = set(value) - set(SPOT_TLS) - SPOT_NEGATIVE - SPOT_EXCLUDED
    if unknown:
        raise ValueError(f"unknown spot labels {sorted(unknown)}; add them to labels.SPOT_* first")
    return pd.DataFrame(
        {
            "tls": value.isin(SPOT_TLS.keys()).astype(int),
            "tls_mature": value.map(SPOT_TLS).astype(float),
            "excluded": value.isin(SPOT_EXCLUDED),
        },
        index=raw.index,
    )
