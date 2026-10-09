import numpy as np
import pandas as pd

from tls_model import labels


def _primary_rows():
    return pd.DataFrame(
        {
            "method": ["pathologist_HE", "pathologist_HE", "pathologist_HE", "HookNet-TLS_a", "HookNet-TLS_b", "pathologist_IHC"],
            "TLS_status": ["positive", "positive", "negative", "positive", "negative", "negative"],
            "maturity_status": [
                "mature (LA2 / GC+)",
                "immature only (LA1, no GC)",
                "lymphoid aggregates only (LA0)",
                "mature (LA2 / GC+)",
                "unknown",
                "no TLS",
            ],
        }
    )


def test_bulk_labels_any_tls_and_la0_is_negative():
    out = labels.bulk_labels(_primary_rows())

    assert out["tls_any"].tolist() == [1, 1, 0, 1, 0, 0]


def test_bulk_labels_maturity_only_from_pathologists_with_tls():
    out = labels.bulk_labels(_primary_rows())

    assert out["tls_mature"].iloc[0] == 1
    assert out["tls_mature"].iloc[1] == 0
    assert np.isnan(out["tls_mature"].iloc[2])  # LA0 only: no TLS -> no maturity
    assert np.isnan(out["tls_mature"].iloc[3])  # HookNet GC is not a pathologist LA2
    assert out["label_source"].tolist() == ["pathologist"] * 3 + ["cnn", "cnn", "pathologist"]


def test_bulk_labels_unknown_status_is_missing_not_negative():
    rows = pd.DataFrame(
        {"method": ["pathologist_HE"] * 3, "TLS_status": ["positive", "not evaluable", None], "maturity_status": ["unknown"] * 3}
    )

    out = labels.bulk_labels(rows)

    assert out["tls_any"].iloc[0] == 1
    assert out["tls_any"].iloc[1:].isna().all()


def test_spot_labels_reject_unknown_strings():
    import pytest

    with pytest.raises(ValueError, match="Mtls"):
        labels.spot_labels(pd.Series(["TLS", "Mtls"]))


def test_spot_labels_tls_maturity_and_exclusions():
    raw = pd.Series(["TLS", "NO_TLS", "", None, "T_agg", "INFL", " TLS ", "ETLS", "MTLS", "GC", "LN", "TUM"])

    out = labels.spot_labels(raw)

    assert out["tls"].tolist() == [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    mature = out["tls_mature"].tolist()
    assert mature[7:10] == [0, 1, 1]
    assert all(np.isnan(m) for m in mature[:7] + mature[10:])
    assert out["excluded"].tolist() == [False] * 10 + [True, False]
