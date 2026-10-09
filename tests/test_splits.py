import numpy as np
import pandas as pd

from tls_model import splits


def _bulk_obs():
    rows = []
    for cohort in ("A", "B"):
        for i in range(50):
            rows.append(
                {
                    "cohort": cohort,
                    "patient": f"{cohort}:{i}",
                    "tls_any": int(i % 3 == 0),
                    "label_source": "cnn" if i < 10 else "pathologist",
                }
            )
    rows += [{"cohort": "OOD", "patient": f"OOD:{i}", "tls_any": i % 2, "label_source": "pathologist"} for i in range(20)]
    return pd.DataFrame(rows, index=[f"s{i}" for i in range(len(rows))])


def test_bulk_split_locks_ood_and_keeps_patients_together():
    obs = _bulk_obs()

    split = splits.split_bulk(obs, ood_cohorts=["OOD"], test_frac=0.2, seed=0)

    assert (split[obs["cohort"] == "OOD"] == "ood").all()
    assert set(split[obs["cohort"] != "OOD"]) == {"train", "test"}
    assert obs.assign(split=split).groupby("patient")["split"].nunique().max() == 1
    test_share = (split[obs["cohort"] == "A"] == "test").mean()
    assert 0.1 <= test_share <= 0.3


def test_bulk_split_never_puts_cnn_only_patients_in_test():
    obs = _bulk_obs()

    split = splits.split_bulk(obs, ood_cohorts=["OOD"], test_frac=0.2, seed=0)

    assert (split[obs["label_source"] == "cnn"] == "train").all()


def test_visium_split_by_patient_stratified_by_source():
    patients = pd.DataFrame(
        {
            "patient": [f"P{i}" for i in range(16)],
            "source": ["USZ"] * 8 + ["GSE"] * 8,
            "has_tls": [1, 1, 1, 1, 1, 1, 0, 0] * 2,
        }
    )

    split = splits.split_visium(patients, test_frac=0.25, seed=0)

    assert (split == "test").sum() == 4
    assert (split[patients["source"].values == "USZ"] == "test").sum() == 2


def test_cv_folds_are_grouped():
    groups = np.repeat(np.arange(40), 3)
    y = np.tile([0, 1, 0], 40)

    folds = splits.cv_folds(groups, y, n_splits=5, seed=0)

    assert pd.DataFrame({"g": groups, "f": folds}).groupby("g")["f"].nunique().max() == 1
    assert set(folds) == set(range(5))


def test_weights_balance_kinds_and_groups():
    kind = np.array(["bulk"] * 6 + ["visium"] * 4)
    group = np.array(["c1"] * 4 + ["c2"] * 2 + ["s1"] * 3 + ["s2"])

    w = splits.balanced_weights(kind, group)

    assert np.isclose(w[kind == "bulk"].sum(), w[kind == "visium"].sum())
    assert np.isclose(w[group == "c1"].sum(), w[group == "c2"].sum())
    assert np.isclose(w[group == "s1"].sum(), w[group == "s2"].sum())
    assert np.isclose(w.mean(), 1.0)
