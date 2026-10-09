import anndata as ad
import numpy as np
import pandas as pd
import pytest

from tls_model import pipeline


def _units(seed=0):
    """Synthetic units: 3 bulk cohorts (one OOD) + 2 Visium sections, label driven by gene G0."""
    rng = np.random.default_rng(seed)
    rows = []
    for cohort, split in (("C1", None), ("C2", None), ("OOD", "ood")):
        for i in range(60):
            rows.append(
                dict(kind="bulk", unit="bulk", group=cohort, cohort=cohort, section="", patient=f"{cohort}:{i}",
                     split=split or ("test" if i % 5 == 0 else "train"), platform="bulk",
                     label_source="cnn" if (cohort == "C1" and i % 7 == 1) else "pathologist",
                     tls_any=int(i % 2), tls_mature=float(i % 4 == 1) if i % 2 else np.nan, k=0, tls_fraction=np.nan)
            )
    for s, split, protocol in (("S1", "train", "frozen"), ("S2", "test", "ffpe")):
        for i in range(80):
            rows.append(
                dict(kind="visium", unit="spot" if i < 60 else "agg7", group=s, cohort="", section=s,
                     patient=f"P{s}", split=split, platform=f"visium_{protocol}", label_source="pathologist",
                     tls_any=int(i % 3 == 0), tls_mature=np.nan, k=1 if i < 60 else 7, tls_fraction=float(i % 3 == 0))
            )
    obs = pd.DataFrame(rows)
    obs.index = [f"u{i}" for i in range(len(obs))]
    X = rng.normal(size=(len(obs), 12)).astype(np.float32) + 5
    X[:, 0] += 2 * obs["tls_any"].to_numpy()
    genes = [f"G{i}" for i in range(12)]
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))


def test_training_rows_never_contain_test_or_ood():
    units = _units()

    for head in ("any", "mature"):
        rows = pipeline.train_rows(units.obs, head)
        train = units.obs[rows]

        assert set(train["split"]) == {"train"}
        held_out = set(units.obs.loc[units.obs["split"] != "train", "patient"])
        assert not held_out & set(train["patient"])


def test_mature_head_uses_only_pathologist_or_visium_maturity():
    units = _units()

    train = units.obs[pipeline.train_rows(units.obs, "mature")]

    assert train["tls_mature"].notna().all()
    assert (train["label_source"] == "pathologist").all()


def test_cv_strata_spread_bulk_positives_across_folds_despite_big_spot_groups():
    from tls_model.splits import cv_folds

    rng = np.random.default_rng(0)
    rows = [dict(kind="visium", cohort="", patient=f"S{g}", tls_mature=float(rng.uniform() < 0.45))
            for g in range(6) for _ in range(400)]
    rows += [dict(kind="bulk", cohort="C1", patient=f"B{i}", tls_mature=float(i < 30)) for i in range(200)]
    obs = pd.DataFrame(rows)

    folds = cv_folds(obs["patient"].to_numpy(), pipeline.cv_strata(obs, "mature"), n_splits=5, seed=0)

    bulk_pos = (obs["kind"] == "bulk").to_numpy() & (obs["tls_mature"] == 1).to_numpy()
    per_fold = np.bincount(folds[bulk_pos], minlength=5)
    assert per_fold.min() >= 3, per_fold


def test_oof_scores_bulk_auroc_is_mean_within_cohort():
    obs = pd.DataFrame({
        "kind": ["bulk"] * 8, "unit": ["bulk"] * 8, "label_source": ["pathologist"] * 8,
        "cohort": ["A"] * 4 + ["B"] * 4, "tls_any": [0, 0, 1, 1, 0, 0, 1, 1],
    })
    # perfect inside each cohort, but cohort B is shifted up -> pooled AUROC < 1
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    out = pipeline.oof_scores(obs, "any", p, np.ones(8, dtype=bool))

    assert out["auroc_bulk"] == 1.0
    assert out["auroc_bulk_pooled"] < 1.0


def test_evaluate_split_without_maturity_reports_head_a_only():
    units = _units()
    p_any = np.where(units.obs["tls_any"].to_numpy() == 1, 0.8, 0.2)

    table = pipeline.evaluate_split(units.obs, p_any, None, "test", n_boot=50)

    assert set(table["head"]) == {"any"}
    assert "bulk:all" in set(table["block"])


def test_calibration_rows_are_pathologist_bulk_and_train_spots_only():
    units = _units()

    rows = pipeline.calibration_rows(units.obs, "any")
    used = units.obs[rows]

    assert (used["split"] == "train").all()
    assert (used["label_source"] == "pathologist").all()
    assert set(used["unit"]) <= {"bulk", "spot"}
    assert set(pipeline.calibration_kind(units.obs)) == {"bulk", "spot"}


def test_feature_matrix_offset_only_touches_frozen_visium():
    units = _units()
    meta = {"genes": list(units.var_names), "signatures": {"sA": ["G0", "G1"], "sB": ["G2", "G3"]},
            "offset": np.full(units.n_vars, 3.0, dtype=np.float32)}

    on, names = pipeline.feature_matrix(units, meta, representation="log", use_offset=True)
    off, _ = pipeline.feature_matrix(units, meta, representation="log", use_offset=False)

    diff = (on[:, names.index("gene:G5")] - off[:, names.index("gene:G5")])
    frozen = (units.obs["platform"] == "visium_frozen").to_numpy()
    assert np.allclose(diff[frozen], 3.0) and np.allclose(diff[~frozen], 0.0)
    assert "sig:sA" in names and "sigpair:sA*sB" in names


def test_run_cv_returns_out_of_fold_scores():
    units = _units()
    meta = {"genes": list(units.var_names), "signatures": {"sA": ["G0", "G1"], "sB": ["G2", "G3"]},
            "offset": np.zeros(units.n_vars, dtype=np.float32)}
    B, names = pipeline.feature_matrix(units, meta, representation="rank", use_offset=False)
    candidates = [{"family": "enet", "selector": "corr"}, {"family": "signature", "selector": None}]

    table, oof = pipeline.run_cv(B, names, units.obs, "any", candidates, meta, n_folds=3, seed=0,
                                 n_panel=5, n_pair_genes=2)

    rows = pipeline.train_rows(units.obs, "any")
    assert len(table) == 2
    assert np.isfinite(oof["enet|corr"][rows]).all()
    assert np.isnan(oof["enet|corr"][~rows]).all()
    assert table.set_index("candidate").loc["enet|corr", "score"] > 0.7
