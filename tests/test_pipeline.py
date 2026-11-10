import anndata as ad
import numpy as np
import pandas as pd
import pytest

from tls_model import pipeline


def _units(seed=0):
    """Synthetic units: 3 bulk cohorts (one OOD) + 3 Visium sections (one OOD), label driven by gene G0."""
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
    for s, split, protocol in (("S1", "train", "frozen"), ("S2", "test", "ffpe"), ("S3", "ood", "ffpe")):
        for i in range(80):
            rows.append(
                dict(kind="visium", unit="spot" if i < 60 else "agg7", group=s, cohort="", section=s,
                     patient=f"P{s}_{i % 4}" if split == "train" else f"P{s}", split=split, platform=f"visium_{protocol}", label_source="pathologist",
                     tls_any=int(i % 3 == 0), tls_mature=float(i % 6 == 0) if i % 3 == 0 else np.nan,
                     k=1 if i < 60 else 7, tls_fraction=float(i % 3 == 0))
            )
    obs = pd.DataFrame(rows)
    obs.index = [f"u{i}" for i in range(len(obs))]
    obs["role"] = pipeline.roles(obs, distill_cohorts=["C1", "C2"])
    obs["tls_three"] = pipeline.three_class_column(obs)
    X = rng.normal(size=(len(obs), 12)).astype(np.float32) + 5
    X[:, 0] += 2 * obs["tls_any"].to_numpy()
    genes = [f"G{i}" for i in range(12)]
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))


def test_training_rows_never_contain_test_or_ood():
    units = _units()

    for head in ("any", "mature", "three"):
        rows = pipeline.train_rows(units.obs, head)
        train = units.obs[rows]

        assert set(train["split"]) == {"train"}
        assert set(train["role"]) == {"train"} and set(train["kind"]) == {"visium"}
        held_out = set(units.obs.loc[units.obs["split"] != "train", "patient"])
        assert not held_out & set(train["patient"])
        assert "PS3" in held_out  # the OOD Visium section is held out like an OOD bulk cohort


def test_roles_layer_on_the_split():
    obs = _units().obs

    assert set(obs.loc[obs["kind"] == "visium", "role"]) == {"train", "test", "holdout"}
    assert (obs.loc[(obs["cohort"] == "C1") & (obs["split"] == "train"), "role"] == "distill").all()
    assert (obs.loc[(obs["cohort"] == "C1") & (obs["split"] == "test"), "role"] == "test").all()
    assert (obs.loc[obs["cohort"] == "OOD", "role"] == "holdout").all()
    with pytest.raises(ValueError, match="C2"):
        pipeline.roles(obs, distill_cohorts=["C1"])


def test_three_class_column_is_nan_for_tls_of_unknown_maturity():
    obs = pd.DataFrame({"tls_any": [0, 1, 1, 1], "tls_mature": [np.nan, np.nan, 0.0, 1.0]})

    out = pipeline.three_class_column(obs)

    assert out.tolist()[0] == 0 and np.isnan(out.tolist()[1]) and out.tolist()[2:] == [1, 2]


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

    table = pipeline.evaluate_split(units.obs, p_any, "test", n_boot=50)

    assert set(table["head"]) == {"any"}
    assert "bulk:all" in set(table["block"])


def test_evaluate_split_with_three_class_probs_scores_every_block():
    units = _units()
    y3 = units.obs["tls_three"].to_numpy(dtype=float)
    probs = np.full((len(y3), 3), 0.1)
    probs[np.arange(len(y3)), np.nan_to_num(y3, nan=1).astype(int)] = 0.8

    table = pipeline.evaluate_split(units.obs, probs, "test", n_boot=20)

    three = table[table["head"] == "3-class"]
    assert {"bulk:all", "bulk:C1", "visium:spots"} <= set(three["block"])
    assert np.isclose(three.loc[three["block"] == "bulk:all", "balanced_accuracy"].iloc[0], 1.0)
    assert {"any", "mature"} <= set(table["head"])


def test_calibration_rows_are_gold_train_spots_and_distill_pathologist_bulk():
    units = _units()

    rows = pipeline.calibration_rows(units.obs, "any")
    used = units.obs[rows]

    assert set(used["role"]) == {"train", "distill"}
    assert (used["label_source"] == "pathologist").all()
    assert set(used.loc[used["role"] == "train", "unit"]) == {"spot"}
    assert set(used.loc[used["role"] == "distill", "unit"]) == {"bulk"}
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

    table, oof, _ = pipeline.run_cv(B, names, units.obs, "any", candidates, meta, n_folds=3, seed=0,
                                    n_panel=5, n_pair_genes=2)

    rows = pipeline.train_rows(units.obs, "any")
    assert len(table) == 2
    assert np.isfinite(oof["enet|corr"][rows]).all()
    assert np.isnan(oof["enet|corr"][~rows]).all()
    assert table.set_index("candidate").loc["enet|corr", "score"] > 0.7


def test_visium_units_fill_genes_missing_from_a_section_with_zeros():
    from tls_model import prepare

    obs = pd.DataFrame(
        {"array_row": [0, 0, 1], "array_col": [0, 2, 1], "tls": [1, 0, 0], "tls_mature": [np.nan] * 3,
         "excluded": [False] * 3, "section": "S", "patient": "P", "protocol": "ffpe", "platform": "visium_ffpe",
         "source": "GSE351843"},
        index=["b1", "b2", "b3"],
    )
    section = ad.AnnData(X=np.array([[4, 0], [0, 4], [2, 2]], dtype=np.float32), obs=obs,
                         var=pd.DataFrame(index=["G0", "G2"]))
    cfg = {"units": {"ks": [1, 7], "cap_per_scale": 300, "max_positive_per_scale": 150}}

    units = prepare._visium_units(section, "ood", ["G0", "G1", "G2"], cfg, np.random.default_rng(0))

    assert list(units.var_names) == ["G0", "G1", "G2"]
    assert (units.obs["split"] == "ood").all() and set(units.obs["unit"]) == {"spot", "section"}
    assert np.all(units.X[:, 1] == 0)
    assert np.isclose(units.X[0, 0], np.log2(1e6 + 1))


def test_bulk_and_visium_unit_blocks_concat_to_the_kept_genes_only():
    from tls_model import prepare

    genes = ["G0", "G1"]
    bulk_obs = pd.DataFrame({"cohort": ["C"], "patient": ["p"], "split": ["train"], "label_source": ["pathologist"],
                             "tls_any": [1.0], "tls_mature": [np.nan]}, index=["b"])
    bulk = ad.AnnData(X=np.ones((1, 3), dtype=np.float32), obs=bulk_obs, var=pd.DataFrame(index=["G0", "G1", "G9"]))
    obs = pd.DataFrame({"array_row": [0], "array_col": [0], "tls": [0], "tls_mature": [np.nan], "excluded": [False],
                        "section": "S", "patient": "P", "protocol": "ffpe", "platform": "visium_ffpe", "source": "X"},
                       index=["s1"])
    section = ad.AnnData(X=np.array([[2, 2]], dtype=np.float32), obs=obs, var=pd.DataFrame(index=genes))
    cfg = {"units": {"ks": [1], "cap_per_scale": 300, "max_positive_per_scale": 150}}

    blocks = [prepare._bulk_units(bulk, genes), prepare._visium_units(section, "ood", genes, cfg, np.random.default_rng(0))]
    units = ad.concat(blocks, join="outer")

    assert list(units.var_names) == genes


def test_predict_in_chunks_equals_one_pass():
    units = _units()
    meta = {"genes": list(units.var_names), "signatures": {"sA": ["G0", "G1"], "sB": ["G2", "G3"]},
            "offset": np.zeros(units.n_vars, dtype=np.float32)}
    B, names = pipeline.feature_matrix(units, meta, representation="rank", use_offset=False)
    rows = pipeline.train_rows(units.obs, "any")
    model = pipeline.make_head({"family": "enet", "selector": "corr"}, meta, 5, 2, 0)
    model.fit(B[rows], names, units.obs["tls_any"].to_numpy(dtype=float)[rows], np.ones(rows.sum()))
    from tls_model.models import PlattCalibrator

    calibrator = PlattCalibrator().fit(model.predict_proba(B[rows], names), units.obs["tls_any"].to_numpy()[rows].astype(int),
                                       pipeline.calibration_kind(units.obs)[rows])
    variant = {"representation": "rank", "offset": False}

    one = pipeline.predict(B, names, units.obs, model, calibrator)
    chunked = pipeline.predict_in_chunks(units, meta, variant, model, calibrator, chunk=37)

    assert np.allclose(one, chunked)
