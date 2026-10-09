import numpy as np
import pandas as pd

from tls_model import ici_stats, io_ici


def _obs():
    return pd.DataFrame(
        {
            "Patient": ["p1", "p2", "p3", "p4", "p5"],
            "Sample_ID": ["s1", "s2", "s3", "s4", "s5"],
            "Cohort": ["A", "A", "B", "B", "C"],
            "Diagnosis_long": ["Skin Cutaneous Melanoma", "Melanoma", "Lung Adenocarcinoma",
                               "Clear Cell Renal Cell Carcinoma", "Non Clear Cell Renal Cell Carcinoma"],
            "F_Therapy_Eval": ["anti-PD1", "anti-CTLA4+anti-PD1", "anti-PD(L)1+Chemo", "PanTK-inh", "anti-PD1"],
            "Response": ["R", "NR", None, "R", "NR"],
            "Recist": ["SD", "PD", "PR", None, "PD"],
            "OS": [100.0, 200.0, np.nan, 50.0, 10.0],
            "OS_FLAG": [1.0, 0.0, np.nan, 1.0, 1.0],
            "Age": [60.0, np.nan, 70.0, 50.0, 40.0],
            "Gender": ["F", "M", None, "M", "F"],
            "RND_USE": ["Allowed"] * 4 + ["Not_Allowed"],
            "PDL1_TC_IHC_cat": ["TC0", "TC2+", None, "TC1", "TC9"],
            "TMB": [12.0, 3.0, None, 10.0, 9.9],
        }
    )


def test_harmonise_maps_diagnosis_and_flags_ici_and_excluded_histology():
    clin = io_ici.harmonise_clinical(_obs(), dataset="X")

    assert clin["diagnosis"].tolist() == ["SKCM", "SKCM", "NSCLC", "ccRCC", None]
    assert clin["is_ici"].tolist() == [True, True, True, False, True]
    assert clin["icb_class"].tolist() == ["anti-PD(L)1", "anti-CTLA4+anti-PD(L)1", "anti-PD(L)1+chemo", None, "anti-PD(L)1"]


def test_harmonise_derives_response_and_dcr():
    clin = io_ici.harmonise_clinical(_obs(), dataset="X")

    # response = ORR from RECIST (SD = no response); annotated R / NR only where RECIST is missing
    assert clin["response"].tolist()[:4] == [0.0, 0.0, 1.0, 1.0]
    assert clin["response_source"].tolist()[:4] == ["RECIST", "RECIST", "RECIST", "annotated R/NR"]
    # DCR from RECIST: CR/PR/SD = 1, PD = 0, missing RECIST = NaN
    assert clin["dcr"].tolist()[:3] == [1.0, 0.0, 1.0]
    assert np.isnan(clin["dcr"].iloc[3])


def test_harmonise_keeps_missing_columns_as_nan():
    clin = io_ici.harmonise_clinical(_obs(), dataset="X")

    assert clin["pfs"].isna().all()
    assert clin["stage"].isna().all()
    assert clin["os_event"].tolist()[:2] == [1.0, 0.0]
    assert clin["pdl1_tc"].tolist()[:2] == [0.0, 2.0] and clin["pdl1_tc"].iloc[2:].isna().tolist() == [True, False, True]


def test_tmb_high_uses_cutoff_10_and_skips_count_scale_studies():
    per_mb = io_ici.harmonise_clinical(_obs(), dataset="X")
    counts = io_ici.harmonise_clinical(_obs(), dataset="X", tmb_per_mb=False)

    assert per_mb["tmb_high"].tolist()[:2] == [1.0, 0.0] and per_mb["tmb_high"].tolist()[3:] == [1.0, 0.0]
    assert np.isnan(per_mb["tmb_high"].iloc[2])
    assert counts["tmb_high"].isna().all()


def test_covariates_unknown_level_and_age_missing_indicator():
    clin = pd.DataFrame({"age": [50.0, np.nan, 70.0, 60.0, 40.0, 55.0],
                         "sex": ["F", None, "M", "M", "F", "M"],
                         "stage": [None] * 6})

    X = ici_stats.covariate_matrix(clin, ["age", "sex", "stage"], min_level=1)

    assert X["age"].tolist()[1] == 55.0  # median of observed ages
    assert X["age_missing"].tolist() == [0, 1, 0, 0, 0, 0]
    assert "sex_unknown" in X.columns  # NaN kept as its own level, not dropped
    assert not any(c.startswith("stage") for c in X.columns)  # constant ("unknown" only) -> dropped


def test_covariates_merge_rare_levels():
    clin = pd.DataFrame({"site": ["a"] * 6 + ["b"] * 6 + ["c"]})

    X = ici_stats.covariate_matrix(clin, ["site"], min_level=5)

    assert "site_c" not in X.columns
    assert X.shape[1] == 2  # levels: a (reference), b, other


def test_zscore_within_groups():
    z = ici_stats.zscore_within(np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0]), np.array(["a"] * 3 + ["b"] * 3))

    assert np.allclose(z[:3], z[3:])
    assert np.isclose(z[:3].mean(), 0)


def test_stratified_cindex_ignores_pairs_across_strata():
    time = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    event = np.ones(6)
    risk = np.array([3.0, 2.0, 1.0, 30.0, 20.0, 10.0])  # high risk dies first in each stratum
    strata = np.array(["a"] * 3 + ["b"] * 3)

    assert ici_stats.stratified_cindex(time, event, risk, strata) == 1.0
    assert ici_stats.stratified_cindex(time, event, -risk, strata) == 0.0


def _survival_frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    score = rng.normal(size=n)
    noise = rng.normal(size=n)
    time = rng.exponential(np.exp(-0.8 * score))
    return pd.DataFrame({"time": time, "event": (rng.uniform(size=n) < 0.8).astype(float),
                         "score": score, "noise": noise, "cohort": rng.choice(["a", "b"], n),
                         "y": (rng.uniform(size=n) < 1 / (1 + np.exp(-1.5 * score))).astype(float)})


def test_cox_recovers_harmful_score_and_bootstrap_delta_favours_it():
    df = _survival_frame()

    fit = ici_stats.fit_cox(df, "time", "event", "score", pd.DataFrame(index=df.index), df["cohort"])
    assert fit["hr"] > 1.5 and fit["hr_lo"] > 1
    assert fit["c_index"] > 0.6

    table = ici_stats.bootstrap_compare(df, "cox", ("time", "event"), "score", ["noise", None],
                                        pd.DataFrame(index=df.index), df["cohort"], n_boot=50, seed=0)
    assert table["other"].tolist() == ["noise", "covariates only"]
    assert (table["delta"] > 0).all() and (table["delta_lo"] > 0).all()


def test_logit_recovers_or_and_auc():
    df = _survival_frame()

    fit = ici_stats.fit_logit(df, "y", "score", pd.DataFrame(index=df.index), df["cohort"])

    assert fit["or"] > 2 and fit["or_lo"] > 1
    assert fit["auc"] > 0.7


def test_logit_reports_nan_on_single_class():
    df = _survival_frame().assign(y=1.0)

    fit = ici_stats.fit_logit(df, "y", "score", pd.DataFrame(index=df.index), df["cohort"])

    assert np.isnan(fit["or"])


def test_score_signatures_zero_for_absent_genes():
    log_tpm = pd.DataFrame(np.arange(12, dtype=float).reshape(3, 4), columns=["A", "B", "C", "D"])

    scores = io_ici.score_signatures(log_tpm, {"s1": ["A", "B"], "s2": ["X"]})

    assert list(scores.columns) == ["s1", "s2"]
    assert (scores["s2"] == 0).all()


def test_dedup_keeps_first_sample():
    clin = pd.DataFrame({"sample": ["s1", "s1", "s2"], "dataset": ["a", "b", "a"]}, index=["a:s1", "b:s1", "a:s2"])

    assert io_ici.dedup_samples(clin).index.tolist() == ["a:s1", "a:s2"]


def test_fit_cox_reports_nan_with_too_few_events():
    df = _survival_frame(n=20).assign(event=0.0)

    fit = ici_stats.fit_cox(df, "time", "event", "score", pd.DataFrame(index=df.index), df["cohort"])

    assert np.isnan(fit["hr"]) and "events" in fit["note"]


def _report_inputs(n=240, seed=1):
    from tls_model import ici_report

    rng = np.random.default_rng(seed)
    clin = pd.DataFrame({
        "is_ici": [True] * (n - 10) + [False] * 10,
        "diagnosis": rng.choice(["SKCM", "NSCLC"], n).astype(object),
        "cohort": rng.choice(["c1", "c2", "c3"], n),
        "age": rng.normal(60, 10, n), "sex": rng.choice(["F", "M"], n),
        "icb_class": "anti-PD(L)1", "stage": None, "site": None, "location": None,
    }, index=[f"s{i}" for i in range(n)])
    p = pd.Series(rng.uniform(0.05, 0.95, n), index=clin.index)
    sig = pd.DataFrame({"sigA": rng.normal(size=n)}, index=clin.index)
    lp = np.log(p / (1 - p)).to_numpy()
    clin["os"] = rng.exponential(np.exp(0.7 * lp))
    clin["os_event"] = 1.0
    clin["pfs"], clin["pfs_event"] = np.nan, np.nan
    clin["response"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-lp))).astype(float)
    clin["dcr"] = clin["response"]
    return ici_report, clin, p, sig


def test_analysis_frame_keeps_ici_and_zscores_within_cohort():
    ici_report, clin, p, sig = _report_inputs()

    df = ici_report.analysis_frame(clin, p, sig)

    assert len(df) == clin["is_ici"].sum()
    assert np.allclose(df.groupby("cohort")[ici_report.OURS].mean(), 0, atol=1e-9)


def test_run_analysis_caches_parts_and_ours_beats_noise(tmp_path, caplog):
    import logging

    ici_report, clin, p, sig = _report_inputs()
    df = ici_report.analysis_frame(clin, p, sig)
    scores = [ici_report.OURS, "sigA", None]
    pairs = ici_report.main_pairs(["sigA"])

    assoc, delta = ici_report.run_analysis(df, "main", scores, ["age", "sex"], pairs, tmp_path, n_boot=20, seed=0)
    with caplog.at_level(logging.INFO):
        again, _ = ici_report.run_analysis(df, "main", scores, ["age", "sex"], pairs, tmp_path, n_boot=20, seed=0)

    pooled_os = assoc[(assoc["diagnosis"] == "All") & (assoc["endpoint"] == "OS") & (assoc["score"] == ici_report.OURS)].iloc[0]
    assert pooled_os["estimate"] < 1  # higher TLS score -> lower hazard in the simulation
    assert assoc[assoc["endpoint"] == "PFS"]["estimate"].isna().all()  # no PFS data -> reported, not crashed
    vs_sig = delta[(delta["diagnosis"] == "All") & (delta["endpoint"] == "Response") & (delta["other"] == "sigA")].iloc[0]
    assert vs_sig["delta"] > 0
    assert "SKIP (cached)" in caplog.text and len(again) == len(assoc)


def test_forest_plot_draws_every_endpoint(tmp_path):
    ici_report, clin, p, sig = _report_inputs()
    df = ici_report.analysis_frame(clin, p, sig)
    assoc, delta = ici_report.run_analysis(df, "main", [ici_report.OURS, "sigA"], ["age"], [(ici_report.OURS, "sigA")],
                                           tmp_path, n_boot=5, seed=0)

    for endpoint in ici_report.ENDPOINTS:
        fig = ici_report.forest_plot(assoc, delta, endpoint, [ici_report.OURS, "sigA"])
        assert fig.axes


def test_clean_design_drops_columns_collinear_within_strata():
    groups = np.array(["a"] * 4 + ["b"] * 4)
    X = pd.DataFrame({"x1": [1, 0, 1, 0, 0, 0, 0, 0], "x2": [0, 1, 0, 1, 0, 0, 0, 0],  # x1 + x2 = 1 inside 'a'
                      "x3": [0, 0, 0, 0, 1, 1, 1, 1],                                   # constant inside each stratum
                      "age": [50, 60, 70, 80, 55, 65, 75, 85]}, dtype=float)

    kept = ici_stats._clean_design(X, np.ones(8, bool), groups)

    assert list(kept.columns) == ["x1", "age"]


def _added_value_frame(n=600, ml_effect=0.6, interaction=0.0, seed=3):
    """12chem and ML correlated (r ≈ 0.5); hazard and response driven by 12chem (+ ML, + 12chem·ML)."""
    rng = np.random.default_rng(seed)
    chem = rng.normal(size=n)
    ml = 0.5 * chem + np.sqrt(0.75) * rng.normal(size=n)
    lp = 0.5 * chem + ml_effect * ml + interaction * chem * ml
    return pd.DataFrame({"chem": chem, "ml": ml, "chem_x_ml": chem * ml, "cohort": rng.choice(["a", "b"], n),
                         "time": rng.exponential(np.exp(-lp)), "event": 1.0,
                         "y": (rng.uniform(size=n) < 1 / (1 + np.exp(-lp))).astype(float)})


def test_multi_term_model_names_and_reports_last_term():
    df = _added_value_frame()
    X = pd.DataFrame(index=df.index)

    fit = ici_stats.fit_model(df, "cox", ("time", "event"), ("chem", "ml"), X, df["cohort"])

    assert ici_stats.model_name(("chem", "ml")) == "chem + ml"
    assert ici_stats.model_name(None) == ici_stats.COVARIATES_ONLY
    assert fit["estimate"] > 1.3  # HR of the last term (ml), adjusted for chem
    assert fit["k"] == 2 and np.isfinite(fit["loglik"])


def test_lrt_and_bootstrap_detect_signal_added_on_top_of_base_score():
    df = _added_value_frame(ml_effect=0.6)
    X, groups = pd.DataFrame(index=df.index), df["cohort"]
    for kind, outcome in (("cox", ("time", "event")), ("logit", "y")):
        small = ici_stats.fit_model(df, kind, outcome, ("chem",), X, groups)
        big = ici_stats.fit_model(df, kind, outcome, ("chem", "ml"), X, groups)

        assert ici_stats.nested_lrt(big, small)["lr_p"] < 0.01
        table = ici_stats.bootstrap_compare(df, kind, outcome, ("chem", "ml"), [("chem",)], X, groups, n_boot=20, seed=0)
        assert table["delta"].iloc[0] > 0 and table["other"].iloc[0] == "chem"


def test_lrt_finds_nothing_when_added_score_is_noise():
    df = _added_value_frame(ml_effect=0.0)
    X, groups = pd.DataFrame(index=df.index), df["cohort"]

    small = ici_stats.fit_model(df, "cox", ("time", "event"), ("chem",), X, groups)
    big = ici_stats.fit_model(df, "cox", ("time", "event"), ("chem", "ml"), X, groups)

    assert ici_stats.nested_lrt(big, small)["lr_p"] > 0.01
    assert abs(big["metric"] - small["metric"]) < 0.02


def test_lrt_detects_interaction_term():
    df = _added_value_frame(ml_effect=0.3, interaction=0.6)
    X, groups = pd.DataFrame(index=df.index), df["cohort"]

    main = ici_stats.fit_model(df, "logit", "y", ("chem", "ml"), X, groups)
    inter = ici_stats.fit_model(df, "logit", "y", ("chem", "ml", "chem_x_ml"), X, groups)

    assert inter["estimate"] > 1.3  # OR of the interaction term
    assert ici_stats.nested_lrt(inter, main)["lr_df"] == 1 and ici_stats.nested_lrt(inter, main)["lr_p"] < 0.01


def test_nested_models_use_the_same_complete_case_rows():
    df = _added_value_frame()
    df.loc[:9, "ml"] = np.nan
    X, groups = pd.DataFrame(index=df.index), df["cohort"]
    from tls_model import ici_report

    rows = ici_report.complete_rows(df, [("chem",), ("chem", "ml")])
    small = ici_stats.fit_model(rows, "cox", ("time", "event"), ("chem",), X.loc[rows.index], groups.loc[rows.index])
    big = ici_stats.fit_model(rows, "cox", ("time", "event"), ("chem", "ml"), X.loc[rows.index], groups.loc[rows.index])

    assert small["n"] == big["n"] == len(df) - 10


def test_run_analysis_adds_lrt_for_nested_pairs_only(tmp_path):
    ici_report, clin, p, sig = _report_inputs()
    df = ici_report.analysis_frame(clin, p, sig)
    models = [("sigA",), ("sigA", ici_report.OURS), ici_report.OURS]
    pairs = [(("sigA", ici_report.OURS), ("sigA",)), (ici_report.OURS, ("sigA",))]

    _, delta = ici_report.run_analysis(df, "added", models, ["age"], pairs, tmp_path, n_boot=5, seed=0)

    pooled = delta[(delta["diagnosis"] == "All") & (delta["endpoint"] == "OS")].set_index("reference")
    assert np.isfinite(pooled.loc[f"sigA + {ici_report.OURS}", "lr_p"])
    assert np.isnan(pooled.loc[ici_report.OURS, "lr_p"])  # not nested -> no LRT


def test_added_value_plot_labels_pairs_and_draws(tmp_path):
    ici_report, clin, p, sig = _report_inputs()
    df = ici_report.analysis_frame(clin, p, sig.rename(columns={"sigA": ici_report.CHEM}))
    df = ici_report.with_interaction(df)

    _, delta = ici_report.run_analysis(df, "added", ici_report.ADDED_MODELS, ["age"],
                                       list(ici_report.ADDED_PAIRS.values()), tmp_path, n_boot=3, seed=0)
    labelled = ici_report.label_pairs(delta, ici_report.ADDED_PAIRS)
    fig = ici_report.added_value_plot(labelled, list(ici_report.ADDED_PAIRS), "test")

    assert set(labelled["comparison"]) == set(ici_report.ADDED_PAIRS)
    assert len(fig.axes) == 3  # OS, Response, DCR (no PFS data in the simulation)
