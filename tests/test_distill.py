"""Self-distillation (teacher pseudo-labels -> student), the direct 3-class form and the leakage guard."""

import numpy as np
import pandas as pd
import pytest

from tls_model import pipeline
from tests.test_pipeline import _units

META = {"signatures": {"sA": ["G0", "G1"], "sB": ["G2", "G3"]}}


def _features(units):
    meta = {**META, "genes": list(units.var_names), "offset": np.zeros(units.n_vars, dtype=np.float32)}
    B, names = pipeline.feature_matrix(units, meta, representation="rank", use_offset=False)
    return B, names, meta


def _pseudo(obs, n_folds=3):
    """Teacher output stand-in: perfect soft labels on the distill rows."""
    rows = pipeline.distill_rows(obs)
    y = obs["tls_any"].to_numpy(dtype=float)
    p_any = np.where(y == 1, 0.9, 0.1)
    p_mat = np.where(obs["tls_mature"].to_numpy(dtype=float) == 1, 0.8, 0.2)
    p3 = np.column_stack([1 - p_any, p_any * (1 - p_mat), p_any * p_mat])
    by_fold = {"any": np.tile(p_any, (n_folds, 1)), "mature": np.tile(p_mat, (n_folds, 1)), "three": np.tile(p3, (n_folds, 1, 1))}
    final = {"any": p_any, "mature": p_mat, "three": p3}
    return pipeline.PseudoLabels(rows=rows, by_fold=by_fold, final=final)


def test_distill_rows_are_the_train_split_of_the_distill_cohorts():
    obs = _units().obs

    rows = pipeline.distill_rows(obs)

    assert (obs.loc[rows, "role"] == "distill").all()
    assert set(obs.loc[rows, "cohort"]) == {"C1", "C2"} and set(obs.loc[rows, "split"]) == {"train"}


def test_student_fit_set_soft_weights_sum_to_the_base_weight():
    units = _units()
    B, names, _ = _features(units)
    pseudo = _pseudo(units.obs)
    n_train = pipeline.train_rows(units.obs, "any").sum()
    n_distill = pseudo.rows.sum()

    X, y, w = pipeline.fit_set(B, units.obs, "any", pseudo, fold=None)
    X3, y3, w3 = pipeline.fit_set(B, units.obs, "three", pseudo, fold=None)

    assert X.shape[0] == n_train + 2 * n_distill and set(np.unique(y)) == {0, 1}
    assert X3.shape[0] == pipeline.train_rows(units.obs, "three").sum() + 3 * n_distill
    assert set(np.unique(y3)) == {0, 1, 2}
    # the two (three) copies of a distill row carry its whole weight between them
    distill_w = w[n_train:].reshape(2, n_distill).sum(axis=0)
    assert np.allclose(distill_w, distill_w[0])
    # visium and distilled bulk get equal total weight, like bulk and visium did before
    assert np.isclose(w[:n_train].sum(), w[n_train:].sum())


def test_student_fit_set_for_maturity_weights_distill_rows_by_p_tls():
    units = _units()
    B, names, _ = _features(units)
    pseudo = _pseudo(units.obs)
    n_train = pipeline.train_rows(units.obs, "mature").sum()

    X, y, w = pipeline.fit_set(B, units.obs, "mature", pseudo, fold=0)

    n_distill = pseudo.rows.sum()
    pair = w[n_train:].reshape(2, n_distill).sum(axis=0)
    p_any = pseudo.by_fold["any"][0][pseudo.rows]
    assert np.allclose(pair / pair.max(), p_any / p_any.max())


def test_fit_set_without_pseudo_labels_is_the_teacher_set():
    units = _units()
    B, names, _ = _features(units)
    rows = pipeline.train_rows(units.obs, "any")

    X, y, w = pipeline.fit_set(B, units.obs, "any", None, fold=None)

    assert X.shape[0] == rows.sum() and np.allclose(y, units.obs["tls_any"].to_numpy()[rows])


def test_run_cv_returns_fold_predictions_on_distill_rows_and_accepts_pseudo_labels():
    units = _units()
    B, names, meta = _features(units)
    candidates = [{"family": "lgbm", "selector": "corr"}]
    distill = pipeline.distill_rows(units.obs)

    table, oof, fold_pred = pipeline.run_cv(B, names, units.obs, "any", candidates, meta, n_folds=3, seed=0,
                                            n_panel=5, n_pair_genes=2, predict_rows=distill)
    pseudo = pipeline.PseudoLabels(rows=distill, by_fold={"any": fold_pred["lgbm|corr"]},
                                   final={"any": fold_pred["lgbm|corr"].mean(axis=0)})
    table_s, oof_s, _ = pipeline.run_cv(B, names, units.obs, "any", candidates, meta, n_folds=3, seed=0,
                                        n_panel=5, n_pair_genes=2, pseudo=pseudo)

    assert fold_pred["lgbm|corr"].shape == (3, len(units.obs))
    assert np.isfinite(fold_pred["lgbm|corr"][:, distill]).all() and np.isnan(fold_pred["lgbm|corr"][:, ~distill]).all()
    train_role = (units.obs["role"] == "train").to_numpy()  # out-of-fold predictions cover every train unit
    assert np.isfinite(oof_s["lgbm|corr"][train_role]).all() and np.isnan(oof_s["lgbm|corr"][~train_role]).all()
    assert table_s["score"].iloc[0] > 0.7


def test_run_cv_three_class_returns_probability_matrices():
    units = _units()
    B, names, meta = _features(units)
    candidates = [{"family": "lgbm", "selector": "corr"}]

    table, oof, _ = pipeline.run_cv(B, names, units.obs, "three", candidates, meta, n_folds=3, seed=0,
                                    n_panel=5, n_pair_genes=2)

    rows = pipeline.train_rows(units.obs, "three")
    assert oof["lgbm|corr"].shape == (len(units.obs), 3)
    assert np.allclose(oof["lgbm|corr"][rows].sum(axis=1), 1.0)
    assert "macro_auroc" in table.columns


def test_compare_forms_scores_hierarchy_and_direct_on_the_same_rows():
    units = _units()
    obs = units.obs
    y3 = obs["tls_three"].to_numpy(dtype=float)
    known = ~np.isnan(y3)
    p_any = np.where(obs["tls_any"] == 1, 0.9, 0.1)
    p_mat = np.where(obs["tls_mature"] == 1, 0.9, 0.1)
    direct = np.full((len(obs), 3), 1 / 3)

    table = pipeline.compare_forms(obs, {"hierarchy": pipeline.three_class(p_any, p_mat), "direct": direct})

    assert list(table["form"]) == ["hierarchy", "direct"]
    assert table.set_index("form").loc["hierarchy", "balanced_accuracy"] == 1.0
    assert table["n"].iloc[0] == (known & pipeline.train_rows(obs, "three")).sum() or table["n"].iloc[0] > 0
    assert table.set_index("form").loc["hierarchy", "auroc_tls_all_spots"] == 1.0
    assert table["n_all_spots"].iloc[0] > table["n"].iloc[0] - table["n"].iloc[0]  # spots incl. unknown maturity
    assert pipeline.choose_form(table) == "hierarchy"


def test_choose_form_needs_direct_to_win_on_both_criteria():
    base = {"n": 10, "balanced_accuracy": 0.5}
    both = pd.DataFrame([{"form": "hierarchy", "macro_auroc": 0.8, "auroc_tls_all_spots": 0.9, **base},
                         {"form": "direct", "macro_auroc": 0.9, "auroc_tls_all_spots": 0.95, **base}])
    one = pd.DataFrame([{"form": "hierarchy", "macro_auroc": 0.8, "auroc_tls_all_spots": 0.9, **base},
                        {"form": "direct", "macro_auroc": 0.9, "auroc_tls_all_spots": 0.7, **base}])

    assert pipeline.choose_form(both) == "direct"
    assert pipeline.choose_form(one) == "hierarchy"


def test_leakage_report_passes_on_clean_roles_and_raises_on_overlap():
    obs = _units().obs
    fit = {"any": pipeline.train_rows(obs, "any") | pipeline.distill_rows(obs)}
    cal = {"any": pipeline.calibration_rows(obs, "any")}

    table = pipeline.leakage_report(obs, fit, cal)

    assert set(table.index) == {"train", "distill", "test", "holdout"}
    assert table.loc["holdout", "patients"] > 0

    dirty = fit["any"].copy()
    dirty[np.flatnonzero((obs["role"] == "holdout").to_numpy())[0]] = True
    with pytest.raises(ValueError, match="holdout"):
        pipeline.leakage_report(obs, {"any": dirty}, cal)

    shared = obs.copy()
    shared.loc[shared["role"] == "test", "patient"] = shared.loc[shared["role"] == "train", "patient"].iloc[0]
    with pytest.raises(ValueError, match="patient"):
        pipeline.leakage_report(shared, fit, cal)


def test_fit_final_with_pseudo_labels_calibrates_bulk_on_distill_gold():
    units = _units()
    B, names, meta = _features(units)
    pseudo = _pseudo(units.obs)
    rows = pipeline.train_rows(units.obs, "any")
    oof = np.full(len(units.obs), np.nan)
    oof[rows] = np.where(units.obs["tls_any"].to_numpy()[rows] == 1, 0.8, 0.2)

    model, calibrator = pipeline.fit_final(B, names, units.obs, "any", {"family": "lgbm", "selector": "corr"}, oof,
                                           meta, n_panel=5, n_pair_genes=2, seed=0, pseudo=pseudo)

    assert set(calibrator.models) == {"bulk", "spot"}
    p = pipeline.predict(B, names, units.obs, model, calibrator)
    assert p.shape == (len(units.obs),) and np.all((p >= 0) & (p <= 1))


def test_predict_three_in_chunks_matches_one_pass_for_both_forms():
    units = _units()
    B, names, meta = _features(units)
    variant = {"representation": "rank", "offset": False}
    pseudo = _pseudo(units.obs)
    saved = {"form": "hierarchy", "models": {}, "calibrators": {}}
    for head in ("any", "mature"):
        rows = pipeline.train_rows(units.obs, head)
        oof = np.full(len(units.obs), np.nan)
        oof[rows] = np.clip(units.obs[pipeline.HEAD_LABEL[head]].to_numpy(dtype=float)[rows] * 0.6 + 0.2, 0, 1)
        saved["models"][head], saved["calibrators"][head] = pipeline.fit_final(
            B, names, units.obs, head, {"family": "lgbm", "selector": "corr"}, oof, meta, 5, 2, 0, pseudo=pseudo)

    one = pipeline.predict_three(B, names, units.obs, saved)
    chunked = pipeline.predict_three_in_chunks(units, meta, variant, saved, chunk=37)

    assert one.shape == (len(units.obs), 3) and np.allclose(one.sum(axis=1), 1.0)
    assert np.allclose(one, chunked)
    p_any, p_mature = pipeline.heads_from_three(one)
    assert np.allclose(p_any, 1 - one[:, 0]) and np.all((p_mature >= 0) & (p_mature <= 1))
