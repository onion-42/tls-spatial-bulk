import numpy as np
import pytest

from tls_model import models


def _toy(n=400, seed=0):
    """Base matrix with genes, signature scores, and a label driven by gene:A × gene:B."""
    rng = np.random.default_rng(seed)
    genes = ["A", "B", "C", "D", "E", "F"]
    G = rng.uniform(size=(n, len(genes))).astype(np.float32)
    S = rng.normal(size=(n, 2)).astype(np.float32)
    B = np.hstack([G, S])
    names = [f"gene:{g}" for g in genes] + ["sig:s1", "sig:s2"]
    y = (G[:, 0] * G[:, 1] + 0.05 * rng.normal(size=n) > 0.25).astype(int)
    return B, names, y, np.ones(n)


@pytest.mark.parametrize("family", ["enet", "lgbm", "ktsp"])
@pytest.mark.parametrize("selector", ["corr", "shap"])
def test_head_learns_and_selects_inside_fit(family, selector):
    B, names, y, w = _toy()
    head = models.HeadModel(family, selector, n_panel=4, n_pair_genes=3, signature_genes={"A", "B", "C"}, seed=0)

    head.fit(B[:300], names, y[:300], w[:300])
    p = head.predict_proba(B[300:], names)

    assert p.shape == (100,) and np.all((p >= 0) & (p <= 1))
    # kTSP rules ("gene i < gene j") cannot express the A×B interaction; it only has to beat chance
    assert models.auroc(y[300:], p) > (0.6 if family == "ktsp" else 0.75)
    assert len(head.panel) <= 4
    assert set(head.pair_genes) <= {"A", "B", "C"}


def test_signature_family_uses_only_signature_scores():
    B, names, y, w = _toy()

    head = models.HeadModel("signature", None, seed=0).fit(B, names, y, w)

    assert head.panel == ["sig:s1", "sig:s2"]


def test_shap_values_add_up_for_linear_and_tree_models():
    B, names, y, w = _toy()
    for family in ("enet", "lgbm"):
        head = models.HeadModel(family, "corr", n_panel=4, n_pair_genes=3, signature_genes={"A", "B"}, seed=0)
        head.fit(B, names, y, w)

        values, columns, X = head.shap_values(B[:50], names)

        assert values.shape == X.shape == (50, len(columns))
        assert np.isfinite(values).all()


def test_ktsp_finds_the_informative_pair():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(500, 5))
    y = (X[:, 2] > X[:, 4]).astype(int)

    model = models.KTSP(k=1).fit(X, y, np.ones(500))

    assert {model.pairs[0][0], model.pairs[0][1]} == {2, 4}
    assert models.auroc(y, model.predict_proba(X)) > 0.99


def test_three_class_probabilities_sum_to_one():
    p_any = np.array([0.1, 0.9, 0.5])
    p_mature = np.array([0.5, 0.2, 1.0])

    probs = models.three_class(p_any, p_mature)

    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(probs[1], [0.1, 0.72, 0.18])


def test_platt_calibration_per_unit_kind():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=400)
    kind = np.array(["bulk", "visium"] * 200)
    y = (rng.uniform(size=400) < np.where(kind == "bulk", p, p**3)).astype(int)

    cal = models.PlattCalibrator().fit(p, y, kind)
    out = cal.transform(np.array([0.5, 0.5]), np.array(["bulk", "visium"]))

    assert out[0] > out[1]


def test_platt_prior_shift_corrects_enriched_sampling():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=2000)
    y = (rng.uniform(size=2000) < p).astype(int)  # sampled prevalence ~0.5
    kind = np.array(["spot"] * 2000)

    plain = models.PlattCalibrator().fit(p, y, kind)
    shifted = models.PlattCalibrator().fit(p, y, kind, prior={"spot": 0.05})

    assert np.isclose(plain.transform(p, kind).mean(), y.mean(), atol=0.02)
    assert shifted.transform(p, kind).mean() < 0.15


@pytest.mark.parametrize("selector", ["corr", "shap"])
def test_lgbm_head_handles_three_classes(selector):
    B, names, y, w = _toy(n=600)
    rng = np.random.default_rng(1)
    y3 = np.where(y == 1, np.where(B[:, 2] > 0.5, 2, 1), 0)  # class 2 = positive with high gene:C

    model = models.HeadModel("lgbm", selector, n_panel=6, n_pair_genes=2, signature_genes={"A", "B", "C"}).fit(B, names, y3, w)
    probs = model.predict_proba(B, names)

    assert probs.shape == (600, 3) and np.allclose(probs.sum(axis=1), 1.0)
    assert (probs.argmax(axis=1) == y3).mean() > 0.9
    assert model.shap_values(B, names) is not None


def test_weighted_resample_follows_the_weights():
    rng = np.random.default_rng(0)
    X = np.arange(2000, dtype=np.float32).reshape(1000, 2)
    y = rng.integers(0, 2, size=1000)
    w = np.where(np.arange(1000) < 100, 9.0, 1.0)  # first 100 rows carry half of the total weight

    Xr, yr = models.weighted_resample(X, y, w, seed=0)

    assert Xr.shape == X.shape and yr.shape == y.shape
    share = np.mean(Xr[:, 0] < 200)  # rows 0..99 hold values 0..199 in column 0
    assert 0.45 < share < 0.55
    Xr2, yr2 = models.weighted_resample(X, y, w, seed=0)
    assert np.array_equal(Xr, Xr2) and np.array_equal(yr, yr2)


def test_weighted_resample_keeps_every_class():
    X = np.zeros((50, 1), dtype=np.float32)
    y = np.array([1] + [0] * 49)
    w = np.array([1e-9] + [1.0] * 49)  # the only positive row is almost never drawn

    _, yr = models.weighted_resample(X, y, w, seed=0)

    assert set(np.unique(yr)) == {0, 1}


def test_weighted_resample_rescue_does_not_remove_another_class():
    X = np.zeros((3, 1), dtype=np.float32)
    y = np.array([0, 1, 2])
    w = np.array([1.0, 1e-9, 1e-9])  # the draw is all class 0; classes 1 and 2 must both be rescued

    _, yr = models.weighted_resample(X, y, w, seed=0)

    assert set(np.unique(yr)) == {0, 1, 2}


def test_weighted_resample_rejects_bad_weights_and_skips_zero_weight_classes():
    X = np.zeros((4, 1), dtype=np.float32)
    y = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError):
        models.weighted_resample(X, y, np.zeros(4))
    with pytest.raises(ValueError):
        models.weighted_resample(X, y, np.array([1.0, -1.0, 1.0, 1.0]))
    _, yr = models.weighted_resample(X, y, np.array([1.0, 1.0, 0.0, 0.0]), seed=0)
    assert set(np.unique(yr)) == {0}  # a class the weights exclude stays excluded


def test_tabpfn_is_a_family_with_shared_selection():
    from tls_model import pipeline

    assert "tabpfn" in models.FAMILIES and "tabpfn" in pipeline.SHARED_SELECTION
    with pytest.raises(ValueError):
        models.HeadModel("tabpfn", None)
    with pytest.raises(ValueError):
        models.HeadModel("tabpfn", "corr", tabpfn_weights="unknown")


def test_tabpfn_head_learns_on_toy_data():
    pytest.importorskip("tabpfn")
    B, names, y, w = _toy()
    head = models.HeadModel("tabpfn", "corr", n_panel=4, n_pair_genes=3, signature_genes={"A", "B", "C"}, seed=0)

    head.fit(B[:300], names, y[:300], w[:300])
    p = head.predict_proba(B[300:], names)

    assert p.shape == (100,) and np.all((p >= 0) & (p <= 1))
    assert models.auroc(y[300:], p) > 0.75
    assert head.shap_values(B[300:], names) is None


def test_selector_all_keeps_every_base_column():
    B, names, y, w = _toy()
    head = models.HeadModel("lgbm", "all", n_panel=4, n_pair_genes=3, signature_genes={"A", "B", "C"}, seed=0)
    head.fit(B, names, y, w)
    assert head.panel == names
    assert models.auroc(y, head.predict_proba(B, names)) > 0.9


def test_selector_coef_ranks_by_the_linear_model_itself():
    B, names, y, w = _toy()
    head = models.HeadModel("enet", "coef", n_panel=3, n_pair_genes=3, signature_genes={"A", "B", "C"}, seed=0)
    head.fit(B, names, y, w)
    assert len(head.panel) == 3
    assert any("A" in col and "B" in col or col in ("gene:A", "gene:B") for col in head.panel)
    assert not {"gene:E", "gene:F"} & set(head.panel)
