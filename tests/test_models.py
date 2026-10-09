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
