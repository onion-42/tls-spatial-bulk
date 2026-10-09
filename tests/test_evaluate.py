import numpy as np

from tls_model import evaluate


def test_binary_metrics_perfect_predictor_with_ci():
    y = np.array([0, 0, 1, 1] * 25)
    p = y * 0.8 + 0.1

    out = evaluate.binary_metrics(y, p, n_boot=200, seed=0)

    assert out["n"] == 100 and out["n_pos"] == 50
    assert out["auroc"] == 1.0 and out["auroc_lo"] == 1.0
    assert np.isclose(out["brier"], 0.01)


def test_bootstrap_by_group_resamples_whole_groups():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 50)
    y = rng.integers(0, 2, 500)
    p = rng.uniform(size=500)

    point, lo, hi = evaluate.bootstrap_ci(evaluate.METRICS["auroc"], y, p, groups=groups, n_boot=200, seed=0)

    assert lo <= point <= hi


def test_three_class_balanced_accuracy():
    y_any = np.array([0, 1, 1, 0])
    y_mature = np.array([np.nan, 0, 1, np.nan])
    probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])

    out = evaluate.three_class_balanced_accuracy(y_any, y_mature, probs)

    assert np.isclose(out, (0.5 + 1 + 1) / 3)


def test_section_spearman():
    sections = np.repeat(["a", "b", "c"], 2)
    p = np.array([0.1, 0.1, 0.5, 0.5, 0.9, 0.9])
    fraction = np.array([0.0, 0.0, 0.1, 0.1, 0.3, 0.3])

    assert np.isclose(evaluate.section_spearman(sections, p, fraction), 1.0)
