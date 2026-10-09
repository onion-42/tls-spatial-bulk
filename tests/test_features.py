import numpy as np
import pandas as pd

from tls_model import features


def _ssgsea_brute(x, genes, signature, alpha=0.25):
    """Running-sum ssGSEA (Barbie 2009) for one sample, no ties."""
    order = np.argsort(-x)
    ranks = np.empty(len(x))
    ranks[np.argsort(x)] = np.arange(1, len(x) + 1)
    in_set = np.isin(np.array(genes)[order], signature)
    weights = ranks[order] ** alpha
    p_hit = np.cumsum(np.where(in_set, weights, 0)) / weights[in_set].sum()
    p_miss = np.cumsum(~in_set) / (~in_set).sum()
    return float(np.sum(p_hit - p_miss))


def test_ssgsea_matches_running_sum():
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(50)]
    X = rng.normal(size=(4, 50))
    signatures = {"a": ["G1", "G7", "G30"], "b": ["G2", "G3", "G4", "G5", "G49"]}

    scores = features.ssgsea(X, genes, signatures)

    for i in range(4):
        assert np.isclose(scores[i, 0], _ssgsea_brute(X[i], genes, signatures["a"]))
        assert np.isclose(scores[i, 1], _ssgsea_brute(X[i], genes, signatures["b"]))


def test_ssgsea_ignores_missing_genes_and_ranks_up_high_expression():
    genes = ["G0", "G1", "G2", "G3"]
    X = np.array([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]])

    scores = features.ssgsea(X, genes, {"top": ["G3", "MISSING"]})

    assert scores[0, 0] > scores[1, 0]


def test_within_sample_rank_is_scale_free():
    X = np.array([[1.0, 5.0, 3.0], [10.0, 50.0, 30.0]])

    out = features.within_sample_rank(X)

    assert np.allclose(out[0], out[1])
    assert np.allclose(out[0], [0.0, 1.0, 0.5])


def test_gene_filter_keeps_protected_and_top_variable_expressed_genes():
    rng = np.random.default_rng(1)
    genes = [f"G{i}" for i in range(6)]
    bulk = np.column_stack(
        [
            rng.normal(5, 3, 200),  # G0 expressed, very variable
            rng.normal(5, 1, 200),  # G1 expressed, variable
            rng.normal(5, 0.01, 200),  # G2 expressed, flat
            rng.normal(0.1, 0.05, 200),  # G3 not expressed
            rng.normal(0.1, 0.05, 200),  # G4 not expressed but protected
            rng.normal(5, 2, 200),  # G5 expressed, variable, never detected in spots
        ]
    )
    spot_detection = np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0])

    kept = features.gene_filter(bulk, spot_detection, genes, n_top=2, protected={"G4"})

    assert kept == ["G0", "G1", "G4"]


def test_platform_offset_is_median_of_paired_differences():
    bulk = pd.DataFrame([[5.0, 1.0], [7.0, 1.0], [6.0, 1.0]], index=["p1", "p2", "p3"], columns=["A", "B"])
    pseudo = pd.DataFrame([[4.0, 2.0], [4.0, 2.0], [4.0, 2.0]], index=["p1", "p2", "p3"], columns=["A", "B"])

    offset = features.platform_offset(bulk, pseudo, genes=["A", "B", "C"])

    assert offset.tolist() == [2.0, -1.0, 0.0]


def test_weighted_abs_corr_respects_weights():
    x = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 0, 1])

    r = features.weighted_abs_corr(x, y, np.ones(4))

    assert np.isclose(r[0], 1.0)
    assert np.isclose(r[1], 0.0)


def test_pair_products_names_and_values():
    X = np.array([[1.0, 2.0, 3.0]])

    values, names = features.pair_products(X, ["a", "b", "c"], prefix="g")

    assert names == ["g:a*b", "g:a*c", "g:b*c"]
    assert values.tolist() == [[2.0, 3.0, 6.0]]
