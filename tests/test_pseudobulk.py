import numpy as np
import scipy.sparse as sp

from tls_model import pseudobulk


def _hex_grid(n_rows=9, n_cols=18):
    """Visium-like grid: on each row only every second column is a spot, offset by row parity."""
    rows, cols = [], []
    for r in range(n_rows):
        for c in range(r % 2, n_cols, 2):
            rows.append(r)
            cols.append(c)
    return np.array(rows), np.array(cols)


def test_neighbourhood_sizes_are_1_7_19_in_the_interior():
    rows, cols = _hex_grid()
    centre = int(np.flatnonzero((rows == 4) & (cols == 8))[0])

    sizes = [pseudobulk.neighbourhood(rows, cols, radius)[centre].nnz for radius in (0, 1, 2)]

    assert sizes == [1, 7, 19]


def _section(n_rows=6, n_cols=12):
    import anndata as ad
    import pandas as pd

    rows, cols = _hex_grid(n_rows, n_cols)
    n = len(rows)
    tls = np.zeros(n, dtype=int)
    mature = np.full(n, np.nan)
    tls[:3] = 1
    mature[:3] = [0.0, 1.0, np.nan]
    excluded = np.zeros(n, dtype=bool)
    excluded[-1] = True
    obs = pd.DataFrame(
        {
            "array_row": rows, "array_col": cols, "tls": tls, "tls_mature": mature, "excluded": excluded,
            "section": "S1", "patient": "P1", "protocol": "ffpe", "platform": "visium_ffpe", "source": "X",
        },
        index=[f"b{i}" for i in range(n)],
    )
    return ad.AnnData(X=sp.csr_matrix(np.ones((n, 3), dtype=np.float32)), obs=obs)


def test_build_units_scales_labels_and_caps():
    section = _section()
    rng = np.random.default_rng(0)

    units = pseudobulk.build_units(section, ks=(1, 7), cap=10, max_positive=2, rng=rng)

    spots = units.obs[units.obs["k"] == 1]
    assert len(spots) == 10 and (spots["tls"] == 1).sum() == 2
    assert "b35" not in set(units.obs["center"])  # excluded spot is never a centre
    whole = units.obs[units.obs["unit"] == "section"].iloc[0]
    assert whole["tls"] == 1 and whole["tls_mature"] == 1
    # excluded (lymph-node) spots contribute neither counts nor label share
    assert np.isclose(whole["tls_fraction"], 3 / 35)
    assert units.X[units.obs_names.get_loc(whole.name)].sum() == 35 * 3


def test_excluded_spots_are_not_summed_into_neighbour_aggregates():
    section = _section()
    obs = section.obs
    last = len(obs) - 1  # the excluded spot
    adjacency = pseudobulk.neighbourhood(obs["array_row"].to_numpy(), obs["array_col"].to_numpy(), 1)
    neighbour = int([j for j in adjacency[last].indices if j != last][0])

    units = pseudobulk.build_units(section, ks=(7,), cap=None, max_positive=None, rng=np.random.default_rng(0))

    row = units.obs.index.get_loc(f"S1:agg7:b{neighbour}")
    assert units.X[row].sum() == (adjacency[neighbour].nnz - 1) * 3


def test_spot_prevalence_counts_non_excluded_spots():
    out = pseudobulk.spot_prevalence(_section())

    assert out == {"n_spots": 35, "n_tls": 3, "n_mature_known": 2, "n_mature": 1}


def test_build_units_without_cap_keeps_every_spot():
    units = pseudobulk.build_units(_section(), ks=(1,), cap=None, max_positive=None, rng=np.random.default_rng(0))

    assert (units.obs["k"] == 1).sum() == 35  # 36 spots minus the excluded one


def test_log_cpm_is_library_size_normalised():
    counts = sp.csr_matrix(np.array([[1, 3], [10, 30]]))

    out = pseudobulk.log2_cpm(counts)

    assert np.allclose(out[0], out[1])
    assert np.allclose(out[0], np.log2(np.array([0.25, 0.75]) * 1e6 + 1))
