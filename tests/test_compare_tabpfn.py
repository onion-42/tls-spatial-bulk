import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import compare_tabpfn  # noqa: E402


def test_paired_delta_is_zero_for_identical_scores_and_positive_for_a_better_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=400)
    good = y + rng.normal(scale=0.5, size=400)
    noise = rng.normal(size=400)

    same = compare_tabpfn.paired_delta(y, good, good, n_boot=100)
    better = compare_tabpfn.paired_delta(y, good, noise, n_boot=100)

    assert same == (0.0, 0.0, 0.0)
    assert better[0] > 0.2 and better[1] > 0  # the whole interval above zero


def test_paired_delta_resamples_whole_groups():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=300)
    a, b = y + rng.normal(scale=1.0, size=300), y + rng.normal(scale=1.0, size=300)
    groups = np.repeat(np.arange(6), 50)

    by_unit = compare_tabpfn.paired_delta(y, a, b, n_boot=200)
    by_group = compare_tabpfn.paired_delta(y, a, b, groups, n_boot=200)

    assert by_unit[0] == by_group[0]            # same point estimate
    assert (by_unit[1], by_unit[2]) != (by_group[1], by_group[2])  # different resampling unit -> different interval
