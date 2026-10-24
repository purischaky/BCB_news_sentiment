import numpy as np
import pytest

from src.stats_utils import fisher_ci, permutation_pvalue

def test_fisher_ci_known_values():
    low, high = fisher_ci(0.5, 30)
    assert low == pytest.approx(0.17, abs=0.01)
    assert high == pytest.approx(0.73, abs=0.01)

def test_fisher_ci_is_symmetric_around_zero():
    low, high = fisher_ci(0.0, 41)
    assert low == pytest.approx(-high)

def test_permutation_pvalue_small_for_strong_relationship():
    x = np.arange(30, dtype=float)
    y = x + np.random.default_rng(0).normal(0, 1, 30)
    assert permutation_pvalue(x, y, n_perm=2000) < 0.01

def test_permutation_pvalue_is_a_probability():
    rng = np.random.default_rng(1)
    p = permutation_pvalue(rng.normal(size=30), rng.normal(size=30), n_perm=500)
    assert 0 < p <= 1