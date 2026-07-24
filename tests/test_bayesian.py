import pytest

from abtest.bayesian import bayesian_conversion


def test_symmetric_when_identical():
    r = bayesian_conversion(100, 1000, 100, 1000)
    assert r.prob_b_beats_a == pytest.approx(0.5, abs=0.02)


def test_confident_when_b_clearly_better():
    r = bayesian_conversion(100, 1000, 160, 1000)
    assert r.prob_b_beats_a > 0.99
    assert r.expected_uplift > 0


def test_reproducible_with_seed():
    a = bayesian_conversion(100, 1000, 130, 1000, seed=42)
    b = bayesian_conversion(100, 1000, 130, 1000, seed=42)
    assert a.prob_b_beats_a == b.prob_b_beats_a
