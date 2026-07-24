import math

import pytest

from abtest.stats import (
    two_proportion_test,
    welch_t_test,
    sample_size_proportions,
    relative_mde_to_absolute,
)
from abtest.diagnostics import srm_check


def test_equal_rates_not_significant():
    r = two_proportion_test(100, 1000, 100, 1000)
    assert r.p_value == pytest.approx(1.0, abs=1e-6)
    assert not r.significant
    assert r.abs_diff == pytest.approx(0.0)


def test_clear_difference_is_significant():
    # 10% vs 13% on 2000 per arm is a strong, real effect.
    r = two_proportion_test(200, 2000, 260, 2000)
    assert r.significant
    assert r.p_value < 0.01
    assert r.rate_b > r.rate_a
    assert r.rel_lift == pytest.approx(0.30, abs=1e-9)


def test_ci_brackets_the_difference():
    r = two_proportion_test(200, 2000, 260, 2000)
    assert r.ci_low < r.abs_diff < r.ci_high


def test_two_proportion_matches_known_z():
    # Hand-checked: 50/100 vs 60/100, pooled z should be about -1.4213.
    r = two_proportion_test(50, 100, 60, 100)
    assert r.z == pytest.approx(1.4213, abs=1e-3)
    assert r.p_value == pytest.approx(0.1552, abs=1e-3)


def test_welch_identical_groups():
    r = welch_t_test(10.0, 2.0, 500, 10.0, 2.0, 500)
    assert r.p_value == pytest.approx(1.0, abs=1e-6)
    assert not r.significant


def test_welch_detects_shift():
    r = welch_t_test(10.0, 2.0, 500, 10.5, 2.0, 500)
    assert r.significant
    assert r.diff == pytest.approx(0.5)


def test_sample_size_shrinks_with_bigger_effect():
    small = sample_size_proportions(0.10, relative_mde_to_absolute(0.10, 0.05))
    big = sample_size_proportions(0.10, relative_mde_to_absolute(0.10, 0.20))
    assert small > big
    assert big > 0


def test_sample_size_rejects_bad_input():
    with pytest.raises(ValueError):
        sample_size_proportions(0.0, 0.01)


def test_srm_flags_bad_split():
    bad = srm_check(1000, 700)
    assert bad.mismatch
    good = srm_check(1000, 1000)
    assert not good.mismatch
