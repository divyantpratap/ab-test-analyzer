"""Frequentist analysis for two-variant experiments.

Everything here works from summary numbers (conversions and sample sizes, or
mean/sd/n), so you can analyse an experiment without the raw event log.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from scipy import stats


@dataclass
class ProportionResult:
    rate_a: float
    rate_b: float
    abs_diff: float          # rate_b - rate_a
    rel_lift: float          # (rate_b - rate_a) / rate_a
    z: float
    p_value: float
    ci_low: float            # CI on the absolute difference
    ci_high: float
    significant: bool

    def summary(self) -> str:
        direction = "up" if self.abs_diff >= 0 else "down"
        return (
            f"B is {direction} {abs(self.rel_lift) * 100:.2f}% vs A "
            f"({self.rate_a * 100:.2f}% to {self.rate_b * 100:.2f}%), "
            f"p = {self.p_value:.4f}, "
            f"95% CI on the absolute change [{self.ci_low * 100:.2f}%, {self.ci_high * 100:.2f}%]."
        )


@dataclass
class MeanResult:
    mean_a: float
    mean_b: float
    diff: float
    t: float
    df: float
    p_value: float
    ci_low: float
    ci_high: float
    significant: bool


def two_proportion_test(
    conv_a: int, n_a: int, conv_b: int, n_b: int, alpha: float = 0.05
) -> ProportionResult:
    """Two-sided z-test for a difference in conversion rates.

    The test statistic uses the pooled proportion. The confidence interval on
    the difference uses the unpooled standard error, which is the usual pairing.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError("sample sizes must be positive")
    if not (0 <= conv_a <= n_a and 0 <= conv_b <= n_b):
        raise ValueError("conversions must be between 0 and n")

    p_a = conv_a / n_a
    p_b = conv_b / n_b
    diff = p_b - p_a

    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se_pool = sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = diff / se_pool if se_pool > 0 else 0.0
    p_value = 2 * stats.norm.sf(abs(z))

    se_unpooled = sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_low = diff - z_crit * se_unpooled
    ci_high = diff + z_crit * se_unpooled

    rel_lift = diff / p_a if p_a > 0 else float("nan")

    return ProportionResult(
        rate_a=p_a,
        rate_b=p_b,
        abs_diff=diff,
        rel_lift=rel_lift,
        z=z,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        significant=p_value < alpha,
    )


def welch_t_test(
    mean_a: float, sd_a: float, n_a: int,
    mean_b: float, sd_b: float, n_b: int,
    alpha: float = 0.05,
) -> MeanResult:
    """Welch's t-test for a difference in means (unequal variances)."""
    if n_a < 2 or n_b < 2:
        raise ValueError("each group needs at least 2 observations")

    va, vb = sd_a ** 2, sd_b ** 2
    se = sqrt(va / n_a + vb / n_b)
    diff = mean_b - mean_a
    t = diff / se if se > 0 else 0.0

    # Welch-Satterthwaite degrees of freedom
    df = (va / n_a + vb / n_b) ** 2 / (
        (va / n_a) ** 2 / (n_a - 1) + (vb / n_b) ** 2 / (n_b - 1)
    )
    p_value = 2 * stats.t.sf(abs(t), df)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_low = diff - t_crit * se
    ci_high = diff + t_crit * se

    return MeanResult(
        mean_a=mean_a, mean_b=mean_b, diff=diff, t=t, df=df,
        p_value=p_value, ci_low=ci_low, ci_high=ci_high,
        significant=p_value < alpha,
    )


def sample_size_proportions(
    baseline: float, abs_mde: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Sample size PER GROUP to detect an absolute lift of `abs_mde`.

    Uses the normal-approximation formula with unpooled variances at the two
    rates. Returns the rounded-up per-group count.
    """
    if not (0 < baseline < 1):
        raise ValueError("baseline must be a probability strictly between 0 and 1")
    p1 = baseline
    p2 = baseline + abs_mde
    if not (0 < p2 < 1):
        raise ValueError("baseline + mde must stay between 0 and 1")

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    numerator = (z_alpha + z_power) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    n = numerator / (p2 - p1) ** 2
    return int(-(-n // 1))  # ceil


def relative_mde_to_absolute(baseline: float, rel_mde: float) -> float:
    """Turn a relative MDE (e.g. 0.05 for +5%) into an absolute rate change."""
    return baseline * rel_mde
