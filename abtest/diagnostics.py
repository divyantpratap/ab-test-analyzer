"""Data-quality checks you should run before trusting an experiment."""
from __future__ import annotations

from dataclasses import dataclass

from scipy import stats


@dataclass
class SRMResult:
    p_value: float
    mismatch: bool
    observed_ratio: float
    expected_ratio: float

    def summary(self) -> str:
        if self.mismatch:
            return (
                f"Sample ratio mismatch. Assignment looks broken "
                f"(p = {self.p_value:.5f}). Do not trust the result until this is fixed."
            )
        return f"Split looks fine (p = {self.p_value:.3f})."


def srm_check(n_a: int, n_b: int, expected_share_a: float = 0.5, threshold: float = 0.001) -> SRMResult:
    """Sample Ratio Mismatch test.

    A chi-square goodness-of-fit test on the two bucket counts against the split
    you intended. A very small p-value means users were not assigned in the ratio
    you expected, which usually points to a logging or assignment bug rather than
    a real effect.
    """
    total = n_a + n_b
    if total == 0:
        raise ValueError("no observations")
    exp_a = total * expected_share_a
    exp_b = total * (1 - expected_share_a)
    chi2, p = stats.chisquare([n_a, n_b], [exp_a, exp_b])
    return SRMResult(
        p_value=float(p),
        mismatch=p < threshold,
        observed_ratio=n_a / total,
        expected_ratio=expected_share_a,
    )
