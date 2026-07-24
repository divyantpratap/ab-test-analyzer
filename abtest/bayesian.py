"""Bayesian read on a conversion experiment.

Beta-Binomial model with a uniform Beta(1, 1) prior on each variant. We draw
from the two posteriors and estimate P(B beats A) and the expected loss of
each choice, which is often easier to act on than a p-value.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BayesianResult:
    prob_b_beats_a: float
    expected_uplift: float      # mean of (rate_b - rate_a) over the posterior
    expected_loss_choose_b: float
    ci_low: float               # 95% credible interval on (rate_b - rate_a)
    ci_high: float

    def summary(self) -> str:
        return (
            f"P(B > A) = {self.prob_b_beats_a * 100:.1f}%. "
            f"Expected uplift {self.expected_uplift * 100:.2f} points, "
            f"95% credible interval [{self.ci_low * 100:.2f}, {self.ci_high * 100:.2f}]."
        )


def bayesian_conversion(
    conv_a: int, n_a: int, conv_b: int, n_b: int,
    prior_alpha: float = 1.0, prior_beta: float = 1.0,
    draws: int = 200_000, seed: int = 7,
) -> BayesianResult:
    if n_a <= 0 or n_b <= 0:
        raise ValueError("sample sizes must be positive")

    rng = np.random.default_rng(seed)
    post_a = rng.beta(prior_alpha + conv_a, prior_beta + n_a - conv_a, draws)
    post_b = rng.beta(prior_alpha + conv_b, prior_beta + n_b - conv_b, draws)

    diff = post_b - post_a
    prob_b = float(np.mean(post_b > post_a))
    # Expected loss if we ship B but A was actually better.
    expected_loss_b = float(np.mean(np.maximum(post_a - post_b, 0.0)))
    ci_low, ci_high = np.percentile(diff, [2.5, 97.5])

    return BayesianResult(
        prob_b_beats_a=prob_b,
        expected_uplift=float(np.mean(diff)),
        expected_loss_choose_b=expected_loss_b,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
    )
