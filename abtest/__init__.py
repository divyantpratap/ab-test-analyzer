"""Small toolkit for planning and reading A/B tests."""
from .stats import (
    ProportionResult,
    MeanResult,
    two_proportion_test,
    welch_t_test,
    sample_size_proportions,
    relative_mde_to_absolute,
)
from .bayesian import BayesianResult, bayesian_conversion
from .diagnostics import srm_check

__all__ = [
    "ProportionResult",
    "MeanResult",
    "two_proportion_test",
    "welch_t_test",
    "sample_size_proportions",
    "relative_mde_to_absolute",
    "BayesianResult",
    "bayesian_conversion",
    "srm_check",
]
