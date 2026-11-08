"""
Track B empirical-analysis package: psychometrics, chronometrics, Conditional Bias Function,
heterogeneity profiling, RT-tail diagnostics (`tails`, Week 3 / C2) and a synthetic mock-data generator for harness testing.
Author: Joshua Koilpillai (2024101128)
"""

from .cbf import (
    bootstrap_cbf,
    cbf_by_subject,
    cbf_curve,
    cbf_slope,
    classify_shape,
    cohort_cbf,
    cohort_slope_test,
    compute_cbf,
)
from . import tails
from .heterogeneity import heterogeneity_summary, mouse_profile, profile_correlations
from .mock import make_mock_cohort, make_mock_trials, mock_truth
from .psychometrics import (
    accuracy_by_rt_bin,
    across_subject,
    chronometric_table,
    fit_psychometric,
    fit_psychometric_by,
    prior_shift,
    psychometric_function,
    psychometric_table,
    rt_quantile_table,
)

__all__ = [
    "bootstrap_cbf", "cbf_by_subject", "cbf_curve", "cbf_slope", "classify_shape", "cohort_cbf",
    "cohort_slope_test", "compute_cbf", "heterogeneity_summary", "mouse_profile",
    "profile_correlations", "make_mock_cohort", "make_mock_trials", "mock_truth", "accuracy_by_rt_bin",
    "across_subject", "chronometric_table", "fit_psychometric", "fit_psychometric_by",
    "prior_shift", "psychometric_function", "psychometric_table", "rt_quantile_table", "tails",
]
