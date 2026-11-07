"""
Visualization module for publication-grade recovery benchmarks and psychometric curves.
"""

from .plots_recovery import (
    plot_parameter_recovery,
    plot_error_correlation_matrix,
    plot_model_recovery_confusion,
    plot_cbf,
    plot_psychometric_chronometric,
)

from .plots_behavior import (
    plot_psychometric_chronometric_cohort,
    plot_psychometric_by_mouse,
    plot_chronometric_by_mouse,
    plot_cbf_cohort,
    plot_cbf_by_mouse,
    plot_cbf_sensitivity,
    plot_rt_distributions,
    plot_heterogeneity,
)

from .plots_c1 import (
    plot_delta_bic,
    plot_cbf_overlay_cohort,
    plot_cbf_overlay_by_mouse,
    plot_cbf_slope_agreement,
)

from .plots_c2 import (
    plot_bound_comparison,
    plot_fitted_bounds,
    plot_invariance,
    plot_tail_by_mouse,
    plot_tail_diagnostics,
)

__all__ = [
    "plot_bound_comparison",
    "plot_fitted_bounds",
    "plot_invariance",
    "plot_tail_by_mouse",
    "plot_tail_diagnostics",
    "plot_delta_bic",
    "plot_cbf_overlay_cohort",
    "plot_cbf_overlay_by_mouse",
    "plot_cbf_slope_agreement",
    "plot_psychometric_chronometric_cohort",
    "plot_psychometric_by_mouse",
    "plot_chronometric_by_mouse",
    "plot_cbf_cohort",
    "plot_cbf_by_mouse",
    "plot_cbf_sensitivity",
    "plot_rt_distributions",
    "plot_heterogeneity",
    "plot_parameter_recovery",
    "plot_error_correlation_matrix",
    "plot_model_recovery_confusion",
    "plot_cbf",
    "plot_psychometric_chronometric",
]
