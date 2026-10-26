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

__all__ = [
    "plot_parameter_recovery",
    "plot_error_correlation_matrix",
    "plot_model_recovery_confusion",
    "plot_cbf",
    "plot_psychometric_chronometric",
]
