"""
Decision boundaries for Drift Diffusion Models (DDM).
Supports fixed and Weibull-collapsing boundaries (C2 urgency hypothesis).
"""

from typing import Union
import numpy as np


def fixed_boundary(t: Union[float, np.ndarray], a: float) -> Union[float, np.ndarray]:
    """
    Fixed decision boundary: a(t) = a.

    Parameters
    ----------
    t : float or np.ndarray
        Elapsed decision time (seconds).
    a : float
        Boundary separation parameter (a > 0).

    Returns
    -------
    float or np.ndarray
        Boundary threshold at time(s) t.
    """
    if isinstance(t, np.ndarray):
        return np.full_like(t, a, dtype=float)
    return float(a)


def weibull_boundary(
    t: Union[float, np.ndarray],
    a: float,
    lam: float = 1.0,
    gamma: float = 2.0,
    min_boundary: float = 0.05
) -> Union[float, np.ndarray]:
    """
    Weibull-collapsing decision boundary:
        a(t) = a * exp( - (t / lam)^gamma )

    Captures dynamic urgency / deadline pressure (Hawkins et al., 2015; Chakravarty et al., 2024).

    Parameters
    ----------
    t : float or np.ndarray
        Elapsed decision time (seconds).
    a : float
        Initial boundary height at t=0 (a > 0).
    lam : float
        Scale parameter (latency of collapse, in seconds, lam > 0).
    gamma : float
        Shape parameter (steepness of collapse, gamma > 0).
    min_boundary : float
        Minimum boundary height to prevent numerical collapse to <= 0.

    Returns
    -------
    float or np.ndarray
        Boundary threshold at time(s) t.
    """
    # Safe exponentiation
    t_safe = np.maximum(t, 0.0)
    decay = np.exp(-np.power(t_safe / np.maximum(lam, 1e-5), gamma))
    bound = a * decay
    return np.maximum(bound, min_boundary)


def get_boundary_function(bound_type: str = "fixed"):
    """
    Factory returning boundary function based on model specification.
    """
    if bound_type.lower() == "fixed":
        return fixed_boundary
    elif bound_type.lower() in ("collapsing", "weibull"):
        return weibull_boundary
    else:
        raise ValueError(f"Unknown boundary type: '{bound_type}'. Must be 'fixed' or 'weibull'.")
