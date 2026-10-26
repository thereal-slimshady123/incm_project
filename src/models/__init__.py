"""
DDM Models module: boundaries, accumulation dynamics, and vectorized Euler-Maruyama simulator.
"""

from .boundaries import fixed_boundary, weibull_boundary
from .simulator import simulate_ddm, DDMSimulator

__all__ = ["fixed_boundary", "weibull_boundary", "simulate_ddm", "DDMSimulator"]
