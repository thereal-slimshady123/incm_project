"""
Unit tests for DDM Simulator, boundary dynamics, and QMP loss surface.
"""

import pytest
import numpy as np
from src.models.boundaries import fixed_boundary, weibull_boundary
from src.models.simulator import simulate_ddm
from src.optimization.qmp import QMPDataset, compute_qmp_nll


def test_weibull_boundary_monotonicity():
    """Verify Weibull boundary decays monotonically with decision time."""
    times = np.linspace(0, 2.0, 50)
    bounds = weibull_boundary(times, a=1.5, lam=1.2, gamma=2.0)

    # Monotonic non-increasing
    diffs = np.diff(bounds)
    assert np.all(diffs <= 1e-6), "Weibull boundary must monotonically collapse over time"
    assert bounds[-1] < bounds[0], "Boundary at t=2.0 s must be strictly lower than at t=0 s"


def test_m1_prior_bias_on_ambiguous_stimuli():
    """Verify M1 starting point bias pushes zero-contrast choices in the direction of the prior."""
    contrasts = np.zeros(2000)
    priors = np.ones(2000)  # Pure right-biased block

    params = {"a": 1.0, "t0": 0.2, "k": 2.5, "z": 0.35}
    df = simulate_ddm(params, model_type="M1", contrasts=contrasts, priors=priors, seed=123)

    p_right = np.mean(df["choice"] == 1)
    assert p_right > 0.60, f"Expected M1 prior bias to elevate P(Right) above 0.60 on 0% contrast, got {p_right:.2f}"


def test_m2_drift_bias_on_ambiguous_stimuli():
    """Verify M2 drift rate prior bias pushes choices in the direction of the prior."""
    contrasts = np.zeros(2000)
    priors = np.ones(2000)  # Pure right-biased block

    params = {"a": 1.0, "t0": 0.2, "k": 2.5, "v_b": 0.6}
    df = simulate_ddm(params, model_type="M2", contrasts=contrasts, priors=priors, seed=456)

    p_right = np.mean(df["choice"] == 1)
    assert p_right > 0.60, f"Expected M2 drift bias to elevate P(Right) above 0.60 on 0% contrast, got {p_right:.2f}"


def test_m3_dynamic_bias_time_dependence():
    """
    Verify M3 dynamic bias produces stronger bias for slow RTs than fast RTs (Hanks et al., 2011).
    """
    contrasts = np.zeros(3000)
    priors = np.random.choice([-1.0, 1.0], size=3000)

    params = {"a": 1.1, "t0": 0.2, "k": 2.5, "b_max": 1.2, "tau_b": 0.6}
    df = simulate_ddm(params, model_type="M3", contrasts=contrasts, priors=priors, seed=789)

    median_rt = df["rt"].median()
    fast_trials = df[df["rt"] <= median_rt]
    slow_trials = df[df["rt"] > median_rt]

    def get_bias(sub):
        rp = sub[sub["prior"] > 0]
        lp = sub[sub["prior"] < 0]
        return np.mean(rp["choice"] == 1) - np.mean(lp["choice"] == 1)

    bias_fast = get_bias(fast_trials)
    bias_slow = get_bias(slow_trials)

    assert bias_slow > bias_fast, (
        f"M3 dynamic bias must produce higher prior bias on slow trials ({bias_slow:.2f}) "
        f"than fast trials ({bias_fast:.2f})"
    )


def test_qmp_loss_minimum_at_true_parameters():
    """Verify QMP NLL is lower at ground truth parameters than distorted parameters."""
    true_params = {"a": 1.1, "t0": 0.22, "k": 2.8, "v_b": 0.5, "v_h": 0.0}
    df_true = simulate_ddm(true_params, model_type="M2", n_trials=1500, seed=10)

    qmp_data = QMPDataset(df_true)

    # Forward evaluate with true vs distorted parameters
    df_eval_true = simulate_ddm(true_params, model_type="M2", n_trials=2500, seed=11)
    distorted_params = {"a": 1.8, "t0": 0.45, "k": 1.0, "v_b": -0.8, "v_h": 0.0}
    df_eval_distorted = simulate_ddm(distorted_params, model_type="M2", n_trials=2500, seed=11)

    nll_true, _ = compute_qmp_nll(df_eval_true, qmp_data)
    nll_distorted, _ = compute_qmp_nll(df_eval_distorted, qmp_data)

    assert nll_true < nll_distorted, (
        f"QMP NLL at true parameters ({nll_true:.1f}) must be substantially lower than "
        f"at distorted parameters ({nll_distorted:.1f})"
    )
