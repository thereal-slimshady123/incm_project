"""
Unit tests for DDM Simulator, boundary dynamics, and QMP loss surface.
"""

import pytest
import numpy as np
from src.models._kernel import HAVE_NUMBA
from src.models.boundaries import fixed_boundary, weibull_boundary
from src.models.simulator import DDMSimulator, simulate_ddm
from src.optimization.fit import fit_ddm
from src.optimization.qmp import QMPDataset, compute_qmp_nll, compute_qmp_nll_blocks


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
    priors = np.random.default_rng(0).choice([-1.0, 1.0], size=3000)

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


def test_choice_history_is_chained():
    """Verify prev_choice is the agent's own previous choice and v_h biases toward repeating it."""
    params = {"a": 1.0, "t0": 0.2, "k": 2.5, "v_h": 1.0}
    df = simulate_ddm(params, model_type="M2", contrasts=np.zeros(3000), priors=np.zeros(3000), seed=21)

    assert df["prev_choice"].iloc[0] == 0
    assert np.array_equal(df["prev_choice"].to_numpy()[1:], df["choice"].to_numpy()[:-1])
    p_repeat = np.mean(df["choice"].to_numpy()[1:] == df["prev_choice"].to_numpy()[1:])
    assert p_repeat > 0.6, f"Expected v_h > 0 to produce choice repetition, got P(repeat)={p_repeat:.2f}"


def test_crn_reproducible_and_parameter_independent_noise():
    """Same seed gives identical output; a parameter change only moves trials it affects."""
    sim = DDMSimulator()
    contrasts = np.full(500, 0.25)
    priors = np.ones(500)
    base = {"a": 1.0, "t0": 0.2, "k": 2.5, "v_b": 0.5}
    df1 = sim.simulate(base, contrasts, priors, model_type="M2", rng=np.random.default_rng(3))
    df2 = sim.simulate(base, contrasts, priors, model_type="M2", rng=np.random.default_rng(3))
    assert df1.equals(df2)

    # Changing only t0 shifts every RT by the same amount (noise streams unchanged)
    df3 = sim.simulate({**base, "t0": 0.3}, contrasts, priors, model_type="M2", rng=np.random.default_rng(3))
    assert np.array_equal(df1["choice"], df3["choice"])
    assert np.allclose(df3["rt"] - df1["rt"], 0.1)


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("model_type,params,bound_type", [
    ("M1", {"a": 1.1, "t0": 0.2, "k": 2.6, "z": 0.4, "v_h": 0.3}, "fixed"),
    ("M2", {"a": 1.1, "t0": 0.2, "k": 2.6, "v_b": 0.6, "s_v": 0.5, "s_z": 0.3}, "fixed"),
    ("M3", {"a": 1.1, "t0": 0.2, "k": 2.6, "b_max": 1.2, "tau_b": 0.5}, "fixed"),
    ("M2", {"a": 1.4, "t0": 0.2, "k": 2.0, "v_b": 0.4, "lam": 0.8, "gamma": 2.0}, "weibull"),
])
def test_numba_matches_numpy_backend(model_type, params, bound_type):
    """Compiled kernel and NumPy fallback must produce the same choice/RT distributions."""
    n = 20000
    rng = np.random.default_rng(0)
    contrasts = rng.choice([-0.25, 0.0, 0.0625, 0.25], n)
    priors = rng.choice([-1.0, 1.0], n)
    prev = rng.choice([-1.0, 1.0], n)

    out = {}
    for backend in ("numba", "numpy"):
        out[backend] = DDMSimulator(backend=backend).simulate(
            params, contrasts, priors, prev, model_type=model_type, bound_type=bound_type,
            rng=np.random.default_rng(1),
        )

    a, b = out["numba"], out["numpy"]
    # P(Right) agrees within ~4 standard errors overall
    assert abs((a["choice"] == 1).mean() - (b["choice"] == 1).mean()) < 0.015
    q = [0.1, 0.3, 0.5, 0.7, 0.9]
    assert np.allclose(np.quantile(a["rt"], q), np.quantile(b["rt"], q), atol=0.02)


def test_qmp_block_nll_matches_groupby_nll():
    """The fitter's fast block QMP must equal the reference groupby implementation."""
    params = {"a": 1.1, "t0": 0.2, "k": 2.6, "v_b": 0.5}
    data = simulate_ddm(params, model_type="M2", n_trials=1500, seed=4)
    qmp_data = QMPDataset(data)

    n_per = 150
    contrasts = np.repeat([c["contrast"] for c in qmp_data.conditions], n_per)
    priors = np.repeat([c["prior"] for c in qmp_data.conditions], n_per)
    sim = DDMSimulator().simulate(params, contrasts, priors, model_type="M2", rng=np.random.default_rng(5))

    nll_ref, _ = compute_qmp_nll(sim, qmp_data)
    nll_fast = compute_qmp_nll_blocks(sim["choice"].to_numpy(), sim["rt"].to_numpy(), qmp_data, n_per)
    assert np.isclose(nll_ref, nll_fast)

    # A vanishing smoothing kernel reduces to hard counts. Simulated and observed RTs share
    # the 1 ms grid, so shift slightly off the bin edges (a kernel splits exact ties 50/50).
    rts = sim["rt"].to_numpy() + 1e-7
    nll_hard = compute_qmp_nll_blocks(sim["choice"].to_numpy(), rts, qmp_data, n_per)
    nll_smooth = compute_qmp_nll_blocks(sim["choice"].to_numpy(), rts, qmp_data, n_per, bandwidth=1e-9)
    assert np.isclose(nll_hard, nll_smooth, rtol=1e-6)


def test_smoothed_qmp_is_continuous_in_t0():
    """With kernel smoothing a tiny t0 change moves the NLL; with hard counts it usually cannot."""
    params = {"a": 1.1, "t0": 0.2, "k": 2.6, "v_b": 0.5}
    data = simulate_ddm(params, model_type="M2", n_trials=800, seed=12)
    qmp_data = QMPDataset(data)
    n_per = 100
    contrasts = np.repeat([c["contrast"] for c in qmp_data.conditions], n_per)
    priors = np.repeat([c["prior"] for c in qmp_data.conditions], n_per)

    def nll(t0, bandwidth):
        sim = DDMSimulator().simulate({**params, "t0": t0}, contrasts, priors, model_type="M2",
                                      rng=np.random.default_rng(13))
        return compute_qmp_nll_blocks(sim["choice"].to_numpy(), sim["rt"].to_numpy(), qmp_data, n_per,
                                      bandwidth=bandwidth)

    assert nll(0.2, 0.01) != nll(0.2 + 1e-6, 0.01)
    assert nll(0.2, None) == nll(0.2 + 1e-6, None)


def test_fit_ddm_runs_and_returns_contract():
    """fit_ddm must run end to end and return the §3.2 result fields (smoke test, few iterations)."""
    params = {"a": 1.1, "t0": 0.22, "k": 2.8, "v_b": 0.5, "v_h": 0.3}
    data = simulate_ddm(params, model_type="M2", n_trials=300, seed=6)
    for model_type in ("M1", "M2", "M3", "JOINT"):
        res = fit_ddm(
            data, model_type=model_type, maxiter=3, n_sim_per_cond=30, n_restarts=0,
            n_screen=2, n_sim_final=60,
        )
        assert {"params", "nll", "bic", "success"} <= set(res)
        assert np.isfinite(res["nll"]) and np.isfinite(res["bic"])
        assert "v_h" in res["params"]


def test_m3_profiles_tau_b_over_grid():
    """M3 fits pick tau_b from the grid, and tau_b still counts in the BIC parameter count."""
    params = {"a": 1.1, "t0": 0.22, "k": 2.8, "b_max": 0.8, "tau_b": 0.35}
    data = simulate_ddm(params, model_type="M3", n_trials=300, seed=8)
    grid = (0.15, 0.8)
    res = fit_ddm(
        data, model_type="M3", maxiter=3, n_sim_per_cond=30, n_restarts=0,
        n_screen=2, n_sim_final=60, tau_b_grid=grid,
    )
    assert res["params"]["tau_b"] in grid
    assert set(res["tau_b_profile"]) == set(grid)
    # a, t0, k, v_h, b_max free + tau_b selected from the grid = 6 parameters
    n_trials = len(data)
    assert np.isclose(res["bic"], 2.0 * res["nll"] + 6 * np.log(n_trials))


def test_demo_cbf_curve_is_contrast_stratified():
    """simulate_ddm ties stimulus side to the block; the CBF used in the demo must not inherit that."""
    from src.visualization.plots_recovery import compute_cbf_curve
    # a model with NO prior mechanism: any raw choice difference between blocks is pure stimulus statistics
    df = simulate_ddm({"a": 1.1, "t0": 0.2, "k": 2.6}, model_type="M2", n_trials=30000, seed=1)
    _, vals = compute_cbf_curve(df)
    assert np.all(np.abs(vals) < 0.08)
