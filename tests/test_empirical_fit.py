"""
Tests for the Week 3 additions that let the DDM family be fitted to real mice:
contrast exponent, trial variabilities, neutral side bias, t_max / quantile pass-through,
and the shared cohort-fitting helpers in src/optimization/empirical.py.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import validate_contract
from src.models.simulator import DDMSimulator, simulate_ddm
from src.optimization.empirical import (
    EMPIRICAL_FIT_DEFAULTS,
    delta_bic,
    fit_cohort,
    fit_mouse,
    fitted_params,
    simulate_from_fit,
)
from src.optimization.fit import DDMFitter, fit_ddm

# keeps every fit in this file to a couple of seconds
TINY = dict(n_screen=2, n_starts=1, n_restarts=0, maxiter=15, n_sim_per_cond=60, n_sim_final=120)


def test_default_parameter_sets_are_unchanged():
    """Track A's recovery pipeline relies on these (the new options are opt-in)."""
    assert DDMFitter("M2").param_names == ["a", "t0", "k", "v_h", "v_b"]
    assert DDMFitter("M1").param_names == ["a", "t0", "k", "v_h", "z"]
    assert DDMFitter("M3").param_names == ["a", "t0", "k", "v_h", "b_max", "tau_b"]
    assert DDMFitter("M2", bound_type="weibull").param_names[-2:] == ["lam", "gamma"]


def test_optional_parameters_are_added_when_requested():
    names = DDMFitter("M2", bound_type="weibull", fit_beta=True, fit_variability=True, bias_type="both").param_names
    for p in ("beta", "s_v", "s_z", "v_0", "z_0", "lam", "gamma", "v_b"):
        assert p in names
    assert "v_0" in DDMFitter("M1", bias_type="drift").param_names
    assert "z_0" not in DDMFitter("M1", bias_type="drift").param_names
    with pytest.raises(ValueError, match="bias_type"):
        DDMFitter("M2", bias_type="sideways")


def test_neutral_side_bias_shifts_choices_independently_of_the_prior():
    n = 4000
    c, prior = np.zeros(n), np.ones(n)
    sim = DDMSimulator(dt=0.002, t_max=2.5)
    base = {"a": 1.0, "t0": 0.2, "k": 3.0}
    p_base = (sim.simulate(base, c, prior, rng=np.random.default_rng(1))["choice"] == 1).mean()
    p_v0 = (sim.simulate({**base, "v_0": -1.0}, c, prior, rng=np.random.default_rng(1))["choice"] == 1).mean()
    p_z0 = (sim.simulate({**base, "z_0": -0.4}, c, prior, rng=np.random.default_rng(1))["choice"] == 1).mean()
    assert abs(p_base - 0.5) < 0.05                    # no prior mechanism in the model -> unbiased at 0%
    assert p_v0 < 0.25 and p_z0 < 0.4


def test_t_max_allows_simulating_beyond_2p5_seconds():
    c = np.zeros(3000)
    p = {"a": 2.0, "t0": 0.1, "k": 1.0}
    slow = DDMSimulator(dt=0.005, t_max=5.0).simulate(p, c, np.zeros(3000), rng=np.random.default_rng(0))
    short = DDMSimulator(dt=0.005, t_max=2.5).simulate(p, c, np.zeros(3000), rng=np.random.default_rng(0))
    assert slow["rt"].max() > 2.7 and short["rt"].max() <= 2.5 + 0.1 + 1e-9


def test_fit_ddm_passes_t_max_and_quantiles_through():
    sim = simulate_ddm({"a": 1.2, "t0": 0.2, "k": 3.0, "v_b": 0.4}, "M2", n_trials=600, seed=3)
    res = fit_ddm(sim, "M2", t_max=5.0, quantiles=(0.1, 0.5, 0.9, 0.97), **TINY)
    assert res["success"] in (True, False) and np.isfinite(res["bic"])
    assert fit_ddm(sim, "M2", bound_type="collapsing", **TINY)["bound_type"] == "collapsing"


def test_fit_mouse_uses_empirical_settings_and_reports_parameter_count():
    sim = simulate_ddm({"a": 1.2, "t0": 0.2, "k": 3.0, "v_b": 0.4}, "M2", n_trials=600, seed=4)
    res = fit_mouse(sim, "M2", "fixed", 2.5, **TINY)
    for p in ("beta", "s_v", "s_z", "v_0", "v_b"):
        assert p in res["params"]
    assert res["n_params"] == len(res["params"])
    assert res["n_trials"] == 600
    assert EMPIRICAL_FIT_DEFAULTS["fit_beta"] and EMPIRICAL_FIT_DEFAULTS["bias_type"] == "drift"
    assert EMPIRICAL_FIT_DEFAULTS["n_screen"] >= 80          # light search is too noisy for delta-BIC (see module docs)


def test_fit_cohort_is_resumable(tmp_path):
    a = simulate_ddm({"a": 1.2, "t0": 0.2, "k": 3.0, "v_b": 0.4}, "M2", n_trials=500, seed=5, subject_id="mA")
    b = simulate_ddm({"a": 1.0, "t0": 0.25, "k": 2.5, "z": 0.3}, "M1", n_trials=500, seed=6, subject_id="mB")
    cohort = pd.concat([a, b], ignore_index=True)
    cache = str(tmp_path / "fits.csv")
    first = fit_cohort(cohort, models=("M1", "M2"), n_jobs=1, cache_path=cache, **TINY)
    assert len(first) == 4 and {"subject_id", "model", "bound", "rt_max", "bic", "p_a"} <= set(first.columns)
    mtime = (tmp_path / "fits.csv").stat().st_mtime_ns
    second = fit_cohort(cohort, models=("M1", "M2"), n_jobs=1, cache_path=cache, **TINY)
    assert len(second) == 4 and (tmp_path / "fits.csv").stat().st_mtime_ns == mtime   # nothing refitted
    third = fit_cohort(cohort, models=("M1", "M2", "M3"), n_jobs=1, cache_path=cache,
                       subjects=["mA"], **TINY)
    assert len(third) == 5                                                           # only mA / M3 was added
    p = fitted_params(first.iloc[0])
    assert "a" in p and "bic" not in p


def test_simulate_from_fit_replays_covariates_and_respects_cutoff():
    real = simulate_ddm({"a": 1.2, "t0": 0.2, "k": 3.0, "v_b": 0.4}, "M2", n_trials=400, seed=7)
    pred = simulate_from_fit(real, {"a": 1.2, "t0": 0.2, "k": 3.0, "v_b": 0.4}, "M2", rt_max=1.0, n_rep=3)
    validate_contract(pred)
    assert pred["rt"].max() <= 1.0
    assert set(pred["contrast"].unique()) <= set(real["contrast"].unique())


def test_delta_bic_is_relative_to_the_reference_level():
    res = pd.DataFrame({
        "subject_id": ["a", "a", "b", "b"], "model": ["M2"] * 4, "rt_max": [2.5] * 4,
        "bound": ["fixed", "weibull", "fixed", "weibull"], "bic": [100.0, 90.0, 50.0, 70.0],
    })
    d = delta_bic(res, by="bound", reference="fixed").set_index("subject_id")["delta_bic"]
    assert d["a"] == -10.0 and d["b"] == 20.0
