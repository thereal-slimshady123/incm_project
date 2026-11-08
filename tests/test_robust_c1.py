"""Tests for the robust C1 additions: inferred prior, lapse mixture, common-seed re-scoring, CBF-slope bootstrap."""

import numpy as np
import pandas as pd
import pytest

from src.data.inferred_prior import attach_inferred_prior, bin_prior, block_hazard, inferred_prior, stimulus_side
from src.models.simulator import simulate_ddm
from src.optimization import c1
from src.optimization.empirical import rescore_fits, simulate_from_fit
from src.optimization.fit import DDMFitter, fit_ddm, score_ddm
from src.optimization.qmp import Lapse, QMPDataset

FAST = dict(n_screen=2, n_starts=1, n_restarts=0, maxiter=10, n_sim_per_cond=80, n_sim_final=150)


# ------------------------------------------------------------------------------------------ inferred prior

def test_block_hazard_matches_the_truncated_exponential_design():
    h = block_hazard()
    assert len(h) == 100 and h[-1] == 1.0
    assert np.all(h[:19] == 0.0) and np.all(h[19:] > 0) and np.all(h <= 1)
    surv = np.cumprod(1 - h)                       # P(block lasts more than r trials)
    p_len = -np.diff(np.r_[1.0, surv])
    assert p_len.sum() == pytest.approx(1.0)
    assert 45 < (np.arange(1, 101) * p_len).sum() < 55     # empirical mean block length in the cohort is 51


def _session(sides_by_block, eid="s1", start=90, drop=()):
    """Extended-style rows: blocks of stimulus sides, with optional missing (unobserved) raw trials."""
    rows, t = [], start
    for side, n in sides_by_block:
        for _ in range(n):
            if t not in drop:
                rows.append({"session_eid": eid, "session_trial": t, "contrast": 0.25 * side, "choice": side, "feedback": 1.0})
            t += 1
    return pd.DataFrame(rows)


def test_observer_adapts_after_a_switch_and_uses_only_past_trials():
    ext = _session([(1, 60), (-1, 40)])
    p = inferred_prior(ext).to_numpy()
    assert p[0] == pytest.approx(0.0)                 # nothing seen yet
    assert p[59] > 0.8                                # deep into a right block
    assert p[60] > 0.5                                # first trial of the new block: still the OLD belief
    assert p[75] < -0.7                               # adapted after ~15 left stimuli
    assert np.all(np.abs(p) < 1)


def test_unobserved_trials_propagate_the_belief_without_updating():
    full_ext = _session([(1, 50)])
    gap_ext = _session([(1, 50)], drop=set(range(91, 111)))          # only trial 90 seen before trial 111
    full = pd.Series(inferred_prior(full_ext).to_numpy(), index=full_ext["session_trial"])
    gap = pd.Series(inferred_prior(gap_ext).to_numpy(), index=gap_ext["session_trial"])
    assert gap.loc[111] < full.loc[111] - 0.3                        # fewer observations -> much less confident
    assert gap.loc[111] > 0                                          # but the one right stimulus still counts


def test_zero_contrast_side_comes_from_feedback():
    df = pd.DataFrame({"contrast": [0.0, 0.0, -0.5], "choice": [1, 1, 1], "feedback": [1.0, -1.0, -1.0]})
    assert stimulus_side(df).tolist() == [1, -1, -1]


def test_bin_prior_and_attach_alignment():
    vals = pd.Series([-0.9, -0.2, 0.3, 0.95, -0.8, 0.6])
    b = bin_prior(vals, pd.Series(["a"] * 6))
    assert b.nunique() == 4 and b.iloc[0] == b.iloc[4] == pytest.approx(-0.85)
    ext = _session([(1, 30), (-1, 30)])
    ext["subject_id"], ext["prior"], ext["prev_choice"], ext["rt"] = "m", 1.0, 0, 0.5
    ext["rt_in_2p5"] = True
    ext.loc[5, "rt_in_2p5"] = False
    cohort = ext[ext["rt_in_2p5"]][["subject_id", "contrast", "prior", "prev_choice", "choice", "rt"]].reset_index(drop=True)
    out = attach_inferred_prior(cohort, ext)
    assert len(out) == len(cohort) and out["prior_model"].notna().all()
    with pytest.raises(ValueError):
        attach_inferred_prior(cohort.iloc[1:], ext)


# ------------------------------------------------------------------------------------------ lapse mixture

def test_lapse_mixture_keeps_probabilities_normalised():
    edges = np.array([0.0, 0.3, 0.5, 0.8, np.inf])
    lp = Lapse(0.2, np.array([0.7]), (0.1, 2.5))
    ddm = {1: np.array([0.1, 0.2, 0.1, 0.1]), -1: np.array([0.2, 0.1, 0.1, 0.1])}     # sums to 1 over both choices
    total = sum(lp.mix(ddm[c], 0, c, edges).sum() for c in (-1, 1))
    assert total == pytest.approx(1.0)
    assert np.allclose(Lapse(0.0, np.array([0.5]), (0.1, 2.5)).mix(ddm[1], 0, 1, edges), ddm[1])
    with pytest.raises(ValueError):
        Lapse(0.1, np.array([0.5]), (1.0, 1.0))


def _data(n=1500, seed=3, lapse=0.0):
    df = simulate_ddm({"a": 1.0, "t0": 0.2, "k": 3.0, "v_b": 0.5, "v_h": 0.2}, "M2", n_trials=n, seed=seed)
    if lapse:
        rng = np.random.default_rng(seed)
        hit = rng.random(len(df)) < lapse
        df.loc[hit, "choice"] = np.where(rng.random(hit.sum()) < 0.5 + 0.3 * df.loc[hit, "prior"], 1, -1).astype(np.int8)
        df.loc[hit, "rt"] = rng.uniform(0.1, 2.5, hit.sum())
    return df


def test_fit_lapse_adds_three_parameters_and_finds_contamination():
    df = _data(lapse=0.15)
    plain = fit_ddm(df, "M2", seed=1, **FAST)
    lap = fit_ddm(df, "M2", seed=1, fit_lapse=True, **FAST)
    assert {"lapse", "lapse_side", "lapse_prior"} <= set(lap["params"])
    assert "lapse" not in plain["params"]
    n_plain = DDMFitter("M2").n_params
    assert DDMFitter("M2", fit_lapse=True).n_params == n_plain + 3
    assert 0.0 <= lap["params"]["lapse"] <= 0.3


def test_score_reproduces_the_likelihood_and_uses_common_random_numbers():
    df = _data()
    res = fit_ddm(df, "M2", seed=5, **FAST)
    p = res["params"]
    s1 = score_ddm(df, p, "M2", n_sim_per_cond=300, seed=9)
    s2 = score_ddm(df, p, "M2", n_sim_per_cond=300, seed=9)
    s3 = score_ddm(df, p, "M2", n_sim_per_cond=300, seed=10)
    assert s1 == s2 and s1["nll"] != s3["nll"]
    k = DDMFitter("M2").n_params
    assert s1["bic"] == pytest.approx(2 * s1["nll"] + k * np.log(len(df)))
    m3 = score_ddm(df, {**p, "b_max": 0.3, "tau_b": 0.35}, "M3", n_sim_per_cond=300, seed=9)
    assert m3["bic"] - 2 * m3["nll"] == pytest.approx((DDMFitter("M3").n_params) * np.log(len(df)))


def test_rescore_fits_replaces_bic_and_keeps_the_originals(tmp_path):
    df = _data().assign(subject_id="m1")
    res = fit_ddm(df, "M2", seed=5, **FAST)
    row = {"subject_id": "m1", "model": "M2", "bound": "fixed", "rt_max": 2.5, "seed": 5, "n_trials": len(df),
           "n_params": 7, "nll": res["nll"], "bic": res["bic"], "aic": res["aic"], "success": True,
           **{f"p_{k}": v for k, v in res["params"].items()}}
    cache = str(tmp_path / "r.csv")
    out = rescore_fits(df, pd.DataFrame([row]), n_sim_per_cond=200, n_jobs=1, cache_path=cache,
                       fit_beta=False, fit_variability=False, bias_type=None)
    assert out.loc[0, "bic_fit"] == pytest.approx(res["bic"]) and out.loc[0, "bic"] != res["bic"]
    again = rescore_fits(df, pd.DataFrame([row]), n_sim_per_cond=200, n_jobs=1, cache_path=cache)
    assert len(again) == 1 and again.loc[0, "bic"] == out.loc[0, "bic"]       # cached, not recomputed


# ------------------------------------------------------------------------------------------ predictive checks

def test_simulate_from_fit_adds_lapses_and_restores_the_true_prior():
    df = _data(n=800)
    df["prior_model"] = 0.5 * df["prior"]
    p = {"a": 1.0, "t0": 0.2, "k": 3.0, "v_b": 0.5, "lapse": 0.3, "lapse_side": 0.0, "lapse_prior": 3.0}
    sim = simulate_from_fit(df, p, "M2", n_rep=5, prior_col="prior_model")
    assert set(np.unique(sim["prior"])) <= {-1.0, 1.0}
    assert sim["replay"].nunique() == 5
    no_lapse = simulate_from_fit(df, {**p, "lapse": 0.0}, "M2", n_rep=5)
    # lapse RTs are uniform up to 2.5 s, so the slow tail gets heavier
    assert (sim["rt"] > 1.5).mean() > (no_lapse["rt"] > 1.5).mean() + 0.05


def test_cbf_slope_bootstrap_and_lr():
    rows, parts = [], []
    for i, s in enumerate(["m1", "m2", "m3"]):
        d = simulate_ddm({"a": 1.0, "t0": 0.2, "k": 3.0, "v_b": 0.6}, "M2", n_trials=4000, seed=i).assign(subject_id=s)
        parts.append(d)
        for m, extra in (("M2", {"p_v_b": 0.6}), ("M3", {"p_b_max": 1.0, "p_tau_b": 0.35})):
            rows.append({"subject_id": s, "model": m, "bound": "fixed", "rt_max": 2.5, "p_a": 1.0, "p_t0": 0.2, "p_k": 3.0, **extra})
    cohort, best = pd.concat(parts, ignore_index=True), pd.DataFrame(rows)
    boot = c1.cbf_slope_bootstrap(cohort, best, n_boot=20)
    assert len(boot) == 6 and boot["slope_null_sd"].gt(0).all() and boot["p_two_sided"].between(0, 1).all()
    lr, summary = c1.cbf_slope_lr(boot)
    assert len(lr) == 3 and summary["n_mice"] == 3
    # data were generated by M2, so its slope distribution should usually fit better than M3's (rising)
    assert summary["sum_loglr"] > 0


def test_prior_covariate_comparison_prefers_the_covariate_that_generated_choices():
    rng = np.random.default_rng(0)
    n = 6000
    true = np.repeat(np.tile([1.0, -1.0], 60), 50)[:n]
    lagged = np.r_[np.zeros(5), true[:-5]]                 # belief lags the block by 5 trials
    c = rng.choice([-0.25, -0.0625, 0.0, 0.0625, 0.25], n)
    choice = np.where(rng.random(n) < 1 / (1 + np.exp(-(6 * c + 1.2 * lagged))), 1, -1)
    df = pd.DataFrame({"subject_id": "m", "contrast": c, "prior": true, "prior_inferred": lagged,
                       "prev_choice": 0, "choice": choice})
    out = c1.prior_covariate_comparison(df)
    assert out.loc[0, "dloglik_prior_inferred_vs_prior"] > 0


# ------------------------------------------------------------------------------------------ fast-guess component

def test_fast_guess_mixture_is_normalised_and_concentrated_early():
    edges = np.array([0.0, 0.15, 0.3, 0.6, 1.0, np.inf])
    ddm = {1: np.array([0.05, 0.15, 0.15, 0.1, 0.05]), -1: np.array([0.05, 0.15, 0.15, 0.1, 0.05])}
    mix = Lapse(0.1, np.array([0.6]), (0.1, 2.5), extra=[(0.2, np.array([0.9]), (0.1, 0.3))])
    out = {c: mix.mix(ddm[c], 0, c, edges) for c in (-1, 1)}
    assert sum(v.sum() for v in out.values()) == pytest.approx(1.0)
    assert out[1][0] - 0.7 * ddm[1][0] > out[1][3] - 0.7 * ddm[1][3]     # the guess mass sits in the fast bins
    with pytest.raises(ValueError):
        Lapse(0.6, np.array([0.5]), (0.1, 2.5), extra=[(0.6, np.array([0.5]), (0.1, 0.3))])


def test_fit_guess_adds_parameters_and_requires_the_lapse():
    assert DDMFitter("M2", fit_lapse=True, fit_guess=True).n_params == DDMFitter("M2", fit_lapse=True).n_params + 3
    with pytest.raises(ValueError):
        DDMFitter("M2", fit_guess=True)
    df = _data(n=800)
    res = fit_ddm(df, "M2", seed=1, fit_lapse=True, fit_guess=True, **FAST)
    assert {"guess", "guess_width", "guess_prior"} <= set(res["params"])


def test_simulate_from_fit_fast_guesses_follow_the_prior():
    df = _data(n=1000)
    p = {"a": 1.2, "t0": 0.3, "k": 3.0, "v_b": 0.0, "lapse": 0.0, "guess": 0.4, "guess_width": 0.1, "guess_prior": 4.0}
    sim = simulate_from_fit(df, p, "M2", n_rep=5)
    fast = sim[sim["rt"] < 0.2]
    assert len(fast) > 0.3 * len(sim)                       # t0 = 0.3 s: everything below 0.2 s is a guess
    agree = (np.sign(fast["choice"]) == np.sign(fast["prior_model"])).mean()
    assert agree > 0.9                                      # sigmoid(4) ~ 0.98


# ------------------------------------------------------------------------------------------ GPU backend

def test_backend_selection_and_errors(monkeypatch):
    from src.models.simulator import DDMSimulator
    with pytest.raises(ValueError):
        DDMSimulator(backend="tpu")
    monkeypatch.setenv("INCM_BACKEND", "numpy")
    assert DDMSimulator().backend == "numpy"


def test_cuda_backend_matches_cpu_distribution():
    from src.models._kernel_cuda import HAVE_CUDA
    if not HAVE_CUDA:
        pytest.skip("no CUDA GPU / numba-cuda")
    from src.models.simulator import DDMSimulator
    rng = np.random.default_rng(0)
    n = 20000
    c, pr = rng.choice([-0.25, -0.0625, 0, 0.0625, 0.25], n), rng.choice([-0.9, 0.9], n)
    p = dict(a=1.1, t0=0.2, k=3.0, beta=0.4, v_b=0.4, s_v=0.8, s_z=0.6)
    a = DDMSimulator(backend="numba").simulate(p, c, pr, rng=np.random.default_rng(1))
    b = DDMSimulator(backend="cuda").simulate(p, c, pr, rng=np.random.default_rng(1))
    assert abs((a.choice > 0).mean() - (b.choice > 0).mean()) < 0.01
    assert np.allclose(np.quantile(a.rt, [0.1, 0.5, 0.9]), np.quantile(b.rt, [0.1, 0.5, 0.9]), atol=0.01)
    b2 = DDMSimulator(backend="cuda").simulate(p, c, pr, rng=np.random.default_rng(1))
    assert np.array_equal(b.rt.to_numpy(), b2.rt.to_numpy())                  # deterministic -> CRN holds on the GPU


# ------------------------------------------------------------------------------------------ replication cohort

def test_select_replication_skips_the_cohort_and_caps_labs():
    from run_track_b import select_replication
    mice = pd.DataFrame({"subject": list("abcdefgh"), "lab": ["x", "x", "x", "y", "y", "y", "z", "x"]})
    picked = select_replication(mice, frozen=["a", "d"], n_mice=4, max_per_lab=2)
    assert picked == ["b", "c", "e", "f"]
    assert select_replication(mice, frozen=["a", "d"], n_mice=10, max_per_lab=1) == ["b", "e", "g"]
