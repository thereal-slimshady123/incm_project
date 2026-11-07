"""Tests for the C2 pipeline: truncation-aware likelihood, warm-started Weibull fits, RT-tail tools, C2 tables and statistics."""

import numpy as np
import pandas as pd
import pytest

from src.empirical import tails
from src.empirical.mock import make_mock_cohort
from src.models.simulator import DDMSimulator, simulate_ddm
from src.optimization import c1, c2
from src.optimization.empirical import (
    COLLAPSE_BOUNDS,
    COLLAPSE_STARTS,
    _warm_starts,
    fit_cohort,
    fit_mouse,
)
from src.optimization.fit import PARAM_SPECS, DDMFitter, fit_ddm
from src.optimization.qmp import QMPDataset

TINY = dict(n_screen=2, n_starts=1, n_restarts=0, maxiter=15, n_sim_per_cond=60, n_sim_final=120)
MICE = [f"m{i}" for i in range(10)]


# --------------------------------------------------------------------------------------------
# truncation-aware likelihood
# --------------------------------------------------------------------------------------------

def _truncated_data(rt_hi=2.5, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.choice([-0.25, -0.0625, 0.0, 0.0625, 0.25], n)
    prior = rng.choice([-1.0, 1.0], n)
    truth = {"a": 1.0, "t0": 0.2, "k": 2.0, "v_b": 0.5, "beta": 1.0, "s_v": 0.5, "s_z": 0.1}
    sim = DDMSimulator(dt=0.002, t_max=8.0).simulate(truth, c, prior, model_type="M2", rng=rng)
    return truth, sim, sim[(sim["rt"] >= 0.1) & (sim["rt"] <= rt_hi)].reset_index(drop=True)


def test_windowed_dataset_uses_window_edges_and_ignores_outside_trials():
    _, sim, d = _truncated_data()
    assert (sim["rt"] > 2.5).mean() > 0.02                      # the generating process really has a tail beyond the cutoff
    q = QMPDataset(sim, group_cols=("contrast", "prior"), window=(0.1, 2.5))
    assert q.total_trials == len(d)                              # trials outside the window are not counted
    edges = q.conditions[0]["choices"][1]["bin_edges"]
    assert edges[0] == 0.1 and edges[-1] == 2.5 and np.all(np.diff(edges) > 0) and np.isfinite(edges).all()
    plain = QMPDataset(d, group_cols=("contrast", "prior"))
    assert plain.conditions[0]["choices"][1]["bin_edges"][0] == 0.0 and np.isinf(plain.conditions[0]["choices"][1]["bin_edges"][-1])
    with pytest.raises(ValueError):
        QMPDataset(d, window=(2.5, 0.1))


@pytest.mark.parametrize("bandwidth", [0.01, None])
def test_windowed_nll_is_lowest_at_the_true_parameters(bandwidth):
    truth, _, d = _truncated_data()
    f = DDMFitter("M2", fit_history=False, fit_beta=True, fit_variability=True, t_max=2.5, rt_window=(0.1, 2.5),
                  n_sim_per_cond=2500, rt_bandwidth=bandwidth)
    q = QMPDataset(d, group_cols=("contrast", "prior"), window=(0.1, 2.5))
    tpl = f._simulation_template(q, 2500)
    nll = {}
    for scale in (0.7, 1.0, 1.4):
        x = np.array([{**truth, "a": truth["a"] * scale}.get(n, 0.0) for n in f.param_names])
        nll[scale] = f._qmp_nll(x, q, tpl, 2500, 5)
    assert nll[1.0] < nll[0.7] and nll[1.0] < nll[1.4]


def test_windowed_probabilities_are_conditional_on_the_window():
    """A model that is exactly right but also produces slow trials is scored on its in-window part only."""
    from src.optimization.qmp import compute_qmp_nll_blocks
    truth, sim, d = _truncated_data(n=3000)
    q = QMPDataset(d, group_cols=("contrast", "prior"), window=(0.1, 2.5))
    n = 500
    # fabricate simulated blocks that REPRODUCE the observed in-window distribution, then add slow trials
    # (rt > 2.5, in the last position of each block): the windowed NLL must not change when they are added
    rng = np.random.default_rng(1)
    ch, rt = [], []
    for cond in q.conditions:
        pool_c = []
        pool_r = []
        for cv in (-1, 1):
            info = cond["choices"][cv]
            if info["n_choice"]:
                e = info["bin_edges"]
                for k, cnt in enumerate(info["observed_counts"]):
                    pool_c += [cv] * int(cnt)
                    pool_r += list(rng.uniform(e[k], e[k + 1], int(cnt)))
        idx = rng.integers(0, len(pool_c), n)
        ch.append(np.array(pool_c)[idx]), rt.append(np.array(pool_r)[idx])
    ch, rt = np.concatenate(ch), np.concatenate(rt)
    base = compute_qmp_nll_blocks(ch, rt, q, n, bandwidth=None)
    # replace 20% of every block by slow trials
    ch2, rt2 = ch.copy(), rt.copy()
    for i in range(len(q.conditions)):
        sl = slice(i * n, i * n + n // 5)
        rt2[sl] = 4.0
    with_slow = compute_qmp_nll_blocks(ch2, rt2, q, n, bandwidth=None)
    # the remaining 80% is still a sample of the same distribution; the two scores differ only by sampling noise
    assert abs(with_slow - base) / q.total_trials < 0.08


def test_default_likelihood_is_unchanged_by_the_window_option():
    sim = simulate_ddm({"a": 1.2, "t0": 0.2, "k": 3.0, "v_b": 0.4}, "M2", n_trials=600, seed=3)
    a = fit_ddm(sim, "M2", **TINY)
    b = fit_ddm(sim, "M2", rt_window=None, **TINY)
    assert a["nll"] == b["nll"] and a["params"] == b["params"]
    w = fit_ddm(sim, "M2", rt_window=(0.1, 2.5), **TINY)
    assert np.isfinite(w["bic"]) and w["nll"] != a["nll"]


def test_fit_mouse_truncate_counts_only_trials_inside_the_window():
    sim = simulate_ddm({"a": 1.2, "t0": 0.2, "k": 3.0, "v_b": 0.4}, "M2", n_trials=700, seed=4)
    n_in = int(((sim["rt"] >= 0.1) & (sim["rt"] <= 1.0)).sum())
    res = fit_mouse(sim, "M2", "fixed", 1.0, truncate=True, **TINY)
    assert res["n_trials"] == n_in and res["n_params"] == len(res["params"])


# --------------------------------------------------------------------------------------------
# Weibull bounds: nesting and warm starts
# --------------------------------------------------------------------------------------------

def test_window_lower_edge_matches_the_preprocessing_cutoff():
    from src.data.preprocess import RT_MAX_EXTENDED, RT_MAX_STANDARD, RT_MIN
    from src.optimization.empirical import RT_MIN as FIT_RT_MIN
    import run_c2
    assert FIT_RT_MIN == RT_MIN                                     # duplicated constant: the two must not drift apart
    assert {v["rt_max"] for v in run_c2.VARIANTS.values()} <= {RT_MAX_STANDARD, RT_MAX_EXTENDED}


def test_default_lam_range_cannot_reach_a_fixed_bound_but_the_c2_range_can():
    lo_hi = PARAM_SPECS["lam"]["bounds"]
    gamma_hi = PARAM_SPECS["gamma"]["bounds"][1]
    assert c2.bound_fraction(5.0, lo_hi[1], gamma_hi) < 0.1             # default: collapses to the floor within 5 s
    assert c2.bound_fraction(2.5, lo_hi[1], gamma_hi) < 0.8
    assert c2.bound_fraction(5.0, COLLAPSE_BOUNDS["lam"][1], gamma_hi) > 0.99     # C2 range: fixed bound is the limit
    assert c2.bound_fraction(0.0, 1.0, 2.0) == 1.0


def test_weibull_fits_use_the_wide_lam_range_and_accept_overrides():
    sim = simulate_ddm({"a": 1.2, "t0": 0.2, "k": 3.0, "v_b": 0.4}, "M2", n_trials=500, seed=5)
    res = fit_mouse(sim, "M2", "weibull", 2.5, **TINY)
    assert 0.4 <= res["params"]["lam"] <= COLLAPSE_BOUNDS["lam"][1] and "gamma" in res["params"]
    f = DDMFitter("M2", bound_type="weibull", param_bounds=COLLAPSE_BOUNDS)
    assert dict(zip(f.param_names, f._bounds().tolist()))["lam"] == [0.4, 20.0]


def test_fit_ddm_accepts_several_starting_guesses():
    sim = simulate_ddm({"a": 1.2, "t0": 0.2, "k": 3.0, "v_b": 0.4}, "M2", n_trials=500, seed=6)
    guesses = [{"a": 1.1, "lam": 20.0, "gamma": 4.0}, {"a": 1.1, "lam": 2.0, "gamma": 2.0}]
    res = fit_ddm(sim, "M2", bound_type="weibull", init_params=guesses, **TINY)
    assert np.isfinite(res["bic"])


def test_warm_starts_come_from_the_matching_fixed_fit(tmp_path):
    a = simulate_ddm({"a": 1.2, "t0": 0.2, "k": 3.0, "v_b": 0.4}, "M2", n_trials=500, seed=7, subject_id="mA")
    cache = str(tmp_path / "f.csv")
    fixed = fit_cohort(a, models=("M2",), bounds=("fixed",), n_jobs=1, cache_path=cache, seeds=(1, 2), **TINY)
    starts = _warm_starts(fixed, COLLAPSE_STARTS, 2.5)
    assert set(starts) == {("mA", "M2", 1), ("mA", "M2", 2)}
    g = starts[("mA", "M2", 1)]
    assert len(g) == len(COLLAPSE_STARTS) and g[0]["lam"] == 20.0 and g[1]["lam"] == 2.5
    assert g[0]["a"] == pytest.approx(fixed[fixed["seed"] == 1]["p_a"].iloc[0]) and "v_b" in g[0]
    assert _warm_starts(fixed, COLLAPSE_STARTS, 5.0) == {}                      # other cutoff: no reference
    wb = fit_cohort(a, models=("M2",), bounds=("weibull",), n_jobs=1, cache_path=cache, seeds=(1, 2),
                    init_from=fixed, init_extra=COLLAPSE_STARTS, truncate=True, **TINY)
    assert sorted(wb["bound"].unique()) == ["fixed", "weibull"] and len(wb) == 4
    assert wb[wb["bound"] == "weibull"]["truncated"].all()


# --------------------------------------------------------------------------------------------
# RT-tail tools
# --------------------------------------------------------------------------------------------

def _cohort_like(rt, n=None, seed=0):
    rng = np.random.default_rng(seed)
    n = len(rt)
    return pd.DataFrame({"subject_id": "s", "trial_idx": np.arange(n), "contrast": rng.choice([-1.0, -0.25, 0.25, 1.0], n),
                         "prior": 1.0, "prev_choice": 0, "choice": rng.choice([-1, 1], n), "rt": rt})


def test_survival_and_hazard_of_an_exponential_tail():
    rng = np.random.default_rng(0)
    rt = 0.1 + rng.exponential(1.0, 200_000)
    df = _cohort_like(rt)
    s = tails.survival_table(df, [0.6, 1.1, 2.1]).set_index("t")["survival"]
    assert s[1.1] == pytest.approx(np.exp(-1.0), abs=0.01) and s[2.1] == pytest.approx(np.exp(-2.0), abs=0.01)
    h = tails.hazard_table(df, [0.5, 1.0, 2.0, 3.0])
    assert np.allclose(h["hazard"], 1.0, atol=0.06)                               # constant hazard = rate 1/s
    r = tails.late_hazard_ratio(df).iloc[0]
    assert r["hazard_ratio"] == pytest.approx(1.0, abs=0.1)
    # survival_distance is zero for identical tables and positive otherwise
    sv = tails.survival_table(df)
    assert tails.survival_distance(sv, sv) == 0.0
    assert tails.survival_distance(sv, tails.survival_table(df.assign(rt=df["rt"] * 0.5))) > 0.01


def test_hazard_rises_under_a_collapsing_bound_and_falls_with_drift_variability():
    """Why the late/early hazard ratio is informative: collapse inflates it, a mixture of slow processes deflates it."""
    n = 40_000
    c, prior = np.zeros(n), np.ones(n)
    sim = DDMSimulator(dt=0.004, t_max=5.0)
    base = {"a": 1.1, "t0": 0.15, "k": 2.0}
    fixed = sim.simulate(base, c, prior, rng=np.random.default_rng(1))
    coll = sim.simulate({**base, "a": 1.3, "lam": 3.0, "gamma": 2.5}, c, prior, bound_type="weibull", rng=np.random.default_rng(1))
    var = sim.simulate({**base, "s_v": 1.5}, c, prior, rng=np.random.default_rng(1))
    r = lambda d: tails.late_hazard_ratio(d[(d["rt"] >= 0.1) & (d["rt"] <= 5.0)].assign(subject_id="s")).iloc[0]["hazard_ratio"]
    assert r(fixed) > 1.0                      # a fixed bound's hazard also rises (towards an asymptote): the ratio must be compared with the model's
    assert r(coll) > 2.5 * r(fixed)            # collapse: the hazard keeps growing
    assert r(var) < r(fixed)                   # drift variability: mixture of slow trials, hazard falls


def test_accuracy_by_rt_window_uses_fixed_windows_and_visible_stimuli():
    df = pd.DataFrame({"subject_id": "s", "contrast": [1.0] * 6 + [0.0625] * 2, "choice": [1, 1, -1, 1, -1, -1, 1, -1],
                       "rt": [0.2, 0.3, 0.7, 0.8, 3.0, 3.2, 0.2, 3.0], "prior": 1.0, "prev_choice": 0, "trial_idx": range(8)})
    a = tails.accuracy_by_rt_window(df, edges=(0.1, 0.5, 1.0, 5.0)).set_index("window")
    assert list(a["n"]) == [2, 2, 2] and list(a["accuracy"]) == [1.0, 0.5, 0.0]    # the |c| = 0.0625 trials are excluded
    summ = tails.tail_summary(_cohort_like(0.1 + np.random.default_rng(2).exponential(1.0, 5000)))
    assert {"hazard_ratio", "accuracy_gt2s", "accuracy_lt1s", "accuracy_drop", "p_gt_2.5"} <= set(summ.columns)
    with pytest.raises(ValueError):
        tails.tail_summary(df, by=())


# --------------------------------------------------------------------------------------------
# C2 tables and statistics (synthetic fit tables)
# --------------------------------------------------------------------------------------------

def _c2_fits(offsets, rt_max=2.5, seeds=(1, 2), jitter=3.0, lam=1.5, gamma=2.0):
    """offsets: {(model, bound): BIC offset}. BIC = 1000 + 30 mouse index + offset + noise."""
    rng = np.random.default_rng(0)
    rows = []
    for i, s in enumerate(MICE):
        for (m, b), off in offsets.items():
            for sd in seeds:
                bic = 1000.0 + 30 * i + off + rng.normal(0, jitter)
                rows.append({"subject_id": s, "model": m, "bound": b, "rt_max": rt_max, "seed": sd, "n_trials": 1000,
                             "n_params": 10 + (b == "weibull") * 2, "nll": bic / 2 - 30, "aic": bic, "bic": bic, "success": True,
                             "p_a": 1.2, "p_lam": lam if b == "weibull" else np.nan, "p_gamma": gamma if b == "weibull" else np.nan})
    return pd.DataFrame(rows)


def test_bound_table_sign_convention_ceiling_flag_and_tests():
    fits = c1.collapse_replicates(_c2_fits({("M2", "fixed"): 0, ("M2", "weibull"): -200}, jitter=1.0))
    t = c2.bound_table(fits, "M2", 2.5, threshold=10.0)
    assert (t["winner"] == "weibull").all() and (t["evidence"] == "decisive").all()
    assert (t["dbic_weibull_vs_fixed"] < -150).all() and (t["dnll_weibull_vs_fixed"] < 0).all()
    assert np.allclose(t["frac_t1"], np.exp(-(1.0 / 1.5) ** 2.0)) and not t["lam_at_ceiling"].any()
    tt = c2.bound_tests(t).iloc[0]
    assert tt["pair"] == "weibull vs fixed" and tt["mean_delta"] < -150 and tt["n_first_better"] == 10 and tt["wilcoxon_p"] < 0.01
    null = c2.bound_table(c1.collapse_replicates(_c2_fits({("M2", "fixed"): 0, ("M2", "weibull"): 0}, jitter=20.0, lam=25.0)), "M2", 2.5)
    assert (null["evidence"] == "inconclusive").all() or null["margin"].max() < 100
    assert null["lam_at_ceiling"].all()
    with pytest.raises(ValueError, match="fits"):
        c2.bound_table(c1.collapse_replicates(_c2_fits({("M2", "fixed"): 0})), "M2", 2.5)


def test_cutoff_consistency_counts_agreement():
    a = c2.bound_table(c1.collapse_replicates(_c2_fits({("M2", "fixed"): 0, ("M2", "weibull"): -50}, jitter=1.0)), "M2", 2.5)
    b = c2.bound_table(c1.collapse_replicates(_c2_fits({("M2", "fixed"): 0, ("M2", "weibull"): -80}, rt_max=5.0, jitter=1.0)), "M2", 5.0)
    out = c2.cutoff_consistency(a, b)
    assert out["n_mice"] == 10 and out["n_same_sign"] == 10 and out["mean_b"] < out["mean_a"] < 0


def _inv_fits(m1_fixed, m1_weib, m3_fixed=50.0, m3_weib=50.0):
    off = {("M1", "fixed"): m1_fixed, ("M2", "fixed"): 0, ("M3", "fixed"): m3_fixed,
           ("M1", "weibull"): m1_weib, ("M2", "weibull"): 0, ("M3", "weibull"): m3_weib}
    return c1.collapse_replicates(_c2_fits(off, jitter=2.0))


def test_invariance_detects_no_change_and_a_change():
    same = _inv_fits(300, 300)
    inv = c2.invariance_table(same, 2.5)
    tests = c2.invariance_tests(same, inv, 2.5)
    shift = tests[(tests["bound"] == "shift") & (tests["pair"] == "M1 vs M2")].iloc[0]
    assert abs(shift["mean_delta"]) < 10 and shift["wilcoxon_p"] > 0.05 and inv["same_winner"].all()
    assert abs(shift["rel_mean"]) < 0.05 and shift["rel_ci_lo"] < 0 < shift["rel_ci_hi"] and shift["rel_ci_hi"] < 0.1   # shift is a few % of the 300-BIC effect
    v_f = c2.c1_verdict(tests[tests["bound"] == "fixed"])
    v_w = c2.c1_verdict(tests[tests["bound"] == "weibull"])
    assert v_f == v_w and v_f["m1_rejected"] and v_f["m2_vs_m3"] == "M2"
    changed = _inv_fits(300, -300)
    inv2 = c2.invariance_table(changed, 2.5)
    t2 = c2.invariance_tests(changed, inv2, 2.5)
    shift2 = t2[(t2["bound"] == "shift") & (t2["pair"] == "M1 vs M2")].iloc[0]
    assert shift2["mean_delta"] < -500 and shift2["wilcoxon_p"] < 0.01 and not inv2["same_winner"].any()
    assert c2.c1_verdict(t2[t2["bound"] == "weibull"])["m1_rejected"] is False
    corr = c2.invariance_correlations(inv)
    assert len(corr) == 3 and (corr["same_sign"] == 10).all()
    with pytest.raises(ValueError, match="need"):
        c2.invariance_table(c1.collapse_replicates(_c2_fits({("M2", "fixed"): 0, ("M2", "weibull"): 0})), 2.5)


def test_c1_verdict_reports_the_cbf_closest_model():
    tests = c2.invariance_tests(_inv_fits(300, 300), c2.invariance_table(_inv_fits(300, 300), 2.5), 2.5)
    fit = pd.DataFrame({"subject_id": np.repeat(MICE, 3), "model": ["M1", "M2", "M3"] * 10, "rmse": [0.3, 0.1, 0.2] * 10})
    assert c2.c1_verdict(tests[tests["bound"] == "fixed"], fit)["cbf_closest"] == "M2"


def test_boundary_curves_and_tail_tables_on_a_mock_cohort():
    cohort = make_mock_cohort(n_subjects=3, n_trials=2500, seed=1)
    subj = sorted(cohort["subject_id"].unique())
    rows = []
    for s in subj:
        for b in c2.BOUNDS:
            rows.append({"subject_id": s, "model": "M2", "bound": b, "rt_max": 5.0, "bic": 1.0, "n_reps": 1,
                         "p_a": 1.2, "p_t0": 0.2, "p_k": 3.0, "p_beta": 0.6, "p_v_b": 0.5,
                         "p_lam": 2.0 if b == "weibull" else np.nan, "p_gamma": 2.0 if b == "weibull" else np.nan})
    best = pd.DataFrame(rows)
    curves = c2.boundary_curves(best, "M2", 5.0)
    assert set(curves["subject_id"]) == set(subj) and curves["fraction"].between(0, 1).all() and curves["fraction"].iloc[0] == 1.0
    tt = c2.tail_tables(cohort, best, "M2", 5.0, n_rep=2)
    assert set(tt["survival"]["source"]) == {"data", "fixed", "weibull"} and set(tt) == {"survival", "hazard", "accuracy", "summary"}
    errs = c2.tail_errors(tt)
    assert len(errs) == 6 and {"survival_error", "hazard_log_err", "accuracy_error"} <= set(errs.columns)
    assert (errs["survival_error"] >= 0).all()
    tests = c2.tail_error_tests(errs)
    assert "survival_error" in set(tests["metric"])
    assert len(c2.empirical_tail_tests(tt["summary"])) == 2
    signed = c2.tail_signed_tests(errs, "fixed")
    assert set(signed["metric"]) <= {"log_hr_diff", "accuracy_diff", "p_gt2p5_diff"} and (signed["n_mice"] == 3).all()
    row = errs[errs["bound"] == "fixed"].iloc[0]
    np.testing.assert_allclose(row["accuracy_diff"], row["accuracy_gt2s_data"] - row["accuracy_gt2s_model"], equal_nan=True)   # sign: data minus model
    np.testing.assert_allclose(row["p_gt2p5_diff"], row["p_gt2p5_data"] - row["p_gt2p5_model"], equal_nan=True)


# --------------------------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------------------------

def test_run_c2_smoke_end_to_end(tmp_path):
    """fit -> analyze on 3 mice with tiny settings: every table and figure family of the C2 pipeline is produced."""
    import run_c2
    run_c2.main(["smoke", "--out", str(tmp_path), "--n-jobs", "2", "--seeds", "1", "2"])
    out = tmp_path / "_smoke"
    for name in ("c2_report.md", "c2_trunc_2p5_bound_table_M2.csv", "c2_trunc_2p5_bound_tests.csv", "c2_trunc_5p0_bound_table_M2.csv",
                 "c2_c1like_2p5_invariance_table.csv", "c2_c1like_2p5_invariance_tests.csv", "c2_c1like_2p5_c1_verdict.csv",
                 "c2_trunc_5p0_tail_errors.csv", "c2_trunc_5p0_tail_data_tests.csv", "c2_trunc_5p0_tail_signed_tests.csv", "c2_variant_consistency.csv",
                 "c2_invariance_verdicts.csv", "c2_tail_5p0_tail_summary.csv"):
        assert (out / name).exists(), name
    for name in ("c2_bound_comparison_M2.png", "c2_trunc_5p0_tail_diagnostics.png", "c2_c1like_2p5_invariance.png",
                 "c2_trunc_2p5_fitted_bounds.png", "c2_c1like_2p5_cbf_overlay_weibull.png"):
        assert (out / "figures" / name).exists(), name
    fits = pd.read_csv(out / "c2_fits_trunc_2p5.csv")
    assert set(fits["bound"]) == {"fixed", "weibull"} and fits["truncated"].all()
    assert len(fits) == 3 * 2 * 2 * 2 * 3        # 3 mice x 2 bounds x (2 seeds + 2 polish replicates) x 3 models
    assert (fits["seed"] >= run_c2.POLISH_SEED_OFFSET).sum() == len(fits) // 2
    assert len(pd.read_csv(out / "c2_trunc_2p5_bound_table_M2.csv")) == 3
