"""Tests for the C1 analysis (BIC tables, across-mouse statistics, model-vs-data CBF)."""

import numpy as np
import pandas as pd
import pytest

from src.empirical.mock import make_mock_cohort
from src.optimization import c1
from src.optimization.empirical import fit_cohort
from src.models.simulator import simulate_ddm

MICE = [f"m{i}" for i in range(10)]


def _fits(offsets, seeds=(1, 2), jitter=5.0, base=1000.0):
    """Synthetic fit table: BIC = base + model offset + mouse effect + replicate noise."""
    rng = np.random.default_rng(0)
    rows = []
    for i, s in enumerate(MICE):
        for m, off in offsets.items():
            for sd in seeds:
                rows.append({"subject_id": s, "model": m, "bound": "fixed", "rt_max": 2.5, "seed": sd,
                             "n_trials": 1000, "n_params": 9 + (m == "M3"), "nll": 0.0, "aic": 0.0, "success": True,
                             "bic": base + 30 * i + off + rng.normal(0, jitter), "p_a": 1.0})
    return pd.DataFrame(rows)


def test_collapse_replicates_keeps_lowest_bic_and_records_spread():
    f = _fits({"M1": 0, "M2": 0, "M3": 0})
    best = c1.collapse_replicates(f)
    assert len(best) == 30 and (best["n_reps"] == 2).all()
    g = f.groupby(c1.KEYS)["bic"]
    got = best.set_index(c1.KEYS)
    assert np.allclose(got["bic"], g.min()) and np.allclose(got["bic_spread"], g.max() - g.min())


def test_noise_floor_recovers_the_replicate_sd():
    f = _fits({"M1": 0, "M2": 0, "M3": 0}, seeds=(1, 2, 3, 4), jitter=20.0)
    assert 15 < c1.noise_floor(f)["sd"] < 26
    assert np.isnan(c1.noise_floor(_fits({"M1": 0}, seeds=(1,)))["sd"])
    assert c1.decisive_threshold(float("nan")) == 10.0
    assert c1.decisive_threshold(2.0) == 10.0 and c1.decisive_threshold(50.0) == pytest.approx(140.0)


def test_bic_table_sign_convention_and_winner():
    best = c1.collapse_replicates(_fits({"M1": 200, "M2": 0, "M3": -300}, jitter=1.0))
    t = c1.bic_table(best, threshold=10.0)
    assert (t["winner"] == "M3").all() and (t["evidence"] == "decisive").all()
    assert (t["dbic_M3_vs_M2"] < -250).all() and (t["dbic_M1_vs_M2"] > 150).all()
    assert np.allclose(t["margin"], 300, atol=15)
    close = c1.bic_table(c1.collapse_replicates(_fits({"M1": 3, "M2": 0, "M3": 5}, jitter=0.5)), threshold=10.0)
    assert (close["evidence"] == "inconclusive").all() and (close["winner_decisive"] == "none").all()
    w = c1.bic_weights(t)
    assert np.allclose(w[["w_M1", "w_M2", "w_M3"]].sum(axis=1), 1.0) and (w["w_M3"] > 0.99).all()


def test_pairwise_tests_detect_a_consistent_effect_and_a_null():
    eff = c1.bic_table(c1.collapse_replicates(_fits({"M1": 0, "M2": 100, "M3": 100}, jitter=3.0)))
    tt = c1.pairwise_tests(eff).set_index("pair")
    assert tt.loc["M1 vs M2", "mean_delta"] < -90 and tt.loc["M1 vs M2", "n_first_better"] == 10
    assert tt.loc["M1 vs M2", "wilcoxon_p"] < 0.01 and tt.loc["M1 vs M2", "sign_p"] < 0.01
    assert tt.loc["M1 vs M2", "ci_hi"] < 0
    assert tt.loc["M1 vs M2", "wilcoxon_p_holm"] >= tt.loc["M1 vs M2", "wilcoxon_p"]
    null = c1.bic_table(c1.collapse_replicates(_fits({"M1": 0, "M2": 0, "M3": 0}, jitter=20.0)))
    assert (c1.pairwise_tests(null)["wilcoxon_p"] > 0.01).all()


def test_holm_adjustment():
    assert np.allclose(c1._holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert (c1._holm([0.5, 0.9]) <= 1).all()


def test_subset_summary_excludes_named_mice():
    t = c1.bic_table(c1.collapse_replicates(_fits({"M1": 0, "M2": 50, "M3": 80})))
    s = c1.subset_summary(t, {"all": None, "drop two": ("m0", "m1")})
    assert set(s["subset"]) == {"all", "drop two"}
    assert set(s.loc[s["subset"] == "drop two", "n_mice"]) == {8}


def test_predicted_cbf_matches_the_empirical_bins_and_fit_table():
    cohort = make_mock_cohort(n_subjects=3, n_trials=2500, seed=1)
    subj = sorted(cohort["subject_id"].unique())
    params = {"a": 1.2, "t0": 0.2, "k": 3.0, "beta": 0.6, "v_h": 0.1, "v_b": 0.5, "z": 0.2, "b_max": 0.6, "tau_b": 0.35}
    rows = [{"subject_id": s, "model": m, "bound": "fixed", "rt_max": 2.5, "bic": 1.0, "n_reps": 1,
             **{f"p_{k}": v for k, v in params.items() if not (k == "z" and m != "M1") and not (k == "v_b" and m != "M2")
                and not (k in ("b_max", "tau_b") and m != "M3")}}
            for s in subj for m in c1.MODELS]
    best = pd.DataFrame(rows)
    emp = c1.empirical_cbf(cohort)
    pred = c1.predicted_cbf(cohort, best, n_rep=3, emp=emp)
    assert len(pred) == 3 * 3 * 5 and set(pred["model"]) == set(c1.MODELS)
    ft = c1.cbf_fit_table(emp, pred)
    assert {"rmse", "chi2_per_bin", "slope_pred", "slope_emp", "slope_err"} <= set(ft.columns) and len(ft) == 9
    assert np.isfinite(ft["rmse"]).all()
    tests = c1.cbf_model_tests(ft)
    assert len(tests) == 3 and "wilcoxon_p_holm" in tests


def test_the_cbf_shape_signatures_are_ordered_M1_below_M2_below_M3():
    """Sanity check on the overlay logic: simulated M1/M2/M3 data give falling / flat / rising slopes."""
    from src.empirical.cbf import cbf_slope, compute_cbf
    slopes = {}
    for m, p in (("M1", {"z": 0.3}), ("M2", {"v_b": 0.7}), ("M3", {"b_max": 1.2, "tau_b": 0.5})):
        df = simulate_ddm({"a": 1.2, "t0": 0.2, "k": 3.0, **p}, m, n_trials=8000, seed=2)
        slopes[m] = cbf_slope(compute_cbf(df)["cbf"])
    assert slopes["M1"] < slopes["M2"] < slopes["M3"]


def test_at_bound_flags_parameters_on_their_limits():
    best = pd.DataFrame([{"subject_id": "x", "model": "M2", "p_a": 0.4, "p_t0": 0.2, "p_s_v": 1.0, "p_tau_b": 0.15}])
    out = c1.at_bound(best)
    assert list(out["param"]) == ["a"] and out["bound"].iloc[0] == "lower"


def test_concordance_and_parameter_table():
    cohort = make_mock_cohort(n_subjects=2, n_trials=1500, seed=3)
    tab = pd.DataFrame({"subject_id": sorted(cohort["subject_id"].unique()), "winner": ["M1", "M2"],
                        "evidence": ["decisive", "inconclusive"]})
    shapes = pd.DataFrame({"subject_id": tab["subject_id"], "shape": ["falling", "rising"], "slope": [-0.05, 0.04]})
    ft = pd.DataFrame({"subject_id": np.repeat(tab["subject_id"], 2), "model": ["M1", "M2"] * 2, "rmse": [0.1, 0.2, 0.3, 0.1]})
    conc = c1.concordance(tab, shapes, ft)
    assert list(conc["shape_agrees"]) == [True, False] and list(conc["cbf_closest"]) == ["M1", "M2"]


def test_fit_cohort_replicate_seeds_are_independent_rows(tmp_path):
    a = simulate_ddm({"a": 1.2, "t0": 0.2, "k": 3.0, "v_b": 0.4}, "M2", n_trials=400, seed=1, subject_id="mA")
    tiny = dict(n_screen=2, n_starts=1, n_restarts=0, maxiter=10, n_sim_per_cond=40, n_sim_final=80)
    cache = str(tmp_path / "f.csv")
    r = fit_cohort(a, models=("M2",), n_jobs=1, cache_path=cache, seeds=(1, 2), **tiny)
    assert sorted(r["seed"]) == [1, 2]
    r2 = fit_cohort(a, models=("M2",), n_jobs=1, cache_path=cache, seeds=(1, 2, 3), **tiny)
    assert sorted(r2["seed"]) == [1, 2, 3]            # only seed 3 was added
    assert len(c1.collapse_replicates(r2)) == 1
