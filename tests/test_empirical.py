"""
Track B mock-data test harness (plan §4, Week 2, task 4).

Verifies that the empirical psychometric / chronometric / CBF tools and the behavioural plots
(a) recover KNOWN ground truth from synthetic mock tables, (b) read ANY DataFrame following the
§3.1 schema (mock, Track A simulator output, real IBL cohort, reordered / extra-column /
categorical variants), and (c) reject tables that violate the schema.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import validate_contract
from src.empirical import (
    accuracy_by_rt_bin,
    across_subject,
    bootstrap_cbf,
    cbf_by_subject,
    cbf_curve,
    chronometric_table,
    classify_shape,
    cohort_cbf,
    cohort_slope_test,
    compute_cbf,
    fit_psychometric,
    fit_psychometric_by,
    heterogeneity_summary,
    make_mock_cohort,
    make_mock_trials,
    mouse_profile,
    prior_shift,
    profile_correlations,
    psychometric_function,
    psychometric_table,
    rt_quantile_table,
)
from src.models.simulator import simulate_ddm
from src.visualization import (
    plot_cbf,
    plot_cbf_by_mouse,
    plot_cbf_cohort,
    plot_cbf_sensitivity,
    plot_chronometric_by_mouse,
    plot_heterogeneity,
    plot_psychometric_by_mouse,
    plot_psychometric_chronometric_cohort,
    plot_rt_distributions,
)

REAL = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cohort_rt2p5.parquet")
REAL_EXT = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cohort_extended.parquet")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def mock_flat():
    return make_mock_trials(30000, cbf_shape="flat", seed=3)


@pytest.fixture(scope="module")
def mock_cohort():
    return make_mock_cohort(n_subjects=6, n_trials=8000, seed=11)


# ---------------------------------------------------------------------------
# Mock generator itself
# ---------------------------------------------------------------------------
def test_mock_table_is_contract_compliant():
    df = make_mock_trials(2000, seed=1)
    validate_contract(df)
    assert df["prior"].isin([-1.0, 1.0]).all()
    # IBL-style statistics: stimulus is on the block's likelier side ~80% of the time
    nz = df[df["contrast"] != 0]
    assert 0.74 < (np.sign(nz["contrast"]) == nz["prior"]).mean() < 0.86
    assert df.attrs["truth"]["sigma"] == 0.12


def test_mock_rejects_unknown_shape():
    with pytest.raises(ValueError):
        make_mock_trials(100, cbf_shape="sideways")


# ---------------------------------------------------------------------------
# Psychometrics / chronometrics recover ground truth
# ---------------------------------------------------------------------------
def test_psychometric_fit_recovers_truth(mock_flat):
    tr = mock_flat.attrs["truth"]
    fit = fit_psychometric(mock_flat[mock_flat["prior"] > 0], n_boot=60, seed=0)
    assert fit["success"]
    assert fit["mu"] == pytest.approx(tr["mu_right_block"], abs=0.02)
    assert fit["sigma"] == pytest.approx(tr["sigma"], rel=0.15)
    assert fit["gamma"] == pytest.approx(tr["gamma"], abs=0.025)
    assert fit["lam"] == pytest.approx(tr["lam"], abs=0.025)
    assert fit["mu_lo"] < tr["mu_right_block"] < fit["mu_hi"] or abs(fit["mu"] - tr["mu_right_block"]) < 0.02


def test_prior_shift_recovers_truth(mock_flat):
    ps = prior_shift(mock_flat, n_boot=100, seed=0).iloc[0]
    assert ps["shift"] == pytest.approx(mock_flat.attrs["truth"]["shift"], abs=0.03)
    assert ps["shift_lo"] < ps["shift"] < ps["shift_hi"]
    assert ps["prior_effect_c0"] > 0.2


def test_no_prior_effect_gives_no_shift():
    df = make_mock_trials(30000, prior_effect=0.0, seed=5)
    ps = prior_shift(df, n_boot=100).iloc[0]
    assert abs(ps["shift"]) < 0.03
    assert ps["shift_lo"] < 0 < ps["shift_hi"]


def test_psychometric_function_properties():
    c = np.array([-1.0, 0.0, 1.0])
    p = psychometric_function(c, mu=0.0, sigma=0.1, gamma=0.1, lam=0.2)
    assert p[0] == pytest.approx(0.1, abs=1e-3)       # left lapse floor
    assert p[2] == pytest.approx(0.8, abs=1e-3)       # 1 - right lapse ceiling
    assert 0.1 < p[1] < 0.8


def test_psychometric_table_counts_and_intervals(mock_flat):
    t = psychometric_table(mock_flat)
    assert t["n"].sum() == len(mock_flat)
    assert ((t["ci_lo"] <= t["p_right"] + 1e-9) | (t["n_right"] == 0)).all()
    assert (t["ci_hi"] >= t["p_right"] - 1e-9).all()
    assert t["ci_lo"].between(0, 1).all() and t["ci_hi"].between(0, 1).all()


def test_chronometric_decreases_with_contrast(mock_flat):
    t = chronometric_table(mock_flat, by=("subject_id",))
    t = t.assign(abs_c=t["contrast"].abs()).groupby("abs_c")["mean_rt"].mean()
    assert t.loc[0.0] > t.loc[1.0], "mean RT must be longer for 0% than 100% contrast"


def test_chronometric_by_outcome_excludes_zero_contrast(mock_flat):
    t = chronometric_table(mock_flat, by_outcome=True)
    assert (t["contrast"] != 0).all()
    assert set(t["correct"].unique()) == {True, False}


def test_rt_quantiles_and_conditional_accuracy(mock_flat):
    q = rt_quantile_table(mock_flat)
    assert (q["q10"] <= q["q50"]).all() and (q["q50"] <= q["q90"]).all()
    a = accuracy_by_rt_bin(mock_flat, n_bins=5)
    assert len(a) == 5 and a["accuracy"].between(0, 1).all()
    assert a["rt_median"].is_monotonic_increasing


def test_across_subject_mean_sem(mock_cohort):
    t = psychometric_table(mock_cohort)
    m = across_subject(t, "p_right", ["prior", "contrast"])
    assert (m["n_subjects"] == 6).all()
    assert m["sem"].notna().all()


# ---------------------------------------------------------------------------
# CBF: shape recovery and the stratification design
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape,expected_sign", [("falling", -1), ("flat", 0), ("rising", 1)])
def test_cbf_shape_classification_on_mock(shape, expected_sign):
    df = make_mock_trials(20000, cbf_shape=shape, seed=1)
    b = bootstrap_cbf(df, n_boot=200, seed=2)
    assert b["shape"] == shape
    if expected_sign:
        assert np.sign(b["slope"]) == expected_sign
    else:
        assert abs(b["slope"]) < 0.02


def test_cbf_table_structure_and_counts(mock_flat):
    t = compute_cbf(mock_flat, n_bins=5)
    assert list(t["bin"]) == [0, 1, 2, 3, 4]
    assert (np.diff(t["rt_lo"]) >= 0).all()
    # quintile bins hold roughly equal numbers of trials
    assert t["n_trials"].max() / t["n_trials"].min() < 1.25
    assert (t["n_right_prior"] + t["n_left_prior"]).sum() == t["n_trials"].sum()
    assert ((t["ci_lo"] < t["cbf"]) & (t["cbf"] < t["ci_hi"])).all()


def test_cbf_is_stratified_by_contrast():
    """With NO prior effect the stratified CBF is ~0, although stimulus statistics (80% of
    stimuli on the block's side) make the naive P(R|R-block) - P(R|L-block) large."""
    df = make_mock_trials(40000, prior_effect=0.0, seed=7)
    naive = (df.loc[df["prior"] > 0, "choice"] == 1).mean() - (df.loc[df["prior"] < 0, "choice"] == 1).mean()
    assert naive > 0.2, "naive difference should be inflated by stimulus side"
    t = compute_cbf(df, max_abs_contrast=1.0)
    assert np.abs(t["cbf"]).max() < 0.08


def test_cbf_contrast_selection_changes_trial_counts(mock_flat):
    n0 = compute_cbf(mock_flat, max_abs_contrast=0.0)["n_trials"].sum()
    n1 = compute_cbf(mock_flat, max_abs_contrast=0.125)["n_trials"].sum()
    n2 = compute_cbf(mock_flat, max_abs_contrast=1.0)["n_trials"].sum()
    assert n0 < n1 < n2 == len(mock_flat)


def test_cbf_fixed_bin_edges_and_validation(mock_flat):
    edges = np.array([0.0, 0.2, 0.3, 0.45, 0.8, 5.0])
    t = compute_cbf(mock_flat, bin_edges=edges)
    np.testing.assert_allclose(t["rt_lo"], edges[:-1])
    with pytest.raises(ValueError):
        compute_cbf(mock_flat, n_bins=5, bin_edges=edges[:-1])
    with pytest.raises(ValueError):
        compute_cbf(mock_flat.head(6), n_bins=5)


def test_cbf_curve_adapter_works_with_track_a_plot(mock_flat):
    centers, values = cbf_curve(mock_flat)
    assert len(centers) == len(values) == 5
    fig = plot_cbf({"Empirical Mouse Data": (centers, values)})
    assert fig.axes


def test_classify_shape_rules():
    assert classify_shape(-0.05, -0.08, -0.02) == "falling"
    assert classify_shape(0.05, 0.02, 0.08) == "rising"
    assert classify_shape(0.0, -0.02, 0.02) == "flat"
    # significant but small trends (below the effect-size floor) are not real trends
    assert classify_shape(-0.01, -0.015, -0.005) == "flat"
    assert classify_shape(-0.027, -0.04, -0.01) == "flat"   # Track A's M2 reference slope
    assert classify_shape(-0.05, -0.08, -0.02, min_abs_slope=0.0) == "falling"
    assert classify_shape(np.nan, np.nan, np.nan) == "undetermined"


def test_cbf_by_subject_and_cohort_tools(mock_cohort):
    curves, slopes = cbf_by_subject(mock_cohort, n_boot=100)
    assert curves["subject_id"].nunique() == 6 and len(slopes) == 6
    coh = cohort_cbf(curves)
    assert len(coh) == 5 and (coh["n_subjects"] == 6).all()
    res = cohort_slope_test(slopes)
    assert res["n_mice"] == 6 and "t_p" in res
    assert sum(res["shape_counts"].values()) == 6


def test_cohort_of_falling_mice_has_negative_mean_slope():
    cohort = make_mock_cohort(n_subjects=6, n_trials=8000, cbf_shape="falling", seed=4)
    _, slopes = cbf_by_subject(cohort, n_boot=100)
    res = cohort_slope_test(slopes)
    assert res["mean_slope"] < -0.03 and res["t_p"] < 0.01


# ---------------------------------------------------------------------------
# Heterogeneity profile
# ---------------------------------------------------------------------------
def test_mouse_profile_tracks_true_heterogeneity(mock_cohort):
    prof = mouse_profile(mock_cohort, n_boot=50).set_index("subject_id")
    truth = mock_cohort.attrs["truth"].set_index("subject_id")
    for col in ("sigma", "prior_shift"):
        true_col = "sigma" if col == "sigma" else "shift"
        r = np.corrcoef(prof[col], truth.loc[prof.index, true_col])[0, 1]
        assert r > 0.9, f"{col} should track the true per-mouse value (r={r:.2f})"
    assert prof["n_trials"].eq(8000).all() or prof["n_trials"].between(7000, 8000).all()
    summ = heterogeneity_summary(prof.reset_index())
    assert {"metric", "mean", "sd", "cv", "outliers"} <= set(summ.columns)
    corr = profile_correlations(prof.reset_index())
    assert corr.shape[0] == corr.shape[1] and np.allclose(np.diag(corr.values), 1.0)


# ---------------------------------------------------------------------------
# Schema generality: tools must read ANY §3.1 DataFrame
# ---------------------------------------------------------------------------
def _run_all_tools(df: pd.DataFrame) -> None:
    psychometric_table(df)
    chronometric_table(df)
    fit_psychometric_by(df)
    compute_cbf(df)
    cbf_by_subject(df, n_boot=20)
    plot_psychometric_chronometric_cohort(df)
    plot_psychometric_by_mouse(df)
    plot_chronometric_by_mouse(df)
    plot_rt_distributions(df)


def test_tools_accept_shuffled_rows_extra_columns_and_dtypes(mock_cohort):
    df = mock_cohort.sample(frac=1.0, random_state=0).reset_index(drop=True)  # shuffled, trial_idx broken
    df["session_eid"] = "x"                                                    # extra column
    df["subject_id"] = df["subject_id"].astype("category")                     # categorical ids
    df["contrast"] = df["contrast"].astype("float32")
    df["choice"] = df["choice"].astype("int64")
    _run_all_tools(df)


def test_tools_accept_track_a_simulator_output():
    for model, params in (
        ("M1", {"a": 1.1, "t0": 0.2, "k": 2.6, "z": 0.4}),
        ("M2", {"a": 1.1, "t0": 0.2, "k": 2.6, "v_b": 0.55}),
        ("M3", {"a": 1.1, "t0": 0.2, "k": 2.6, "b_max": 1.2, "tau_b": 0.5}),
    ):
        df = simulate_ddm(params, model_type=model, n_trials=4000, seed=5)
        validate_contract(df)
        _run_all_tools(df)


def test_cbf_discriminates_track_a_models():
    """Track B's CBF must show M1 falling and M3 rising on Track A's simulator (the Week 3 premise)."""
    slopes = {}
    for model, params in (
        ("M1", {"a": 1.1, "t0": 0.2, "k": 2.6, "z": 0.4}),
        ("M2", {"a": 1.1, "t0": 0.2, "k": 2.6, "v_b": 0.55}),
        ("M3", {"a": 1.1, "t0": 0.2, "k": 2.6, "b_max": 1.2, "tau_b": 0.5}),
    ):
        df = simulate_ddm(params, model_type=model, n_trials=20000, seed=101)
        slopes[model] = bootstrap_cbf(df, n_boot=100, seed=0)
    assert slopes["M1"]["shape"] == "falling"
    assert slopes["M2"]["shape"] == "flat"
    assert slopes["M3"]["shape"] == "rising"
    assert slopes["M1"]["slope"] < slopes["M2"]["slope"] < slopes["M3"]["slope"]
    assert abs(slopes["M2"]["slope"]) < 0.05


def test_tools_handle_single_block_and_single_subject_edge_cases(mock_flat):
    one_block = mock_flat[mock_flat["prior"] > 0]
    assert len(psychometric_table(one_block)) > 0
    b = bootstrap_cbf(one_block, n_boot=10)           # no left-prior trials: CBF undefined, no crash
    assert b["shape"] == "undetermined"
    plot_psychometric_chronometric_cohort(one_block)  # only one prior drawn


def test_neutral_prior_trials_are_ignored(mock_flat):
    extra = mock_flat.head(500).assign(prior=0.0)
    mixed = pd.concat([mock_flat, extra], ignore_index=True)
    assert psychometric_table(mixed)["n"].sum() == len(mock_flat)
    assert compute_cbf(mixed)["n_trials"].sum() == compute_cbf(mock_flat)["n_trials"].sum()


def test_tools_reject_tables_violating_the_schema(mock_flat):
    for col in ("prior", "choice", "rt", "contrast", "subject_id"):
        with pytest.raises(ValueError, match="§3.1"):
            psychometric_table(mock_flat.drop(columns=[col]))
    with pytest.raises(ValueError, match="§3.1"):
        compute_cbf(mock_flat.drop(columns=["rt"]))
    with pytest.raises(ValueError, match="§3.1"):
        plot_psychometric_chronometric_cohort(mock_flat.drop(columns=["choice"]))
    with pytest.raises(TypeError):
        psychometric_table([1, 2, 3])


# ---------------------------------------------------------------------------
# Plots: return figures and save files
# ---------------------------------------------------------------------------
def test_plots_save_files(tmp_path, mock_cohort):
    curves, slopes = cbf_by_subject(mock_cohort, n_boot=30)
    prof = mouse_profile(mock_cohort, n_boot=20)
    figs = {
        "a": plot_psychometric_chronometric_cohort(mock_cohort, str(tmp_path / "a.png")),
        "b": plot_psychometric_by_mouse(mock_cohort, save_path=str(tmp_path / "b.png"), ncols=3),
        "c": plot_chronometric_by_mouse(mock_cohort, save_path=str(tmp_path / "c.png"), ncols=3),
        "d": plot_cbf_cohort(curves, overlay={"M2": [0.3] * 5}, save_path=str(tmp_path / "d.png")),
        "e": plot_cbf_by_mouse(curves, slopes, ncols=3, save_path=str(tmp_path / "e.png")),
        "f": plot_heterogeneity(prof, save_path=str(tmp_path / "f.png")),
        "g": plot_cbf_sensitivity(mock_cohort, {"zero": {"max_abs_contrast": 0.0},
                                                "low": {"max_abs_contrast": 0.125}},
                                  n_boot=10, save_path=str(tmp_path / "g.png")),
    }
    for name, fig in figs.items():
        assert isinstance(fig, plt.Figure)
        assert (tmp_path / f"{name}.png").stat().st_size > 5000


# ---------------------------------------------------------------------------
# Real cohort (skipped if data/processed has not been built)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(REAL), reason="processed cohort not built")
def test_tools_accept_real_ibl_cohort_and_show_known_effects():
    df = pd.read_parquet(REAL)
    _run_all_tools(df)
    shifts = prior_shift(df, n_boot=50)
    assert len(shifts) == 10
    assert (shifts["shift"] > 0).all(), "every mouse must be biased toward its block's likelier side"
    assert (shifts["shift_lo"] > 0).sum() >= 9
    chrono = chronometric_table(df, by=("subject_id",))
    zero = chrono[chrono["contrast"] == 0]["mean_rt"].mean()
    full = chrono[chrono["contrast"].abs() == 1.0]["mean_rt"].mean()
    assert zero > full


# ---------------------------------------------------------------------------
# Robustness additions: session-level bootstrap, per-block lapses, history, leave-one-out
# ---------------------------------------------------------------------------
def _sessioned_mouse(n_sessions=20, n_per=500, seed=0):
    """One mouse whose prior strength differs from session to session (heterogeneous sessions)."""
    rng = np.random.default_rng(seed)
    frames = []
    for j in range(n_sessions):
        f = make_mock_trials(n_trials=n_per, subject_id="m", prior_effect=float(rng.uniform(0.2, 1.8)),
                             seed=int(rng.integers(0, 2 ** 31 - 1)))
        frames.append(f.assign(session_eid=f"s{j:02d}"))
    out = pd.concat(frames, ignore_index=True)
    out["trial_idx"] = np.arange(len(out))
    return out


def test_session_bootstrap_gives_wider_shift_intervals_than_trial_bootstrap():
    df = _sessioned_mouse()
    trial = prior_shift(df, n_boot=150, seed=1).iloc[0]
    clus = prior_shift(df, n_boot=150, seed=1, cluster_col="session_eid").iloc[0]
    assert clus["shift"] == pytest.approx(trial["shift"])            # same point estimate
    assert (clus["shift_hi"] - clus["shift_lo"]) > 1.3 * (trial["shift_hi"] - trial["shift_lo"])
    assert clus["shift_lo"] < clus["shift"] < clus["shift_hi"]


def test_session_bootstrap_cbf_runs_and_widens_slope_interval():
    df = _sessioned_mouse(n_sessions=20, n_per=700)
    a = bootstrap_cbf(df, n_boot=150, seed=3)
    b = bootstrap_cbf(df, n_boot=150, seed=3, cluster_col="session_eid")
    assert a["slope"] == pytest.approx(b["slope"])
    assert np.isfinite(b["slope_lo"]) and b["slope_lo"] <= b["slope"] <= b["slope_hi"]
    with pytest.raises(ValueError, match="§3.1"):
        bootstrap_cbf(df, n_boot=5, cluster_col="not_a_column")
    _, slopes = cbf_by_subject(df, n_boot=20, cluster_col="session_eid")
    assert len(slopes) == 1


def test_prior_shift_reports_per_block_lapses(mock_flat):
    ps = prior_shift(mock_flat, n_boot=0).iloc[0]
    truth = mock_flat.attrs["truth"]
    for col, true in (("gamma_right_block", truth["gamma"]), ("gamma_left_block", truth["gamma"]),
                      ("lam_right_block", truth["lam"]), ("lam_left_block", truth["lam"])):
        assert abs(ps[col] - true) < 0.06


def test_lapse_block_difference_detects_prior_acting_on_the_asymptote():
    """P(Right) at far-left stimuli raised in right blocks must show up as gamma_block_diff > 0."""
    df = make_mock_trials(n_trials=20000, gamma=0.02, lam=0.02, seed=5)
    rng = np.random.default_rng(5)
    far_left_right_block = (df["contrast"] == -1.0) & (df["prior"] > 0)
    flip = far_left_right_block & (rng.random(len(df)) < 0.3)
    df.loc[flip, "choice"] = 1
    prof = mouse_profile(df, n_boot=10, seed=0).iloc[0]
    assert prof["gamma_block_diff"] > 0.1
    assert prof["gamma_right_block"] > prof["gamma_left_block"]


def test_history_effect_is_not_confounded_with_the_block(mock_flat):
    """Choices here depend on the block but never on the previous choice, so the effect must be ~0."""
    prof = mouse_profile(mock_flat, n_boot=10).iloc[0]
    c0 = mock_flat[(mock_flat["contrast"] == 0) & (mock_flat["prev_choice"] != 0)]
    pooled = (c0.loc[c0["prev_choice"] > 0, "choice"] == 1).mean() - (c0.loc[c0["prev_choice"] < 0, "choice"] == 1).mean()
    assert pooled > 0.1                       # the naive pooled contrast is inflated by the block
    assert abs(prof["history_effect_c0"]) < 0.12


def test_cohort_slope_test_exclude(mock_cohort):
    _, slopes = cbf_by_subject(mock_cohort, n_boot=20)
    full = cohort_slope_test(slopes)
    drop = cohort_slope_test(slopes, exclude=["mock_001"])
    n = mock_cohort["subject_id"].nunique()
    assert full["n_mice"] == n and drop["n_mice"] == n - 1
    assert cohort_slope_test(slopes, exclude=[])["n_mice"] == n


@pytest.mark.skipif(not os.path.exists(REAL_EXT), reason="extended cohort not built")
def test_real_cohort_response_time_keeps_fast_trials_and_has_block_positions():
    ext = pd.read_parquet(REAL_EXT)
    assert (ext["rt"] == ext["rt_response"]).all()
    assert ext["rt"].median() > 0.3          # response times, not anticipatory first movements
    assert ext["block_pos"].min() == 0 and ext["block_pos"].max() > 50
