"""
Track B tests: IBL preprocessing logic (offline, synthetic raw tables) and §3.1 contract
compliance of the processed cohort files in data/processed/ (skipped if not yet built).
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    CONTRACT_COLUMNS,
    build_subject_table,
    filter_trials,
    parse_session_trials,
    session_qc,
    validate_contract,
)
from src.models.simulator import simulate_ddm

PROCESSED = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def make_raw_session() -> pd.DataFrame:
    """Hand-built raw IBL trials table mimicking `_ibl_trials.table.pqt`."""
    return pd.DataFrame({
        # trial:        0     1     2     3     4     5     6     7
        "contrastLeft":  [np.nan, 0.25, np.nan, np.nan, 1.0, np.nan, 0.0, np.nan],
        "contrastRight": [1.0, np.nan, 0.0, 0.125, np.nan, 0.0625, np.nan, 1.0],
        "probabilityLeft": [0.5, 0.5, 0.2, 0.2, 0.8, 0.8, 0.8, 0.2],
        # IBL: choice -1 = right-side response, +1 = left-side response, 0 = no-go
        "choice": [-1.0, 1.0, -1.0, 0.0, 1.0, -1.0, 1.0, -1.0],
        "feedbackType": [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0],
        "stimOn_times": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        "firstMovement_times": [10.3, 20.05, 30.5, 40.6, 50.4, 66.0, 70.3, 80.4],
        "response_times": [10.5, 20.3, 30.9, 41.0, 50.9, 66.5, 70.9, 80.8],
    })


def test_parse_conventions():
    p = parse_session_trials(make_raw_session(), "mouse_x", eid="e1", date="2020-01-01")
    # Signed contrast: positive = right stimulus
    assert p["contrast"].tolist() == [1.0, -0.25, 0.0, 0.125, -1.0, 0.0625, 0.0, 1.0]
    # prior: p_left 0.2 -> +1 (right-biased), 0.8 -> -1, 0.5 -> 0
    assert p["prior"].tolist() == [0, 0, 1, 1, -1, -1, -1, 1]
    # choice: IBL -1 -> +1 (Right); no-go stays 0
    assert p["choice"].tolist() == [1, -1, 1, 0, -1, 1, -1, 1]
    # prev_choice: 1-back on the unfiltered sequence, 0 at start and after a no-go
    assert p["prev_choice"].tolist() == [0, 1, -1, 1, 0, -1, 1, -1]
    # rt = firstMovement - stimOn
    np.testing.assert_allclose(p["rt"].iloc[:3], [0.3, 0.05, 0.5])


def test_correct_trials_match_stimulus_side():
    """Feedback-correct trials must pick the stimulus side under our sign convention."""
    p = parse_session_trials(make_raw_session(), "mouse_x")
    nonzero = (p["contrast"] != 0) & (p["feedback"] == 1)
    assert (np.sign(p.loc[nonzero, "contrast"]) == p.loc[nonzero, "choice"]).all()


def test_filter_waterfall_counts():
    p = parse_session_trials(make_raw_session(), "mouse_x")
    kept, counts = filter_trials(p, rt_min=0.1, rt_max=2.5)
    assert counts["n_input"] == 8
    assert counts["dropped_neutral_block"] == 2          # trials 0, 1
    assert counts["dropped_nogo_or_missing_rt"] == 1     # trial 3
    assert counts["dropped_rt_too_fast"] == 0
    assert counts["dropped_rt_too_slow"] == 1            # trial 5 (rt = 6.0 s)
    assert counts["n_kept"] == 4 == len(kept)
    assert kept["rt"].between(0.1, 2.5).all()


def test_rt_cutoffs_nested():
    p = parse_session_trials(make_raw_session(), "mouse_x")
    t25, _ = build_subject_table([p], rt_max=2.5)
    t50, _ = build_subject_table([p], rt_max=5.0)
    assert len(t50) >= len(t25)
    assert set(t25["session_trial"]).issubset(set(t50["session_trial"]))


def test_prev_choice_preserved_across_filtering():
    """prev_choice must come from the unfiltered sequence, not the filtered one."""
    p = parse_session_trials(make_raw_session(), "mouse_x")
    t, _ = build_subject_table([p])
    row = t[t["session_trial"] == 4].iloc[0]       # follows the no-go trial 3
    assert row["prev_choice"] == 0
    row = t[t["session_trial"] == 6].iloc[0]       # trial 5 was dropped (slow) but was a Right choice
    assert row["prev_choice"] == 1


def test_build_subject_table_meets_contract_across_sessions():
    s1 = parse_session_trials(make_raw_session(), "mouse_x", eid="e1")
    s2 = parse_session_trials(make_raw_session(), "mouse_x", eid="e2")
    t, _ = build_subject_table([s1, s2])
    validate_contract(t)
    assert t["trial_idx"].tolist() == list(range(len(t)))
    # prev_choice resets at the start of each session
    assert (t[t["session_trial"] == 2].groupby("session_eid")["prev_choice"].first() == -1).all()


def test_session_qc_fields():
    qc = session_qc(make_raw_session())
    assert qc["n_trials"] == 8
    assert qc["n_nogo"] == 1
    assert qc["has_both_blocks"] is True
    assert 0.0 <= qc["easy_accuracy"] <= 1.0


def test_validator_accepts_simulator_output():
    """Track A synthetic data and Track B empirical data pass the same validator."""
    df = simulate_ddm({"a": 1.0, "t0": 0.2, "k": 2.5}, n_trials=100, seed=3)
    validate_contract(df)


def test_validator_rejects_bad_tables():
    good = parse_session_trials(make_raw_session(), "m")
    t, _ = build_subject_table([good])
    bad = t.copy()
    bad.loc[0, "choice"] = 0
    with pytest.raises(AssertionError):
        validate_contract(bad)
    with pytest.raises(AssertionError):
        validate_contract(t.drop(columns=["rt"]))


# ---------------------------------------------------------------------------
# Processed cohort (requires `python run_track_b.py --all` to have been run)
# ---------------------------------------------------------------------------
cohort_files = [os.path.join(PROCESSED, f"cohort_rt{t}.parquet") for t in ("2p5", "5p0")]
needs_data = pytest.mark.skipif(
    not all(os.path.exists(f) for f in cohort_files), reason="processed cohort not built"
)


@needs_data
@pytest.mark.parametrize("path", cohort_files)
def test_processed_cohort_contract(path):
    df = pd.read_parquet(path)
    assert list(df.columns) == CONTRACT_COLUMNS
    validate_contract(df)
    assert df["subject_id"].nunique() == 10


@needs_data
def test_processed_rt_filters():
    d25 = pd.read_parquet(cohort_files[0])
    d50 = pd.read_parquet(cohort_files[1])
    assert d25["rt"].min() >= 0.1 and d25["rt"].max() <= 2.5
    assert d50["rt"].min() >= 0.1 and d50["rt"].max() <= 5.0
    assert len(d50) >= len(d25)


@needs_data
def test_processed_prior_bias_visible_at_zero_contrast():
    """Mice should choose Right more often in right-biased blocks (Findling et al., 2023)."""
    df = pd.read_parquet(cohort_files[0])
    c0 = df[df["contrast"] == 0]
    p_r = c0[c0["prior"] > 0]["choice"].eq(1).mean()
    p_l = c0[c0["prior"] < 0]["choice"].eq(1).mean()
    assert p_r > p_l + 0.1


@needs_data
def test_processed_csv_matches_parquet():
    pq = pd.read_parquet(cohort_files[0])
    csv = pd.read_csv(cohort_files[0].replace(".parquet", ".csv"))
    assert len(pq) == len(csv)
    np.testing.assert_allclose(pq["rt"].to_numpy(), csv["rt"].to_numpy(), rtol=1e-9)
