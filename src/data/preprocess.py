"""
Behavioural preprocessing: raw IBL trials table -> §3.1 Trial DataFrame.

Track B (Joshua Koilpillai). Conventions (verified against `feedbackType` on real data):
- contrast:    signed, contrastRight - contrastLeft  (positive = stimulus on the right)
- prior:       +1 right-biased block (p_left = 0.2), -1 left-biased block (p_left = 0.8);
               0 for the unbiased p_left = 0.5 opening trials (dropped from the cohort files)
- choice:      +1 = mouse chose the Right side, -1 = Left. IBL codes wheel direction, where
               choice_IBL = -1 corresponds to the right-side response, so choice = -choice_IBL
- prev_choice: 1-back choice on the unfiltered session sequence; 0 on the first trial of a
               session and after a no-go trial
- rt:          firstMovement_times - stimOn_times (wheel onset minus stimulus onset), seconds
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

CONTRACT_COLUMNS = ["subject_id", "trial_idx", "contrast", "prior", "prev_choice", "choice", "rt"]
EXTENDED_COLUMNS = [
    "lab", "session_eid", "session_date", "session_trial", "p_left",
    "rt_response", "feedback", "rt_in_2p5", "rt_in_5p0",
]

RT_MIN = 0.1
RT_MAX_STANDARD = 2.5
RT_MAX_EXTENDED = 5.0
EASY_CONTRAST = 0.25  # the biased protocol presents {0, .0625, .125, .25, 1}

_REQUIRED_RAW = [
    "choice", "contrastLeft", "contrastRight", "probabilityLeft", "feedbackType",
    "stimOn_times", "firstMovement_times", "response_times",
]


def parse_session_trials(
    raw: pd.DataFrame,
    subject_id: str,
    eid: str = "",
    date: str = "",
    lab: str = "",
) -> pd.DataFrame:
    """
    Convert one session's raw IBL trials table into unfiltered derived trial variables.

    Returns every trial of the session (including no-go trials and neutral-block trials),
    so `prev_choice` reflects the true 1-back sequence. Filtering happens in `filter_trials`.
    """
    missing = [c for c in _REQUIRED_RAW if c not in raw.columns]
    if missing:
        raise KeyError(missing[0])

    contrast_r = raw["contrastRight"].fillna(0.0).to_numpy(dtype=float)
    contrast_l = raw["contrastLeft"].fillna(0.0).to_numpy(dtype=float)
    p_left = raw["probabilityLeft"].to_numpy(dtype=float)
    ibl_choice = raw["choice"].fillna(0.0).to_numpy(dtype=float)

    choice = (-ibl_choice).astype(np.int8)  # IBL -1 (right response) -> +1
    prev_choice = np.concatenate(([0], choice[:-1])).astype(np.int8)
    prior = np.sign(np.round(0.5 - p_left, 3))

    stim_on = raw["stimOn_times"].to_numpy(dtype=float)
    return pd.DataFrame({
        "subject_id": subject_id,
        "lab": lab,
        "session_eid": eid,
        "session_date": date,
        "session_trial": np.arange(len(raw), dtype=np.int32),
        "contrast": contrast_r - contrast_l,
        "prior": prior,
        "prev_choice": prev_choice,
        "choice": choice,
        "rt": raw["firstMovement_times"].to_numpy(dtype=float) - stim_on,
        "rt_response": raw["response_times"].to_numpy(dtype=float) - stim_on,
        "feedback": raw["feedbackType"].to_numpy(dtype=float),
        "p_left": p_left,
    })


def filter_trials(
    df: pd.DataFrame,
    rt_min: float = RT_MIN,
    rt_max: float = RT_MAX_STANDARD,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply the exclusion filters in sequence and report how many trials each removed.

    Order: neutral (p_left = 0.5) block -> no-go / missing RT -> RT < rt_min -> RT > rt_max.
    """
    counts: Dict[str, int] = {"n_input": int(len(df))}
    keep = df["prior"].to_numpy() != 0
    counts["dropped_neutral_block"] = int((~keep).sum())

    valid = (df["choice"].to_numpy() != 0) & np.isfinite(df["rt"].to_numpy())
    counts["dropped_nogo_or_missing_rt"] = int((keep & ~valid).sum())
    keep &= valid

    rt = df["rt"].to_numpy()
    with np.errstate(invalid="ignore"):
        too_fast = rt < rt_min
        too_slow = rt > rt_max
    counts["dropped_rt_too_fast"] = int((keep & too_fast).sum())
    keep &= ~too_fast
    counts["dropped_rt_too_slow"] = int((keep & too_slow).sum())
    keep &= ~too_slow

    counts["n_kept"] = int(keep.sum())
    return df[keep].copy(), counts


def build_subject_table(
    sessions: List[pd.DataFrame],
    rt_min: float = RT_MIN,
    rt_max: float = RT_MAX_STANDARD,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Concatenate (time-ordered) parsed sessions of one mouse, filter, and index trials 0..N-1.

    Returns the extended table (contract columns first, then extended columns) and
    summed exclusion counts.
    """
    kept, totals = [], {}
    for sess in sessions:
        sub, counts = filter_trials(sess, rt_min=rt_min, rt_max=rt_max)
        kept.append(sub)
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
    df = pd.concat(kept, ignore_index=True)
    df["trial_idx"] = np.arange(len(df), dtype=np.int32)
    df["rt_in_2p5"] = df["rt"] <= RT_MAX_STANDARD
    df["rt_in_5p0"] = df["rt"] <= RT_MAX_EXTENDED
    return df[CONTRACT_COLUMNS + EXTENDED_COLUMNS], totals


def session_qc(raw: pd.DataFrame) -> Dict[str, float]:
    """
    Per-session quality-control statistics used for session / mouse selection.

    easy_accuracy: fraction of responded trials with |contrast| >= 0.25 where the
    mouse chose the stimulus side.
    """
    parsed = parse_session_trials(raw, subject_id="qc")
    responded = parsed["choice"] != 0
    easy = responded & (parsed["contrast"].abs() >= EASY_CONTRAST)
    if easy.sum() > 0:
        easy_acc = float(
            (np.sign(parsed.loc[easy, "contrast"]) == parsed.loc[easy, "choice"]).mean()
        )
    else:
        easy_acc = np.nan
    kept, _ = filter_trials(parsed)
    p_left = np.unique(np.round(parsed["p_left"].dropna().to_numpy(), 3))
    return {
        "n_trials": int(len(parsed)),
        "n_nogo": int((~responded).sum()),
        "easy_accuracy": easy_acc,
        "has_both_blocks": bool(0.2 in p_left and 0.8 in p_left),
        "n_valid_trials": int(len(kept)),
    }


def validate_contract(df: pd.DataFrame) -> None:
    """
    Assert that `df` conforms to the §3.1 Trial DataFrame schema.

    Works on both synthetic (Track A) and empirical (Track B) tables.
    Raises AssertionError describing the first violated rule.
    """
    for col in CONTRACT_COLUMNS:
        assert col in df.columns, f"Missing required contract column: {col}"
    core = df[CONTRACT_COLUMNS]
    assert not core.isna().any().any(), "Contract columns must not contain NaN"
    assert pd.api.types.is_numeric_dtype(core["contrast"]), "contrast must be numeric"
    assert pd.api.types.is_numeric_dtype(core["prior"]), "prior must be numeric"
    assert pd.api.types.is_integer_dtype(core["choice"]), "choice must be an integer dtype"
    assert pd.api.types.is_integer_dtype(core["prev_choice"]), "prev_choice must be an integer dtype"
    assert pd.api.types.is_integer_dtype(core["trial_idx"]), "trial_idx must be an integer dtype"
    assert pd.api.types.is_float_dtype(core["rt"]), "rt must be a float dtype"
    assert core["contrast"].between(-1.0, 1.0).all(), "contrast outside [-1, 1]"
    assert core["prior"].isin([-1.0, 1.0]).all(), "prior must be -1 or +1"
    assert core["choice"].isin([-1, 1]).all(), "choice must be -1 or +1"
    assert core["prev_choice"].isin([-1, 0, 1]).all(), "prev_choice must be -1, 0 or +1"
    assert (core["rt"] > 0).all(), "rt must be positive"
    for subj, g in core.groupby("subject_id", sort=False):
        assert np.array_equal(g["trial_idx"].to_numpy(), np.arange(len(g))), (
            f"trial_idx for {subj} must run 0..N-1 in order"
        )
