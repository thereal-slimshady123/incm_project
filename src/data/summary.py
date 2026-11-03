"""
Summary statistics for the processed IBL cohort (one row per mouse + cohort total).
"""

from typing import Dict

import numpy as np
import pandas as pd

from .preprocess import EASY_CONTRAST


def _stats(df: pd.DataFrame) -> Dict[str, float]:
    c0 = df[df["contrast"] == 0]
    easy = df[df["contrast"].abs() >= EASY_CONTRAST]
    p_right = lambda sub: float((sub["choice"] == 1).mean()) if len(sub) else np.nan  # noqa: E731
    return {
        "n_trials": int(len(df)),
        "n_right_block": int((df["prior"] > 0).sum()),
        "n_left_block": int((df["prior"] < 0).sum()),
        "p_right": p_right(df),
        "p_right_c0_right_block": p_right(c0[c0["prior"] > 0]),
        "p_right_c0_left_block": p_right(c0[c0["prior"] < 0]),
        "prior_effect_c0": p_right(c0[c0["prior"] > 0]) - p_right(c0[c0["prior"] < 0]),
        "easy_accuracy": float((np.sign(easy["contrast"]) == easy["choice"]).mean()),
        "repeat_rate": float((df.loc[df["prev_choice"] != 0, "choice"]
                              == df.loc[df["prev_choice"] != 0, "prev_choice"]).mean()),
        "rt_median": float(df["rt"].median()),
        "rt_q10": float(df["rt"].quantile(0.10)),
        "rt_q90": float(df["rt"].quantile(0.90)),
        "rt_mean": float(df["rt"].mean()),
    }


def summarize_cohort(
    cohort: pd.DataFrame,
    exclusion_counts: Dict[str, Dict[str, int]],
    session_counts: Dict[str, int],
    labs: Dict[str, str],
    rt_max: float,
) -> pd.DataFrame:
    """
    Build the per-mouse summary table for one RT cutoff.

    Parameters
    ----------
    cohort : pd.DataFrame
        Contract-conformant table of all mice at this cutoff.
    exclusion_counts : dict
        subject -> counts returned by `preprocess.build_subject_table`.
    session_counts : dict
        subject -> number of included sessions.
    labs : dict
        subject -> lab name.
    rt_max : float
        RT cutoff this table was built with (recorded in the output).
    """
    rows = []
    for subj, g in cohort.groupby("subject_id", sort=False):
        ex = exclusion_counts[subj]
        rows.append({
            "subject_id": subj,
            "lab": labs[subj],
            "rt_max": rt_max,
            "n_sessions": session_counts[subj],
            "n_raw_trials": ex["n_input"],
            "dropped_neutral_block": ex["dropped_neutral_block"],
            "dropped_nogo_or_missing_rt": ex["dropped_nogo_or_missing_rt"],
            "dropped_rt_too_fast": ex["dropped_rt_too_fast"],
            "dropped_rt_too_slow": ex["dropped_rt_too_slow"],
            **_stats(g),
        })
    out = pd.DataFrame(rows)

    total = {"subject_id": "ALL", "lab": "", "rt_max": rt_max,
             "n_sessions": int(out["n_sessions"].sum())}
    for col in ["n_raw_trials", "dropped_neutral_block", "dropped_nogo_or_missing_rt",
                "dropped_rt_too_fast", "dropped_rt_too_slow"]:
        total[col] = int(out[col].sum())
    total.update(_stats(cohort))
    return pd.concat([out, pd.DataFrame([total])], ignore_index=True)
