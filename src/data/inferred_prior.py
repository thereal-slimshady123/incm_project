"""
Trial-by-trial INFERRED block prior: what an ideal observer who knows the task statistics would believe.

The C1 fits originally used the true block label (prior = +-1). Mice never see that label: right after a
block switch they still expect the old side, and they need several trials to adapt. Using the true label
therefore mis-states the prior on exactly the trials that carry information about it. This module replaces
it with the Bayes-optimal estimate (as in Findling et al., 2023, "Bayes-optimal" prior):

* Hidden state: block side (L: P(left stim) = 0.8, R: P(right stim) = 0.8) and position in the block.
* Block lengths follow the IBL design: exponential with scale 60 trials, truncated to [20, 100] (the
  empirical lengths in cohort_extended are 20-100, mean 51, matching this); the first biased block starts at
  raw trial 90 with either side equally likely.
* Observation on each trial: the stimulus side (sign of the contrast; on 0% trials the side the task rewarded,
  i.e. the choice if the feedback was positive, otherwise the other side).
* Trials missing from the processed table (RT outside the window, no-go: ~8%) are treated as unobserved:
  the belief is propagated through them without an update. The mouse did see those stimuli, so this slightly
  under-estimates how fast it could adapt; it never uses information the mouse did not have.

`prior_inferred` = 2 P(R block | stimuli before this trial) - 1, in (-1, 1), on the same scale as the block
label (+1 = right-biased). The QMP likelihood groups trials by condition, so `bin_prior` maps it to a few
levels (default: confident / uncertain on each side), each represented by the mouse's mean value in that bin.
"""

from typing import Sequence

import numpy as np
import pandas as pd

BIASED_START = 90          # raw trial index of the first biased block (the 90 opening trials are unbiased)
BLOCK_MIN, BLOCK_MAX, BLOCK_SCALE = 20, 100, 60.0
P_STIM = 0.8               # P(stimulus on the block's preferred side)
DEFAULT_BIN_EDGES = (-1.0, -0.5, 0.0, 0.5, 1.0)


def block_hazard(lo: int = BLOCK_MIN, hi: int = BLOCK_MAX, scale: float = BLOCK_SCALE) -> np.ndarray:
    """h[r-1] = P(block ends after trial r | it lasted r trials), r = 1..hi, for the truncated-exponential design."""
    lengths = np.arange(1, hi + 1)
    p = np.where(lengths >= lo, np.exp(-lengths / scale), 0.0)
    p /= p.sum()
    surv = p[::-1].cumsum()[::-1]                       # P(L >= r)
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(surv > 0, p / surv, 1.0)
    h[-1] = 1.0
    return h


def stimulus_side(df: pd.DataFrame) -> np.ndarray:
    """+1 right / -1 left stimulus; for 0% contrast the rewarded side (from choice and feedback)."""
    side = np.sign(df["contrast"].to_numpy(dtype=float))
    zero = side == 0
    if zero.any():
        ch = df["choice"].to_numpy(dtype=float)
        fb = df["feedback"].to_numpy(dtype=float)
        side[zero] = np.where(fb[zero] > 0, ch[zero], -ch[zero])
    return side


def _filter_session(trial_numbers: np.ndarray, sides: np.ndarray, hazard: np.ndarray,
                    start: int = BIASED_START) -> np.ndarray:
    """P(R block) before each listed trial of one session (trial_numbers sorted, raw indices)."""
    n_r = len(hazard)
    p = np.zeros((2, n_r))            # rows: 0 = L block, 1 = R block; cols: position in block - 1
    p[:, 0] = 0.5
    lik = np.array([[P_STIM, 1 - P_STIM],     # L block: P(left stim), P(right stim)
                    [1 - P_STIM, P_STIM]])    # R block
    out = np.empty(len(trial_numbers))
    observed = dict(zip(trial_numbers.tolist(), sides.tolist()))
    row_of = {t: i for i, t in enumerate(trial_numbers.tolist())}
    last = int(trial_numbers.max())
    for t in range(min(start, int(trial_numbers.min())), last + 1):
        if t in row_of:
            out[row_of[t]] = p[1].sum()
        y = observed.get(t)
        if y is not None and y != 0:
            p = p * lik[:, 1 if y > 0 else 0][:, None]
            p /= p.sum()
        # transition to the next trial
        stay = p * (1.0 - hazard)[None, :]
        switch = (p * hazard[None, :]).sum(axis=1)
        nxt = np.zeros_like(p)
        nxt[:, 1:] = stay[:, :-1]
        nxt[0, 0] = switch[1]          # an R block ended -> a new L block
        nxt[1, 0] = switch[0]
        p = nxt / nxt.sum()
    return out


def inferred_prior(ext: pd.DataFrame, hazard: np.ndarray = None) -> pd.Series:
    """
    Ideal-observer prior for every row of `cohort_extended` (needs session_eid, session_trial, contrast,
    choice, feedback). Returns a Series aligned with `ext.index`, values in (-1, 1).
    """
    hazard = block_hazard() if hazard is None else hazard
    side = stimulus_side(ext)
    out = pd.Series(np.nan, index=ext.index, name="prior_inferred")
    for _, g in ext.groupby("session_eid", sort=False):
        g = g.sort_values("session_trial")
        idx = g.index
        belief = _filter_session(g["session_trial"].to_numpy(), side[ext.index.get_indexer(idx)], hazard)
        out.loc[idx] = 2.0 * belief - 1.0
    return out


def bin_prior(values: pd.Series, subjects: pd.Series, edges: Sequence[float] = DEFAULT_BIN_EDGES) -> pd.Series:
    """
    Discretise the inferred prior for the condition-based QMP likelihood: bins by `edges` (default confident /
    uncertain on each side), each trial gets its mouse's mean inferred prior within its bin.
    """
    edges = np.asarray(edges, dtype=float)
    b = np.clip(np.searchsorted(edges, values.to_numpy(dtype=float), side="right") - 1, 0, len(edges) - 2)
    key = pd.DataFrame({"s": subjects.to_numpy(), "b": b, "v": values.to_numpy(dtype=float)}, index=values.index)
    means = key.groupby(["s", "b"])["v"].transform("mean")
    return means.rename("prior_model").round(4)


def attach_inferred_prior(cohort: pd.DataFrame, ext: pd.DataFrame, rt_flag: str = "rt_in_2p5",
                          edges: Sequence[float] = DEFAULT_BIN_EDGES) -> pd.DataFrame:
    """
    Copy of `cohort` (cohort_rt2p5 or cohort_rt5p0) with `prior_inferred` and its binned `prior_model`.

    The cohort tables are `cohort_extended` filtered by `rt_flag`, in the same order; the alignment is checked.
    The filter runs on ALL rows of `ext` (so trials outside this cohort's RT window still update the belief).
    """
    pri = inferred_prior(ext)
    sub = ext[ext[rt_flag].astype(bool)]
    if len(sub) != len(cohort):
        raise ValueError(f"cohort has {len(cohort)} rows but cohort_extended[{rt_flag}] has {len(sub)}")
    for col in ("subject_id", "contrast", "choice", "rt"):
        if not np.array_equal(sub[col].to_numpy(), cohort[col].to_numpy()):
            raise ValueError(f"cohort and cohort_extended[{rt_flag}] are not aligned (column {col})")
    out = cohort.copy()
    out["prior_inferred"] = pri.loc[sub.index].to_numpy()
    out["prior_model"] = bin_prior(out["prior_inferred"], out["subject_id"], edges).to_numpy()
    return out
