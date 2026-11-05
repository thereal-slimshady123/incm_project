"""
Synthetic mock-data generator with KNOWN ground truth for the Track B test harness.
Track B (Joshua Koilpillai).

Independent of Track A's DDM simulator on purpose (plan §1: Track B needs 0 simulation-engine
code): trials follow the IBL block structure and the §3.1 schema exactly, but choices come from
a lapse-corrected logistic whose parameters, block-prior bias and *RT-dependence of the bias*
are set by the caller. The harness can therefore check that the empirical tools recover the
truth, and that the CBF shape classifier labels falling / flat / rising bias correctly.

Generative model (per trial):
    rt        = 0.1 + LogNormal(m - 0.5 * |c|^0.5, rt_sd)            (faster for stronger stimuli)
    u         = within-contrast RT percentile of that draw              (0 = fast, 1 = slow)
    w(u)      = 1 + a * (1 - 2u),  a = +bias_modulation (falling), 0 (flat), -bias_modulation (rising)
    logit(P_R)= (c - mu) / sigma + prior_effect * w(u) * prior
    P(Right)  = gamma + (1 - gamma - lam) * expit(logit(P_R))
For `cbf_shape='flat'` the choice model is an exact lapse-corrected logistic with
mu_block = mu - sigma * prior_effect * prior.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

CONTRAST_SET = np.array([0.0, 0.0625, 0.125, 0.25, 1.0])
_MODULATION = {"falling": +1.0, "flat": 0.0, "rising": -1.0}


def make_mock_trials(
    n_trials: int = 6000,
    subject_id: str = "mock_001",
    mu: float = 0.0,
    sigma: float = 0.12,
    gamma: float = 0.05,
    lam: float = 0.05,
    prior_effect: float = 1.0,
    cbf_shape: str = "flat",
    bias_modulation: float = 0.8,
    rt_log_mean: float = -0.9,
    rt_log_sd: float = 0.5,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a §3.1 trial table with known psychometric / bias parameters.

    Parameters
    ----------
    prior_effect : float
        Strength of the prior in log-odds units (0 = no prior effect).
    cbf_shape : {'falling', 'flat', 'rising'}
        How the prior's influence changes with RT percentile (bias_modulation sets the depth).
    The ground truth is stored in `df.attrs['truth']`.
    """
    if cbf_shape not in _MODULATION:
        raise ValueError(f"cbf_shape must be one of {sorted(_MODULATION)}")
    rng = np.random.default_rng(seed)

    # IBL-like blocks: lengths 20-100, alternating 80/20 and 20/80 (prior +1 = right-biased)
    priors = []
    block_prior = rng.choice([-1.0, 1.0])
    while len(priors) < n_trials:
        priors.extend([block_prior] * int(rng.integers(20, 101)))
        block_prior = -block_prior
    priors = np.asarray(priors[:n_trials])

    # Stimulus side follows the block (80% on the likelier side); contrast uniform over the set
    side = np.where(rng.random(n_trials) < 0.8, priors, -priors)
    contrast = side * rng.choice(CONTRAST_SET, size=n_trials)
    contrast = np.where(contrast == 0, 0.0, contrast)  # remove -0.0

    # RT first (choice depends on RT percentile, never the reverse)
    loc = rt_log_mean - 0.5 * np.sqrt(np.abs(contrast))
    z = rng.standard_normal(n_trials)
    rt = np.clip(0.1 + np.exp(loc + rt_log_sd * z), 0.1, 2.5)
    u = norm.cdf(z)  # RT percentile within contrast

    w = 1.0 + _MODULATION[cbf_shape] * bias_modulation * (1.0 - 2.0 * u)
    logit = (contrast - mu) / sigma + prior_effect * w * priors
    p_right = gamma + (1.0 - gamma - lam) * expit(logit)
    choice = np.where(rng.random(n_trials) < p_right, 1, -1).astype(np.int8)

    prev_choice = np.concatenate(([0], choice[:-1])).astype(np.int8)
    df = pd.DataFrame({
        "subject_id": subject_id,
        "trial_idx": np.arange(n_trials, dtype=np.int32),
        "contrast": contrast.astype(float),
        "prior": priors.astype(float),
        "prev_choice": prev_choice,
        "choice": choice,
        "rt": rt.astype(float),
    })
    df.attrs["truth"] = {
        "mu": mu, "sigma": sigma, "gamma": gamma, "lam": lam, "prior_effect": prior_effect,
        "cbf_shape": cbf_shape, "bias_modulation": bias_modulation,
        # PSE of each block for the flat model; shift = mu_left_block - mu_right_block
        "mu_right_block": mu - sigma * prior_effect, "mu_left_block": mu + sigma * prior_effect,
        "shift": 2.0 * sigma * prior_effect,
    }
    return df


def make_mock_cohort(
    n_subjects: int = 10,
    n_trials: int = 6000,
    cbf_shape: str = "flat",
    seed: int = 0,
    **kwargs,
) -> pd.DataFrame:
    """Several mock mice with heterogeneous parameters (random sigma, lapses, prior strength)."""
    rng = np.random.default_rng(seed)
    frames, truths = [], []
    for i in range(n_subjects):
        frames.append(make_mock_trials(
            n_trials=n_trials,
            subject_id=f"mock_{i + 1:03d}",
            sigma=float(rng.uniform(0.08, 0.2)),
            gamma=float(rng.uniform(0.0, 0.1)),
            lam=float(rng.uniform(0.0, 0.1)),
            prior_effect=float(rng.uniform(0.6, 1.6)),
            cbf_shape=cbf_shape,
            seed=int(rng.integers(0, 2 ** 31 - 1)),
            **kwargs,
        ))
        truths.append({"subject_id": f"mock_{i + 1:03d}", **frames[-1].attrs["truth"]})
    cohort = pd.concat(frames, ignore_index=True)
    cohort.attrs["truth"] = pd.DataFrame(truths)
    return cohort
