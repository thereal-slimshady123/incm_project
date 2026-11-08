"""
Quantile Maximum Probability (QMP; Heathcote et al., 2002) Objective Function.

Divides reaction time distributions into quantiles {0.1, 0.3, 0.5, 0.7, 0.9}
per stimulus condition and response boundary, computing multinomial log-likelihood
and BIC for model adjudication.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.special import ndtr


class QMPDataset:
    """
    Prepared empirical dataset with precomputed RT quantile bin edges
    and observed frequency counts per condition and choice.
    """

    def __init__(
        self,
        trials_df: pd.DataFrame,
        quantiles: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
        min_trials_per_choice: int = 4,
        group_cols: Tuple[str, ...] = ("contrast", "prior"),
        window: Optional[Tuple[float, float]] = None,
    ):
        """
        Parameters
        ----------
        trials_df : pd.DataFrame
            Trial table containing 'contrast', 'prior', 'choice', 'rt'.
        quantiles : tuple of float
            Quantile cutoff probabilities (default 0.1, 0.3, 0.5, 0.7, 0.9).
        min_trials_per_choice : int
            Minimum trials to compute quantiles; fallback to uniform if fewer.
        group_cols : tuple of str
            Columns defining a stimulus condition. Add 'prev_choice' when fitting
            the 1-back history weight v_h, so the likelihood depends on it.
        window : (rt_lo, rt_hi), optional
            Truncation-aware mode (used for C2). The data were filtered to rt_lo <= RT <= rt_hi, so the
            model must predict the RT distribution CONDITIONAL on that window: bins become
            [rt_lo, q1, ..., qk, rt_hi] and `compute_qmp_nll_blocks` renormalises the simulated bin
            probabilities by the in-window mass of each condition. Without a window (the C1 behaviour),
            the bins are [0, q1, ..., qk, inf) and simulated RTs slower than the cutoff all land in the
            last bin, which forces a heavy-tailed model to squash its tail (see `src/optimization/c2.py`).
            Trials outside the window are ignored.
        """
        self.quantiles = np.array(quantiles)
        self.n_bins = len(quantiles) + 1
        self.min_trials = min_trials_per_choice
        self.group_cols = list(group_cols)
        self.window = None if window is None else (float(window[0]), float(window[1]))
        if self.window is not None:
            lo, hi = self.window
            if not 0.0 <= lo < hi:
                raise ValueError(f"invalid RT window {window}")
            trials_df = trials_df[(trials_df["rt"] >= lo) & (trials_df["rt"] <= hi)]
        self.total_trials = len(trials_df)

        # Build condition tables
        self.conditions: List[Dict[str, Any]] = []
        self._prepare_conditions(trials_df)

    def _prepare_conditions(self, df: pd.DataFrame):
        """
        Group trials by condition (group_cols) and compute empirical quantile bin edges.
        """
        grouped = df.groupby(self.group_cols, observed=False)

        for key, group in grouped:
            n_cond = len(group)
            if n_cond == 0:
                continue

            cond_dict: Dict[str, Any] = {
                "key": tuple(float(v) for v in key),
                "n_trials": n_cond,
                "choices": {},
            }
            for col, val in zip(self.group_cols, cond_dict["key"]):
                cond_dict[col] = val

            for choice_val in (-1, 1):
                sub = group[group["choice"] == choice_val]
                n_choice = len(sub)

                if n_choice == 0:
                    cond_dict["choices"][choice_val] = {
                        "n_choice": 0,
                        "bin_edges": None,
                        "observed_counts": np.zeros(self.n_bins, dtype=int),
                    }
                    continue

                rts = sub["rt"].values
                if n_choice >= self.min_trials:
                    q_vals = np.quantile(rts, self.quantiles)
                else:
                    # Fallback for very few trials
                    q_vals = np.linspace(np.min(rts), np.max(rts), len(self.quantiles))

                # Bin edges: [0, q1, q2, q3, q4, q5, inf]  (windowed: [rt_lo, q1, ..., q5, rt_hi])
                if self.window is None:
                    bin_edges = np.concatenate(([0.0], q_vals, [np.inf]))
                else:
                    bin_edges = np.concatenate(([self.window[0]], q_vals, [self.window[1]]))
                # Ensure strictly increasing edges to prevent histogram collapse
                for i in range(1, len(bin_edges) - 1):
                    if bin_edges[i] <= bin_edges[i - 1]:
                        bin_edges[i] = bin_edges[i - 1] + 1e-4
                if self.window is not None and bin_edges[-1] <= bin_edges[-2]:
                    bin_edges[-1] = bin_edges[-2] + 1e-4

                # Observed counts in each quantile bin
                obs_counts, _ = np.histogram(rts, bins=bin_edges)

                cond_dict["choices"][choice_val] = {
                    "n_choice": n_choice,
                    "bin_edges": bin_edges,
                    "observed_counts": obs_counts,
                }

            self.conditions.append(cond_dict)


def compute_qmp_nll(
    simulated_df: pd.DataFrame,
    qmp_data: QMPDataset,
    eps: float = 1e-5,
) -> Tuple[float, float]:
    """
    Compute Quantile Maximum Probability Negative Log-Likelihood (NLL) and G^2.

    Parameters
    ----------
    simulated_df : pd.DataFrame
        Simulated trials from DDM forward pass.
    qmp_data : QMPDataset
        Target dataset with empirical bin edges and observed counts.
    eps : float
        Numerical epsilon to prevent log(0).

    Returns
    -------
    nll : float
        Multinomial negative log-likelihood.
    g_squared : float
        Neyman / Pearson G^2 statistic.
    """
    if qmp_data.window is not None:
        raise NotImplementedError("compute_qmp_nll does not support windowed datasets; use compute_qmp_nll_blocks")
    total_nll = 0.0
    total_g2 = 0.0

    # Group simulated trials by the same condition columns as the target data
    sim_grouped = simulated_df.groupby(qmp_data.group_cols, observed=False)
    sim_keys = {tuple(float(v) for v in k): k for k in sim_grouped.groups}

    for cond in qmp_data.conditions:
        n_obs_cond = cond["n_trials"]

        # Retrieve simulated group
        key = sim_keys.get(cond["key"])
        if key is not None:
            sim_sub = sim_grouped.get_group(key)
            n_sim_cond = len(sim_sub)
        else:
            sim_sub = None
            n_sim_cond = 0

        for choice_val in (-1, 1):
            target_info = cond["choices"][choice_val]
            obs_counts = target_info["observed_counts"]
            n_obs_choice = target_info["n_choice"]

            if n_obs_choice == 0:
                continue

            bin_edges = target_info["bin_edges"]

            if sim_sub is not None and n_sim_cond > 0:
                sim_choice_sub = sim_sub[sim_sub["choice"] == choice_val]
                if len(sim_choice_sub) > 0:
                    sim_rts = sim_choice_sub["rt"].values
                    sim_counts, _ = np.histogram(sim_rts, bins=bin_edges)
                    # Probability of choosing choice_val in bin i within condition
                    pred_prob = sim_counts / float(n_sim_cond)
                else:
                    pred_prob = np.zeros(len(obs_counts))
            else:
                pred_prob = np.zeros(len(obs_counts))

            # Apply epsilon floor and re-normalize condition probabilities safely
            pred_prob = np.maximum(pred_prob, eps)

            # NLL contribution: - sum( O_i * ln(P_i) )
            total_nll -= np.sum(obs_counts * np.log(pred_prob))

            # Expected counts: E_i = N_obs * P_i
            expected_counts = n_obs_cond * pred_prob
            nonzero = obs_counts > 0
            if np.any(nonzero):
                total_g2 += 2.0 * np.sum(
                    obs_counts[nonzero] * np.log(obs_counts[nonzero] / expected_counts[nonzero])
                )

    return float(total_nll), float(total_g2)


class Lapse:
    """
    Lapse / contaminant mixture applied analytically to the QMP bin probabilities.

    With probability `rate` a trial is not a diffusion decision: its choice is Right with probability
    `p_right[i]` (one value per condition i of the dataset, so a lapse may depend on the block prior) and its
    RT is uniform on `rt_range`. The predicted probability of (choice, RT bin) in condition i is then
        (1 - rate) * P_ddm(choice, bin) + rate * P_lapse(choice) * |bin ∩ rt_range| / |rt_range|.
    Because the uniform covers both fast and slow RTs, the same component also absorbs fast guesses and
    slow disengaged responses that no diffusion process produces.

    `extra` adds further components of the same form, as (rate, p_right, rt_range) tuples, e.g. a FAST-GUESS
    component (anticipatory responses: RT uniform on a short window above the RT cutoff, choice biased by the
    prior). The diffusion weight is then 1 - rate - sum of the extra rates.
    """

    def __init__(self, rate: float, p_right: np.ndarray, rt_range: Tuple[float, float], extra=None):
        self.components = []
        for r, pr, rng in [(rate, p_right, rt_range)] + list(extra or []):
            lo, hi = float(rng[0]), float(rng[1])
            if not hi > lo:
                raise ValueError(f"invalid lapse RT range {rng}")
            self.components.append((float(r), np.asarray(pr, dtype=float), lo, hi))
        self.rate = sum(c[0] for c in self.components)
        if self.rate > 1.0:
            raise ValueError("mixture rates sum to more than 1")
        self.p_right, self.lo, self.hi = self.components[0][1], self.components[0][2], self.components[0][3]

    def mix(self, pred_prob: np.ndarray, cond_idx: int, choice_val: int, bin_edges: np.ndarray) -> np.ndarray:
        if self.rate <= 0.0:
            return pred_prob
        out = (1.0 - self.rate) * pred_prob
        for r, pr, lo, hi in self.components:
            if r <= 0.0:
                continue
            width = np.clip(np.minimum(bin_edges[1:], hi) - np.maximum(bin_edges[:-1], lo), 0.0, None)
            p_choice = pr[cond_idx] if choice_val > 0 else 1.0 - pr[cond_idx]
            out = out + r * p_choice * width / (hi - lo)
        return out


def compute_qmp_nll_blocks(
    sim_choices: np.ndarray,
    sim_rts: np.ndarray,
    qmp_data: QMPDataset,
    n_sim_per_cond: int,
    eps: float = 1e-5,
    bandwidth: Optional[float] = None,
    lapse: Optional[Lapse] = None,
) -> float:
    """
    Fast QMP negative log-likelihood for simulations laid out in condition blocks.

    Equivalent to compute_qmp_nll when simulated trial j belongs to condition
    j // n_sim_per_cond, in the order of qmp_data.conditions (the layout built by
    DDMFitter). Avoids a pandas groupby on every objective evaluation.

    bandwidth : float, optional
        If given, each simulated RT is spread over the bins with a Gaussian kernel of
        this width (seconds) instead of being counted in exactly one bin. Under CRN a
        hard count is a step function of the parameters (nothing changes until an RT
        crosses a bin edge), which stalls Nelder-Mead on flat steps; the kernel makes
        the objective continuous. Keep it small relative to the bin widths.
    lapse : Lapse, optional
        Mixes a lapse / contaminant component into every bin probability (see `Lapse`). None = pure DDM.
    """
    if qmp_data.window is not None:
        return _qmp_nll_blocks_windowed(sim_choices, sim_rts, qmp_data, n_sim_per_cond, eps, bandwidth, lapse)
    if bandwidth:
        return _qmp_nll_blocks_smoothed(sim_choices, sim_rts, qmp_data, n_sim_per_cond, eps, bandwidth, lapse)

    total_nll = 0.0
    for i, cond in enumerate(qmp_data.conditions):
        block = slice(i * n_sim_per_cond, (i + 1) * n_sim_per_cond)
        block_choices = sim_choices[block]
        block_rts = sim_rts[block]

        for choice_val in (-1, 1):
            target_info = cond["choices"][choice_val]
            if target_info["n_choice"] == 0:
                continue

            rts = block_rts[block_choices == choice_val]
            # Bin index per simulated RT; edges are [0, q1..q5, inf] so bins are 1..6
            bins = np.searchsorted(target_info["bin_edges"], rts, side="right") - 1
            sim_counts = np.bincount(bins, minlength=qmp_data.n_bins)[: qmp_data.n_bins]
            pred_prob = sim_counts / float(n_sim_per_cond)
            if lapse is not None:
                pred_prob = lapse.mix(pred_prob, i, choice_val, target_info["bin_edges"])
            pred_prob = np.maximum(pred_prob, eps)
            total_nll -= np.sum(target_info["observed_counts"] * np.log(pred_prob))

    return float(total_nll)


def _qmp_nll_blocks_smoothed(
    sim_choices: np.ndarray,
    sim_rts: np.ndarray,
    qmp_data: QMPDataset,
    n_sim_per_cond: int,
    eps: float,
    bandwidth: float,
    lapse: Optional[Lapse] = None,
) -> float:
    """Kernel-smoothed bin probabilities: P(bin k) = mean_j [Phi((e_k+1 - rt_j)/h) - Phi((e_k - rt_j)/h)]."""
    total_nll = 0.0
    for i, cond in enumerate(qmp_data.conditions):
        block = slice(i * n_sim_per_cond, (i + 1) * n_sim_per_cond)
        block_choices = sim_choices[block]
        block_rts = sim_rts[block]

        for choice_val in (-1, 1):
            target_info = cond["choices"][choice_val]
            if target_info["n_choice"] == 0:
                continue

            rts = block_rts[block_choices == choice_val]
            if len(rts) == 0:
                pred_prob = np.zeros(qmp_data.n_bins)
            else:
                # CDF of every simulated RT at every bin edge; edges are [0, q1..q5, inf]
                cdf = ndtr((target_info["bin_edges"][None, :] - rts[:, None]) / bandwidth)
                bin_mass = np.diff(cdf, axis=1).sum(axis=0)
                pred_prob = bin_mass / float(n_sim_per_cond)
            if lapse is not None:
                pred_prob = lapse.mix(pred_prob, i, choice_val, target_info["bin_edges"])
            pred_prob = np.maximum(pred_prob, eps)
            total_nll -= np.sum(target_info["observed_counts"] * np.log(pred_prob))

    return float(total_nll)


def _qmp_nll_blocks_windowed(
    sim_choices: np.ndarray,
    sim_rts: np.ndarray,
    qmp_data: QMPDataset,
    n_sim_per_cond: int,
    eps: float,
    bandwidth: Optional[float],
    lapse: Optional[Lapse] = None,
) -> float:
    """
    Truncation-aware QMP NLL for a dataset built with `window=(rt_lo, rt_hi)`.

    The observed counts are conditional on rt_lo <= RT <= rt_hi, so the model probability of
    (choice, bin) is its simulated mass in that bin DIVIDED BY the simulated mass of the whole window
    (both choices) in the same condition: P(choice, bin | RT in window). Simulated RTs outside the
    window (anticipations below rt_lo, slow responses and timeouts above rt_hi) are removed from both
    numerator and denominator, exactly as the data filter removed them. With a `bandwidth`, each
    simulated RT is spread over the bins with a Gaussian kernel as in the unwindowed smoothed version.
    """
    lo, hi = qmp_data.window
    total_nll = 0.0
    for i, cond in enumerate(qmp_data.conditions):
        block = slice(i * n_sim_per_cond, (i + 1) * n_sim_per_cond)
        block_choices = sim_choices[block]
        block_rts = sim_rts[block]

        masses = {}
        window_mass = 0.0
        for choice_val in (-1, 1):
            rts = block_rts[block_choices == choice_val]
            info = cond["choices"][choice_val]
            if bandwidth:
                window_mass += float(np.sum(ndtr((hi - rts) / bandwidth) - ndtr((lo - rts) / bandwidth)))
                if info["n_choice"] > 0 and len(rts):
                    cdf = ndtr((info["bin_edges"][None, :] - rts[:, None]) / bandwidth)
                    masses[choice_val] = np.diff(cdf, axis=1).sum(axis=0)
            else:
                inside = rts[(rts >= lo) & (rts <= hi)]
                window_mass += float(len(inside))
                if info["n_choice"] > 0 and len(inside):
                    masses[choice_val] = np.histogram(inside, bins=info["bin_edges"])[0].astype(float)

        denom = max(window_mass, 1e-9)
        for choice_val in (-1, 1):
            info = cond["choices"][choice_val]
            if info["n_choice"] == 0:
                continue
            mass = masses.get(choice_val, np.zeros(qmp_data.n_bins))
            pred_prob = mass / denom
            if lapse is not None:      # lapse RTs are uniform inside the window, so the mixture stays conditional on it
                pred_prob = lapse.mix(pred_prob, i, choice_val, info["bin_edges"])
            pred_prob = np.maximum(pred_prob, eps)
            total_nll -= np.sum(info["observed_counts"] * np.log(pred_prob))
    return float(total_nll)


def compute_bic(nll: float, n_params: int, n_trials: int) -> float:
    """
    Bayesian Information Criterion (BIC):
        BIC = 2 * NLL + k * ln(N)

    Parameters
    ----------
    nll : float
        Negative log-likelihood.
    n_params : int
        Number of free model parameters (k).
    n_trials : int
        Total number of observed trials (N).

    Returns
    -------
    float
        BIC score (lower is better).
    """
    if n_trials <= 0:
        return np.inf
    return 2.0 * nll + n_params * np.log(n_trials)


def compute_aic(nll: float, n_params: int) -> float:
    """
    Akaike Information Criterion (AIC):
        AIC = 2 * NLL + 2 * k
    """
    return 2.0 * nll + 2.0 * n_params
