"""
Quantile Maximum Probability (QMP; Heathcote et al., 2002) Objective Function.

Divides reaction time distributions into quantiles {0.1, 0.3, 0.5, 0.7, 0.9}
per stimulus condition and response boundary, computing multinomial log-likelihood
and BIC for model adjudication.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


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
        """
        self.quantiles = np.array(quantiles)
        self.n_bins = len(quantiles) + 1
        self.min_trials = min_trials_per_choice
        self.total_trials = len(trials_df)

        # Build condition tables
        self.conditions: List[Dict[str, Any]] = []
        self._prepare_conditions(trials_df)

    def _prepare_conditions(self, df: pd.DataFrame):
        """
        Group trials by (contrast, prior) and compute empirical quantile bin edges.
        """
        grouped = df.groupby(["contrast", "prior"], observed=False)

        for (c, p), group in grouped:
            n_cond = len(group)
            if n_cond == 0:
                continue

            cond_dict: Dict[str, Any] = {
                "contrast": float(c),
                "prior": float(p),
                "n_trials": n_cond,
                "choices": {},
            }

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

                # Bin edges: [0, q1, q2, q3, q4, q5, inf]
                bin_edges = np.concatenate(([0.0], q_vals, [np.inf]))
                # Ensure strictly increasing edges to prevent histogram collapse
                for i in range(1, len(bin_edges) - 1):
                    if bin_edges[i] <= bin_edges[i - 1]:
                        bin_edges[i] = bin_edges[i - 1] + 1e-4

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
    total_nll = 0.0
    total_g2 = 0.0

    # Group simulated trials by (contrast, prior)
    sim_grouped = simulated_df.groupby(["contrast", "prior"], observed=False)

    for cond in qmp_data.conditions:
        c = cond["contrast"]
        p = cond["prior"]
        n_obs_cond = cond["n_trials"]

        # Retrieve simulated group
        key = (c, p)
        if key in sim_grouped.groups:
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
