"""
Wilson & Collins (2019) Parameter Recovery Pipeline (Rule 4).

Generates synthetic datasets with known ground-truth parameters, fits them
via QMP Nelder-Mead, and quantifies:
1. Pearson recovery correlation (r > 0.8)
2. Root-Mean-Square Error (RMSE)
3. Linear recovery slopes (beta_1 ~ 1.0)
4. Parameter error correlation matrix Corr(Delta theta_i, Delta theta_j)
   to prove absence of parameter trade-offs / cross-talk.
"""

from typing import Dict, List, Optional
import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..models.simulator import simulate_ddm
from ..optimization.fit import fit_ddm
from ..visualization.plots_recovery import plot_parameter_recovery, plot_error_correlation_matrix


def sample_random_parameters(
    model_type: str = "M2",
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Sample physiologically plausible ground truth parameters uniformly from bounded ranges.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Core DDM parameters
    params = {
        "a": float(rng.uniform(0.8, 1.8)),
        "t0": float(rng.uniform(0.15, 0.35)),
        "k": float(rng.uniform(1.8, 4.5)),
        "v_h": float(rng.uniform(-0.6, 0.6)),
    }

    model_key = model_type.upper()
    if model_key == "M1":
        # Starting point bias
        params["z"] = float(rng.uniform(-0.5, 0.5))
    elif model_key == "M2":
        # Drift rate prior bias
        params["v_b"] = float(rng.uniform(-1.2, 1.2))
    elif model_key == "M3":
        # Dynamic time-growing bias
        params["b_max"] = float(rng.uniform(0.2, 1.5))
        params["tau_b"] = float(rng.uniform(0.15, 0.8))
    elif model_key == "JOINT":
        # Joint parameter vector to test z vs v_b trade-off directly!
        params["z"] = float(rng.uniform(-0.4, 0.4))
        params["v_b"] = float(rng.uniform(-1.0, 1.0))
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'")

    return params


def _run_single_recovery_step(
    sim_idx: int,
    true_params: Dict[str, float],
    model_type: str,
    n_trials: int,
    maxiter: int,
    n_sim_per_cond: int,
    seed: int,
) -> Dict[str, float]:
    """
    Single simulation + fitting step.
    """
    # 1. Forward simulation with ground-truth
    sim_df = simulate_ddm(
        params=true_params,
        model_type=model_type,
        n_trials=n_trials,
        seed=seed,
    )

    # 2. Fit model
    fit_res = fit_ddm(
        trials_df=sim_df,
        model_type=model_type,
        maxiter=maxiter,
        n_sim_per_cond=n_sim_per_cond,
        seed=seed + 1000,
    )

    record = {"sim_idx": sim_idx, "n_trials": n_trials, "success": fit_res["success"], "bic": fit_res["bic"]}

    for p, val in true_params.items():
        record[f"true_{p}"] = val
    for p, val in fit_res["params"].items():
        record[f"fit_{p}"] = val

    return record


def run_parameter_recovery(
    model_type: str = "M2",
    n_iterations: int = 100,
    trial_counts: List[int] = (1000,),
    n_jobs: int = -1,
    maxiter: int = 120,
    n_sim_per_cond: int = 300,
    output_dir: str = "results/recovery",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Execute full parameter recovery pipeline across specified trial counts.

    Parameters
    ----------
    model_type : str
        'M1', 'M2', 'M3', or 'JOINT'.
    n_iterations : int
        Number of synthetic parameter vectors (e.g. 100).
    trial_counts : list of int
        Sample sizes to benchmark, e.g. [200, 500, 1000, 2000].
    n_jobs : int
        Number of parallel CPU workers (-1 uses all available cores).
    maxiter : int
        Nelder-Mead max iterations.
    n_sim_per_cond : int
        Simulation resolution per condition.
    output_dir : str
        Directory to save results CSV and figures.
    seed : int
        Master random seed.

    Returns
    -------
    pd.DataFrame
        Complete table of true and recovered parameters.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    print(f"=== Starting Parameter Recovery: Model={model_type}, N_sims={n_iterations}, Trials={trial_counts} ===")
    t_start = time.time()

    all_tasks = []
    task_idx = 0
    for n_trials in trial_counts:
        for it in range(n_iterations):
            p_true = sample_random_parameters(model_type, rng=rng)
            sim_seed = int(rng.integers(1, 1000000))
            all_tasks.append((task_idx, p_true, model_type, n_trials, maxiter, n_sim_per_cond, sim_seed))
            task_idx += 1

    print(f"Dispatching {len(all_tasks)} optimization jobs across {n_jobs} workers...")
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_run_single_recovery_step)(*task) for task in all_tasks
    )

    df_results = pd.DataFrame(results)

    # Save CSV
    csv_path = os.path.join(output_dir, f"param_recovery_{model_type}.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Saved parameter recovery data to: {csv_path}")

    # Generate plots for each trial count
    for n_trials in trial_counts:
        sub = df_results[df_results["n_trials"] == n_trials]
        fig_path = os.path.join(output_dir, f"param_recovery_{model_type}_N{n_trials}.png")
        plot_parameter_recovery(
            sub,
            save_path=fig_path,
            title=f"Wilson & Collins Parameter Recovery: Model {model_type} (N={n_trials} trials)",
        )

        corr_fig_path = os.path.join(output_dir, f"error_corr_{model_type}_N{n_trials}.png")
        plot_error_correlation_matrix(
            sub,
            save_path=corr_fig_path,
            title=f"Error Correlation Matrix: Model {model_type} (N={n_trials} trials)",
        )

    elapsed = time.time() - t_start
    print(f"=== Parameter Recovery Complete in {elapsed:.1f} s ({elapsed/60:.2f} min) ===")
    return df_results
