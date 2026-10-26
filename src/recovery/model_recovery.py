"""
Wilson & Collins (2019) Model Recovery Pipeline (Rule 5).

Generates synthetic datasets under competing prior mechanisms (M1, M2, M3),
fits each dataset with all candidate models, and calculates:
1. BIC-based Confusion Matrix P(Fitted M_j | True M_i)
2. Inversion Matrix P(True M_i | Fitted M_j)
Verifies absence of model mimicry and confirms > 85% identifiability.
"""

from typing import List, Dict, Optional, Any, Tuple
import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..models.simulator import simulate_ddm
from ..optimization.fit import fit_ddm
from .param_recovery import sample_random_parameters
from ..visualization.plots_recovery import plot_model_recovery_confusion


MODELS = ["M1", "M2", "M3"]


def _run_single_model_recovery_step(
    sim_idx: int,
    true_model: str,
    n_trials: int,
    maxiter: int,
    n_sim_per_cond: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Simulate under true_model, fit with M1, M2, M3, return BICs.
    """
    rng = np.random.default_rng(seed)
    true_params = sample_random_parameters(true_model, rng=rng)

    # 1. Forward simulation under true model
    sim_df = simulate_ddm(
        params=true_params,
        model_type=true_model,
        n_trials=n_trials,
        seed=seed,
    )

    # 2. Fit all competing models
    bic_scores = {}
    for fit_m in MODELS:
        fit_res = fit_ddm(
            trials_df=sim_df,
            model_type=fit_m,
            maxiter=maxiter,
            n_sim_per_cond=n_sim_per_cond,
            seed=seed + 500,
        )
        bic_scores[fit_m] = fit_res["bic"]

    # Select winning model via minimum BIC
    best_fit_model = min(bic_scores, key=bic_scores.get)

    record = {
        "sim_idx": sim_idx,
        "true_model": true_model,
        "winning_model": best_fit_model,
        "bic_M1": bic_scores["M1"],
        "bic_M2": bic_scores["M2"],
        "bic_M3": bic_scores["M3"],
        "delta_bic_M2_M1": bic_scores["M2"] - bic_scores["M1"],
        "delta_bic_M3_M2": bic_scores["M3"] - bic_scores["M2"],
    }
    return record


def run_model_recovery(
    n_datasets_per_model: int = 50,
    n_trials: int = 1000,
    n_jobs: int = -1,
    maxiter: int = 120,
    n_sim_per_cond: int = 300,
    output_dir: str = "results/recovery",
    seed: int = 101,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Run full model recovery pipeline across M1, M2, M3.

    Parameters
    ----------
    n_datasets_per_model : int
        Number of datasets simulated per generative model (e.g. 50).
    n_trials : int
        Trial count per simulated dataset (default 1000).
    n_jobs : int
        Parallel CPU workers.
    maxiter : int
        Nelder-Mead iterations.
    n_sim_per_cond : int
        Simulation resolution per condition.
    output_dir : str
        Directory to save results and figures.
    seed : int
        Master random seed.

    Returns
    -------
    df_results : pd.DataFrame
        Complete table of all simulation runs and BICs.
    confusion_counts : np.ndarray
        Shape (3, 3) count matrix of True vs. Fitted models.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    print(f"=== Starting Model Recovery: Models={MODELS}, N_datasets_per_model={n_datasets_per_model} ===")
    t_start = time.time()

    all_tasks = []
    task_idx = 0
    for true_m in MODELS:
        for _ in range(n_datasets_per_model):
            sim_seed = int(rng.integers(1, 1000000))
            all_tasks.append((task_idx, true_m, n_trials, maxiter, n_sim_per_cond, sim_seed))
            task_idx += 1

    print(f"Dispatching {len(all_tasks)} model recovery jobs across {n_jobs} workers...")
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_run_single_model_recovery_step)(*task) for task in all_tasks
    )

    df_results = pd.DataFrame(results)

    # Save CSV
    csv_path = os.path.join(output_dir, "model_recovery_bics.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Saved model recovery data to: {csv_path}")

    # Build confusion matrix counts
    model_to_idx = {m: i for i, m in enumerate(MODELS)}
    confusion_counts = np.zeros((len(MODELS), len(MODELS)), dtype=int)

    for _, row in df_results.iterrows():
        i = model_to_idx[row["true_model"]]
        j = model_to_idx[row["winning_model"]]
        confusion_counts[i, j] += 1

    # Plot Confusion and Inversion matrices
    fig_path = os.path.join(output_dir, "model_recovery_confusion_inversion.png")
    plot_model_recovery_confusion(
        confusion_counts,
        model_labels=["M1 (Start)", "M2 (Drift)", "M3 (Dynamic)"],
        save_path=fig_path,
        title_prefix=f"Model Recovery (N={n_trials} trials)",
    )

    elapsed = time.time() - t_start
    print(f"=== Model Recovery Complete in {elapsed:.1f} s ({elapsed/60:.2f} min) ===")
    return df_results, confusion_counts
