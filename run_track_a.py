"""
Command-Line Runner for Track A: Computational Modeling & Identifiability Pipeline.
Author: Parth Dhodapkar (2024111009)
Course: Introduction to Neural and Cognitive Modeling (INCM)

Usage:
    python run_track_a.py --demo
    python run_track_a.py --full-param-recovery --trials 200 500 1000 2000 --n-sims 100
    python run_track_a.py --full-model-recovery --n-datasets 50
    python run_track_a.py --test
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.models.simulator import simulate_ddm
from src.optimization.fit import fit_ddm
from src.visualization.plots_recovery import (
    plot_parameter_recovery,
    plot_error_correlation_matrix,
    plot_model_recovery_confusion,
    plot_cbf,
    plot_psychometric_chronometric,
    compute_cbf_curve,
)
from src.recovery.param_recovery import run_parameter_recovery, sample_random_parameters
from src.recovery.model_recovery import run_model_recovery


def run_demo(output_dir: str = "results"):
    """
    Executes a fast, end-to-end demonstration of all Track A deliverables:
    1. Forward simulation & Psychometric/Chronometric curves.
    2. Qualitative diagnostic CBF curves (M1 falling, M2 flat, M3 rising).
    3. Mini parameter recovery benchmark with scatter & error correlation plots.
    4. Mini model recovery benchmark with confusion & inversion matrices.
    """
    fig_dir = os.path.join(output_dir, "figures")
    rec_dir = os.path.join(output_dir, "recovery")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(rec_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(" TRACK A: COMPUTATIONAL MODELING & RECOVERY PIPELINE DEMO ")
    print("=" * 70)
    t0 = time.time()

    # -------------------------------------------------------------
    # 1. Forward Simulation & Psychometrics / Chronometrics
    # -------------------------------------------------------------
    print("\n[Step 1/4] Generating Psychometric & Chronometric curves...")
    baseline_params = {"a": 1.15, "t0": 0.22, "k": 3.0, "v_b": 0.5, "v_h": 0.15}
    demo_df = simulate_ddm(baseline_params, model_type="M2", n_trials=3000, seed=42)

    psych_path = os.path.join(fig_dir, "demo_psychometric_chronometric.png")
    plot_psychometric_chronometric(demo_df, save_path=psych_path)
    print(f"  -> Saved psychometric/chronometric curves to: {psych_path}")

    # -------------------------------------------------------------
    # 2. Diagnostic Conditional Bias Functions (CBF; White & Poldrack, 2014)
    # -------------------------------------------------------------
    print("\n[Step 2/4] Generating Diagnostic Conditional Bias Functions (CBF)...")
    cbf_results = {}

    models_config = [
        ("M1 (Starting Point)", "M1", {"a": 1.1, "t0": 0.2, "k": 2.6, "z": 0.40}),
        ("M2 (Drift Bias)", "M2", {"a": 1.1, "t0": 0.2, "k": 2.6, "v_b": 0.55}),
        ("M3 (Dynamic Bias)", "M3", {"a": 1.1, "t0": 0.2, "k": 2.6, "b_max": 1.2, "tau_b": 0.5}),
    ]

    for label, m_type, m_params in models_config:
        sim_df = simulate_ddm(m_params, model_type=m_type, n_trials=5000, seed=101)
        centers, cbf_vals = compute_cbf_curve(sim_df, n_bins=5)
        cbf_results[label] = (centers, cbf_vals)
        print(f"  -> {label} quintile bias values: {[round(float(v), 3) for v in cbf_vals]}")

    cbf_path = os.path.join(fig_dir, "demo_conditional_bias_functions.png")
    plot_cbf(cbf_results, save_path=cbf_path, title="Diagnostic Conditional Bias Functions (M1 vs M2 vs M3)")
    print(f"  -> Saved diagnostic CBF comparison plot to: {cbf_path}")

    # -------------------------------------------------------------
    # 3. Mini Parameter Recovery Benchmark (Wilson & Collins Rule 4)
    # -------------------------------------------------------------
    print("\n[Step 3/4] Running Mini Parameter Recovery Benchmark (N=8 parameter sets)...")
    param_df = run_parameter_recovery(
        model_type="M2",
        n_iterations=8,
        trial_counts=[1000],
        n_jobs=-1,
        maxiter=100,
        n_sim_per_cond=250,
        output_dir=rec_dir,
        seed=42,
    )
    print(f"  -> Parameter recovery complete. Samples evaluated: {len(param_df)}")

    # -------------------------------------------------------------
    # 4. Mini Model Recovery Benchmark (Wilson & Collins Rule 5)
    # -------------------------------------------------------------
    print("\n[Step 4/4] Running Mini Model Recovery Benchmark (3 datasets per model x 3 models)...")
    model_df, conf_counts = run_model_recovery(
        n_datasets_per_model=3,
        n_trials=1000,
        n_jobs=-1,
        maxiter=80,
        n_sim_per_cond=250,
        output_dir=rec_dir,
        seed=202,
    )
    print(f"  -> Model recovery complete. Total fits evaluated: {len(model_df) * 3}")

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f" DEMO COMPLETED SUCCESSFULLY IN {elapsed:.1f} SECONDS ({elapsed/60:.2f} MIN) ")
    print(f" All publication figures saved to: {os.path.abspath(fig_dir)}")
    print(f" All CSV benchmarks saved to:      {os.path.abspath(rec_dir)}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Track A Computational Modeling & Recovery Benchmark Harness (Parth Dhodapkar)"
    )
    parser.add_argument("--demo", action="store_true", help="Run fast end-to-end demonstration")
    parser.add_argument("--full-param-recovery", action="store_true", help="Run full parameter recovery benchmark")
    parser.add_argument("--full-model-recovery", action="store_true", help="Run full model recovery benchmark")
    parser.add_argument("--test", action="store_true", help="Run pytest unit test suite")
    parser.add_argument("--n-sims", type=int, default=100, help="Number of synthetic parameter vectors")
    parser.add_argument("--trials", nargs="+", type=int, default=[200, 500, 1000, 2000], help="Trial counts to benchmark")
    parser.add_argument("--n-datasets", type=int, default=50, help="Datasets per model for model recovery")
    parser.add_argument("--model", type=str, default="M2", help="Model type ('M1', 'M2', 'M3', or 'JOINT')")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel CPU workers (-1 uses all cores)")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")

    args = parser.parse_args()

    if len(sys.argv) == 1 or args.demo:
        run_demo(output_dir=args.output_dir)
    elif args.test:
        import pytest
        sys.exit(pytest.main(["-v", "tests/"]))
    elif args.full_param_recovery:
        run_parameter_recovery(
            model_type=args.model,
            n_iterations=args.n_sims,
            trial_counts=args.trials,
            n_jobs=args.n_jobs,
            output_dir=os.path.join(args.output_dir, "recovery"),
        )
    elif args.full_model_recovery:
        run_model_recovery(
            n_datasets_per_model=args.n_datasets,
            n_jobs=args.n_jobs,
            output_dir=os.path.join(args.output_dir, "recovery"),
        )


if __name__ == "__main__":
    main()
