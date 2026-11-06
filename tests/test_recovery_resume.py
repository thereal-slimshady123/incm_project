"""The plan-scale recovery runs take hours, so they checkpoint every finished fit and can resume."""

import os

import pandas as pd
import pytest

from src.recovery.model_recovery import run_model_recovery
from src.recovery.param_recovery import run_parameter_recovery, run_tasks_resumable


def _worker(i, scale):
    if i == 0:
        raise AssertionError("task 0 was already checkpointed and must not be re-run")
    return {"sim_idx": i, "value": i * scale}


def test_run_tasks_resumable_skips_finished_tasks(tmp_path):
    partial = str(tmp_path / "p.csv")
    pd.DataFrame([{"sim_idx": 0, "value": 99}]).to_csv(partial, index=False)
    out = run_tasks_resumable([(i, 2) for i in range(4)], _worker, n_jobs=1, partial_path=partial)
    assert list(out["sim_idx"]) == [0, 1, 2, 3] and list(out["value"]) == [99, 2, 4, 6]
    assert len(pd.read_csv(partial)) == 4                     # every finished task was checkpointed


def test_parameter_recovery_is_deterministic_and_cleans_up_its_checkpoint(tmp_path):
    kw = dict(model_type="M2", n_iterations=2, trial_counts=[200], n_jobs=1, maxiter=5, n_sim_per_cond=40, seed=3)
    a = run_parameter_recovery(output_dir=str(tmp_path / "a"), **kw)
    # resume from a checkpoint holding the first task only: the second must reproduce the uninterrupted run
    b_dir = tmp_path / "b"
    os.makedirs(b_dir)
    a.iloc[:1].to_csv(b_dir / "param_recovery_M2.partial.csv", index=False)
    b = run_parameter_recovery(output_dir=str(b_dir), **kw)
    assert len(b) == 2 and b["fit_a"].round(8).tolist() == a["fit_a"].round(8).tolist()
    assert not (b_dir / "param_recovery_M2.partial.csv").exists() and (b_dir / "param_recovery_M2.csv").exists()


def test_model_recovery_writes_final_csv_without_checkpoint(tmp_path):
    df, conf = run_model_recovery(n_datasets_per_model=1, n_trials=200, n_jobs=1, maxiter=5, n_sim_per_cond=30,
                                  output_dir=str(tmp_path), seed=1)
    assert conf.sum() == 3 and (tmp_path / "model_recovery_bics.csv").exists()
    assert not (tmp_path / "model_recovery_bics.partial.csv").exists()
