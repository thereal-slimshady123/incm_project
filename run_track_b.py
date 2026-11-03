"""
Command-Line Runner for Track B: IBL Data Ingestion & Behavioural Preprocessing.
Author: Joshua Koilpillai (2024101128)
Course: Introduction to Neural and Cognitive Modeling (INCM)

Usage:
    python run_track_b.py --all                 # list -> download + QC -> select -> process
    python run_track_b.py --qc                  # list candidate sessions, download all, compute QC
    python run_track_b.py --select              # select the 10 mice from cached QC
    python run_track_b.py --process             # build data/processed/ for the selected mice
    python run_track_b.py --test                # run pytest suite
"""

import argparse
import os
import sys
import time

import pandas as pd

from src.data.ibl_loader import (
    evaluate_sessions,
    get_one,
    list_candidate_sessions,
    load_selected_sessions,
    select_mice,
)
from src.data.preprocess import (
    CONTRACT_COLUMNS,
    RT_MAX_EXTENDED,
    RT_MAX_STANDARD,
    build_subject_table,
    parse_session_trials,
    validate_contract,
)
from src.data.summary import summarize_cohort


def _paths(data_dir: str) -> dict:
    return {
        "raw": os.path.join(data_dir, "raw"),
        "processed": os.path.join(data_dir, "processed"),
        "qc": os.path.join(data_dir, "raw", "session_qc.csv"),
        "candidates": os.path.join(data_dir, "raw", "candidate_sessions.csv"),
        "mice": os.path.join(data_dir, "processed", "mice_selection.csv"),
        "manifest": os.path.join(data_dir, "processed", "session_manifest.csv"),
    }


def step_qc(data_dir: str, cache_dir: str, n_workers: int) -> pd.DataFrame:
    """Discover candidate sessions and download + QC every session of mice with >= 5 sessions."""
    p = _paths(data_dir)
    os.makedirs(p["raw"], exist_ok=True)
    one = get_one(cache_dir)

    print("\n[Step 1/4] Listing standard biased-protocol sessions on openalyx...")
    cand = list_candidate_sessions(one)
    cand.to_csv(p["candidates"], index=False)
    print(f"  -> {len(cand)} sessions, {cand['subject'].nunique()} mice, {cand['lab'].nunique()} labs")

    n_per_mouse = cand.groupby("subject")["eid"].transform("count")
    cand = cand[n_per_mouse >= 5]
    print(f"  -> {len(cand)} sessions from {cand['subject'].nunique()} mice with >= 5 sessions kept for QC")

    print("\n[Step 2/4] Downloading trial tables and computing session QC...")
    qc = evaluate_sessions(one, cand, qc_path=p["qc"], n_workers=n_workers)
    print(f"  -> status counts: {qc['status'].value_counts().to_dict()}")
    return qc


def step_select(data_dir: str, n_mice: int):
    p = _paths(data_dir)
    os.makedirs(p["processed"], exist_ok=True)
    cand = pd.read_csv(p["candidates"])
    qc = pd.read_csv(p["qc"])
    merged = cand.merge(qc, on="eid", how="inner")

    print("\n[Step 3/4] Selecting mice...")
    mice, sessions = select_mice(merged, n_mice=n_mice)
    mice.to_csv(p["mice"], index=False)
    print(f"  -> {len(mice)} mice qualify; selected {int(mice['selected'].sum())}:")
    print(mice[mice["selected"]].to_string(index=False))
    return mice, sessions


def step_process(data_dir: str, cache_dir: str, n_mice: int):
    p = _paths(data_dir)
    mice, sessions = step_select(data_dir, n_mice)
    subjects = mice.loc[mice["selected"], "subject"].tolist()
    labs = dict(zip(mice["subject"], mice["lab"]))

    print("\n[Step 4/4] Building processed cohort files...")
    one = get_one(cache_dir)
    loaded = load_selected_sessions(one, sessions, subjects)

    manifest_rows = []
    parsed: dict = {}
    for subj, items in loaded.items():
        parsed[subj] = []
        for meta, raw in items:
            parsed[subj].append(
                parse_session_trials(raw, subj, eid=meta["eid"], date=meta["date"], lab=meta["lab"])
            )
            manifest_rows.append({
                "subject_id": subj, "lab": meta["lab"], "session_eid": meta["eid"],
                "date": meta["date"], "number": meta["number"],
                "task_protocol": meta["task_protocol"], "n_raw_trials": len(raw),
                "easy_accuracy": meta["easy_accuracy"],
            })
    pd.DataFrame(manifest_rows).to_csv(p["manifest"], index=False)

    summaries = []
    extended_all = None
    for tag, rt_max in (("2p5", RT_MAX_STANDARD), ("5p0", RT_MAX_EXTENDED)):
        out_dir = os.path.join(p["processed"], f"rt_{tag}")
        os.makedirs(out_dir, exist_ok=True)
        tables, counts = [], {}
        for subj in subjects:
            ext, ex = build_subject_table(parsed[subj], rt_max=rt_max)
            counts[subj] = ex
            core = ext[CONTRACT_COLUMNS]
            validate_contract(core)
            core.to_parquet(os.path.join(out_dir, f"{subj}.parquet"), index=False)
            core.to_csv(os.path.join(out_dir, f"{subj}.csv"), index=False)
            tables.append(ext)
        cohort_ext = pd.concat(tables, ignore_index=True)
        cohort = cohort_ext[CONTRACT_COLUMNS]
        validate_contract(cohort)
        cohort.to_parquet(os.path.join(p["processed"], f"cohort_rt{tag}.parquet"), index=False)
        cohort.to_csv(os.path.join(p["processed"], f"cohort_rt{tag}.csv"), index=False)
        n_sess = {s: len(parsed[s]) for s in subjects}
        summaries.append(summarize_cohort(cohort, counts, n_sess, labs, rt_max))
        if tag == "5p0":
            extended_all = cohort_ext
        print(f"  -> rt <= {rt_max}s: {len(cohort)} trials, {cohort['subject_id'].nunique()} mice")

    extended_all.to_parquet(os.path.join(p["processed"], "cohort_extended.parquet"), index=False)
    extended_all.to_csv(os.path.join(p["processed"], "cohort_extended.csv"), index=False)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(os.path.join(p["processed"], "summary_stats.csv"), index=False)
    print(f"  -> summary statistics: {os.path.join(p['processed'], 'summary_stats.csv')}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Track B IBL Data Pipeline (Joshua Koilpillai)")
    parser.add_argument("--all", action="store_true", help="Run the full pipeline")
    parser.add_argument("--qc", action="store_true", help="Download all candidate sessions and compute QC")
    parser.add_argument("--select", action="store_true", help="Select mice from cached QC")
    parser.add_argument("--process", action="store_true", help="Select mice and build data/processed/")
    parser.add_argument("--test", action="store_true", help="Run pytest unit test suite")
    parser.add_argument("--n-mice", type=int, default=10, help="Number of mice to select")
    parser.add_argument("--n-workers", type=int, default=6, help="Parallel download threads")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--cache-dir", type=str, default=None, help="ONE cache dir (default data/raw/ONE)")
    args = parser.parse_args()
    cache_dir = args.cache_dir or os.path.join(args.data_dir, "raw", "ONE")
    cache_dir = os.path.abspath(cache_dir)

    t0 = time.time()
    if args.test:
        import pytest
        sys.exit(pytest.main(["-v", "tests/"]))
    elif args.qc:
        step_qc(args.data_dir, cache_dir, args.n_workers)
    elif args.select:
        step_select(args.data_dir, args.n_mice)
    elif args.process:
        step_process(args.data_dir, cache_dir, args.n_mice)
    else:  # --all or no arguments
        step_qc(args.data_dir, cache_dir, args.n_workers)
        step_process(args.data_dir, cache_dir, args.n_mice)
    print(f"\nDone in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
