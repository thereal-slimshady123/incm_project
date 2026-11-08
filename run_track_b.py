"""
Command-Line Runner for Track B: IBL Data Ingestion & Behavioural Preprocessing.
Author: Joshua Koilpillai (2024101128)
Course: Introduction to Neural and Cognitive Modeling (INCM)

Usage:
    python run_track_b.py --all                 # list -> download + QC -> select -> process
    python run_track_b.py --qc                  # list candidate sessions, download all, compute QC
    python run_track_b.py --select              # select the 10 mice from cached QC (keeps an existing cohort)
    python run_track_b.py --refresh-qc          # recompute session QC from the local cache (offline)
    python run_track_b.py --process             # build data/processed/ for the selected mice
    python run_track_b.py --empirical           # execute the Week 2 empirical baseline notebook
    python run_track_b.py --test                # run pytest suite
    python run_track_b.py --replication         # replication cohort: next 10 ranked mice -> data/processed/replication/

Add --reselect to rebuild the cohort ranking from scratch instead of keeping the cohort already
stored in data/processed/mice_selection.csv, and --rt-source first_movement for the alternative
RT definition (default: response time, see src/data/preprocess.py).
"""

import argparse
import os
import shutil
import sys
import time

import pandas as pd

from src.data.ibl_loader import (
    evaluate_sessions,
    get_one,
    list_candidate_sessions,
    load_selected_sessions,
    refresh_qc_from_cache,
    select_mice,
)
from src.data.preprocess import (
    CONTRACT_COLUMNS,
    DEFAULT_RT_SOURCE,
    RT_SOURCES,
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
        # small, version-controlled copies so a fresh clone can re-select without re-downloading
        "qc_tracked": os.path.join(data_dir, "processed", "session_qc.csv"),
        "candidates_tracked": os.path.join(data_dir, "processed", "candidate_sessions.csv"),
        "manifest": os.path.join(data_dir, "processed", "session_manifest.csv"),
    }


def _read_cached(p: dict, key: str) -> pd.DataFrame:
    """Read the raw QC / candidate table, falling back to the tracked copy in data/processed/."""
    path = p[key] if os.path.exists(p[key]) else p[key + "_tracked"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"{p[key]} not found; run `python run_track_b.py --qc` first")
    return pd.read_csv(path)


def _save_tracked_copies(p: dict) -> None:
    os.makedirs(p["processed"], exist_ok=True)
    for key in ("qc", "candidates"):
        if os.path.exists(p[key]):
            shutil.copyfile(p[key], p[key + "_tracked"])


def step_refresh_qc(data_dir: str, cache_dir: str) -> pd.DataFrame:
    """Recompute session QC from the cached trial tables (offline) and refresh the tracked copy."""
    p = _paths(data_dir)
    cand = _read_cached(p, "candidates")
    if not os.path.exists(p["qc"]):
        os.makedirs(p["raw"], exist_ok=True)
        _read_cached(p, "qc").to_csv(p["qc"], index=False)
    qc = refresh_qc_from_cache(cand, cache_dir, qc_path=p["qc"])
    _save_tracked_copies(p)
    return qc


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
    _save_tracked_copies(p)
    return qc


def step_select(data_dir: str, n_mice: int, reselect: bool = False):
    """
    Rank qualifying mice and pick the cohort. An existing cohort in mice_selection.csv is kept
    (the ranking columns are refreshed) unless `reselect` is set: the cohort must not change
    silently when the QC definition does.
    """
    p = _paths(data_dir)
    os.makedirs(p["processed"], exist_ok=True)
    cand = _read_cached(p, "candidates")
    qc = _read_cached(p, "qc")
    merged = cand.merge(qc, on="eid", how="inner")

    print("\n[Step 3/4] Selecting mice...")
    mice, sessions = select_mice(merged, n_mice=n_mice)
    if os.path.exists(p["mice"]) and not reselect:
        kept = pd.read_csv(p["mice"])
        frozen = kept.loc[kept["selected"], "subject"].tolist()
        if frozen:
            would = mice.loc[mice["selected"], "subject"].tolist()
            mice["selected"] = mice["subject"].isin(frozen)
            if set(would) != set(frozen):
                print(f"  NOTE: keeping the stored cohort; a fresh ranking would swap in "
                      f"{sorted(set(would) - set(frozen))} for {sorted(set(frozen) - set(would))} "
                      f"(use --reselect to adopt it)")
    mice.to_csv(p["mice"], index=False)
    print(f"  -> {len(mice)} mice qualify; selected {int(mice['selected'].sum())}:")
    print(mice[mice["selected"]].to_string(index=False))
    return mice, sessions


def build_cohort_files(loaded: dict, subjects: list, labs: dict, processed_dir: str, rt_source: str = DEFAULT_RT_SOURCE):
    """
    Parse the downloaded sessions of `subjects` and write the §3.1 cohort files (2.5 s and 5.0 s cutoffs, per-mouse
    files, cohort_extended, manifest, summary) into `processed_dir`. Shared by the main cohort and the replication set.
    """
    manifest_rows = []
    parsed: dict = {}
    for subj, items in loaded.items():
        parsed[subj] = []
        for meta, raw in items:
            parsed[subj].append(
                parse_session_trials(raw, subj, eid=meta["eid"], date=meta["date"], lab=meta["lab"],
                                     rt_source=rt_source)
            )
            manifest_rows.append({
                "subject_id": subj, "lab": meta["lab"], "session_eid": meta["eid"],
                "date": meta["date"], "number": meta["number"],
                "task_protocol": meta["task_protocol"], "n_raw_trials": len(raw),
                "easy_accuracy": meta["easy_accuracy"],
            })
    os.makedirs(processed_dir, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(os.path.join(processed_dir, "session_manifest.csv"), index=False)

    summaries = []
    extended_all = None
    for tag, rt_max in (("2p5", RT_MAX_STANDARD), ("5p0", RT_MAX_EXTENDED)):
        out_dir = os.path.join(processed_dir, f"rt_{tag}")
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
        cohort.to_parquet(os.path.join(processed_dir, f"cohort_rt{tag}.parquet"), index=False)
        cohort.to_csv(os.path.join(processed_dir, f"cohort_rt{tag}.csv"), index=False)
        n_sess = {s: len(parsed[s]) for s in subjects}
        summaries.append(summarize_cohort(cohort, counts, n_sess, labs, rt_max))
        if tag == "5p0":
            extended_all = cohort_ext
        print(f"  -> rt <= {rt_max}s: {len(cohort)} trials, {cohort['subject_id'].nunique()} mice")

    # parquet only: the extended table is Track B's own analysis input (the CSV copy was 17 MB)
    extended_all.to_parquet(os.path.join(processed_dir, "cohort_extended.parquet"), index=False)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(os.path.join(processed_dir, "summary_stats.csv"), index=False)
    print(f"  -> summary statistics: {os.path.join(processed_dir, 'summary_stats.csv')}")
    return summary


def step_process(data_dir: str, cache_dir: str, n_mice: int, rt_source: str = DEFAULT_RT_SOURCE,
                 reselect: bool = False):
    p = _paths(data_dir)
    mice, sessions = step_select(data_dir, n_mice, reselect)
    subjects = mice.loc[mice["selected"], "subject"].tolist()
    labs = dict(zip(mice["subject"], mice["lab"]))

    print("\n[Step 4/4] Building processed cohort files...")
    one = get_one(cache_dir)
    loaded = load_selected_sessions(one, sessions, subjects)

    return build_cohort_files(loaded, subjects, labs, p["processed"], rt_source)


REPLICATION_DIR = os.path.join("processed", "replication")


def select_replication(mice: pd.DataFrame, frozen: list, n_mice: int = 10, max_per_lab: int = 2) -> list:
    """
    Replication cohort: the next `n_mice` qualifying mice in the SAME ranking that chose the original cohort
    (total valid trials), skipping the frozen original mice, with at most `max_per_lab` new mice per lab.
    Deterministic and decided before any of their data are fitted.
    """
    picked, per_lab = [], {}
    for row in mice.itertuples():
        if row.subject in frozen or len(picked) >= n_mice:
            continue
        if per_lab.get(row.lab, 0) < max_per_lab:
            picked.append(row.subject)
            per_lab[row.lab] = per_lab.get(row.lab, 0) + 1
    return picked


def step_replication(data_dir: str, cache_dir: str, n_mice: int = 10, rt_source: str = DEFAULT_RT_SOURCE):
    """Download and build an independent replication cohort into data/processed/replication/ (original cohort untouched)."""
    p = _paths(data_dir)
    out_dir = os.path.join(data_dir, REPLICATION_DIR)
    merged = _read_cached(p, "candidates").merge(_read_cached(p, "qc"), on="eid", how="inner")
    mice, sessions = select_mice(merged, n_mice=n_mice)
    frozen = pd.read_csv(p["mice"]).query("selected")["subject"].tolist()
    subjects = select_replication(mice, frozen, n_mice)
    labs = dict(zip(mice["subject"], mice["lab"]))
    os.makedirs(out_dir, exist_ok=True)
    mice.assign(selected=mice["subject"].isin(subjects)).to_csv(os.path.join(out_dir, "mice_selection.csv"), index=False)
    print(f"[replication] {len(subjects)} mice: {subjects}")
    one = get_one(cache_dir)
    loaded = load_selected_sessions(one, sessions, subjects)
    return build_cohort_files(loaded, subjects, labs, out_dir, rt_source)


def step_empirical(timeout: int = 1800) -> str:
    """
    Execute notebooks/02_track_b_empirical_baseline.ipynb in place. The notebook writes the
    Week 2 tables to results/empirical/ and figures to results/figures/empirical/.
    """
    import nbformat
    from nbclient import NotebookClient

    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, "notebooks", "02_track_b_empirical_baseline.ipynb")
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=timeout,
                   resources={"metadata": {"path": os.path.join(root, "notebooks")}}).execute()
    nbformat.write(nb, path)
    print(f"  -> executed {path}; tables in results/empirical/, figures in results/figures/empirical/")
    return path


def main():
    parser = argparse.ArgumentParser(description="Track B IBL Data Pipeline (Joshua Koilpillai)")
    parser.add_argument("--all", action="store_true", help="Run the full pipeline")
    parser.add_argument("--qc", action="store_true", help="Download all candidate sessions and compute QC")
    parser.add_argument("--select", action="store_true", help="Select mice from cached QC")
    parser.add_argument("--process", action="store_true", help="Select mice and build data/processed/")
    parser.add_argument("--empirical", action="store_true", help="Execute the Week 2 empirical baseline notebook")
    parser.add_argument("--refresh-qc", action="store_true", help="Recompute session QC from the local cache")
    parser.add_argument("--reselect", action="store_true", help="Rebuild the cohort ranking instead of keeping the stored cohort")
    parser.add_argument("--rt-source", choices=sorted(RT_SOURCES), default=DEFAULT_RT_SOURCE,
                        help="Definition of the contract `rt` column")
    parser.add_argument("--test", action="store_true", help="Run pytest unit test suite")
    parser.add_argument("--replication", action="store_true",
                        help="Build an independent replication cohort (next 10 ranked mice) in data/processed/replication/")
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
    elif args.empirical:
        step_empirical()
    elif args.qc:
        step_qc(args.data_dir, cache_dir, args.n_workers)
    elif args.refresh_qc:
        step_refresh_qc(args.data_dir, cache_dir)
    elif args.select:
        step_select(args.data_dir, args.n_mice, args.reselect)
    elif args.replication:
        step_replication(args.data_dir, cache_dir, args.n_mice, args.rt_source)
    elif args.process:
        step_process(args.data_dir, cache_dir, args.n_mice, args.rt_source, args.reselect)
    else:  # --all or no arguments
        step_qc(args.data_dir, cache_dir, args.n_workers)
        step_process(args.data_dir, cache_dir, args.n_mice, args.rt_source, args.reselect)
    print(f"\nDone in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
