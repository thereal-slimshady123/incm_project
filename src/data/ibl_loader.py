"""
IBL ONE API loader: session discovery, trial-table download and mouse selection.

Track B (Joshua Koilpillai): connects to the public IBL Alyx database (openalyx),
finds standard `biasedChoiceWorld` training sessions (alternating 20/80 and 80/20
blocks), downloads their trial tables and selects the 10-mouse cohort.

Only `_ibl_trials.table.pqt` is needed, so the lighter `ONE-api` package suffices
and the heavy `ibllib` package is not required. Legacy sessions that lack this table
(and therefore `firstMovement_times`, which would need raw wheel data and `ibllib`) are
excluded; they are recorded with status 'no_table' in the QC output.
"""

import glob
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .preprocess import session_qc

OPENALYX_URL = "https://openalyx.internationalbrainlab.org"
OPENALYX_USER = "intbrainlab"
OPENALYX_PASSWORD = "international"

# Standard IBL training project; excludes opto / ASD / reverse-contingency / serotonin variants
IBL_PROJECT = "ibl_neuropixel_brainwide_01"
# e.g. '_iblrig_tasks_biasedChoiceWorld6.4.0' (rejects ...biasedChoiceWorldRPE, opto_..., nate_...)
STANDARD_PROTOCOL_RE = re.compile(r"^_iblrig_tasks_biasedChoiceWorld\d")
TRIALS_TABLE = "_ibl_trials.table.pqt"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "ONE")


def get_one(cache_dir: str = DEFAULT_CACHE_DIR):
    """
    Connect to the public IBL Alyx database with the shared public credentials.

    Parameters
    ----------
    cache_dir : str
        Local directory where ONE caches downloaded datasets (default `data/raw/ONE`).
    """
    from one.api import ONE

    os.makedirs(cache_dir, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # shared public account retirement notice
        return ONE(
            base_url=OPENALYX_URL,
            username=OPENALYX_USER,
            password=OPENALYX_PASSWORD,
            silent=True,
            cache_dir=cache_dir,
        )


def list_candidate_sessions(one) -> pd.DataFrame:
    """
    List all standard biased-protocol training sessions on openalyx.

    Returns
    -------
    pd.DataFrame
        One row per session: ['eid', 'subject', 'lab', 'date', 'number', 'task_protocol'],
        sorted by subject, date and session number.
    """
    eids, details = one.search(task_protocol="biasedChoiceWorld", project=IBL_PROJECT, details=True)
    rows = []
    for eid, d in zip(eids, details):
        if not STANDARD_PROTOCOL_RE.match(d["task_protocol"]):
            continue
        if list(d["projects"]) != [IBL_PROJECT]:
            continue
        rows.append({
            "eid": str(eid),
            "subject": d["subject"],
            "lab": d["lab"],
            "date": str(d["date"]),
            "number": int(d["number"]),
            "task_protocol": d["task_protocol"],
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["subject", "date", "number"]).reset_index(drop=True)


def load_session_trials(one, eid: str) -> Optional[pd.DataFrame]:
    """
    Download (or read from cache) the raw ALF trials table of one session.

    Returns None if the session has no `_ibl_trials.table.pqt` (legacy sessions).
    """
    datasets = sorted(one.list_datasets(eid, filename=TRIALS_TABLE))
    if not datasets:
        return None
    collection = datasets[0].rsplit("/", 1)[0]
    table = one.load_dataset(eid, TRIALS_TABLE, collection=collection)
    return pd.DataFrame(table).reset_index(drop=True)


def _load_with_retry(one, eid: str, retries: int = 3) -> Tuple[Optional[pd.DataFrame], str]:
    last_err = ""
    for attempt in range(retries):
        try:
            return load_session_trials(one, eid), "ok"
        except Exception as exc:  # network hiccup, partial download, ...
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(2.0 * (attempt + 1))
    return None, last_err


def evaluate_sessions(
    one,
    sessions: pd.DataFrame,
    qc_path: Optional[str] = None,
    n_workers: int = 6,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Download the trials table of every session and compute per-session QC statistics.

    Results are cached in `qc_path` (CSV) so interrupted runs resume where they stopped;
    ONE itself caches the downloaded files.

    Returns
    -------
    pd.DataFrame
        `sessions` plus the columns produced by `preprocess.session_qc` and a `status`
        column ('ok', 'no_table', 'bad_table', or an error message).
    """
    done: Dict[str, dict] = {}
    if qc_path and os.path.exists(qc_path):
        prev = pd.read_csv(qc_path)
        prev = prev[prev["status"].isin(["ok", "no_table", "bad_table"])]
        done = {r["eid"]: r for r in prev.to_dict("records")}

    todo = [e for e in sessions["eid"] if e not in done]
    if verbose:
        print(f"  [QC] {len(done)} sessions cached, {len(todo)} to download "
              f"({n_workers} workers)", flush=True)

    def work(eid: str) -> dict:
        raw, status = _load_with_retry(one, eid)
        if raw is None:
            return {"eid": eid, "status": "no_table" if status == "ok" else status}
        try:
            return {"eid": eid, "status": "ok", **session_qc(raw)}
        except KeyError as exc:
            return {"eid": eid, "status": "bad_table", "error": f"missing column {exc}"}

    t0 = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(work, e): e for e in todo}
        for fut in as_completed(futures):
            res = fut.result()
            done[res["eid"]] = res
            n_done += 1
            if verbose and (n_done % 50 == 0 or n_done == len(todo)):
                rate = n_done / (time.time() - t0)
                print(f"  [QC] {n_done}/{len(todo)} sessions ({rate:.1f}/s)", flush=True)
                if qc_path:
                    pd.DataFrame(list(done.values())).to_csv(qc_path, index=False)

    # Sessions that failed with a network error get one slow sequential retry pass
    failed = [e for e, r in done.items() if r["status"] not in ("ok", "no_table", "bad_table")]
    for eid in failed:
        done[eid] = work(eid)

    qc = pd.DataFrame(list(done.values()))
    if qc_path:
        qc.to_csv(qc_path, index=False)
    return sessions.merge(qc, on="eid", how="left")


def refresh_qc_from_cache(
    sessions: pd.DataFrame,
    cache_dir: str = DEFAULT_CACHE_DIR,
    qc_path: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Recompute the per-session QC statistics from the locally cached trial tables (no network).

    Use after the QC definition changes (e.g. the RT definition behind `n_valid_trials`) so the
    cached numbers stay consistent with `preprocess.session_qc`. Sessions whose table is not in
    the cache keep their previous row. `sessions` needs ['eid', 'subject', 'lab', 'date', 'number'].
    """
    rows, missing = [], 0
    for r in sessions.itertuples():
        hits = glob.glob(os.path.join(
            cache_dir, r.lab, "Subjects", r.subject, str(r.date), f"{int(r.number):03d}", "**", TRIALS_TABLE
        ), recursive=True)
        if not hits:
            missing += 1
            continue
        try:
            rows.append({"eid": r.eid, "status": "ok", **session_qc(pd.read_parquet(hits[0]))})
        except KeyError as exc:
            rows.append({"eid": r.eid, "status": "bad_table", "error": f"missing column {exc}"})
    fresh = pd.DataFrame(rows)
    if qc_path and os.path.exists(qc_path):
        old = pd.read_csv(qc_path)
        old = old[~old["eid"].isin(fresh["eid"])]
        fresh = pd.concat([old, fresh], ignore_index=True)
    if qc_path:
        fresh.to_csv(qc_path, index=False)
    if verbose:
        print(f"  [QC] refreshed {len(rows)} sessions from cache ({missing} not cached)")
    return fresh


def select_mice(
    session_qc_df: pd.DataFrame,
    n_mice: int = 10,
    min_sessions: int = 5,
    min_trials: int = 400,
    min_easy_accuracy: float = 0.80,
    max_per_lab: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deterministically select the cohort of trained mice.

    A session qualifies if its table loaded, it has >= `min_trials` trials, both block
    types (p_left 0.2 and 0.8) occur, and accuracy on easy trials (|contrast| >= 0.25)
    is >= `min_easy_accuracy`. A mouse qualifies with >= `min_sessions` qualifying
    sessions. Mice are ranked by total valid trials (descending, ties broken by name)
    and picked greedily with at most `max_per_lab` mice per lab.

    Returns
    -------
    mice : pd.DataFrame
        Ranking of all qualifying mice with a `selected` flag.
    sessions : pd.DataFrame
        `session_qc_df` with a boolean `qualifies` column.
    """
    s = session_qc_df.copy()
    s["qualifies"] = (
        (s["status"] == "ok")
        & (s["n_trials"] >= min_trials)
        & (s["has_both_blocks"] == True)  # noqa: E712 (NaN-safe comparison)
        & (s["easy_accuracy"] >= min_easy_accuracy)
    )
    q = s[s["qualifies"]]
    mice = (
        q.groupby("subject")
        .agg(
            lab=("lab", "first"),
            n_sessions=("eid", "count"),
            n_trials=("n_trials", "sum"),
            n_valid_trials=("n_valid_trials", "sum"),
            mean_easy_accuracy=("easy_accuracy", "mean"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .reset_index()
    )
    mice = mice[mice["n_sessions"] >= min_sessions]
    mice = mice.sort_values(["n_valid_trials", "subject"], ascending=[False, True]).reset_index(drop=True)

    selected, lab_counts = [], {}
    for row in mice.itertuples():
        take = sum(selected) < n_mice and lab_counts.get(row.lab, 0) < max_per_lab
        if take:
            lab_counts[row.lab] = lab_counts.get(row.lab, 0) + 1
        selected.append(take)
    mice["selected"] = selected
    return mice, s


def load_selected_sessions(
    one, sessions: pd.DataFrame, subjects: List[str]
) -> Dict[str, List[Tuple[dict, pd.DataFrame]]]:
    """
    Load raw trial tables for all qualifying sessions of the selected subjects.

    Returns
    -------
    dict
        subject -> list of (session metadata dict, raw trials DataFrame), in time order.
    """
    out: Dict[str, List[Tuple[dict, pd.DataFrame]]] = {}
    for subj in subjects:
        sub = sessions[(sessions["subject"] == subj) & (sessions["qualifies"])]
        sub = sub.sort_values(["date", "number"])
        frames = []
        for meta in sub.to_dict("records"):
            raw, status = _load_with_retry(one, meta["eid"])
            if raw is None:
                raise RuntimeError(f"Could not load trials for {subj} {meta['eid']}: {status}")
            frames.append((meta, raw))
        out[subj] = frames
    return out
