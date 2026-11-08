"""
Fitting the DDM family to the empirical IBL cohort (Week 3: C1 prior mechanism, C2 boundaries).

Thin layer over `fit_ddm` so that C1 (M1/M2/M3, fixed bounds) and C2 (fixed vs Weibull bounds,
2.5 s vs 5.0 s cutoffs) use IDENTICAL settings, and so that the ~100 fits can run in parallel
and resume after an interruption.

Settings used for real mice (`EMPIRICAL_FIT_DEFAULTS`), and why:
    fit_beta=True         contrast exponent: mouse psychometric functions saturate at low contrast;
                          with beta = 1 the fit under-predicts accuracy at 6% contrast by ~25 points.
    fit_variability=True  s_v, s_z (plan §4: "with trial variabilities s_v, s_z").
    bias_type='drift'     neutral side bias v_0 shared by every model; 7/10 mice choose one side by
                          10-25 points at 0% contrast, which z / v_b / b_max would otherwise absorb.
    t_max = RT cutoff     the simulator must be able to produce RTs up to the data cutoff.
"""

import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit

from ..models.simulator import DDMSimulator
from .fit import fit_ddm

EMPIRICAL_FIT_DEFAULTS: Dict[str, object] = {
    "fit_beta": True,
    "fit_variability": True,
    "bias_type": "drift",
    # Search effort. With fit_ddm's defaults (n_screen=24, n_starts=2, n_restarts=1) the BIC of the
    # SAME fit varied by ~660 across seeds on a real mouse (93,675-94,338), larger than the
    # M1/M2/M3 differences being adjudicated; this heavier search cuts the spread to ~60 at ~2.3x
    # the cost (~8 min per 22k-trial fit). Treat |delta BIC| below ~100 as inconclusive.
    "n_screen": 80,
    "n_starts": 5,
    "n_restarts": 3,
}

# Search bounds that are NOT binding for real mice: with the defaults (t0 >= 0.08 s, s_z <= 0.6) fits sit on
# the bounds and improve by 60-770 BIC when they are relaxed. Passed as `bounds=` for the C1/C2 sensitivity runs.
WIDE_BOUNDS = {"t0": (0.02, 0.6), "s_z": (0.0, 1.2)}

# C2 (collapsing bounds). The default lam range (0.4, 3.5) cannot reproduce a fixed bound: even at lam = 3.5,
# gamma = 4 the Weibull bound has fallen to 0.77 a at 2.5 s and 0.05 a at 5 s, so "fixed" would not be nested in
# "Weibull". With lam up to 20 and gamma up to 4 the collapse is < 1% over 5 s, i.e. the fixed bound is the limit.
COLLAPSE_BOUNDS = {"lam": (0.4, 20.0)}
# Warm starts of a Weibull fit, each combined with the mouse's fixed-bound solution: the fixed-bound limit and two
# genuinely collapsing bounds (a(t)/a = 0.64 / 0.43 at t = 1 s). Needed because the run-to-run noise of a
# cold-started fit (SD ~250 BIC in C1) is larger than the effects C2 tests; with the fixed solution among the starts a
# Weibull fit cannot do worse than its fixed-bound special case, up to the likelihood's Monte-Carlo noise.
COLLAPSE_STARTS = ({"lam": 20.0, "gamma": 4.0}, {"lam": 2.5, "gamma": 2.0}, {"lam": 1.2, "gamma": 1.5})
RT_MIN = 0.1       # the data's lower RT cutoff (= src.data.preprocess.RT_MIN, checked in tests/test_c2.py): lower edge of the truncation window

RESULT_KEYS = ("subject_id", "model", "bound", "rt_max")
FIT_SEED_DEFAULT = 12345      # fit_ddm's default CRN / Latin-hypercube seed, recorded in every result row

# Rough relative cost of one fit job (M3 profiles three tau_b values); used to dispatch the longest
# jobs first so that a parallel run is not held up by a straggler at the end.
_MODEL_COST = {"M3": 3.0, "JOINT": 1.3}


def fit_mouse(
    trials_df: pd.DataFrame,
    model_type: str,
    bound_type: str = "fixed",
    rt_max: float = 2.5,
    truncate: bool = False,
    rt_min: float = RT_MIN,
    **overrides,
) -> Dict[str, object]:
    """
    Fit ONE mouse's trials (a §3.1 table) with the empirical settings; returns the `fit_ddm`
    dict plus bookkeeping fields (n_trials, n_params). `overrides` replace any `fit_ddm`
    keyword (e.g. quantiles=(0.1, 0.3, 0.5, 0.7, 0.9, 0.97) to weight the RT tail).

    `truncate=True` scores the model on its RT distribution CONDITIONAL on the data window
    [rt_min, rt_max] (see `QMPDataset(window=...)`); the default reproduces the C1 likelihood.
    For Weibull bounds the search range of `lam` is widened to `COLLAPSE_BOUNDS` unless overridden.
    """
    kwargs = {**EMPIRICAL_FIT_DEFAULTS, "t_max": float(rt_max), **overrides}
    if truncate:
        kwargs["rt_window"] = (float(rt_min), float(rt_max))
    if bound_type.lower() in ("weibull", "collapsing"):
        kwargs["param_bounds"] = {**COLLAPSE_BOUNDS, **(kwargs.get("param_bounds") or {})}
    res = fit_ddm(trials_df, model_type=model_type, bound_type=bound_type, **kwargs)
    n = len(trials_df)
    if truncate:      # the likelihood (and so BIC's N) only counts trials inside the window
        n = int(((trials_df["rt"] >= rt_min) & (trials_df["rt"] <= rt_max)).sum())
    res["n_trials"] = n
    res["n_params"] = int(round((res["bic"] - 2.0 * res["nll"]) / np.log(n)))
    return res


def _flatten(subject: str, model: str, bound: str, rt_max: float, res: Dict[str, object], seed: int,
             truncated: bool = False) -> Dict[str, object]:
    row = {"subject_id": subject, "model": model, "bound": bound, "rt_max": rt_max, "seed": seed, "truncated": truncated}
    for key in ("n_trials", "n_params", "nll", "bic", "aic", "success", "nit", "nfev"):
        row[key] = res[key]
    for name, val in res["params"].items():
        row[f"p_{name}"] = val
    return row


def _job(subject: str, df_mouse: pd.DataFrame, model: str, bound: str, rt_max: float, overrides: dict):
    res = fit_mouse(df_mouse, model, bound, rt_max, **overrides)
    return _flatten(subject, model, bound, rt_max, res, overrides.get("seed", FIT_SEED_DEFAULT),
                    bool(overrides.get("truncate", False)))


def fit_cohort(
    cohort: pd.DataFrame,
    models: Sequence[str] = ("M1", "M2", "M3"),
    bounds: Sequence[str] = ("fixed",),
    rt_max: float = 2.5,
    n_jobs: int = -1,
    cache_path: Optional[str] = None,
    subjects: Optional[Iterable[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    init_from: Optional[pd.DataFrame] = None,
    init_extra: Optional[Sequence[Dict[str, float]]] = None,
    **overrides,
) -> pd.DataFrame:
    """
    Fit every (mouse, model, bound[, seed]) combination in parallel; one row per fit.

    `seeds` runs independent replicate fits (different Latin-hypercube screen and CRN noise) in the
    same pool; their BIC spread is the optimizer-noise estimate (see `src.optimization.c1`).

    `cohort` is a cohort table matching `rt_max` (cohort_rt2p5 for 2.5, cohort_rt5p0 for 5.0).
    With `cache_path` (CSV) every finished fit is written immediately and skipped on the next call,
    so an interrupted ~hour-long job resumes. Jobs are dispatched longest-first (M3 before M1/M2).
    Changing `overrides` between calls into the same cache is NOT detected: use a different
    `cache_path` for a different configuration.

    `init_from` (a table of finished fits, e.g. the fixed-bound rows) warm-starts every task from the
    fit of the same (mouse, model, rt_max, seed): the starting guess is that fit's parameters, once for
    each dict in `init_extra` (e.g. `COLLAPSE_STARTS` adds lam / gamma for Weibull fits). Tasks without a
    matching reference fit start as usual.
    """
    subs = list(subjects) if subjects is not None else sorted(cohort["subject_id"].unique())
    done = pd.read_csv(cache_path) if cache_path and os.path.exists(cache_path) else pd.DataFrame()
    if len(done) and "seed" not in done.columns:
        done["seed"] = FIT_SEED_DEFAULT
    keys = list(RESULT_KEYS) + ["seed"]
    finished = set(map(tuple, done[keys].astype({"rt_max": float, "seed": int}).values.tolist())) if len(done) else set()
    seed_list = [int(x) for x in seeds] if seeds is not None else [int(overrides.get("seed", FIT_SEED_DEFAULT))]
    overrides = {k: v for k, v in overrides.items() if k != "seed"}

    starts = _warm_starts(init_from, init_extra, float(rt_max))
    tasks = []
    for s in subs:
        d = cohort[cohort["subject_id"] == s].reset_index(drop=True)
        for m in models:
            for b in bounds:
                for sd in seed_list:
                    if (s, m, b, float(rt_max), sd) not in finished:
                        extra = {"init_params": starts[(s, m, sd)]} if (s, m, sd) in starts else {}
                        tasks.append((s, d, m, b, float(rt_max), {**overrides, **extra, "seed": sd}))
    tasks.sort(key=lambda t: -_MODEL_COST.get(t[2].upper(), 1.0) * (1.15 if t[3] != "fixed" else 1.0))

    # Results are saved as each fit completes, so a killed run loses at most the fits in flight.
    rows = done.to_dict("records")
    if tasks:
        stream = Parallel(n_jobs=n_jobs, verbose=5, return_as="generator_unordered")(
            delayed(_job)(*t) for t in tasks
        )
        for row in stream:
            rows.append(row)
            if cache_path:
                _atomic_csv(pd.DataFrame(rows), cache_path)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _warm_starts(init_from: Optional[pd.DataFrame], extras: Optional[Sequence[Dict[str, float]]],
                 rt_max: float) -> Dict[tuple, List[Dict[str, float]]]:
    """{(subject, model, seed): [guess, ...]} from reference fits at this cutoff (first row wins)."""
    if init_from is None or not len(init_from):
        return {}
    ref = init_from[init_from["rt_max"].astype(float) == rt_max]
    if "seed" not in ref.columns:
        ref = ref.assign(seed=FIT_SEED_DEFAULT)
    out: Dict[tuple, List[Dict[str, float]]] = {}
    for _, r in ref.iterrows():
        key = (r["subject_id"], r["model"], int(r["seed"]))
        if key not in out:
            base = fitted_params(r)
            out[key] = [{**base, **e} for e in (extras or [{}])]
    return out


def _atomic_csv(frame: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def fitted_params(row: "pd.Series | Dict[str, float]") -> Dict[str, float]:
    """Parameter dict from a `fit_cohort` result row (columns prefixed `p_`, NaN = not used)."""
    items = row.items() if hasattr(row, "items") else dict(row).items()
    return {k[2:]: float(v) for k, v in items if k.startswith("p_") and pd.notna(v)}


def simulate_from_fit(
    trials_df: pd.DataFrame,
    params: Dict[str, float],
    model_type: str,
    bound_type: str = "fixed",
    rt_max: float = 2.5,
    n_rep: int = 5,
    seed: int = 0,
    dt: float = 0.002,
    rt_min: float = 0.1,
    prior_col: str = "prior",
) -> pd.DataFrame:
    """
    Posterior-predictive §3.1 table: replay a mouse's actual trial covariates (contrast, prior,
    previous choice) `n_rep` times through the fitted model. Feed it to `src.empirical.compute_cbf`
    (with the mouse's `bin_edges` from results/empirical/cbf_bin_edges.csv) for the model-vs-data
    CBF overlay, or to the psychometric / chronometric tools. Simulated RTs above `rt_max` are
    dropped (and RTs below `rt_min`, the data's lower cutoff) to mirror the data's exclusion filter.

    `prior_col` is the covariate the model was FITTED with (e.g. 'prior_model', the binned inferred prior);
    the output's `prior` column is always the true block label, so CBFs stay comparable with the data.
    If `params` has a lapse rate (`lapse` > 0), that fraction of trials is replaced by lapses (choice Right
    with probability sigmoid(lapse_side + lapse_prior * prior), RT uniform on [rt_min, rt_max]), as in the
    fitted likelihood; a fast-guess rate (`guess`) adds responses with RT uniform on [rt_min, rt_min + guess_width]
    and P(Right) = sigmoid(lapse_side + guess_prior * prior). The `replay` column numbers the `n_rep` copies and
    `prior_model` holds the covariate the model used.
    """
    sim = DDMSimulator(dt=dt, t_max=rt_max)
    n = len(trials_df)
    model_prior = np.tile(trials_df[prior_col].to_numpy(dtype=float), n_rep)
    out = sim.simulate(
        params,
        np.tile(trials_df["contrast"].to_numpy(dtype=float), n_rep),
        model_prior,
        np.tile(trials_df["prev_choice"].to_numpy(dtype=float), n_rep),
        model_type=model_type,
        bound_type=bound_type,
        subject_id=str(trials_df["subject_id"].iloc[0]),
        rng=np.random.default_rng(seed),
    )
    out["prior"] = np.tile(trials_df["prior"].to_numpy(dtype=float), n_rep)
    out["replay"] = np.repeat(np.arange(n_rep, dtype=np.int32), n)
    out["prior_model"] = model_prior
    rate = float(params.get("lapse", 0.0) or 0.0)
    g_rate = float(params.get("guess", 0.0) or 0.0)
    if rate > 0.0 or g_rate > 0.0:
        rng = np.random.default_rng([seed, 7919])
        u = rng.random(len(out))
        side = float(params.get("lapse_side", 0.0))
        hit = u < rate                                    # lapses: RT uniform on the whole window
        p_right = expit(side + float(params.get("lapse_prior", 0.0)) * model_prior[hit])
        out.loc[hit, "choice"] = np.where(rng.random(hit.sum()) < p_right, 1, -1).astype(np.int8)
        out.loc[hit, "rt"] = rng.uniform(rt_min, rt_max, hit.sum())
        g = (u >= rate) & (u < rate + g_rate)             # fast guesses: RT uniform just above the cutoff
        if g.any():
            p_right = expit(side + float(params.get("guess_prior", 0.0)) * model_prior[g])
            out.loc[g, "choice"] = np.where(rng.random(g.sum()) < p_right, 1, -1).astype(np.int8)
            out.loc[g, "rt"] = rng.uniform(rt_min, rt_min + float(params.get("guess_width", 0.2)), g.sum())
    out = out[(out["rt"] <= rt_max) & (out["rt"] >= rt_min)].reset_index(drop=True)
    out["trial_idx"] = np.arange(len(out), dtype=np.int32)
    return out


def rescore_fits(
    cohort: pd.DataFrame,
    fits: pd.DataFrame,
    n_sim_per_cond: int = 4000,
    seed: int = 2024,
    n_jobs: int = -1,
    cache_path: Optional[str] = None,
    **overrides,
) -> pd.DataFrame:
    """
    Re-score every finished fit (every replicate seed of every model) with ONE high-precision likelihood
    evaluation that uses the same `seed` for all of a mouse's fits.

    Why: each fit's own final BIC comes from its own Monte-Carlo seed, so a model difference mixes optimizer
    differences with simulation noise. With a common seed (common random numbers across models) the simulation
    noise largely cancels in a difference, and the larger `n_sim_per_cond` shrinks what remains. The original
    values are kept as `nll_fit` / `bic_fit` / `aic_fit`; `nll` / `bic` / `aic` become the re-scored ones, so the
    C1 analysis (best replicate, noise floor, tests) uses them unchanged. `overrides` must be the SAME
    configuration the fits were run with (fit_lapse, param_bounds, fit_history, ...).
    Cached in `cache_path` (one row per fit; rows already there are reused).
    """
    from .fit import score_ddm

    keys = list(RESULT_KEYS) + ["seed"]
    done = pd.read_csv(cache_path) if cache_path and os.path.exists(cache_path) else pd.DataFrame()
    have = set(map(tuple, done[keys].astype({"rt_max": float, "seed": int}).values.tolist())) if len(done) else set()
    cfg = {**EMPIRICAL_FIT_DEFAULTS, **overrides}

    def work(row: Dict[str, object], d: pd.DataFrame) -> Dict[str, object]:
        kw = {**cfg, "t_max": float(row["rt_max"])}
        sc = score_ddm(d, fitted_params(row), model_type=row["model"], bound_type=row["bound"],
                       n_sim_per_cond=n_sim_per_cond, seed=seed, **kw)
        out = dict(row)
        for k in ("nll", "bic", "aic"):
            out[f"{k}_fit"] = row[k]
            out[k] = sc[k]
        out["rescore_seed"], out["rescore_n_sim"] = seed, n_sim_per_cond
        return out

    todo = [r for r in fits.to_dict("records")
            if (r["subject_id"], r["model"], r["bound"], float(r["rt_max"]), int(r.get("seed", FIT_SEED_DEFAULT))) not in have]
    rows = done.to_dict("records")
    if todo:
        by_mouse = {s: g.reset_index(drop=True) for s, g in cohort.groupby("subject_id", observed=True)}
        stream = Parallel(n_jobs=n_jobs, verbose=5, return_as="generator_unordered")(
            delayed(work)(r, by_mouse[r["subject_id"]]) for r in todo)
        for row in stream:
            rows.append(row)
            if cache_path:
                _atomic_csv(pd.DataFrame(rows), cache_path)
    return pd.DataFrame(rows)


def delta_bic(results: pd.DataFrame, by: str, reference: str, value: str = "bic") -> pd.DataFrame:
    """
    Per-mouse BIC differences relative to a reference level of `by` ('model' or 'bound'),
    e.g. delta_bic(res[res.rt_max == 2.5], by='bound', reference='fixed'): positive = worse
    than the reference. Returns one row per (mouse, other-factors) and level of `by`.
    """
    other = [k for k in RESULT_KEYS if k != by]
    ref = results[results[by] == reference].set_index(other)[value].rename("ref")
    out = results[results[by] != reference].join(ref, on=other)
    out["delta_" + value] = out[value] - out["ref"]
    return out.drop(columns="ref")
