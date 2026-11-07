"""
Drift Diffusion Model Parameter Fitting Engine via Quantile Maximum Probability (QMP).

Uses SciPy Nelder-Mead simplex optimization with Common Random Numbers (CRN)
for smooth variance-reduced objective surfaces.
Adheres strictly to §3.2 Model API specification.
"""

from typing import Dict, List, Tuple, Any, Optional, Sequence
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc

from ..models.simulator import DDMSimulator
from .qmp import QMPDataset, compute_qmp_nll_blocks, compute_bic, compute_aic


# Parameter boundaries and default initial guesses
PARAM_SPECS = {
    "a": {"init": 1.1, "bounds": (0.4, 2.5), "desc": "Boundary separation"},
    "t0": {"init": 0.22, "bounds": (0.08, 0.60), "desc": "Non-decision time (s)"},
    "k": {"init": 2.5, "bounds": (0.5, 7.5), "desc": "Sensory drift sensitivity"},
    "v_h": {"init": 0.0, "bounds": (-2.0, 2.0), "desc": "Choice history bias"},
    # M1 specific
    "z": {"init": 0.2, "bounds": (-0.85, 0.85), "desc": "Starting point bias (M1)"},
    # M2 specific
    "v_b": {"init": 0.4, "bounds": (-2.5, 2.5), "desc": "Drift rate prior bias (M2)"},
    # M3 specific
    "b_max": {"init": 0.5, "bounds": (0.0, 2.5), "desc": "Dynamic bias ceiling (M3)"},
    "tau_b": {"init": 0.35, "bounds": (0.05, 1.5), "desc": "Dynamic bias latency (M3)"},
    # Collapsing bounds (C2)
    "lam": {"init": 1.5, "bounds": (0.4, 3.5), "desc": "Weibull scale / collapse latency"},
    "gamma": {"init": 2.0, "bounds": (1.0, 4.0), "desc": "Weibull shape / steepness"},
    # Contrast non-linearity of the sensory drift k * sign(c) * |c|^beta (plan §4, Week 1)
    "beta": {"init": 0.5, "bounds": (0.05, 1.5), "desc": "Contrast exponent"},
    # Neutral side bias (independent of the block prior); real mice have large ones
    "v_0": {"init": 0.0, "bounds": (-1.5, 1.5), "desc": "Neutral drift offset"},
    "z_0": {"init": 0.0, "bounds": (-0.8, 0.8), "desc": "Neutral starting-point offset (fraction of a)"},
    # Trial-to-trial variabilities (plan §4, Week 1 / Week 3 C2)
    "s_v": {"init": 0.5, "bounds": (0.0, 3.0), "desc": "Drift-rate variability (SD)"},
    "s_z": {"init": 0.15, "bounds": (0.0, 0.6), "desc": "Starting-point variability (range as fraction of a)"},
}

# Build initial simplex with physiologically scaled step sizes
# Prevents Nelder-Mead from premature contraction in 5D parameter space
SIMPLEX_STEPS = {
    "a": 0.3,
    "t0": 0.08,
    "k": 0.8,
    "v_h": 0.3,
    "z": 0.25,
    "v_b": 0.45,
    "b_max": 0.4,
    "tau_b": 0.2,
    "lam": 0.4,
    "gamma": 0.5,
    "beta": 0.25,
    "v_0": 0.3,
    "z_0": 0.2,
    "s_v": 0.4,
    "s_z": 0.12,
}

# tau_b is not identifiable jointly with b_max (pilot recovery: r ~ 0), so M3 fits
# b_max with tau_b fixed at each grid value and keeps the best (plan §6 contingency)
DEFAULT_TAU_B_GRID = (0.15, 0.35, 0.8)


class DDMFitter:
    """
    QMP Fitting harness for Drift Diffusion Models.
    """

    def __init__(
        self,
        model_type: str = "M2",
        bound_type: str = "fixed",
        fit_history: bool = True,
        n_sim_per_cond: int = 400,
        dt: float = 0.002,
        t_max: float = 2.5,
        crn_seed: int = 12345,
        fixed_params: Optional[Dict[str, float]] = None,
        n_sim_final: int = 2000,
        n_extra_params: int = 0,
        rt_bandwidth: Optional[float] = 0.01,
        fit_beta: bool = False,
        fit_variability: bool = False,
        quantiles: Optional[Sequence[float]] = None,
        bias_type: Optional[str] = None,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        rt_window: Optional[Tuple[float, float]] = None,
    ):
        """
        Parameters
        ----------
        model_type : str
            'M1' (starting point), 'M2' (drift bias), 'M3' (dynamic bias), or
            'JOINT' (starting point + drift bias).
        bound_type : str
            'fixed' or 'weibull'.
        fit_history : bool
            Whether to fit 1-back choice history weight v_h.
        n_sim_per_cond : int
            Simulated trials per condition per objective evaluation.
        dt : float
            Euler-Maruyama step size for internal simulator during fitting.
        t_max : float
            Maximum decision time cutoff.
        crn_seed : int
            Common Random Numbers seed for deterministic objective evaluations.
        fixed_params : dict, optional
            Model parameters held at fixed values (excluded from optimization).
        n_sim_final : int
            Simulated trials per condition for the final likelihood evaluation. Uses an
            independent seed so the reported NLL/BIC is not fitted to the CRN noise.
        n_extra_params : int
            Parameters chosen outside this fit (e.g. tau_b picked from a grid), added to
            the BIC/AIC parameter count.
        rt_bandwidth : float, optional
            Gaussian kernel width (s) for smoothing simulated RTs across QMP bins, so the
            CRN objective is continuous rather than a step function. None = hard counts.
        fit_beta : bool
            Also fit the contrast exponent beta of the sensory drift (otherwise beta = 1,
            linear in contrast, unless given in `fixed_params`). Needed for real mouse data,
            whose psychometric function saturates at low contrast.
        fit_variability : bool
            Also fit the trial-to-trial drift (s_v) and starting-point (s_z) variabilities
            (otherwise both 0, unless given in `fixed_params`).
        bias_type : {None, 'drift', 'start', 'both'}
            Adds a neutral side-bias term shared by ALL models (so it cannot favour one prior
            mechanism): 'drift' fits v_0, 'start' fits z_0, 'both' fits both. Real mice have
            side biases of 10-25 percentage points at 0% contrast; without this term z, v_b
            and b_max absorb them.
        quantiles : sequence of float, optional
            QMP quantile cut points (default 0.1, 0.3, 0.5, 0.7, 0.9). Add e.g. 0.95 / 0.99
            to make the likelihood sensitive to the RT tail (the C2 urgency question).
        param_bounds : dict, optional
            Per-parameter (low, high) search bounds replacing the defaults in PARAM_SPECS for this fit,
            e.g. {'t0': (0.02, 0.6), 's_z': (0.0, 1.0)} to check whether a fit is limited by its bounds.
        rt_window : (rt_lo, rt_hi), optional
            Truncation-aware likelihood (see `QMPDataset`): the data were filtered to this RT window, so
            the model is scored on its RT distribution conditional on the window. None (default) = the
            original behaviour used for C1, where mass beyond the cutoff falls into the last QMP bin.
        """
        self.model_type = model_type.upper()
        self.bound_type = bound_type.lower()
        self.fit_history = fit_history
        self.n_sim_per_cond = n_sim_per_cond
        self.dt = dt
        self.t_max = t_max
        self.crn_seed = crn_seed
        self.fixed_params = dict(fixed_params or {})
        self.n_sim_final = n_sim_final
        self.n_extra_params = n_extra_params
        self.rt_bandwidth = rt_bandwidth
        self.fit_beta = fit_beta
        self.fit_variability = fit_variability
        self.quantiles = tuple(quantiles) if quantiles is not None else None
        self.rt_window = None if rt_window is None else (float(rt_window[0]), float(rt_window[1]))
        if bias_type not in (None, "drift", "start", "both"):
            raise ValueError("bias_type must be None, 'drift', 'start' or 'both'")
        self.bias_type = bias_type
        self.bounds_map = {p: tuple(PARAM_SPECS[p]["bounds"]) for p in PARAM_SPECS}
        for name, (lo, hi) in (param_bounds or {}).items():
            if name not in PARAM_SPECS or not lo < hi:
                raise ValueError(f"invalid bounds override for '{name}': {(lo, hi)}")
            self.bounds_map[name] = (float(lo), float(hi))

        self.param_names = [p for p in self._determine_param_names() if p not in self.fixed_params]
        self.simulator = DDMSimulator(dt=dt, t_max=t_max)

    def _determine_param_names(self) -> List[str]:
        """
        Select active parameter names for this model architecture.
        """
        names = ["a", "t0", "k"]
        if self.fit_beta:
            names.append("beta")
        if self.fit_history:
            names.append("v_h")
        if self.bias_type in ("drift", "both"):
            names.append("v_0")
        if self.bias_type in ("start", "both"):
            names.append("z_0")

        if self.model_type == "M1":
            names.append("z")
        elif self.model_type == "M2":
            names.append("v_b")
        elif self.model_type == "M3":
            names.extend(["b_max", "tau_b"])
        elif self.model_type == "JOINT":
            # Starting point + drift bias together, to measure z vs v_b trade-off
            names.extend(["z", "v_b"])
        else:
            raise ValueError(f"Unknown model_type: '{self.model_type}'")

        if self.bound_type in ("collapsing", "weibull"):
            names.extend(["lam", "gamma"])

        if self.fit_variability:
            names.extend(["s_v", "s_z"])

        return names

    @property
    def n_params(self) -> int:
        return len(self.param_names) + self.n_extra_params

    def _pack_params(self, param_dict: Dict[str, float]) -> np.ndarray:
        return np.array([param_dict.get(p, PARAM_SPECS[p]["init"]) for p in self.param_names])

    def _unpack_params(self, x: np.ndarray) -> Dict[str, float]:
        params = {p: float(val) for p, val in zip(self.param_names, x)}
        params.update(self.fixed_params)
        return params

    def _bounds(self) -> np.ndarray:
        return np.array([self.bounds_map[p] for p in self.param_names])

    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        bounds = self._bounds()
        return np.clip(x, bounds[:, 0], bounds[:, 1])

    def _compute_penalty(self, x: np.ndarray) -> float:
        """
        Quadratic penalty for parameters outside physiological bounds.
        """
        penalty = 0.0
        for p, val in zip(self.param_names, x):
            low, high = self.bounds_map[p]
            if val < low:
                penalty += 1000.0 * (low - val) ** 2
            elif val > high:
                penalty += 1000.0 * (high - val) ** 2
        return penalty

    def _build_simplex(self, center: np.ndarray) -> np.ndarray:
        simplex = [center.copy()]
        for idx, p in enumerate(self.param_names):
            vertex = center.copy()
            step = SIMPLEX_STEPS.get(p, 0.25)
            # Ensure vertex remains within bounds
            low, high = self.bounds_map[p]
            if vertex[idx] + step <= high:
                vertex[idx] += step
            else:
                vertex[idx] -= step
            simplex.append(vertex)
        return np.array(simplex)

    def _simulation_template(self, qmp_data: QMPDataset, n_per_cond: int) -> Tuple[np.ndarray, ...]:
        """Condition-blocked simulation inputs (the observed prev_choice is a covariate)."""
        conds = qmp_data.conditions
        contrasts = np.repeat([c["contrast"] for c in conds], n_per_cond).astype(np.float64)
        priors = np.repeat([c["prior"] for c in conds], n_per_cond).astype(np.float64)
        prev = np.repeat([c.get("prev_choice", 0.0) for c in conds], n_per_cond).astype(np.float64)
        return contrasts, priors, prev

    def _qmp_nll(self, x_free: np.ndarray, qmp_data: QMPDataset, template, n_per_cond: int, seed: int) -> float:
        contrasts, priors, prev = template
        sim_df = self.simulator.simulate(
            params=self._unpack_params(x_free),
            contrasts=contrasts,
            priors=priors,
            prev_choices=prev,
            model_type=self.model_type,
            bound_type=self.bound_type,
            rng=np.random.default_rng(seed),
        )
        return compute_qmp_nll_blocks(
            sim_df["choice"].to_numpy(), sim_df["rt"].to_numpy(), qmp_data, n_per_cond,
            bandwidth=self.rt_bandwidth,
        )

    def fit(
        self,
        trials_df: pd.DataFrame,
        init_params: Optional[Any] = None,
        maxiter: int = 400,
        tol: float = 0.5,
        xatol: float = 0.01,
        n_restarts: int = 1,
        n_screen: int = 24,
        n_starts: int = 2,
    ) -> Dict[str, Any]:
        """
        Fit model parameters to empirical/target dataset via QMP Nelder-Mead.

        Search: the initial guess plus n_screen Latin-hypercube points are each
        evaluated once; Nelder-Mead runs from the n_starts best; the winner is then
        polished with n_restarts fresh-simplex restarts.

        Parameters
        ----------
        trials_df : pd.DataFrame
            Target trials DataFrame conforming to §3.1 contract.
        init_params : dict or list of dict, optional
            Initial parameter guess; a list gives several guesses, all screened with the Latin-hypercube
            points (used to warm-start Weibull fits from the fixed-bound solution).
        maxiter : int
            Maximum iterations for each Nelder-Mead run.
        tol : float
            Convergence tolerance on the total NLL (BIC differences below ~1 are not
            meaningful, and a simulated likelihood is not smoother than that).
        xatol : float
            Convergence tolerance on parameter values.
        n_restarts : int
            Polishing restarts from the best point with a fresh simplex.
        n_screen : int
            Latin-hypercube screening points (0 disables multi-start).
        n_starts : int
            Number of best screened points to run Nelder-Mead from.

        Returns
        -------
        dict
            Results dictionary conforming to §3.2 Model API:
            {'params': dict, 'nll': float, 'bic': float, 'aic': float, 'success': bool, 'nit': int}
        """
        # Condition on previous choice whenever v_h is fitted, so it is identifiable
        group_cols = ("contrast", "prior", "prev_choice") if self.fit_history else ("contrast", "prior")
        qmp_kwargs = {"quantiles": self.quantiles} if self.quantiles is not None else {}
        qmp_data = QMPDataset(trials_df, group_cols=group_cols, window=self.rt_window, **qmp_kwargs)
        n_total = float(qmp_data.total_trials)
        template = self._simulation_template(qmp_data, self.n_sim_per_cond)
        n_fev = [0]

        def objective(x_vec: np.ndarray) -> float:
            n_fev[0] += 1
            # Check boundary penalty
            penalty = self._compute_penalty(x_vec)
            if penalty > 1e4:
                return 1e6 + penalty
            # Simulate at the nearest in-bounds point (fixed CRN seed); the penalty pushes the simplex back
            nll = self._qmp_nll(self._clip_to_bounds(x_vec), qmp_data, template, self.n_sim_per_cond, self.crn_seed)
            # Normalize by trial count for stable tolerance evaluation
            return (nll + penalty) / n_total

        def run_nelder_mead(start: np.ndarray):
            return minimize(
                objective,
                start,
                method="Nelder-Mead",
                options={
                    "initial_simplex": self._build_simplex(start),
                    "maxiter": maxiter,
                    "fatol": tol / n_total,
                    "xatol": xatol,
                    "adaptive": True,
                    "disp": False,
                },
            )

        # Screen the initial guess plus Latin-hypercube points across the bounds
        guesses = init_params if isinstance(init_params, (list, tuple)) else [init_params or {}]
        candidates = [self._clip_to_bounds(self._pack_params(g or {})) for g in guesses]
        if n_screen > 0:
            bounds = self._bounds()
            lhs = qmc.LatinHypercube(d=len(self.param_names), seed=self.crn_seed).random(n_screen)
            candidates.extend(qmc.scale(lhs, bounds[:, 0], bounds[:, 1]))
        screen_losses = [objective(c) for c in candidates]
        starts = [candidates[i] for i in np.argsort(screen_losses)[: max(1, n_starts)]]

        # Nelder-Mead from each start, keep the best, then polish it
        best = min((run_nelder_mead(s) for s in starts), key=lambda r: r.fun)
        total_nit = int(best.nit)
        converged = bool(best.success)
        for _ in range(max(0, n_restarts)):
            res = run_nelder_mead(self._clip_to_bounds(best.x))
            total_nit += int(res.nit)
            converged = bool(res.success)
            if res.fun <= best.fun:
                best = res

        best_x = self._clip_to_bounds(best.x)
        best_params = self._unpack_params(best_x)

        # Final likelihood: more simulations and an independent seed
        final_template = self._simulation_template(qmp_data, self.n_sim_final)
        final_nll = self._qmp_nll(best_x, qmp_data, final_template, self.n_sim_final, self.crn_seed + 1)
        bic = compute_bic(final_nll, n_params=self.n_params, n_trials=qmp_data.total_trials)
        aic = compute_aic(final_nll, n_params=self.n_params)

        return {
            "params": best_params,
            "nll": float(final_nll),
            "bic": float(bic),
            "aic": float(aic),
            "success": converged,
            "nit": total_nit,
            "nfev": n_fev[0],
            "message": str(best.message),
            "model_type": self.model_type,
            "bound_type": self.bound_type,
        }


def fit_ddm(
    trials_df: pd.DataFrame,
    model_type: str = "M2",
    bound_type: str = "fixed",
    fit_history: bool = True,
    init_params: Optional[Any] = None,
    maxiter: int = 400,
    n_sim_per_cond: int = 400,
    seed: int = 12345,
    n_restarts: int = 1,
    n_screen: int = 24,
    n_starts: int = 2,
    n_sim_final: int = 2000,
    tau_b_grid: Optional[Sequence[float]] = DEFAULT_TAU_B_GRID,
    rt_bandwidth: Optional[float] = 0.01,
    t_max: float = 2.5,
    dt: float = 0.002,
    fit_beta: bool = False,
    fit_variability: bool = False,
    quantiles: Optional[Sequence[float]] = None,
    fixed_params: Optional[Dict[str, float]] = None,
    bias_type: Optional[str] = None,
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    rt_window: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """
    Standard top-level functional API adhering to §3.2 Model API specification.

    For M3, tau_b is profiled over tau_b_grid (one fit per value, b_max free) and the
    best final NLL is kept; tau_b still counts as a parameter in BIC/AIC. Pass
    tau_b_grid=None to fit tau_b freely.

    For real data pass `t_max` equal to the RT cutoff of the data (2.5 or 5.0 s; the default
    2.5 s cannot simulate the 5.0 s set), and `fit_beta=True, fit_variability=True,
    bias_type='drift'`.
    `bound_type` is 'fixed' or 'weibull' ('collapsing' is accepted as an alias).
    `rt_window=(0.1, cutoff)` switches to the truncation-aware likelihood used for C2 (default None = C1).
    `init_params` may be one dict or a list of dicts (several warm starts).
    """
    base_fixed = dict(fixed_params or {})

    def make_fitter(fixed_params=None, n_extra_params=0):
        return DDMFitter(
            model_type=model_type,
            bound_type=bound_type,
            fit_history=fit_history,
            n_sim_per_cond=n_sim_per_cond,
            dt=dt,
            t_max=t_max,
            crn_seed=seed,
            fixed_params={**base_fixed, **(fixed_params or {})},
            n_sim_final=n_sim_final,
            n_extra_params=n_extra_params,
            rt_bandwidth=rt_bandwidth,
            fit_beta=fit_beta,
            fit_variability=fit_variability,
            quantiles=quantiles,
            bias_type=bias_type,
            param_bounds=param_bounds,
            rt_window=rt_window,
        )

    fit_kwargs = dict(
        init_params=init_params, maxiter=maxiter, n_restarts=n_restarts, n_screen=n_screen, n_starts=n_starts,
    )

    if model_type.upper() == "M3" and tau_b_grid:
        grid_results = [
            make_fitter({"tau_b": float(tau)}, n_extra_params=1).fit(trials_df, **fit_kwargs)
            for tau in tau_b_grid
        ]
        best = min(grid_results, key=lambda r: r["nll"])
        best["tau_b_profile"] = {float(t): r["nll"] for t, r in zip(tau_b_grid, grid_results)}
        best["nfev"] = sum(r["nfev"] for r in grid_results)
        return best

    return make_fitter().fit(trials_df, **fit_kwargs)
