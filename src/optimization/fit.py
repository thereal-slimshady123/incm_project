"""
Drift Diffusion Model Parameter Fitting Engine via Quantile Maximum Probability (QMP).

Uses SciPy Nelder-Mead simplex optimization with Common Random Numbers (CRN)
for smooth variance-reduced objective surfaces.
Adheres strictly to §3.2 Model API specification.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..models.simulator import DDMSimulator
from .qmp import QMPDataset, compute_qmp_nll, compute_bic, compute_aic


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
}


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
    ):
        """
        Parameters
        ----------
        model_type : str
            'M1' (starting point), 'M2' (drift bias), or 'M3' (dynamic bias).
        bound_type : str
            'fixed' or 'weibull'.
        fit_history : bool
            Whether to fit 1-back choice history weight v_h.
        n_sim_per_cond : int
            Simulated trials per (contrast, prior) condition per objective evaluation.
        dt : float
            Euler-Maruyama step size for internal simulator during fitting.
        t_max : float
            Maximum decision time cutoff.
        crn_seed : int
            Common Random Numbers seed for deterministic objective evaluations.
        """
        self.model_type = model_type.upper()
        self.bound_type = bound_type.lower()
        self.fit_history = fit_history
        self.n_sim_per_cond = n_sim_per_cond
        self.dt = dt
        self.t_max = t_max
        self.crn_seed = crn_seed

        self.param_names = self._determine_param_names()
        self.simulator = DDMSimulator(dt=dt, t_max=t_max)

    def _determine_param_names(self) -> List[str]:
        """
        Select active parameter names for this model architecture.
        """
        names = ["a", "t0", "k"]
        if self.fit_history:
            names.append("v_h")

        if self.model_type == "M1":
            names.append("z")
        elif self.model_type == "M2":
            names.append("v_b")
        elif self.model_type == "M3":
            names.extend(["b_max", "tau_b"])
        else:
            raise ValueError(f"Unknown model_type: '{self.model_type}'")

        if self.bound_type in ("collapsing", "weibull"):
            names.extend(["lam", "gamma"])

        return names

    def _pack_params(self, param_dict: Dict[str, float]) -> np.ndarray:
        return np.array([param_dict.get(p, PARAM_SPECS[p]["init"]) for p in self.param_names])

    def _unpack_params(self, x: np.ndarray) -> Dict[str, float]:
        return {p: float(val) for p, val in zip(self.param_names, x)}

    def _compute_penalty(self, x: np.ndarray) -> float:
        """
        Quadratic penalty for parameters outside physiological bounds.
        """
        penalty = 0.0
        for p, val in zip(self.param_names, x):
            low, high = PARAM_SPECS[p]["bounds"]
            if val < low:
                penalty += 1000.0 * (low - val) ** 2
            elif val > high:
                penalty += 1000.0 * (high - val) ** 2
        return penalty

    def fit(
        self,
        trials_df: pd.DataFrame,
        init_params: Optional[Dict[str, float]] = None,
        maxiter: int = 150,
        tol: float = 1e-2,
    ) -> Dict[str, Any]:
        """
        Fit model parameters to empirical/target dataset via QMP Nelder-Mead.

        Parameters
        ----------
        trials_df : pd.DataFrame
            Target trials DataFrame conforming to §3.1 contract.
        init_params : dict, optional
            Initial parameter guess.
        maxiter : int
            Maximum iterations for Nelder-Mead.
        tol : float
            Convergence tolerance.

        Returns
        -------
        dict
            Results dictionary conforming to §3.2 Model API:
            {'params': dict, 'nll': float, 'bic': float, 'aic': float, 'success': bool, 'nit': int}
        """
        qmp_data = QMPDataset(trials_df)

        # Build condition simulation template
        cond_contrasts = []
        cond_priors = []
        for cond in qmp_data.conditions:
            cond_contrasts.extend([cond["contrast"]] * self.n_sim_per_cond)
            cond_priors.extend([cond["prior"]] * self.n_sim_per_cond)

        sim_contrasts = np.array(cond_contrasts, dtype=np.float64)
        sim_priors = np.array(cond_priors, dtype=np.float64)
        sim_prev_choices = np.zeros_like(sim_contrasts)  # Controlled dummy history during fitting

        # Build initial simplex with physiologically scaled step sizes
        # Prevents Nelder-Mead from premature contraction in 5D parameter space
        simplex_steps = {
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
        }

        simplex = [x0.copy()]
        for idx, p in enumerate(self.param_names):
            vertex = x0.copy()
            step = simplex_steps.get(p, 0.25)
            # Ensure vertex remains within bounds
            low, high = PARAM_SPECS[p]["bounds"]
            if vertex[idx] + step <= high:
                vertex[idx] += step
            else:
                vertex[idx] -= step
            simplex.append(vertex)
        initial_simplex = np.array(simplex)

        def objective(x_vec: np.ndarray) -> float:
            # Check boundary penalty
            penalty = self._compute_penalty(x_vec)
            if penalty > 1e4:
                return 1e6 + penalty

            p_dict = self._unpack_params(x_vec)
            # Use fixed CRN generator for smooth evaluation
            rng = np.random.default_rng(self.crn_seed)

            sim_df = self.simulator.simulate(
                params=p_dict,
                contrasts=sim_contrasts,
                priors=sim_priors,
                prev_choices=sim_prev_choices,
                model_type=self.model_type,
                bound_type=self.bound_type,
                rng=rng,
            )

            nll, _ = compute_qmp_nll(sim_df, qmp_data)
            # Normalize by trial count for stable tolerance evaluation
            loss = (nll + penalty) / float(qmp_data.total_trials)
            return loss

        # Run adaptive Nelder-Mead optimization
        res = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={
                "initial_simplex": initial_simplex,
                "maxiter": maxiter,
                "adaptive": True,
                "disp": False,
            },
        )

        best_params = self._unpack_params(res.x)
        # Final evaluation with clean simulator
        rng_eval = np.random.default_rng(self.crn_seed)
        sim_eval = self.simulator.simulate(
            params=best_params,
            contrasts=sim_contrasts,
            priors=sim_priors,
            prev_choices=sim_prev_choices,
            model_type=self.model_type,
            bound_type=self.bound_type,
            rng=rng_eval,
        )
        final_nll, _ = compute_qmp_nll(sim_eval, qmp_data)
        bic = compute_bic(final_nll, n_params=len(self.param_names), n_trials=qmp_data.total_trials)
        aic = compute_aic(final_nll, n_params=len(self.param_names))

        return {
            "params": best_params,
            "nll": float(final_nll),
            "bic": float(bic),
            "aic": float(aic),
            "success": bool(res.success),
            "nit": int(res.nit),
            "nfev": int(res.nfev),
            "message": str(res.message),
            "model_type": self.model_type,
            "bound_type": self.bound_type,
        }


def fit_ddm(
    trials_df: pd.DataFrame,
    model_type: str = "M2",
    bound_type: str = "fixed",
    fit_history: bool = True,
    init_params: Optional[Dict[str, float]] = None,
    maxiter: int = 150,
    n_sim_per_cond: int = 400,
    seed: int = 12345,
) -> Dict[str, Any]:
    """
    Standard top-level functional API adhering to §3.2 Model API specification.
    """
    fitter = DDMFitter(
        model_type=model_type,
        bound_type=bound_type,
        fit_history=fit_history,
        n_sim_per_cond=n_sim_per_cond,
        crn_seed=seed,
    )
    return fitter.fit(trials_df=trials_df, init_params=init_params, maxiter=maxiter)
