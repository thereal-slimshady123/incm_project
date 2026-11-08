"""
Vectorized Euler-Maruyama Drift Diffusion Model (DDM) Simulator.

Supports:
- Three prior mechanisms:
    * M1: Starting Point Bias (x(0) = a * z * prior)
    * M2: Drift Bias (v(t) = v_sensory + v_h * y_{t-1} + v_b * prior)
    * M3: Dynamic Bias (x_eff(t) = x(t) + prior * b_max * t / (t + tau_b))
- Boundary dynamics: Fixed (+-a) and Weibull collapsing boundaries
- Trial-to-trial variabilities: Starting point (s_z) and drift rate (s_v)
- Standardized §3.1 Trial DataFrame output schema
"""

import os
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd

from .boundaries import get_boundary_function
from ._kernel import HAVE_NUMBA

if HAVE_NUMBA:
    from ._kernel import simulate_kernel


class DDMSimulator:
    """
    High-performance Euler-Maruyama DDM Simulator.
    Simulates thousands of trials vectorized across time steps.
    """

    def __init__(
        self,
        dt: float = 0.001,
        t_max: float = 2.5,
        sigma: float = 1.0,
        backend: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        dt : float
            Simulation time step in seconds (default 1 ms = 0.001 s).
        t_max : float
            Maximum decision time in seconds (default 2.5 s).
        sigma : float
            Diffusion noise scale (fixed to 1.0 for identifiability; Bogacz et al., 2006).
        backend : str, optional
            'numba' (compiled per-trial loop on the CPU), 'cuda' (the same loop on an NVIDIA GPU, float32; needs
            numba-cuda, see `_kernel_cuda`) or 'numpy' (vectorized fallback). Defaults to the environment variable
            INCM_BACKEND if set, otherwise 'numba' when numba is installed.
        """
        self.dt = dt
        self.t_max = t_max
        self.sigma = sigma
        if backend is None:
            backend = os.environ.get("INCM_BACKEND") or ("numba" if HAVE_NUMBA else "numpy")
        backend = backend.lower()
        if backend == "numba" and not HAVE_NUMBA:
            raise ImportError("backend='numba' requested but numba is not installed")
        if backend == "cuda":
            from ._kernel_cuda import HAVE_CUDA
            if not HAVE_CUDA:
                raise ImportError("backend='cuda' requested but numba-cuda or an NVIDIA GPU is not available")
        elif backend not in ("numba", "numpy"):
            raise ValueError(f"unknown backend '{backend}' (use 'numba', 'cuda' or 'numpy')")
        self.backend = backend
        self.n_steps = int(np.ceil(t_max / dt))
        # Time at the start of each Euler step (exact spacing dt)
        self.time_grid = np.arange(self.n_steps) * dt

    def simulate(
        self,
        params: Dict[str, float],
        contrasts: np.ndarray,
        priors: np.ndarray,
        prev_choices: Optional[np.ndarray] = None,
        model_type: str = "M2",
        bound_type: str = "fixed",
        subject_id: str = "sim_agent",
        rng: Optional[np.random.Generator] = None,
        stream_ids: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Simulate a batch of trials.

        Parameters
        ----------
        params : dict
            Model parameters:
            - 'a': boundary height (> 0)
            - 't0': non-decision time in seconds (> 0)
            - 'k': sensory sensitivity (> 0)
            - 'beta': contrast non-linearity exponent (default 1.0)
            - 'v_h': choice history weight (optional, default 0.0)
            - 'v_0': neutral side-bias drift offset, independent of the prior (default 0.0)
            - 'z_0': neutral side-bias starting-point offset as a fraction of a (default 0.0)
            - 'z': starting point bias parameter (for M1; in [-1, 1], default 0.0)
            - 'v_b': drift bias parameter (for M2; default 0.0)
            - 'b_max': dynamic bias ceiling (for M3; default 0.0)
            - 'tau_b': dynamic bias growth rate (for M3; seconds, default 0.3)
            - 's_v': trial-to-trial drift variability std (optional, default 0.0)
            - 's_z': trial-to-trial starting point variability range (optional, default 0.0)
            - 'lam': Weibull scale parameter (for collapsing bounds; default 1.5)
            - 'gamma': Weibull shape parameter (for collapsing bounds; default 2.0)
        contrasts : np.ndarray
            1D array of signed contrasts, c in [-1.0, 1.0].
        priors : np.ndarray
            1D array of block priors (-1.0 for left-biased, +1.0 for right-biased, 0.0 for neutral).
        prev_choices : np.ndarray, optional
            1D array of previous choices (-1 for Left, +1 for Right, 0 for neutral/start).
        model_type : str
            'M1' (Starting Point), 'M2' (Drift Bias), 'M3' (Dynamic Bias), or
            'JOINT' (starting point + drift bias, for z vs v_b trade-off tests).
        bound_type : str
            'fixed' or 'weibull'.
        subject_id : str
            Identifier for generated trials.
        rng : np.random.Generator, optional
            Numpy random generator for reproducibility / CRN.
        stream_ids : np.ndarray, optional
            Noise stream index per trial. Trials sharing a stream receive identical
            noise and variability draws (used to chain choice history exactly).
            Defaults to one stream per trial.

        Notes
        -----
        Each trial's noise sequence is fixed by the seed and its stream id, regardless
        of parameters (numba: private per-stream generator; numpy: noise drawn for every
        stream at every step). This keeps Common Random Numbers valid across objective
        evaluations.

        Returns
        -------
        pd.DataFrame
            DataFrame adhering to §3.1 contract:
            columns: ['subject_id', 'trial_idx', 'contrast', 'prior', 'prev_choice', 'choice', 'rt']
        """
        contrasts = np.asarray(contrasts, dtype=np.float64)
        priors = np.asarray(priors, dtype=np.float64)
        n_trials = len(contrasts)

        if prev_choices is None:
            prev_choices = np.zeros(n_trials, dtype=np.float64)
        else:
            prev_choices = np.asarray(prev_choices, dtype=np.float64)

        if rng is None:
            rng = np.random.default_rng()

        if stream_ids is None:
            stream_ids = np.arange(n_trials)
        else:
            stream_ids = np.asarray(stream_ids, dtype=np.int64)
        n_streams = int(stream_ids.max()) + 1 if n_trials > 0 else 0

        # Extract parameters with safe defaults
        a = float(params.get("a", 1.0))
        t0 = float(params.get("t0", 0.2))
        k = float(params.get("k", 2.5))
        beta = float(params.get("beta", 1.0))
        v_h = float(params.get("v_h", 0.0))
        v_0 = float(params.get("v_0", 0.0))
        z_0 = float(params.get("z_0", 0.0))
        z = float(params.get("z", 0.0))
        v_b = float(params.get("v_b", 0.0))
        b_max = float(params.get("b_max", 0.0))
        tau_b = float(params.get("tau_b", 0.3))
        s_v = float(params.get("s_v", 0.0))
        s_z = float(params.get("s_z", 0.0))
        lam = float(params.get("lam", 1.5))
        gamma = float(params.get("gamma", 2.0))

        # Sensory drift calculation: k * sign(c) * |c|^beta
        abs_c = np.abs(contrasts)
        v_sensory = k * np.sign(contrasts) * (abs_c ** beta)

        # Baseline drift for each trial
        drift_rate = v_sensory + v_h * prev_choices + v_0

        model_key = model_type.upper()
        if model_key in ("M2", "JOINT"):
            drift_rate = drift_rate + v_b * priors

        # Starting point calculation
        if model_key in ("M1", "JOINT"):
            # Starting point shifted by prior: x0 = a * z * prior
            # where z in [-1, 1] scales starting position relative to boundary a
            mean_x0 = a * np.clip(z, -0.95, 0.95) * priors
        else:
            mean_x0 = np.zeros(n_trials, dtype=np.float64)
        if z_0 != 0.0:
            mean_x0 = mean_x0 + a * np.clip(z_0, -0.95, 0.95)

        # Boundary function
        bound_fn = get_boundary_function(bound_type)
        if bound_type.lower() in ("collapsing", "weibull"):
            upper_bounds = bound_fn(self.time_grid, a=a, lam=lam, gamma=gamma)
        else:
            upper_bounds = bound_fn(self.time_grid, a=a)

        # Precompute dynamic bias factor if M3: prior * b_max * t / (t + tau_b)
        # Time-dependent dynamic prior drift: grows with elapsed decision time t
        # Hanks et al. (2011): uncertain/slow trials increasingly rely on the prior
        if model_key == "M3" and b_max > 1e-6:
            dynamic_factors = b_max * self.time_grid / (self.time_grid + np.maximum(tau_b, 1e-4))
        else:
            dynamic_factors = np.zeros(self.n_steps, dtype=np.float64)

        if self.backend == "cuda":
            from ._kernel_cuda import simulate_kernel_cuda
            choices, decision_time = simulate_kernel_cuda(
                drift_rate, mean_x0, priors, stream_ids, int(rng.integers(0, 2**63 - 1)), s_v, s_z * a, 0.95 * a,
                self.dt, self.sigma, self.t_max, upper_bounds, dynamic_factors,
            )
        elif self.backend == "numba":
            choices = np.zeros(n_trials, dtype=np.int8)
            decision_time = np.empty(n_trials, dtype=np.float64)
            simulate_kernel(
                np.ascontiguousarray(drift_rate), mean_x0, priors, stream_ids,
                int(rng.integers(0, 2**63 - 1)), s_v, s_z * a, 0.95 * a,
                self.dt, self.sigma, self.t_max, upper_bounds, dynamic_factors,
                choices, decision_time,
            )
        else:
            choices, decision_time = self._simulate_numpy(
                drift_rate, mean_x0, priors, stream_ids, n_streams, s_v, s_z, a,
                upper_bounds, dynamic_factors, rng,
            )

        # Total Reaction Time (RT) = decision time + non-decision time t0
        rt = decision_time + t0

        df = pd.DataFrame({
            "subject_id": subject_id,
            "trial_idx": np.arange(n_trials, dtype=np.int32),
            "contrast": contrasts,
            "prior": priors,
            "prev_choice": prev_choices.astype(np.int8),
            "choice": choices,
            "rt": rt,
        })
        return df

    def _simulate_numpy(
        self, drift_rate, mean_x0, priors, stream_ids, n_streams, s_v, s_z, a,
        upper_bounds, dynamic_factors, rng,
    ):
        """Pure-NumPy fallback, vectorized across trials (used when numba is unavailable)."""
        n_trials = len(drift_rate)
        lower_bounds = -upper_bounds

        # Variability draws are always taken (parameter-independent RNG consumption for CRN)
        eta_v = rng.standard_normal(n_streams)[stream_ids]
        eta_z = rng.uniform(-0.5, 0.5, size=n_streams)[stream_ids]

        # Trial-to-trial drift variability s_v: normal(0, s_v)
        drift_rate = drift_rate + s_v * eta_v

        # Trial-to-trial starting point variability s_z: uniform(-s_z*a/2, +s_z*a/2)
        # Clip so initial starting point is strictly within initial boundaries (-0.95*a, 0.95*a)
        x0 = np.clip(mean_x0 + s_z * a * eta_z, -0.95 * a, 0.95 * a)

        # Simulation state
        x = x0.copy()
        decision_time = np.full(n_trials, np.nan, dtype=np.float64)
        choices = np.zeros(n_trials, dtype=np.int8)  # -1 (Left), +1 (Right), 0 (Undecided)
        active_indices = np.arange(n_trials)

        sqrt_dt = np.sqrt(self.dt)
        std_noise = self.sigma * sqrt_dt

        # Vectorized Euler-Maruyama time-stepping
        for s in range(self.n_steps):
            if len(active_indices) == 0:
                break

            t = self.time_grid[s]
            upper_b = upper_bounds[s]
            lower_b = lower_bounds[s]

            # Draw increments for all streams, then pick those of active trials (CRN-stable)
            step_noise = rng.standard_normal(n_streams)
            noise = std_noise * step_noise[stream_ids[active_indices]]

            # Determine instantaneous drift rate (plus M3 time-dependent dynamic bias)
            current_drift = drift_rate[active_indices] + priors[active_indices] * dynamic_factors[s]

            # Update accumulated evidence
            x[active_indices] += current_drift * self.dt + noise
            x_check = x[active_indices]

            # Boundary crossing check
            hit_upper = x_check >= upper_b
            hit_lower = x_check <= lower_b
            crossed = hit_upper | hit_lower

            if np.any(crossed):
                crossed_idx = active_indices[crossed]
                # If hitting upper boundary -> Right (+1), if lower -> Left (-1)
                crossed_choices = np.where(hit_upper[crossed], 1, -1)

                choices[crossed_idx] = crossed_choices
                # Crossing happens at the end of this step
                decision_time[crossed_idx] = t + self.dt

                # Remove crossed trials from active pool
                active_indices = active_indices[~crossed]

        # Handle any timeout / uncrossed trials (assign nearest boundary or random choice at t_max)
        if len(active_indices) > 0:
            undecided = active_indices
            decision_time[undecided] = self.t_max
            # Choose according to sign of final state
            final_sign = np.sign(x[undecided])
            # If exactly 0, coin flip
            final_sign[final_sign == 0] = rng.choice([-1, 1], size=np.sum(final_sign == 0))
            choices[undecided] = final_sign.astype(np.int8)

        return choices, decision_time


def simulate_ddm(
    params: Dict[str, float],
    model_type: str = "M2",
    contrasts: Optional[np.ndarray] = None,
    priors: Optional[np.ndarray] = None,
    prev_choices: Optional[np.ndarray] = None,
    n_trials: int = 1000,
    bound_type: str = "fixed",
    subject_id: str = "sim_001",
    seed: Optional[int] = None,
    dt: float = 0.001,
    t_max: float = 2.5,
) -> pd.DataFrame:
    """
    Standard top-level functional API adhering to §3.2 Model API specification.
    """
    rng = np.random.default_rng(seed)

    if contrasts is None or priors is None:
        # Standard IBL contrast set and block prior structure
        ibl_contrasts = np.array([-1.0, -0.5, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 0.5, 1.0])
        # Alternating blocks of 80/20 (+1.0) and 20/80 (-1.0)
        block_len = 50
        n_blocks = int(np.ceil(n_trials / block_len))
        block_priors = np.array([1.0 if (b % 2 == 0) else -1.0 for b in range(n_blocks)])
        priors_full = np.repeat(block_priors, block_len)[:n_trials]

        # Draw contrasts according to prior probabilities
        sampled_contrasts = []
        for p in priors_full:
            # In 80/20 block (p = +1), 80% positive contrasts, 20% negative
            if p > 0:
                p_pos = 0.8
            elif p < 0:
                p_pos = 0.2
            else:
                p_pos = 0.5

            if rng.random() < p_pos:
                c = rng.choice(ibl_contrasts[ibl_contrasts >= 0])
            else:
                c = rng.choice(ibl_contrasts[ibl_contrasts <= 0])
            sampled_contrasts.append(c)

        contrasts = np.array(sampled_contrasts)
        priors = priors_full

    simulator = DDMSimulator(dt=dt, t_max=t_max)

    if prev_choices is None:
        # Chain 1-back history: prev_choice[t] is the simulated choice on trial t-1
        return _simulate_chained(
            simulator, params, model_type, np.asarray(contrasts, dtype=np.float64),
            np.asarray(priors, dtype=np.float64), bound_type, subject_id, rng,
        )

    return simulator.simulate(
        params=params,
        contrasts=contrasts,
        priors=priors,
        prev_choices=prev_choices,
        model_type=model_type,
        bound_type=bound_type,
        subject_id=subject_id,
        rng=rng,
    )


def _simulate_chained(
    simulator: DDMSimulator,
    params: Dict[str, float],
    model_type: str,
    contrasts: np.ndarray,
    priors: np.ndarray,
    bound_type: str,
    subject_id: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Simulate a session in which prev_choice is the agent's own previous choice.

    Each trial t >= 1 is simulated twice in one vectorized pass, once with
    prev_choice = -1 and once with +1, sharing the same noise stream. Walking the
    chain then picks the copy matching the choice actually made on trial t-1,
    which is exactly equivalent to sequential trial-by-trial simulation.
    """
    n_trials = len(contrasts)
    rest = np.arange(1, n_trials)
    trial_of_row = np.concatenate(([0], rest, rest))
    prev_of_row = np.concatenate(([0.0], -np.ones(n_trials - 1), np.ones(n_trials - 1)))

    sim = simulator.simulate(
        params=params,
        contrasts=contrasts[trial_of_row],
        priors=priors[trial_of_row],
        prev_choices=prev_of_row,
        model_type=model_type,
        bound_type=bound_type,
        subject_id=subject_id,
        rng=rng,
        stream_ids=trial_of_row,
    )
    sim_choice = sim["choice"].to_numpy()
    sim_rt = sim["rt"].to_numpy()

    choices = np.empty(n_trials, dtype=np.int8)
    rts = np.empty(n_trials, dtype=np.float64)
    prev = np.zeros(n_trials, dtype=np.int8)
    choices[0], rts[0] = sim_choice[0], sim_rt[0]
    for t in range(1, n_trials):
        prev[t] = choices[t - 1]
        # Row t holds the prev=-1 copy, row t + (n_trials - 1) the prev=+1 copy
        row = t if prev[t] < 0 else t + n_trials - 1
        choices[t], rts[t] = sim_choice[row], sim_rt[row]

    return pd.DataFrame({
        "subject_id": subject_id,
        "trial_idx": np.arange(n_trials, dtype=np.int32),
        "contrast": contrasts,
        "prior": priors,
        "prev_choice": prev,
        "choice": choices,
        "rt": rts,
    })
