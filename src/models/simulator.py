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

from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd

from .boundaries import get_boundary_function


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
        """
        self.dt = dt
        self.t_max = t_max
        self.sigma = sigma
        self.n_steps = int(np.ceil(t_max / dt))
        self.time_grid = np.linspace(0, t_max, self.n_steps)

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
            'M1' (Starting Point), 'M2' (Drift Bias), or 'M3' (Dynamic Bias).
        bound_type : str
            'fixed' or 'weibull'.
        subject_id : str
            Identifier for generated trials.
        rng : np.random.Generator, optional
            Numpy random generator for reproducibility / CRN.

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

        # Extract parameters with safe defaults
        a = float(params.get("a", 1.0))
        t0 = float(params.get("t0", 0.2))
        k = float(params.get("k", 2.5))
        beta = float(params.get("beta", 1.0))
        v_h = float(params.get("v_h", 0.0))
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
        drift_rate = v_sensory + v_h * prev_choices

        model_key = model_type.upper()
        if model_key == "M2":
            drift_rate = drift_rate + v_b * priors

        # Trial-to-trial drift variability s_v: normal(0, s_v)
        if s_v > 1e-6:
            drift_rate = drift_rate + rng.normal(0.0, s_v, size=n_trials)

        # Starting point calculation
        if model_key == "M1":
            # Starting point shifted by prior: x0 = a * z * prior
            # where z in [-1, 1] scales starting position relative to boundary a
            mean_x0 = a * np.clip(z, -0.95, 0.95) * priors
        else:
            mean_x0 = np.zeros(n_trials, dtype=np.float64)

        # Trial-to-trial starting point variability s_z: uniform(-s_z*a/2, +s_z*a/2)
        if s_z > 1e-6:
            half_range = 0.5 * s_z * a
            x0 = mean_x0 + rng.uniform(-half_range, half_range, size=n_trials)
            # Clip so initial starting point is strictly within initial boundaries (-0.95*a, 0.95*a)
            x0 = np.clip(x0, -0.95 * a, 0.95 * a)
        else:
            x0 = np.clip(mean_x0, -0.95 * a, 0.95 * a)

        # Boundary function
        bound_fn = get_boundary_function(bound_type)
        if bound_type.lower() in ("collapsing", "weibull"):
            upper_bounds = bound_fn(self.time_grid, a=a, lam=lam, gamma=gamma)
        else:
            upper_bounds = bound_fn(self.time_grid, a=a)
        lower_bounds = -upper_bounds

        # Precompute dynamic bias factor if M3: prior * b_max * t / (t + tau_b)
        is_m3 = (model_key == "M3") and (b_max > 1e-6)
        if is_m3:
            # Time-dependent dynamic prior drift: grows with elapsed decision time t
            # Hanks et al. (2011): uncertain/slow trials increasingly rely on the prior
            t_over_tau = self.time_grid / (self.time_grid + np.maximum(tau_b, 1e-4))
            dynamic_factors = b_max * t_over_tau  # (n_steps,)

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

            # Generate Gaussian increment for active trials
            n_active = len(active_indices)
            noise = rng.normal(0.0, std_noise, size=n_active)

            # Determine instantaneous drift rate
            current_drift = drift_rate[active_indices]
            if is_m3:
                # Add time-dependent dynamic bias drift offset
                current_drift = current_drift + priors[active_indices] * dynamic_factors[s]

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
                decision_time[crossed_idx] = t

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

    if prev_choices is None:
        # Initial dummy history; in sequential simulation this can be chained
        prev_choices = np.zeros(len(contrasts), dtype=int)

    simulator = DDMSimulator(dt=dt, t_max=t_max)
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
