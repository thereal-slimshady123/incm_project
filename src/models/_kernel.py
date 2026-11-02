"""
Numba-compiled Euler-Maruyama kernel for the DDM simulator.

Each trial runs its own time loop with a private SplitMix64 random stream seeded
from (seed, stream_id). A trial's noise sequence therefore depends only on the
seed and its stream id, never on the parameters or on other trials, which keeps
Common Random Numbers valid across objective evaluations without storing a noise
matrix. Trials that share a stream id receive identical noise.
"""

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - fallback path is exercised where numba is absent
    HAVE_NUMBA = False

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_S30 = np.uint64(30)
_S27 = np.uint64(27)
_S31 = np.uint64(31)
_S11 = np.uint64(11)
_INV53 = 1.0 / 9007199254740992.0  # 2**-53


if HAVE_NUMBA:

    @njit(cache=True, inline="always")
    def _mix(z):
        z = (z ^ (z >> _S30)) * _MIX1
        z = (z ^ (z >> _S27)) * _MIX2
        return z ^ (z >> _S31)

    @njit(cache=True, inline="always")
    def _uniform(state):
        """Advance the stream; return (new_state, u) with u in the open interval (0, 1)."""
        state = state + _GOLDEN
        u = (float(_mix(state) >> _S11) + 0.5) * _INV53
        return state, u

    @njit(cache=True, inline="always")
    def _normal_pair(state):
        """Marsaglia polar method: two independent standard normals, no trig calls."""
        while True:
            state, u1 = _uniform(state)
            state, u2 = _uniform(state)
            v1 = 2.0 * u1 - 1.0
            v2 = 2.0 * u2 - 1.0
            r2 = v1 * v1 + v2 * v2
            if 0.0 < r2 < 1.0:
                f = np.sqrt(-2.0 * np.log(r2) / r2)
                return state, v1 * f, v2 * f

    @njit(cache=True)
    def simulate_kernel(
        base_drift, mean_x0, priors, stream_ids, seed,
        s_v, z_range, x0_lim, dt, sigma, t_max, upper, dyn,
        choices_out, dtime_out,
    ):
        """
        Simulate every trial to a boundary crossing or t_max.

        base_drift : drift without across-trial variability, per trial
        mean_x0    : starting point without across-trial variability, per trial
        z_range    : width of the uniform starting-point variability (s_z * a)
        x0_lim     : starting points are clipped to [-x0_lim, x0_lim]
        upper      : upper boundary at the start of each step (lower is -upper)
        dyn        : M3 dynamic bias factor per step (zeros for other models)
        """
        n_trials = base_drift.shape[0]
        n_steps = upper.shape[0]
        noise_scale = sigma * np.sqrt(dt)
        seed_u = np.uint64(seed)

        for i in range(n_trials):
            state = _mix(seed_u ^ _mix(np.uint64(stream_ids[i]) + _GOLDEN))

            # Variability draws come first, so every stream consumes them identically
            state, eta_v, spare = _normal_pair(state)
            has_spare = True
            state, u_z = _uniform(state)
            drift = base_drift[i] + s_v * eta_v
            x = mean_x0[i] + z_range * (u_z - 0.5)
            if x > x0_lim:
                x = x0_lim
            elif x < -x0_lim:
                x = -x0_lim

            prior = priors[i]
            choice = 0
            for s in range(n_steps):
                # Use both normals of each polar pair
                if has_spare:
                    eps = spare
                    has_spare = False
                else:
                    state, eps, spare = _normal_pair(state)
                    has_spare = True
                x += (drift + prior * dyn[s]) * dt + noise_scale * eps
                if x >= upper[s]:
                    choice = 1
                    dtime_out[i] = (s + 1) * dt
                    break
                if x <= -upper[s]:
                    choice = -1
                    dtime_out[i] = (s + 1) * dt
                    break

            if choice == 0:
                # Timeout: choose by the sign of the final state, coin flip if exactly 0
                dtime_out[i] = t_max
                if x > 0.0:
                    choice = 1
                elif x < 0.0:
                    choice = -1
                else:
                    state, u = _uniform(state)
                    choice = 1 if u < 0.5 else -1
            choices_out[i] = choice
