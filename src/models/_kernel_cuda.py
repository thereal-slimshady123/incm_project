"""
CUDA (GPU) version of the Euler-Maruyama DDM kernel in `_kernel.py`, via numba-cuda.

One GPU thread simulates one trial. The random streams are the same SplitMix64 streams as the CPU kernel
(seeded from (seed, stream_id)), so Common Random Numbers hold exactly as on the CPU: a trial's noise depends only
on the seed and its stream id. The arithmetic is single precision (float32): consumer / older GPUs run float64 at
1/32-1/64 of the float32 rate, and float32 is ample for a walk of at most a few thousand steps. GPU and CPU results
are therefore statistically equivalent but not bit-identical (the CPU kernel draws 53-bit uniforms, this one 24-bit).

Enable with the environment variable INCM_BACKEND=cuda (or DDMSimulator(backend='cuda')). Each process picks one of
the visible GPUs (os.getpid() % n_gpus), so a pool of fit workers spreads over all GPUs of a node.
"""

import math
import os

import numpy as np

try:
    from numba import cuda
    HAVE_CUDA = bool(cuda.is_available())
except Exception:  # pragma: no cover - numba-cuda missing or no driver
    cuda = None
    HAVE_CUDA = False

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_S30 = np.uint64(30)
_S27 = np.uint64(27)
_S31 = np.uint64(31)
_S40 = np.uint64(40)
_INV24 = np.float32(1.0 / 16777216.0)  # 2**-24

_selected = False
THREADS = 128


if HAVE_CUDA:

    @cuda.jit(device=True, inline=True)
    def _mix(z):
        z = (z ^ (z >> _S30)) * _MIX1
        z = (z ^ (z >> _S27)) * _MIX2
        return z ^ (z >> _S31)

    @cuda.jit(device=True, inline=True)
    def _uniform(state):
        state = state + _GOLDEN
        u = (np.float32(_mix(state) >> _S40) + np.float32(0.5)) * _INV24
        return state, u

    @cuda.jit(device=True, inline=True)
    def _normal_pair(state):
        while True:
            state, u1 = _uniform(state)
            state, u2 = _uniform(state)
            v1 = np.float32(2.0) * u1 - np.float32(1.0)
            v2 = np.float32(2.0) * u2 - np.float32(1.0)
            r2 = v1 * v1 + v2 * v2
            if r2 > np.float32(0.0) and r2 < np.float32(1.0):
                f = math.sqrt(np.float32(-2.0) * math.log(r2) / r2)
                return state, v1 * f, v2 * f

    @cuda.jit
    def _kernel(base_drift, mean_x0, priors, stream_ids, seed, s_v, z_range, x0_lim, dt, noise_scale,
                t_max, upper, dyn, choices_out, dtime_out):
        i = cuda.grid(1)
        if i >= base_drift.shape[0]:
            return
        n_steps = upper.shape[0]
        state = _mix(seed ^ _mix(np.uint64(stream_ids[i]) + _GOLDEN))
        state, eta_v, spare = _normal_pair(state)
        has_spare = True
        state, u_z = _uniform(state)
        drift = base_drift[i] + s_v * eta_v
        x = mean_x0[i] + z_range * (u_z - np.float32(0.5))
        if x > x0_lim:
            x = x0_lim
        elif x < -x0_lim:
            x = -x0_lim
        prior = priors[i]
        choice = 0
        for s in range(n_steps):
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
            dtime_out[i] = t_max
            if x > np.float32(0.0):
                choice = 1
            elif x < np.float32(0.0):
                choice = -1
            else:
                state, u = _uniform(state)
                choice = 1 if u < np.float32(0.5) else -1
        choices_out[i] = choice


def _select_device() -> None:
    """Pin this process to one visible GPU (spread a worker pool over all GPUs of the node)."""
    global _selected
    if _selected:
        return
    n = len(cuda.gpus)
    if n > 1:
        cuda.select_device(os.getpid() % n)
    _selected = True


def simulate_kernel_cuda(base_drift, mean_x0, priors, stream_ids, seed, s_v, z_range, x0_lim, dt, sigma, t_max,
                         upper, dyn):
    """Same inputs as `_kernel.simulate_kernel` (without the output arrays); returns (choices int8, dtime float64)."""
    if not HAVE_CUDA:
        raise RuntimeError("CUDA backend requested but numba-cuda / a GPU is not available")
    _select_device()
    f32 = np.float32
    n = base_drift.shape[0]
    d_choice = cuda.device_array(n, dtype=np.int8)
    d_dtime = cuda.device_array(n, dtype=np.float32)
    blocks = (n + THREADS - 1) // THREADS
    _kernel[blocks, THREADS](
        cuda.to_device(np.ascontiguousarray(base_drift, dtype=f32)),
        cuda.to_device(np.ascontiguousarray(mean_x0, dtype=f32)),
        cuda.to_device(np.ascontiguousarray(priors, dtype=f32)),
        cuda.to_device(np.ascontiguousarray(stream_ids, dtype=np.int64)),
        np.uint64(seed), f32(s_v), f32(z_range), f32(x0_lim), f32(dt), f32(sigma * math.sqrt(dt)), f32(t_max),
        cuda.to_device(np.ascontiguousarray(upper, dtype=f32)),
        cuda.to_device(np.ascontiguousarray(dyn, dtype=f32)),
        d_choice, d_dtime,
    )
    return d_choice.copy_to_host(), d_dtime.copy_to_host().astype(np.float64)
