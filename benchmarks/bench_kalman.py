"""Benchmark: Kalman filter speed across series lengths and state dimensions.

Measures wall-clock time for the JAX Kalman filter (JIT-compiled) and
the pure NumPy backend. Reports speedup from JIT compilation and GPU.

Usage::

    uv run python benchmarks/bench_kalman.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from dynaris.backends.numpy_backend import kalman_filter_numpy
from dynaris.core.state_space import StateSpaceModel
from dynaris.filters.kalman import kalman_filter

# --- Configuration ---
SERIES_LENGTHS = [100, 500, 1_000, 5_000, 10_000]
STATE_DIMS = [1, 5, 10]
N_REPEATS = 5


def _make_model(n: int) -> StateSpaceModel:
    """Create a random walk model with state_dim=n, obs_dim=n."""
    return StateSpaceModel(
        observation_matrix=jnp.eye(n),
        system_matrix=jnp.eye(n),
        obs_cov=jnp.eye(n),
        evolution_cov=jnp.eye(n) * 0.1,
    )


def _bench_jax(model: StateSpaceModel, obs: jax.Array, n_repeats: int) -> float:
    """Benchmark JAX Kalman filter (excludes first JIT compile)."""
    # Warm up JIT
    _ = kalman_filter(model, obs)
    jax.block_until_ready(kalman_filter(model, obs).log_likelihood)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = kalman_filter(model, obs)
        jax.block_until_ready(result.log_likelihood)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _bench_numpy(
    f: np.ndarray,
    g: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    obs: np.ndarray,
    n_repeats: int,
) -> float:
    """Benchmark NumPy Kalman filter."""
    # Warm up
    _ = kalman_filter_numpy(f, g, v, w, obs)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = kalman_filter_numpy(f, g, v, w, obs)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main() -> None:
    print("=" * 70)
    print("Dynaris Kalman Filter Benchmark")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print("=" * 70)
    print()

    print(f"{'T':>8} {'n':>4} {'JAX (ms)':>10} {'NumPy (ms)':>12} {'Speedup':>8}")
    print("-" * 50)

    for n in STATE_DIMS:
        model = _make_model(n)
        f_np = np.asarray(model.F)
        g_np = np.asarray(model.G)
        v_np = np.asarray(model.V)
        w_np = np.asarray(model.W)

        for t_len in SERIES_LENGTHS:
            key = jax.random.PRNGKey(0)
            obs_jax = jax.random.normal(key, (t_len, n))
            obs_np = np.asarray(obs_jax)

            jax_time = _bench_jax(model, obs_jax, N_REPEATS)
            numpy_time = _bench_numpy(f_np, g_np, v_np, w_np, obs_np, N_REPEATS)
            speedup = numpy_time / jax_time if jax_time > 0 else float("inf")

            print(
                f"{t_len:>8} {n:>4} {jax_time * 1000:>10.2f} "
                f"{numpy_time * 1000:>12.2f} {speedup:>7.1f}x"
            )

        print()

    print("Done.")


if __name__ == "__main__":
    main()
