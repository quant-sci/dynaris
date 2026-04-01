"""Benchmark: Batch filtering via jax.vmap.

Measures the speedup from vectorized batch processing compared to
sequential filtering of multiple series.

Usage::

    uv run python benchmarks/bench_batch.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from dynaris.core.state_space import StateSpaceModel
from dynaris.filters.kalman import kalman_filter
from dynaris.forecast.forecast import fit_batch

# --- Configuration ---
N_SERIES_LIST = [1, 10, 50, 100, 500]
T = 200
STATE_DIM = 2
N_REPEATS = 3


def _make_model() -> StateSpaceModel:
    return StateSpaceModel(
        observation_matrix=jnp.eye(STATE_DIM),
        system_matrix=jnp.eye(STATE_DIM),
        obs_cov=jnp.eye(STATE_DIM),
        evolution_cov=jnp.eye(STATE_DIM) * 0.1,
    )


def _bench_sequential(model: StateSpaceModel, obs_batch: jax.Array) -> float:
    """Run Kalman filter sequentially on each series."""
    n_series = obs_batch.shape[0]

    # Warm up
    for i in range(min(2, n_series)):
        result = kalman_filter(model, obs_batch[i])
        jax.block_until_ready(result.log_likelihood)

    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        for i in range(n_series):
            result = kalman_filter(model, obs_batch[i])
            jax.block_until_ready(result.log_likelihood)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _bench_vmap(model: StateSpaceModel, obs_batch: jax.Array) -> float:
    """Run batch Kalman filter via jax.vmap."""
    # Warm up
    result = fit_batch(model, obs_batch)
    jax.block_until_ready(result.log_likelihood)

    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        result = fit_batch(model, obs_batch)
        jax.block_until_ready(result.log_likelihood)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main() -> None:
    print("=" * 60)
    print("Dynaris Batch Filtering Benchmark (vmap vs sequential)")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"T={T}, state_dim={STATE_DIM}")
    print("=" * 60)
    print()

    model = _make_model()

    print(f"{'N_series':>10} {'Sequential (ms)':>16} {'vmap (ms)':>12} {'Speedup':>8}")
    print("-" * 50)

    for n_series in N_SERIES_LIST:
        key = jax.random.PRNGKey(0)
        obs_batch = jax.random.normal(key, (n_series, T, STATE_DIM))

        seq_time = _bench_sequential(model, obs_batch)
        vmap_time = _bench_vmap(model, obs_batch)
        speedup = seq_time / vmap_time if vmap_time > 0 else float("inf")

        print(f"{n_series:>10} {seq_time * 1000:>16.2f} {vmap_time * 1000:>12.2f} {speedup:>7.1f}x")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
