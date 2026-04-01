"""Dynamic Factor Model factory and utilities.

Provides a factory function to create DFM state-space models,
PCA-based initialization, and varimax rotation for interpretability.

References:
    Stock, J.H. and Watson, M.W. (2002). "Forecasting Using Principal
    Components from a Large Number of Predictors." JASA, 97(460).

    Kaiser, H.F. (1958). "The Varimax Criterion for Analytic Rotation
    in Factor Analysis." Psychometrika, 23(3), 187-200.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from dynaris.core.state_space import StateSpaceModel


def DynamicFactorModel(  # noqa: N802
    n_factors: int,
    n_variables: int,
    factor_order: int = 1,
    loadings: Array | None = None,
    obs_noise: float | Array = 1.0,
) -> StateSpaceModel:
    r"""Create a Dynamic Factor Model as a StateSpaceModel.

    Factor dynamics:

    .. math::

        f_t = G_f f_{t-1} + w_t, \quad w_t \sim N(0, I_r)

    Observations:

    .. math::

        y_t = \Lambda f_t + e_t, \quad e_t \sim N(0, R)

    where R is diagonal.

    Args:
        n_factors: Number of latent factors (r).
        n_variables: Number of observed variables (m).
        factor_order: AR order for factor dynamics. 1 = random walk.
        loadings: Initial loading matrix, shape (m, r). If ``None``,
            initialized randomly.
        obs_noise: Observation noise. Scalar (broadcast to diagonal)
            or array of shape (m,) for per-variable noise.

    Returns:
        StateSpaceModel with state_dim=r (or r*factor_order),
        obs_dim=m.

    Example::

        from dynaris.models import DynamicFactorModel

        model = DynamicFactorModel(n_factors=2, n_variables=10)
    """
    r, m = n_factors, n_variables

    # Loading matrix
    if loadings is None:
        loadings = jax.random.normal(jax.random.PRNGKey(0), (m, r)) / jnp.sqrt(r)
    loadings = jnp.asarray(loadings)

    # Observation noise (diagonal)
    if jnp.ndim(obs_noise) == 0:
        obs_cov = jnp.eye(m) * float(obs_noise)
    else:
        obs_cov = jnp.diag(jnp.asarray(obs_noise))

    # Factor transition
    if factor_order == 1:
        system_matrix = jnp.eye(r)
        evolution_cov = jnp.eye(r)
        observation_matrix = loadings
    else:
        # Companion form for VAR(p) factors
        state_dim = r * factor_order
        system_matrix = jnp.zeros((state_dim, state_dim))
        system_matrix = system_matrix.at[:r, :r].set(jnp.eye(r))
        if factor_order > 1:
            system_matrix = system_matrix.at[r:, : r * (factor_order - 1)].set(
                jnp.eye(r * (factor_order - 1))
            )
        evolution_cov = jnp.zeros((state_dim, state_dim))
        evolution_cov = evolution_cov.at[:r, :r].set(jnp.eye(r))
        # Observation maps only the first r state components
        observation_matrix = jnp.zeros((m, state_dim))
        observation_matrix = observation_matrix.at[:, :r].set(loadings)

    return StateSpaceModel(
        observation_matrix=observation_matrix,
        system_matrix=system_matrix,
        obs_cov=obs_cov,
        evolution_cov=evolution_cov,
    )


def initialize_loadings_pca(
    observations: Array,
    n_factors: int,
) -> tuple[Array, Array]:
    """Initialize factor loadings via PCA (SVD of standardized data).

    Args:
        observations: Data matrix, shape (T, m).
        n_factors: Number of factors to extract.

    Returns:
        Tuple of (loadings, initial_factors).
        loadings: shape (m, r).
        initial_factors: shape (T, r).
    """
    observations = jnp.asarray(observations)
    # Standardize columns
    means = jnp.mean(observations, axis=0)
    stds = jnp.std(observations, axis=0)
    stds = jnp.maximum(stds, 1e-8)
    x_std = (observations - means) / stds

    # SVD
    u, s, vh = jnp.linalg.svd(x_std, full_matrices=False)

    # Loadings: scale by singular values / sqrt(T) and undo standardization
    t = observations.shape[0]
    loadings = vh[:n_factors, :].T * s[:n_factors] / jnp.sqrt(t)  # (m, r)
    loadings = loadings * stds[:, None]  # undo standardization scaling

    # Initial factors
    initial_factors = u[:, :n_factors] * s[:n_factors]  # (T, r)

    return loadings, initial_factors


def rotate_loadings(
    loadings: Array,
    method: str = "varimax",
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[Array, Array]:
    """Apply rotation to factor loadings for interpretability.

    Args:
        loadings: Loading matrix, shape (m, r).
        method: Rotation method. Currently supports ``"varimax"``.
        max_iter: Maximum iterations for the rotation algorithm.
        tol: Convergence tolerance.

    Returns:
        Tuple of (rotated_loadings, rotation_matrix).
        rotated_loadings: shape (m, r).
        rotation_matrix: shape (r, r), orthogonal.
    """
    if method != "varimax":
        msg = f"Unknown rotation method: {method!r}. Use 'varimax'."
        raise ValueError(msg)

    return _varimax(loadings, max_iter, tol)


def _varimax(loadings: Array, max_iter: int, tol: float) -> tuple[Array, Array]:
    """Kaiser varimax rotation via iterative pairwise Givens rotations."""
    m, r = loadings.shape
    rotation = jnp.eye(r)
    rotated = loadings.copy()

    for _ in range(max_iter):
        old_rotated = rotated
        for i in range(r):
            for j in range(i + 1, r):
                # Compute optimal rotation angle for columns i, j
                u_val = rotated[:, i] ** 2 - rotated[:, j] ** 2  # (m,)
                v_val = 2.0 * rotated[:, i] * rotated[:, j]  # (m,)

                a_val = jnp.sum(u_val)
                b_val = jnp.sum(v_val)
                c_val = jnp.sum(u_val**2 - v_val**2)
                d_val = 2.0 * jnp.sum(u_val * v_val)

                num = d_val - 2.0 * a_val * b_val / m
                den = c_val - (a_val**2 - b_val**2) / m

                angle = 0.25 * jnp.arctan2(num, den)

                # Apply Givens rotation
                cos_a = jnp.cos(angle)
                sin_a = jnp.sin(angle)
                col_i = rotated[:, i] * cos_a + rotated[:, j] * sin_a
                col_j = -rotated[:, i] * sin_a + rotated[:, j] * cos_a
                rotated = rotated.at[:, i].set(col_i)
                rotated = rotated.at[:, j].set(col_j)

                # Update rotation matrix
                rot_i = rotation[:, i] * cos_a + rotation[:, j] * sin_a
                rot_j = -rotation[:, i] * sin_a + rotation[:, j] * cos_a
                rotation = rotation.at[:, i].set(rot_i)
                rotation = rotation.at[:, j].set(rot_j)

        # Check convergence
        change = jnp.max(jnp.abs(rotated - old_rotated))
        if change < tol:
            break

    return rotated, rotation


def apply_identification_constraints(loadings: Array) -> Array:
    """Enforce identification constraints on the loading matrix.

    Makes the upper-left r x r block lower-triangular with positive
    diagonal entries. This fixes the rotational indeterminacy.

    Args:
        loadings: Loading matrix, shape (m, r).

    Returns:
        Constrained loading matrix, shape (m, r).
    """
    m, r = loadings.shape
    k = min(m, r)

    # Zero out upper triangle in the first k rows
    mask = jnp.tril(jnp.ones((k, r)))
    constrained = loadings.at[:k, :].set(loadings[:k, :] * mask)

    # Ensure positive diagonal
    diag_signs = jnp.sign(jnp.diag(constrained[:k, :k]))
    diag_signs = jnp.where(diag_signs == 0, 1.0, diag_signs)
    constrained = constrained * diag_signs[None, :]

    return constrained
