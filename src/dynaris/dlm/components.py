"""Modular DLM building blocks.

Each function returns a :class:`StateSpaceModel` that can be composed
with the ``+`` operator for block-diagonal superposition.

Example::

    model = LocalLevel(sigma_level=1.0, sigma_obs=1.0) + Seasonal(period=12)
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from dynaris.core.state_space import StateSpaceModel

# ---------------------------------------------------------------------------
# Local Level (random walk + noise)
# ---------------------------------------------------------------------------


def LocalLevel(  # noqa: N802
    sigma_level: float = 1.0,
    sigma_obs: float = 1.0,
) -> StateSpaceModel:
    """Local level model (random walk plus noise).

    State:  x_t = x_{t-1} + w_t,   w_t ~ N(0, sigma_level^2)
    Obs:    y_t = x_t + v_t,        v_t ~ N(0, sigma_obs^2)

    State dimension: 1.
    """
    return StateSpaceModel(
        system_matrix=jnp.array([[1.0]]),
        observation_matrix=jnp.array([[1.0]]),
        evolution_cov=jnp.array([[sigma_level**2]]),
        obs_cov=jnp.array([[sigma_obs**2]]),
    )


# ---------------------------------------------------------------------------
# Local Linear Trend (level + slope)
# ---------------------------------------------------------------------------


def LocalLinearTrend(  # noqa: N802
    sigma_level: float = 1.0,
    sigma_slope: float = 0.1,
    sigma_obs: float = 1.0,
) -> StateSpaceModel:
    """Local linear trend model (level + slope).

    State: [level_t, slope_t]
      level_t = level_{t-1} + slope_{t-1} + w1_t
      slope_t = slope_{t-1} + w2_t

    Obs: y_t = level_t + v_t

    State dimension: 2.
    """
    return StateSpaceModel(
        system_matrix=jnp.array([[1.0, 1.0], [0.0, 1.0]]),
        observation_matrix=jnp.array([[1.0, 0.0]]),
        evolution_cov=jnp.diag(jnp.array([sigma_level**2, sigma_slope**2])),
        obs_cov=jnp.array([[sigma_obs**2]]),
    )


# ---------------------------------------------------------------------------
# Seasonal component
# ---------------------------------------------------------------------------


def Seasonal(  # noqa: N802
    period: int,
    sigma_seasonal: float = 0.1,
    sigma_obs: float = 0.0,
    form: str = "dummy",
) -> StateSpaceModel:
    """Seasonal component.

    Args:
        period: Length of the seasonal cycle (e.g. 12 for monthly data).
        sigma_seasonal: Std dev of seasonal state noise.
        sigma_obs: Observation noise contribution (often 0 when
            composed with other components).
        form: ``"dummy"`` for dummy-variable form (state_dim = period - 1),
            ``"fourier"`` for trigonometric/Fourier form.

    Returns:
        StateSpaceModel representing the seasonal component.
    """
    if form == "dummy":
        return _seasonal_dummy(period, sigma_seasonal, sigma_obs)
    if form == "fourier":
        return _seasonal_fourier(period, sigma_seasonal, sigma_obs)
    msg = f"Unknown seasonal form: {form!r}. Use 'dummy' or 'fourier'."
    raise ValueError(msg)


def _seasonal_dummy(period: int, sigma_seasonal: float, sigma_obs: float) -> StateSpaceModel:
    """Dummy-variable seasonal: s_t = -s_{t-1} - s_{t-2} - ... - s_{t-p+1}.

    State dimension: period - 1.
    """
    s = period - 1
    # Transition: first row is [-1, -1, ..., -1], rest is identity shift
    transition = jnp.zeros((s, s))
    transition = transition.at[0, :].set(-1.0)
    transition = transition.at[1:, :-1].set(jnp.eye(s - 1))

    observation = jnp.zeros((1, s))
    observation = observation.at[0, 0].set(1.0)

    state_noise = jnp.zeros((s, s))
    state_noise = state_noise.at[0, 0].set(sigma_seasonal**2)

    return StateSpaceModel(
        system_matrix=transition,
        observation_matrix=observation,
        evolution_cov=state_noise,
        obs_cov=jnp.array([[sigma_obs**2]]),
    )


def _seasonal_fourier(period: int, sigma_seasonal: float, sigma_obs: float) -> StateSpaceModel:
    """Fourier-form seasonal using trigonometric harmonics.

    Uses floor(period / 2) harmonics. For each harmonic j:
      freq = 2 * pi * j / period
      Block: [[cos(freq), sin(freq)], [-sin(freq), cos(freq)]]

    If period is even, the last harmonic has state_dim=1 (cos(pi) = -1).

    State dimension: period - 1 (same as dummy form).
    """
    n_harmonics = period // 2
    blocks: list[Array] = []

    for j in range(1, n_harmonics + 1):
        freq = 2.0 * jnp.pi * j / period
        cos_f = jnp.cos(freq)
        sin_f = jnp.sin(freq)

        if j == n_harmonics and period % 2 == 0:
            # Last harmonic for even period: scalar -1 block
            blocks.append(jnp.array([[float(cos_f)]]))
        else:
            blocks.append(
                jnp.array(
                    [
                        [float(cos_f), float(sin_f)],
                        [float(-sin_f), float(cos_f)],
                    ]
                )
            )

    # Build block-diagonal transition matrix
    state_dim = sum(b.shape[0] for b in blocks)
    transition = jnp.zeros((state_dim, state_dim))
    idx = 0
    for b in blocks:
        d = b.shape[0]
        transition = transition.at[idx : idx + d, idx : idx + d].set(b)
        idx += d

    # Observation: picks the cosine component from each harmonic
    observation = jnp.zeros((1, state_dim))
    idx = 0
    for b in blocks:
        observation = observation.at[0, idx].set(1.0)
        idx += b.shape[0]

    state_noise = jnp.eye(state_dim) * sigma_seasonal**2
    obs_noise = jnp.array([[sigma_obs**2]])

    return StateSpaceModel(
        system_matrix=transition,
        observation_matrix=observation,
        evolution_cov=state_noise,
        obs_cov=obs_noise,
    )


# ---------------------------------------------------------------------------
# Regression component
# ---------------------------------------------------------------------------


def Regression(  # noqa: N802
    n_regressors: int,
    sigma_coeff: float = 0.0,
    sigma_obs: float = 0.0,
) -> StateSpaceModel:
    """Dynamic regression component.

    State: coefficient vector beta_t of length ``n_regressors``.
    Transition: beta_t = beta_{t-1} + w_t (random walk if sigma_coeff > 0).
    Observation: y_t = X_t @ beta_t + v_t.

    The observation matrix H should be updated at each time step with the
    regressor values. This factory sets H = I as a placeholder; the user
    should replace H with actual regressor rows when running the filter.

    For static coefficients, set ``sigma_coeff=0``.

    State dimension: n_regressors.
    """
    n = n_regressors
    return StateSpaceModel(
        system_matrix=jnp.eye(n),
        observation_matrix=jnp.ones((1, n)),
        evolution_cov=jnp.eye(n) * sigma_coeff**2,
        obs_cov=jnp.array([[sigma_obs**2]]),
    )


# ---------------------------------------------------------------------------
# Autoregressive component
# ---------------------------------------------------------------------------


def Autoregressive(  # noqa: N802
    order: int,
    coefficients: Array | None = None,
    sigma_ar: float = 1.0,
    sigma_obs: float = 0.0,
) -> StateSpaceModel:
    """Autoregressive (AR) component of given order.

    State: ``[x_t, x_{t-1}, ..., x_{t-p+1}]``.
    Transition uses the companion form with coefficients in the first row.

    Args:
        order: AR order p.
        coefficients: AR coefficients [phi_1, ..., phi_p]. Defaults to zeros.
        sigma_ar: Std dev of innovation noise (applied to first state only).
        sigma_obs: Observation noise contribution.

    State dimension: order.
    """
    p = order
    if coefficients is None:
        coefficients = jnp.zeros(p)

    # Companion-form transition matrix
    transition = jnp.zeros((p, p))
    transition = transition.at[0, :].set(coefficients)
    if p > 1:
        transition = transition.at[1:, :-1].set(jnp.eye(p - 1))

    observation = jnp.zeros((1, p))
    observation = observation.at[0, 0].set(1.0)

    state_noise = jnp.zeros((p, p))
    state_noise = state_noise.at[0, 0].set(sigma_ar**2)

    return StateSpaceModel(
        system_matrix=transition,
        observation_matrix=observation,
        evolution_cov=state_noise,
        obs_cov=jnp.array([[sigma_obs**2]]),
    )


# ---------------------------------------------------------------------------
# Cycle component
# ---------------------------------------------------------------------------


def Cycle(  # noqa: N802
    period: float,
    damping: float = 1.0,
    sigma_cycle: float = 1.0,
    sigma_obs: float = 0.0,
) -> StateSpaceModel:
    r"""Stochastic cycle component.

    A damped sinusoidal cycle with frequency 2*pi/period:

    Transition: rho * [[cos(freq), sin(freq)],
                        [-sin(freq), cos(freq)]]

    Args:
        period: Cycle period (in time steps). Must be > 0.
        damping: Damping factor rho in (0, 1]. 1.0 = undamped.
        sigma_cycle: Std dev of cycle state noise.
        sigma_obs: Observation noise contribution.

    State dimension: 2.
    """
    freq = 2.0 * jnp.pi / period
    cos_f = float(jnp.cos(freq))
    sin_f = float(jnp.sin(freq))

    transition = damping * jnp.array(
        [
            [cos_f, sin_f],
            [-sin_f, cos_f],
        ]
    )

    observation = jnp.array([[1.0, 0.0]])

    state_noise = jnp.eye(2) * sigma_cycle**2
    obs_noise = jnp.array([[sigma_obs**2]])

    return StateSpaceModel(
        system_matrix=transition,
        observation_matrix=observation,
        evolution_cov=state_noise,
        obs_cov=obs_noise,
    )
