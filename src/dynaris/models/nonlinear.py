"""Built-in nonlinear state-space models.

Factory functions that return :class:`~dynaris.core.nonlinear.NonlinearSSM`
instances for well-known nonlinear systems.

References:
    Kim, S., Shephard, N. and Chib, S. (1998). "Stochastic Volatility:
    Likelihood Inference and Comparison with ARCH Models."
    Review of Economic Studies, 65(3), 361-393.

    Bar-Shalom, Y., Li, X.R. and Kirubarajan, T. (2001).
    "Estimation with Applications to Tracking and Navigation." Wiley.

    Lorenz, E.N. (1963). "Deterministic Nonperiodic Flow."
    Journal of the Atmospheric Sciences, 20(2), 130-141.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from dynaris.core.nonlinear import NonlinearSSM

# ---------------------------------------------------------------------------
# Stochastic Volatility
# ---------------------------------------------------------------------------


def StochasticVolatility(  # noqa: N802
    mu: float = 0.0,
    phi: float = 0.97,
    sigma: float = 0.15,
    obs_var: float | None = None,
) -> NonlinearSSM:
    r"""Stochastic volatility model (Kim-Shephard-Chib linearization).

    The latent log-volatility follows an AR(1) process:

    .. math::

        h_t = \mu + \phi (h_{t-1} - \mu) + \sigma \eta_t, \quad
        \eta_t \sim N(0, 1)

    Observations are log-squared returns (use :func:`transform_returns`):

    .. math::

        y^*_t = h_t + z_t, \quad z_t \sim N(-1.27, \pi^2 / 2)

    Args:
        mu: Long-run mean of log-volatility.
        phi: Persistence parameter (abs(phi) < 1 for stationarity).
        sigma: Volatility of log-volatility.
        obs_var: Observation noise variance. Defaults to pi^2 / 2
            (the KSC approximation for log-chi-squared).

    Returns:
        NonlinearSSM with state_dim=1, obs_dim=1.

    Example::

        from dynaris.models import StochasticVolatility, transform_returns
        import jax.numpy as jnp

        model = StochasticVolatility(mu=0.0, phi=0.97, sigma=0.15)
        y_star = transform_returns(returns)
    """
    if obs_var is None:
        obs_var = jnp.pi**2 / 2.0

    def f(h: Array) -> Array:
        return jnp.array([mu + phi * (h[0] - mu)])

    def h(state: Array) -> Array:
        return state

    return NonlinearSSM(
        transition_fn=f,
        observation_fn=h,
        transition_cov=jnp.array([[sigma**2]]),
        observation_cov=jnp.array([[obs_var]]),
        state_dim=1,
        obs_dim=1,
    )


def transform_returns(y: Array, offset: float = 1e-8) -> Array:
    r"""Transform financial returns to log-squared form for stochastic volatility.

    Applies the Kim-Shephard-Chib (1998) linearization:

    .. math::

        y^*_t = \log(y_t^2 + \epsilon) + 1.2704

    The constant 1.2704 removes the mean of the log-chi-squared(1)
    distribution, centering the observation noise at zero.

    Args:
        y: Raw returns, shape (T,) or (T, 1).
        offset: Small constant to avoid log(0). Default 1e-8.

    Returns:
        Transformed observations, same shape as input.
    """
    return jnp.log(y**2 + offset) + 1.2704


# ---------------------------------------------------------------------------
# Bearings-Only Tracking
# ---------------------------------------------------------------------------


def BearingsTracking(  # noqa: N802
    sensor_pos: tuple[float, float] = (0.0, 0.0),
    dt: float = 1.0,
    sigma_accel: float = 0.1,
    sigma_bearing: float = 0.01,
) -> NonlinearSSM:
    r"""Bearings-only tracking model in 2D.

    A target moves with near-constant velocity in 2D, observed via bearing
    (angle) from a sensor at a known position.

    State: :math:`[x, v_x, y, v_y]` (position and velocity).

    Transition (constant velocity):

    .. math::

        x_t = x_{t-1} + \Delta t \, v_{x,t-1} + \text{noise}

    Observation (bearing angle):

    .. math::

        \theta_t = \text{atan2}(y_t - s_y, \, x_t - s_x) + \text{noise}

    Args:
        sensor_pos: Sensor (x, y) position.
        dt: Time step between observations.
        sigma_accel: Acceleration noise standard deviation.
        sigma_bearing: Bearing measurement noise std (radians).

    Returns:
        NonlinearSSM with state_dim=4, obs_dim=1.

    Example::

        from dynaris.models import BearingsTracking

        model = BearingsTracking(sensor_pos=(0, 0), sigma_bearing=0.05)
    """
    sx, sy = sensor_pos

    def f(state: Array) -> Array:
        x, vx, y, vy = state[0], state[1], state[2], state[3]
        return jnp.array([x + dt * vx, vx, y + dt * vy, vy])

    def h(state: Array) -> Array:
        dx = state[0] - sx
        dy = state[2] - sy
        return jnp.array([jnp.arctan2(dy, dx)])

    # Discrete white-noise acceleration model for each axis
    q1d = sigma_accel**2 * jnp.array(
        [
            [dt**3 / 3.0, dt**2 / 2.0],
            [dt**2 / 2.0, dt],
        ]
    )
    transition_cov = jnp.zeros((4, 4))
    transition_cov = transition_cov.at[:2, :2].set(q1d)
    transition_cov = transition_cov.at[2:, 2:].set(q1d)

    return NonlinearSSM(
        transition_fn=f,
        observation_fn=h,
        transition_cov=transition_cov,
        observation_cov=jnp.array([[sigma_bearing**2]]),
        state_dim=4,
        obs_dim=1,
    )


# ---------------------------------------------------------------------------
# Lorenz Attractor
# ---------------------------------------------------------------------------


def LorenzAttractor(  # noqa: N802
    lorenz_sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    dt: float = 0.01,
    process_noise: float = 0.1,
    obs_noise: float = 1.0,
    obs_dims: int = 3,
) -> NonlinearSSM:
    r"""Lorenz attractor state-space model (Euler discretization).

    The Lorenz system of ordinary differential equations:

    .. math::

        \dot{x} &= \sigma (y - x) \\
        \dot{y} &= x (\rho - z) - y \\
        \dot{z} &= x y - \beta z

    discretized with an Euler step of size ``dt``.

    Args:
        lorenz_sigma: Lorenz sigma parameter (Prandtl number).
        rho: Lorenz rho parameter (Rayleigh number).
        beta: Lorenz beta parameter (geometric factor).
        dt: Euler discretization time step.
        process_noise: Process noise standard deviation.
        obs_noise: Observation noise standard deviation.
        obs_dims: Number of observed dimensions (1, 2, or 3).

    Returns:
        NonlinearSSM with state_dim=3, obs_dim=obs_dims.

    Example::

        from dynaris.models import LorenzAttractor

        model = LorenzAttractor(dt=0.01, obs_noise=2.0)
    """
    s, r, b = lorenz_sigma, rho, beta

    def f(state: Array) -> Array:
        x, y, z = state[0], state[1], state[2]
        dx = s * (y - x)
        dy = x * (r - z) - y
        dz = x * y - b * z
        return jnp.array([x + dt * dx, y + dt * dy, z + dt * dz])

    def h(state: Array) -> Array:
        return state[:obs_dims]

    return NonlinearSSM(
        transition_fn=f,
        observation_fn=h,
        transition_cov=jnp.eye(3) * process_noise**2,
        observation_cov=jnp.eye(obs_dims) * obs_noise**2,
        state_dim=3,
        obs_dim=obs_dims,
    )
