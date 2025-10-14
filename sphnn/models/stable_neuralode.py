"""Stable neural ODEs.

This file implements stable neural ODEs introduced by Kolter and Manek (2019):
https://proceedings.neurips.cc/paper_files/paper/2019/file/0a4bbceda17a6253386bc9eb45240e25-Paper.pdf
"""

from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from dynax._misc import default_floating_dtype
from jax.nn.initializers import Initializer, zeros
from jaxtyping import Array, PRNGKeyArray, Scalar, Shaped


def smoothed_relu(d: float = 0.1) -> Callable[[Array], Array]:
    """Smoothed ReLU activation function from Kolter and Manek (2019).

    Args:
        d: Width of the quadratic transition region. Defaults to 0.1.

    Returns:
        Callable activation function.

    """

    @jax.jit
    def func(x: Array) -> Array:
        condlist = [x <= 0.0, jnp.logical_and(0.0 < x, x < d), d <= x]
        funclist = [0.0, lambda x: 0.5 / d * x**2, lambda x: x - 0.5 * d]
        return jnp.piecewise(x, condlist, funclist)

    func.__doc__ = f"Smoothed ReLU activation with smoothing parameter d={d}."
    return func


class SNODELyapunov(eqx.Module):
    """Lyapunov function for stable neural ODEs as described by Kolter and Manek (2019).

    Does not implement the optional warped input space.
    """

    func: Callable[[Shaped[Array, "n"]], Scalar]
    activation: Callable[[Array], Array]
    epsilon: float
    minimum: Shaped[Array, "n"]
    minimum_learnable: bool

    def __init__(
        self,
        func: Callable[[Shaped[Array, "n"]], Scalar],
        activation: Callable[[Array], Array],
        state_size: int,
        minimum_init: Initializer = zeros,
        minimum_learnable: bool = False,
        epsilon: float = 1e-3,
        dtype: type | None = None,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize the Lyapunov function module.

        Args:
            func: A callable representing the Lyapunov function.
                It takes an input of shape `(n,)` and returns a scalar.
            activation: A callable representing the activation function applied to
                the Lyapunov function output.
            state_size: The size of the state space `n`.
            minimum_init: An initializer for the minimum state. Defaults to zeros.
            minimum_learnable: A boolean indicating whether the minimum state is learnable.
                Defaults to False.
            epsilon: A small positive value added to ensure numerical stability.
                Defaults to 1e-3.
            dtype: The data type for the minimum state.
                Defaults to the default floating-point dtype.
            key: A PRNG key used for initializing the minimum state.

        """
        dtype = default_floating_dtype() if dtype is None else dtype

        self.func = func
        self.activation = activation
        self.minimum = minimum_init(key, (state_size,), dtype)
        self.minimum_learnable = minimum_learnable
        self.epsilon = epsilon

    def __call__(self, x: Array) -> Scalar:
        minimum = self.minimum if self.minimum_learnable else jax.lax.stop_gradient(self.minimum)

        z = self.func(x) - self.func(minimum)
        z = self.activation(z)

        dx = x - minimum
        z += jax.lax.stop_gradient(self.epsilon) * jnp.inner(dx, dx)
        return z


class SNODEProjection(eqx.Module):
    """Projection layer for stable neural ODEs as described by Kolter and Manek (2019)."""

    dynamics_fn: Callable[[Shaped[Array, "n"]], Shaped[Array, "n"]]
    lyapunov_fn: Callable[[Shaped[Array, "n"]], Scalar]
    alpha: float

    def __init__(
        self,
        dynamics_fn: Callable[[Shaped[Array, "n"]], Shaped[Array, "n"]],
        lyapunov_fn: Callable[[Shaped[Array, "n"]], Scalar],
        alpha: float = 0.0,
    ):
        self.dynamics_fn = dynamics_fn
        self.lyapunov_fn = lyapunov_fn
        self.alpha = alpha

    def __call__(self, t: Scalar | None, x: Array, u: Array | None):
        f_hat = self.dynamics_fn(x)
        v, grad_v = jax.value_and_grad(self.lyapunov_fn)(x)

        correction_switch = jnn.relu(
            jnp.inner(grad_v, f_hat) + jax.lax.stop_gradient(self.alpha) * v
        )
        norm_grad_v = jnp.inner(grad_v, grad_v)

        # Avoid division by zero in a jit traceable way
        projection = jnp.where(
            norm_grad_v == jnp.array(0.0),
            jnp.array(0.0),
            (correction_switch / norm_grad_v) * grad_v,
        )
        projection = cast(Array, projection)

        f = f_hat - projection
        return f
