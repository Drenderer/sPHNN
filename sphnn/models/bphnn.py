"""OnsagerNet Potential.

This file implements a coercive potential function as used in the OnsagerNet by Haijun et al. (2021):
https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.6.114402
"""

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, Scalar
from klax.nn import MLP


class OnsagerNetPotential(eqx.Module):
    """Coercive Potential from OnsagerNet Haijun et al. (2021).

    Here we use a standard MLP instead of a ResNet.
    """

    mlp: Callable[[Array], Scalar]
    gamma: Array 
    beta: Scalar
    beta_learnable: bool

    def __init__(
        self,
        state_size: int,
        width_sizes: Sequence[int], 
        activation: Callable = jnn.softplus,
        weight_initializer: jnn.initializers.Initializer = jnn.initializers.he_normal(),
        bias_initializer: jnn.initializers.Initializer = jnn.initializers.zeros,
        beta: float = 0.1,
        beta_learnable: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        mlp_key, gamma_key = jax.random.split(key, 2)
        self.mlp = MLP(
            state_size,
            state_size,
            width_sizes=width_sizes,
            activation=activation,
            weight_init=weight_initializer,
            bias_init=bias_initializer,
            key=mlp_key,
        )
        self.gamma = weight_initializer(key=gamma_key, shape=(state_size, state_size))
        self.beta = jnp.array(beta)
        self.beta_learnable = beta_learnable

    def __call__(self, h: Array):
        beta = self.beta if self.beta_learnable else jax.lax.stop_gradient(self.beta)
        out = self.mlp(h) + self.gamma @ h
        out = 0.5 * jnp.inner(out, out) + beta * jnp.inner(h, h)
        return out
