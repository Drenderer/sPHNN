from collections.abc import Callable
from typing import overload

import equinox as eqx
from dynax import Trajectory, normalization_coefficients
from jax import numpy as jnp
from jaxtyping import Array, PyTree


def also_accept_none(func):
    def wrapper(self, arg: Array | None) -> Array | None:
        if arg is None:
            return None
        else:
            return func(self, arg)

    return wrapper


class Normalization(eqx.Module):
    """Normalized data using affine transformations for states `y`, inputs `u` and time `t`."""

    mean_y: Array
    alpha_y: Array
    tau_t: Array
    mean_u: Array
    alpha_u: Array

    @classmethod
    def init_identity(cls, state_size: int, input_size: int):
        """Initialize a normalization that does not change the data.

        Args:
            state_size: Dimensionality of the state
            input_size: Dimensionality of the input

        Returns:
            An identity normalization.

        """
        mean_y = jnp.zeros((state_size,))
        alpha_y = jnp.ones((state_size,))
        tau_t = jnp.ones(())
        mean_u = jnp.zeros((input_size,))
        alpha_u = jnp.ones((input_size,))
        return cls(mean_y, alpha_y, tau_t, mean_u, alpha_u)
    
    @classmethod
    def fit(cls, data: Trajectory, tol: float = 1e-6) -> "Normalization":
        """Fit the normalization parameters to the given data.

        Args:
            data: Trajectory to fit the normalization to.
            tol: Tolerance for the `normalization_coefficient` calculation. Defaults to 1e-6.

        Returns:
            Normalization fitted to the data.

        """
        def _semi_flatten(x: Array) -> Array:
            return x.reshape(-1, x.shape[-1])

        mean_y = _semi_flatten(data.ys).mean(axis=0)
        std_y = _semi_flatten(data.ys).std(axis=0)
        std_y_t = None if data._y_ts is None else _semi_flatten(data._y_ts).std(axis=0)
        mean_u = jnp.array(0) if data._us is None else _semi_flatten(data._us).mean(axis=0)
        std_u = 1 if data._us is None else _semi_flatten(data._us).std(axis=0)

        alpha_q, tau_t = normalization_coefficients(std_y, std_y_t, tol=tol)
        alpha_u, _ = normalization_coefficients(std_u, tol=tol)

        return cls(mean_y, alpha_q, tau_t, mean_u, alpha_u)

    @overload
    def transform_t(self, ts: Array) -> Array: ...
    @overload
    def transform_t(self, ts: None) -> None: ...
    def transform_t(self, ts: Array | None) -> Array | None:
        return None if ts is None else ts / self.tau_t

    @overload
    def inverse_transform_t(self, ts: Array) -> Array: ...
    @overload
    def inverse_transform_t(self, ts: None) -> None: ...
    def inverse_transform_t(self, ts: Array | None) -> Array | None:
        return None if ts is None else ts * self.tau_t

    @overload
    def transform_y(self, ys: Array) -> Array: ...
    @overload
    def transform_y(self, ys: None) -> None: ...
    def transform_y(self, ys: Array | None) -> Array | None:
        return None if ys is None else self.alpha_y * (ys - self.mean_y)

    @overload
    def inverse_transform_y(self, ys: Array) -> Array: ...
    @overload
    def inverse_transform_y(self, ys: None) -> None: ...
    def inverse_transform_y(self, ys: Array | None) -> Array | None:
        return None if ys is None else ys / self.alpha_y + self.mean_y

    @overload
    def transform_y_t(self, y_ts: Array) -> Array: ...
    @overload
    def transform_y_t(self, y_ts: None) -> None: ...
    def transform_y_t(self, y_ts: Array | None) -> Array | None:
        return None if y_ts is None else self.tau_t * self.alpha_y * y_ts

    @overload
    def inverse_transform_y_t(self, y_ts: Array) -> Array: ...
    @overload
    def inverse_transform_y_t(self, y_ts: None) -> None: ...
    def inverse_transform_y_t(self, y_ts: Array | None) -> Array | None:
        return None if y_ts is None else y_ts / (self.tau_t * self.alpha_y)

    @overload
    def transform_u(self, us: Array) -> Array: ...
    @overload
    def transform_u(self, us: None) -> None: ...
    def transform_u(self, us: Array | None) -> Array | None:
        return None if us is None else self.alpha_u * (us - self.mean_u)

    @overload
    def inverse_transform_u(self, us: Array) -> Array: ...
    @overload
    def inverse_transform_u(self, us: None) -> None: ...
    def inverse_transform_u(self, us: Array | None) -> Array | None:
        return None if us is None else us / self.alpha_u + self.mean_u

    def transform(self, data: Trajectory) -> Trajectory:
        return Trajectory(
            ts=self.transform_t(data._ts),
            ys=self.transform_y(data._ys),
            y_ts=self.transform_y_t(data._y_ts),
            us=self.transform_t(data._us),
        )

    def inverse_transform(self, data: Trajectory) -> Trajectory:
        return Trajectory(
            ts=self.inverse_transform_t(data._ts),
            ys=self.inverse_transform_y(data._ys),
            y_ts=self.inverse_transform_y_t(data._y_ts),
            us=self.inverse_transform_t(data._us),
        )


class NormalizationWrapper(eqx.Module):
    """Wrapper around an ODESolver that normalizes the data before solving."""
    
    model: Callable[[Array, Array, Array | None, PyTree], Array]
    normalizer: Normalization

    def __call__(
        self,
        ts: Array,
        y0: Array,
        us: Array | None = None,
        funcargs: PyTree = None,
    ) -> Array:
        ts_ = self.normalizer.transform_t(ts)
        y0_ = self.normalizer.transform_y(y0)
        us_ = self.normalizer.transform_u(us)
        ys_ = self.model(ts_, y0_, us_, funcargs)
        return self.normalizer.inverse_transform_y(ys_)
