from typing import Literal, cast

import numpy as np
from jaxtyping import Array, Shaped
from numpy.typing import NDArray


def get_svd(
    snapshots: Shaped[NDArray | Array, "n_batch n_time n_space"],
) -> tuple[
    Shaped[NDArray, "n_batch n_time n_time"],
    Shaped[NDArray, "n_time"],
    Shaped[NDArray, "n_time n_space"],
]:
    """Calculate the (stacked) svd for snapshot matrices.

    Snapshots must be 3D. Useful for calculating the modes of multiple trajectories.
    U then is the trajectories of the latent variables,
    and modes = s[:, None] * V the modes.

    Args:
        snapshots (Array): Array of trajectories, shape=(n_batch, n_time, n_space).

    Returns:
        SVD-matrices:
        - U - Left singular matrix, shape=(n_batch, n_time, n_batch*n_time)
        - s - singular values, shape=(n_batch*n_time)
        - V - right singular matrix, shape=(n_batch*n_time, n_space)

    """
    n_traj, n_time, _ = snapshots.shape
    traj_stacked = np.concatenate(snapshots, axis=0)
    U, s, V = np.linalg.svd(traj_stacked, full_matrices=False)  # noqa: N806
    U = np.reshape(U, (n_traj, n_time, -1))  # noqa: N806

    return U, s, V


class PODLatentSpace:
    """Map snapshots to and from a latent space using POD."""

    shift: Shaped[NDArray, "1 1 n_space"]
    scale: float
    modes: Shaped[NDArray, "n_time n_space"]
    singular_values: Shaped[NDArray, "n_time"]
    num_modes: int

    def __init__(
        self,
        snapshot: Shaped[NDArray | Array, "n_batch n_time n_space"],
        num_modes: int,
        shift: float | Shaped[NDArray | Array, "#n_space"] | Literal["mean"] | None = None,
    ) -> None:
        """Initialize the POD latent space.

        Args:
            snapshot: Snapshots of shape (n_batch, n_time, n_space).
            num_modes: Number of modes to use in the latent space.
            shift: Array to shift a snapshot by, before transforming. 
                By setting `shift=<special state>`, the special state is 
                transformed to the origin of the latent space.

        """
        if shift is None:
            self.shift = np.zeros((1,1,1))
        elif shift == "mean":
            self.shift = np.mean(snapshot, axis=(0,1), keepdims=True)
        else:
            shift = np.atleast_1d(shift)
            assert shift.ndim <= 1, "Shift must be 1-dimensional."
            self.shift = shift[None, None, :]
        _snapshot = snapshot - self.shift

        left_singular_vectors, singular_values, right_singluar_vectors = (
            get_svd(_snapshot)
        )

        # Compute scaling such that the latent variables have std=1
        latent_variables = (
            left_singular_vectors * singular_values[None, None, :]
        )
        scale = np.std(latent_variables[:, :, :num_modes])

        self.scale = float(scale)
        self.modes = right_singluar_vectors
        self.singular_values = singular_values
        self.num_modes = num_modes

    def to_latent[T: NDArray | Array](
        self,
        snapshot: Shaped[T, "n_batch n_time n_space"],
    ) -> Shaped[T, "n_batch n_time n_modes"]:
        _snapshot = snapshot - self.shift
        latent_variables = (
            _snapshot @ self.modes[: self.num_modes].T / self.scale
        )
        return cast(T, latent_variables)

    def to_physical[T: NDArray | Array](
        self,
        latent_variables: Shaped[T, "n_batch n_time n_modes"],
    ) -> Shaped[T, "n_batch n_time n_space"]:
        snapshot = (
            latent_variables @ self.modes[: self.num_modes] * self.scale
            + self.shift
        )
        return cast(T, snapshot)
