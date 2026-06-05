"""Core data structure for a 3D HKL intensity volume."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class HKLVolume:
    """3D reciprocal-space intensity grid in fractional HKL coordinates.

    Attributes
    ----------
    data:
        Shape (nh, nk, nl) intensity array.
    sigma:
        Shape (nh, nk, nl) standard-deviation array (same units as data).
    mask:
        Boolean array; True = voxel is valid (not masked).
    h_axis, k_axis, l_axis:
        1D coordinate arrays giving the h, k, l value of each voxel centre.
    ub_matrix:
        (3, 3) UB matrix in Å^-1 (columns = reciprocal-lattice vectors).
    instrument:
        Free-text instrument name for provenance.
    """

    data: NDArray[np.float64]
    sigma: NDArray[np.float64]
    mask: NDArray[np.bool_]
    h_axis: NDArray[np.float64]
    k_axis: NDArray[np.float64]
    l_axis: NDArray[np.float64]
    ub_matrix: NDArray[np.float64] = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )
    instrument: str = ""

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        data: NDArray[np.float64],
        h_range: tuple[float, float],
        k_range: tuple[float, float],
        l_range: tuple[float, float],
        sigma: NDArray[np.float64] | None = None,
        ub_matrix: NDArray[np.float64] | None = None,
    ) -> HKLVolume:
        nh, nk, nl = data.shape
        h_axis = np.linspace(h_range[0], h_range[1], nh)
        k_axis = np.linspace(k_range[0], k_range[1], nk)
        l_axis = np.linspace(l_range[0], l_range[1], nl)
        if sigma is None:
            sigma = np.sqrt(np.abs(data))
        mask = np.ones(data.shape, dtype=bool)
        ub = ub_matrix if ub_matrix is not None else np.eye(3, dtype=np.float64)
        return cls(
            data=data,
            sigma=sigma,
            mask=mask,
            h_axis=h_axis,
            k_axis=k_axis,
            l_axis=l_axis,
            ub_matrix=ub,
        )

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def hkl_grid(self) -> tuple[NDArray, NDArray, NDArray]:
        """Return (H, K, L) meshgrid arrays of shape (nh, nk, nl)."""
        return np.meshgrid(self.h_axis, self.k_axis, self.l_axis, indexing="ij")

    def q_magnitude(self) -> NDArray[np.float64]:
        """Return |Q| in Å^-1 for every voxel, shape (nh, nk, nl)."""
        H, K, L = self.hkl_grid()
        hkl = np.stack([H, K, L], axis=-1)  # (..., 3)
        q_cart = hkl @ self.ub_matrix.T      # (..., 3) in Å^-1
        return np.linalg.norm(q_cart, axis=-1)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    def apply_mask(self, new_mask: NDArray[np.bool_]) -> None:
        """Update mask in place; existing False voxels remain masked."""
        self.mask &= new_mask

    def masked_data(self) -> NDArray[np.float64]:
        """Return data with masked voxels set to NaN."""
        out = self.data.copy()
        out[~self.mask] = np.nan
        return out
