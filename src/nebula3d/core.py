# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tsung-han Yang

"""Core data structure for a 3D HKL intensity volume."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


def q_magnitude_from_axes(
    h_axis: NDArray[np.float64],
    k_axis: NDArray[np.float64],
    l_axis: NDArray[np.float64],
    ub_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """|Q| in Å^-1 on the (nh, nk, nl) grid spanned by three 1-D axes.

    Accumulates |Q|² from broadcast 1-D axes instead of materialising the full
    (nh, nk, nl) meshgrid and the (..., 3) Cartesian stack: peak memory is
    2 volume-sized arrays instead of ~10, which is what lets full-resolution
    volumes run inside the browser's WASM heap (see docs/web.md).
    """
    h = np.asarray(h_axis, dtype=np.float64)[:, None, None]
    k = np.asarray(k_axis, dtype=np.float64)[None, :, None]
    l_ = np.asarray(l_axis, dtype=np.float64)[None, None, :]
    ub = np.asarray(ub_matrix, dtype=np.float64)
    q2: NDArray[np.float64] | None = None
    for row in ub:
        qc = row[0] * h + row[1] * k   # (nh, nk, 1) — small
        qc = qc + row[2] * l_          # one full-volume array per component
        np.square(qc, out=qc)
        q2 = qc if q2 is None else np.add(q2, qc, out=q2)
    assert q2 is not None
    return np.sqrt(q2, out=q2)


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
        h_axis = np.linspace(h_range[0], h_range[1], nh).astype(np.float64)
        k_axis = np.linspace(k_range[0], k_range[1], nk).astype(np.float64)
        l_axis = np.linspace(l_range[0], l_range[1], nl).astype(np.float64)
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
        H, K, L = np.meshgrid(self.h_axis, self.k_axis, self.l_axis, indexing="ij")
        return H, K, L

    def q_magnitude(self) -> NDArray[np.float64]:
        """Return |Q| in Å^-1 for every voxel, shape (nh, nk, nl).

        Memory-lean: see :func:`q_magnitude_from_axes` (2 volume-sized arrays
        peak instead of the ~10 a meshgrid + (..., 3) stack would allocate).
        """
        return q_magnitude_from_axes(
            self.h_axis, self.k_axis, self.l_axis, self.ub_matrix)

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
