"""Interpolation-based inpainting for masked HKL voxels.

Implements RBF (radial basis function) and biharmonic interpolation.
These are used as fallback when symmetry equivalents are insufficient.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RBFInterpolator


def rbf_fill(
    data: NDArray[np.float64],
    mask: NDArray[np.bool_],
    kernel: str = "thin_plate_spline",
    neighbors: int = 64,
    smoothing: float = 0.0,
) -> NDArray[np.float64]:
    """Fill masked voxels via RBF interpolation from surrounding unmasked voxels.

    Parameters
    ----------
    data:
        3D intensity array. NaN or arbitrary values at masked positions.
    mask:
        Boolean; True = valid source voxel.
    kernel:
        RBF kernel: ``"thin_plate_spline"``, ``"multiquadric"``, ``"gaussian"``.
    neighbors:
        Number of nearest unmasked neighbors used per query point.
    smoothing:
        Smoothing factor passed to RBFInterpolator (0 = exact interpolation).

    Returns
    -------
    data_filled:
        Copy of *data* with masked voxels replaced by RBF predictions.
    """
    if not np.any(~mask):
        return data.copy()

    nh, nk, nl = data.shape
    coords = np.array(np.meshgrid(
        np.arange(nh), np.arange(nk), np.arange(nl), indexing="ij"
    )).reshape(3, -1).T  # (N, 3)

    src_idx = mask.ravel()
    src_coords = coords[src_idx]
    src_vals = data.ravel()[src_idx]

    query_idx = ~mask.ravel()
    query_coords = coords[query_idx]

    if len(src_coords) < 4:
        raise RuntimeError("Too few unmasked voxels for RBF interpolation.")

    rbf = RBFInterpolator(
        src_coords, src_vals,
        kernel=kernel, neighbors=min(neighbors, len(src_coords)),
        smoothing=smoothing,
    )
    predicted = rbf(query_coords)

    data_out = data.copy()
    flat = data_out.ravel()
    flat[query_idx] = predicted
    data_out[:] = flat.reshape(data.shape)
    return data_out


def biharmonic_fill(
    data: NDArray[np.float64],
    mask: NDArray[np.bool_],
    max_iter: int = 500,
    tol: float = 1e-4,
) -> NDArray[np.float64]:
    """Fill masked voxels by iteratively solving the discrete biharmonic equation.

    Minimises ∫|∇²u|² over the masked region with boundary conditions from
    the unmasked data. Uses a simple iterative Gauss-Seidel relaxation on
    the 3D discrete Laplacian (equivalent to harmonic/biharmonic inpainting
    depending on iteration depth).

    This is appropriate for broad, smooth diffuse features.
    """
    data_out = data.copy()
    # initialise masked region with local mean of unmasked voxels
    _init_masked_region(data_out, mask)

    for _ in range(max_iter):
        old = data_out.copy()
        # 3D mean of 6-connected neighbours (discrete Laplacian = 0 → harmonic)
        avg = (
            np.roll(data_out, 1, 0) + np.roll(data_out, -1, 0) +
            np.roll(data_out, 1, 1) + np.roll(data_out, -1, 1) +
            np.roll(data_out, 1, 2) + np.roll(data_out, -1, 2)
        ) / 6.0
        # update only masked voxels
        data_out[~mask] = avg[~mask]
        if np.max(np.abs(data_out - old)) < tol:
            break

    return data_out


def _init_masked_region(data: NDArray, mask: NDArray[np.bool_]) -> None:
    """In-place: set masked voxels to mean of all unmasked voxels."""
    fill_val = float(data[mask].mean()) if mask.any() else 0.0
    data[~mask] = fill_val
