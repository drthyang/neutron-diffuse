"""Backfill powder-ring holes in the **subtracted** diffuse signal.

Context
-------
After :func:`powder_rings.subtract_rings`, the volume contains:

    I_diffuse_est(Q) = I_measured(Q) − I_ring(|Q|)

with a mask that flags voxels where the powder ring dominated and the
subtraction quality is poor. The goal here is to fill those voxels with
physically plausible diffuse signal values.

Why this is non-trivial
-----------------------
Powder ring holes form **thin spherical shells** in |Q|-space. Every voxel
in a shell is at the same |Q|, so the hole wraps all the way around the
sphere. This means:

* Simple radial interpolation fails (no source data at the same |Q|).
* Symmetry-equivalent voxels are also on the same shell → also masked.

What DOES work
--------------
The diffuse signal is **smooth in 3D HKL space**. The shell hole is thin
(typically < 0.1 Å^-1 wide). The signal just inside and just outside the
shell is clean (post-subtraction). We can therefore fill by:

1. **3D smooth interpolation** from unmasked voxels at slightly different |Q|.
   TV inpainting and RBF both handle thin-shell geometry well.
2. **Local shell polynomial fit**: for each masked voxel, fit a low-order
   polynomial to unmasked voxels in a local HKL neighbourhood that straddles
   the shell, and evaluate at the masked point.

TV inpainting (default) is preferred because:
- It minimises total variation → respects anisotropic diffuse streaks.
- It naturally handles the thin-shell topology.
- The diffuse signal is smooth enough that a moderate λ gives excellent results.

The filled values represent our best estimate of the diffuse scattering that
was underneath the powder ring — NOT a reconstruction of the ring itself.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray

from ndiff.core import HKLVolume
from ndiff.inpainting.tv_inpainting import tv_inpaint
from ndiff.inpainting.interpolation import rbf_fill, biharmonic_fill

Method = Literal["tv", "rbf", "biharmonic"]


def backfill(
    vol: HKLVolume,
    method: Method = "tv",
    tv_lam: float = 0.08,
    tv_iter: int = 500,
    rbf_kernel: str = "thin_plate_spline",
    rbf_neighbors: int = 64,
    uncertainty_scale: float = 2.0,
) -> HKLVolume:
    """Fill masked (powder-ring-subtracted) holes in *vol*.

    The fill is performed on the **already subtracted** diffuse signal, so
    the sources used for interpolation are clean I_diffuse values, not
    contaminated I_total values.

    Parameters
    ----------
    vol : HKLVolume
        Output of :func:`~powder_rings.subtract_rings`. The mask marks valid
        post-subtraction voxels.
    method : {"tv", "rbf", "biharmonic"}
        Filling algorithm.

        - ``"tv"`` (default): Total-variation inpainting — best for structured
          diffuse signals (anisotropic streaks, broad features).
        - ``"rbf"``: Radial-basis-function interpolation — good for smooth,
          nearly isotropic diffuse backgrounds.
        - ``"biharmonic"``: Iterative biharmonic relaxation — smoothest result,
          best for featureless backgrounds.
    tv_lam : float
        TV regularisation strength. Larger → smoother fill.
        0.05–0.15 is appropriate for diffuse scattering.
    tv_iter : int
        Maximum TV iterations (500 is enough for thin-shell holes).
    rbf_kernel : str
        RBF kernel type (see :func:`~inpainting.interpolation.rbf_fill`).
    rbf_neighbors : int
        Number of nearest unmasked voxels used per query point in RBF.
    uncertainty_scale : float
        Filled voxels are assigned σ = *uncertainty_scale* × local σ of
        neighbouring unmasked voxels, to flag them as estimated in
        downstream analysis.

    Returns
    -------
    HKLVolume
        New volume with masked holes filled. The mask is reset to all-True.
        Filled voxels carry inflated σ values.
    """
    import dataclasses

    mask = vol.mask
    data = vol.data.copy()
    sigma = vol.sigma.copy()

    if method == "tv":
        data_filled = tv_inpaint(data, mask, lam=tv_lam, max_iter=tv_iter)
    elif method == "rbf":
        data_filled = rbf_fill(data, mask, kernel=rbf_kernel, neighbors=rbf_neighbors)
    elif method == "biharmonic":
        data_filled = biharmonic_fill(data, mask)
    else:
        raise ValueError(f"Unknown backfill method: {method!r}")

    # Assign inflated uncertainty at filled positions
    sigma_filled = _local_sigma_estimate(sigma, mask) * uncertainty_scale
    sigma_out = sigma.copy()
    sigma_out[~mask] = sigma_filled[~mask]

    return dataclasses.replace(
        vol,
        data=data_filled,
        sigma=sigma_out,
        mask=np.ones(vol.shape, dtype=bool),
    )


def _local_sigma_estimate(
    sigma: NDArray[np.float64],
    mask: NDArray[np.bool_],
    radius: int = 3,
) -> NDArray[np.float64]:
    """Return per-voxel estimate of local σ from unmasked neighbours."""
    from scipy.ndimage import uniform_filter
    # dilate the valid sigma into masked regions via box-filter mean
    sigma_valid = np.where(mask, sigma, 0.0)
    count_valid = mask.astype(np.float64)
    box = 2 * radius + 1
    sigma_sum = uniform_filter(sigma_valid, size=box, mode="nearest") * box**3
    count_sum = uniform_filter(count_valid, size=box, mode="nearest") * box**3
    local_mean = np.where(count_sum > 0, sigma_sum / (count_sum + 1e-10), sigma.mean())
    return local_mean
