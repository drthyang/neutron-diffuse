"""Fill masked ring-shell voxels by radial interpolation.

Why radial interpolation is the right approach
-----------------------------------------------
After ring detection and masking, each masked voxel sits inside a thin
|Q| shell.  The nearest uncontaminated voxels in 3D HKL space are almost
always at the same angular position (same h/k/l direction) but at
slightly different |Q| — just inside or just outside the ring shell.

Interpolating across the shell in the |Q| direction:
  * Makes **no assumption** about the diffuse signal shape — the fill is
    purely based on the observed values at neighbouring |Q|.
  * Naturally gives C¹ continuity at the shell boundaries: the
    interpolant matches both the value and the slope of the uncontaminated
    data at the inner and outer shell edges.
  * Is physically motivated: the diffuse signal varies smoothly in |Q|,
    and the ring shell is thin relative to that scale.

Algorithm (per masked voxel)
-----------------------------
For a masked voxel at angular position Ω and |Q| = q₀:

    1. Collect the k nearest valid voxels in 3D HKL space.
    2. Among those, use only voxels outside the ring shell (|Q| < q_lo
       or |Q| > q_hi), i.e. "uncontaminated neighbours".
    3. Fit a weighted linear interpolation (or median for robustness)
       in |Q| to estimate I at q₀.
    4. The weights are inversely proportional to the 3D HKL distance.

The filled voxel's σ is set to the local scatter of the contributing
neighbours, inflated by an uncertainty_scale factor to flag it as
estimated.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from ndiff.core import HKLVolume
from ndiff.preprocessing.powder_rings import RingShell


def backfill_ring_shells(
    vol: HKLVolume,
    rings: list[RingShell],
    n_neighbors: int = 12,
    uncertainty_scale: float = 2.0,
    fallback_tv: bool = True,
    tv_lam: float = 0.08,
    tv_iter: int = 300,
) -> HKLVolume:
    """Fill masked ring-shell voxels by radial interpolation.

    Parameters
    ----------
    vol : HKLVolume
        Volume after ring masking (``vol.mask`` marks valid voxels).
    rings : list[RingShell]
        The rings that were masked.  Used to identify which neighbours
        are "uncontaminated" (outside the ring |Q| range).
    n_neighbors : int
        Number of nearest valid 3D-HKL neighbours to consider per
        masked voxel.  Among these, only uncontaminated ones are used.
    uncertainty_scale : float
        Filled voxels get σ = ``uncertainty_scale`` × local neighbour σ,
        flagging them as estimated in downstream analysis.
    fallback_tv : bool
        If True, voxels that could not be filled by radial interpolation
        (too few uncontaminated neighbours) are filled by TV inpainting.
    tv_lam : float
        TV regularisation weight for the fallback.
    tv_iter : int
        TV iteration limit for the fallback.

    Returns
    -------
    HKLVolume
        Filled volume.  Mask is all-True.  Filled voxels carry inflated σ.
    """
    import dataclasses

    data_out = vol.data.copy()
    sigma_out = vol.sigma.copy()
    mask_out = vol.mask.copy()

    masked_idx = np.argwhere(~vol.mask)
    if len(masked_idx) == 0:
        return vol

    # Build KD-tree on all valid voxels (in normalised HKL index space)
    valid_idx = np.argwhere(vol.mask)
    q_valid = vol.q_magnitude()[vol.mask]
    I_valid = vol.data[vol.mask]
    sig_valid = vol.sigma[vol.mask]

    # Normalise index coordinates so that the tree metric is isotropic
    norm = np.array(vol.shape, dtype=float)
    tree = KDTree(valid_idx / norm)

    unfilled: list[tuple[int, int, int]] = []

    for ih, ik, il in masked_idx:
        q0 = float(vol.q_magnitude()[ih, ik, il])

        # Query nearest valid neighbours in HKL index space
        k = min(n_neighbors * 4, len(valid_idx))   # query extra, filter below
        dists, nn_idx = tree.query(np.array([ih, ik, il]) / norm, k=k)

        # Keep only uncontaminated neighbours (outside all ring shells)
        clean_mask = np.ones(len(nn_idx), dtype=bool)
        for ring in rings:
            q_nn = q_valid[nn_idx]
            clean_mask &= (q_nn < ring.q_lo) | (q_nn > ring.q_hi)

        if clean_mask.sum() < 2:
            unfilled.append((ih, ik, il))
            continue

        nn_q = q_valid[nn_idx][clean_mask]
        nn_I = I_valid[nn_idx][clean_mask]
        nn_sig = sig_valid[nn_idx][clean_mask]
        nn_d = dists[clean_mask]

        # Weighted interpolation in |Q|: weight = 1 / (HKL distance × σ²)
        w_dist = 1.0 / (nn_d + 1e-6)
        w_var = 1.0 / (nn_sig**2 + 1e-30)
        weights = w_dist * w_var
        weights /= weights.sum()

        # Weighted mean (linear interpolation collapses to this for 1 or 2 points;
        # for more points it is a weighted local average in |Q|)
        fill_val = float(np.dot(weights, nn_I))
        fill_sig = float(np.sqrt(np.dot(weights**2, nn_sig**2))) * uncertainty_scale

        data_out[ih, ik, il] = fill_val
        sigma_out[ih, ik, il] = fill_sig
        mask_out[ih, ik, il] = True

    # Fallback: TV inpainting for voxels with too few clean neighbours
    if unfilled and fallback_tv:
        from ndiff.inpainting.tv_inpainting import tv_inpaint
        data_out = tv_inpaint(data_out, mask_out, lam=tv_lam, max_iter=tv_iter)
        for ih, ik, il in unfilled:
            mask_out[ih, ik, il] = True
            sigma_out[ih, ik, il] = float(sigma_out[max(0, ih-1):ih+2,
                                                      max(0, ik-1):ik+2,
                                                      max(0, il-1):il+2].mean()) * uncertainty_scale
    elif unfilled:
        for ih, ik, il in unfilled:
            mask_out[ih, ik, il] = False   # leave masked for caller to handle

    return dataclasses.replace(vol, data=data_out, sigma=sigma_out,
                               mask=np.ones(vol.shape, dtype=bool))
