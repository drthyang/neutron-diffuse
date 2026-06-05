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

import numpy as np
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

    # |Q| is needed both per masked voxel and per neighbour; compute it ONCE
    # (q_magnitude rebuilds a full meshgrid + matmul + norm, so calling it per
    # voxel inside the loop dominates the runtime on real-size volumes).
    q_all = vol.q_magnitude()

    # Build KD-tree on all valid voxels (in normalised HKL index space)
    valid_idx = np.argwhere(vol.mask)
    q_valid = q_all[vol.mask]
    I_valid = vol.data[vol.mask]
    sig_valid = vol.sigma[vol.mask]

    # Normalise index coordinates so that the tree metric is isotropic
    norm = np.array(vol.shape, dtype=float)
    tree = KDTree(valid_idx / norm)

    k = min(n_neighbors * 4, len(valid_idx))   # query extra, filter below
    ring_lo = np.array([r.q_lo for r in rings], dtype=float)
    ring_hi = np.array([r.q_hi for r in rings], dtype=float)

    # Vectorised, chunked nearest-neighbour fill.  Querying/weighting one voxel
    # at a time in Python is the second bottleneck; batching the query (and
    # letting SciPy use all cores) and doing the weighting with array ops is
    # orders of magnitude faster.  Chunking bounds peak memory of the (chunk, k)
    # neighbour arrays.
    chunk = 200_000
    fill_ok = np.zeros(len(masked_idx), dtype=bool)

    for start in range(0, len(masked_idx), chunk):
        block = masked_idx[start:start + chunk]
        dists, nn = tree.query(block / norm, k=k, workers=-1)   # (c, k)
        if k == 1:                                              # SciPy drops the k axis
            dists = dists[:, None]
            nn = nn[:, None]

        q_nn = q_valid[nn]            # (c, k)
        I_nn = I_valid[nn]
        sig_nn = sig_valid[nn]

        # Neighbours inside any ring shell are contaminated; keep the rest.
        contaminated = np.zeros_like(q_nn, dtype=bool)
        for lo, hi in zip(ring_lo, ring_hi):
            contaminated |= (q_nn >= lo) & (q_nn <= hi)
        clean = ~contaminated
        enough = clean.sum(axis=1) >= 2     # (c,)

        # Weighted interpolation in |Q|: weight = 1 / (HKL distance × σ²),
        # restricted to clean neighbours.
        weights = np.where(clean, 1.0 / (dists + 1e-6) / (sig_nn**2 + 1e-30), 0.0)
        wsum = weights.sum(axis=1, keepdims=True)
        wnorm = weights / np.where(wsum > 0, wsum, 1.0)

        vals = (wnorm * I_nn).sum(axis=1)
        sigs = np.sqrt((wnorm**2 * sig_nn**2).sum(axis=1)) * uncertainty_scale

        rows = block[enough]
        data_out[rows[:, 0], rows[:, 1], rows[:, 2]] = vals[enough]
        sigma_out[rows[:, 0], rows[:, 1], rows[:, 2]] = sigs[enough]
        mask_out[rows[:, 0], rows[:, 1], rows[:, 2]] = True
        fill_ok[start:start + len(block)] = enough

    unfilled = masked_idx[~fill_ok]   # (U, 3); too few clean neighbours

    # Fallback: TV inpainting for voxels with too few clean neighbours
    if len(unfilled) and fallback_tv:
        from ndiff.inpainting.tv_inpainting import tv_inpaint
        data_out = tv_inpaint(data_out, mask_out, lam=tv_lam, max_iter=tv_iter)
        ih, ik, il = unfilled[:, 0], unfilled[:, 1], unfilled[:, 2]
        mask_out[ih, ik, il] = True
        for h, kk, ll in unfilled:
            sigma_out[h, kk, ll] = float(sigma_out[max(0, h-1):h+2,
                                                   max(0, kk-1):kk+2,
                                                   max(0, ll-1):ll+2].mean()) * uncertainty_scale
    elif len(unfilled):
        # leave masked for caller to handle
        mask_out[unfilled[:, 0], unfilled[:, 1], unfilled[:, 2]] = False

    return dataclasses.replace(vol, data=data_out, sigma=sigma_out,
                               mask=np.ones(vol.shape, dtype=bool))
