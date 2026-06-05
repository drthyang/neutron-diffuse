"""Step 2 of powder ring removal: residual ring detection via Laue symmetry.

Why this is needed
------------------
The empty-scan subtraction (step 1) removes the cryostat/environment ring but
leaves behind the contribution from the **sample holder**, which is present
during the sample scan but absent from the empty scan.

Why symmetry works here
-----------------------
The residual ring is **not uniform around the |Q| shell** — its amplitude
varies with direction due to detector coverage, absorption geometry, and
normalisation errors.  This non-uniformity is exploitable:

    Within a set of Laue-equivalent voxels (all at the same |Q|), voxels
    that happen to lie in a high-amplitude ring sector will be anomalously
    high compared to voxels in low-amplitude sectors.

    By detecting outlier-high voxels within each equivalent group and
    filling them from the remaining clean equivalents, we remove the
    residual ring without needing to know its profile.

This approach requires the **input to already be symmetrised by Mantid**
(or equivalent), so that the only remaining asymmetry within an equivalent
group is contamination rather than an uncorrected systematic.

Algorithm
---------
For each voxel (h,k,l) and its Laue equivalents {Rg | R in Laue group}:

    1. Collect intensities of all valid (unmasked) equivalents.
    2. Compute median and MAD (median absolute deviation) — robust to outliers.
    3. Flag the voxel if:
           (I(hkl) - median) > threshold * MAD    AND    I(hkl) > median
       i.e. it is anomalously *high* (not low) compared to its equivalents.
    4. Fill flagged voxels with the inverse-variance weighted mean of the
       unflagged equivalents.

Step 3 (flag voxels where ALL equivalents are contaminated) is beyond the
scope of a simple local algorithm; those cases are handled by the general
3D inpainting fallback in backfill.py.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ndiff.core import HKLVolume
from ndiff.inpainting.symmetry import LAUE_CLASSES


def detect_and_fill_residual(
    vol: HKLVolume,
    symmetry_ops: Sequence[NDArray] | None = None,
    laue_class: str = "m3m",
    mad_threshold: float = 5.0,
    min_clean_equivalents: int = 2,
) -> tuple[HKLVolume, NDArray[np.bool_]]:
    """Detect and fill outlier-high voxels using Laue symmetry.

    Parameters
    ----------
    vol : HKLVolume
        Volume after empty-scan subtraction.  Should be Mantid-symmetrised
        so that clean voxels are consistent within each equivalent group.
    symmetry_ops : list of (3,3) int arrays, optional
        Point-group rotation matrices.  If None, *laue_class* is used.
    laue_class : str
        Key into the built-in Laue class library (``"m3m"``, ``"4/mmm"``,
        ``"mmm"``).  Ignored if *symmetry_ops* is provided.
    mad_threshold : float
        A voxel is flagged as contaminated if its excess above the group
        median exceeds *mad_threshold* × MAD of the group.
        Higher → more conservative (fewer flags).  Typical range: 4–8.
    min_clean_equivalents : int
        Minimum number of unflagged equivalents required to fill a flagged
        voxel.  Voxels with too few clean equivalents are left masked and
        handled downstream by inpainting.

    Returns
    -------
    vol_cleaned : HKLVolume
        Volume with residual-ring voxels replaced by symmetry-averaged values.
        Filled voxels carry inflated σ.
    contamination_flag : NDArray[bool]
        Shape = vol.shape; True where a voxel was identified as contaminated
        (regardless of whether it could be filled).
    """
    ops = list(symmetry_ops) if symmetry_ops is not None else LAUE_CLASSES[laue_class]

    h_arr, k_arr, l_arr = vol.h_axis, vol.k_axis, vol.l_axis
    dh = (h_arr[-1] - h_arr[0]) / max(len(h_arr) - 1, 1)
    dk = (k_arr[-1] - k_arr[0]) / max(len(k_arr) - 1, 1)
    dl = (l_arr[-1] - l_arr[0]) / max(len(l_arr) - 1, 1)

    data_out = vol.data.copy()
    sigma_out = vol.sigma.copy()
    mask_out = vol.mask.copy()
    contamination_flag = np.zeros(vol.shape, dtype=bool)

    # Precompute grid of equivalent indices for every voxel
    # Shape: (n_ops, nh, nk, nl, 3) → expensive; we iterate over masked voxels only
    H, K, L = vol.hkl_grid()

    # Process every valid voxel
    valid_idx = np.argwhere(vol.mask)

    for ih, ik, il in valid_idx:
        hkl0 = np.array([h_arr[ih], k_arr[ik], l_arr[il]])

        equiv_vals: list[float] = []
        equiv_vars: list[float] = []
        equiv_pos: list[tuple[int, int, int]] = []

        for op in ops:
            hkl_eq = op.astype(float) @ hkl0
            ji = int(round((hkl_eq[0] - h_arr[0]) / (dh + 1e-15)))
            jk = int(round((hkl_eq[1] - k_arr[0]) / (dk + 1e-15)))
            jl = int(round((hkl_eq[2] - l_arr[0]) / (dl + 1e-15)))
            if (0 <= ji < vol.shape[0] and 0 <= jk < vol.shape[1]
                    and 0 <= jl < vol.shape[2] and vol.mask[ji, jk, jl]):
                equiv_vals.append(float(vol.data[ji, jk, jl]))
                equiv_vars.append(float(vol.sigma[ji, jk, jl]) ** 2)
                equiv_pos.append((ji, jk, jl))

        if len(equiv_vals) < 2:
            continue

        vals = np.array(equiv_vals)
        median = float(np.median(vals))
        mad = float(np.median(np.abs(vals - median))) + 1e-12

        I_here = float(vol.data[ih, ik, il])
        # Asymmetric flag: only flag if anomalously HIGH (ring adds signal)
        if (I_here - median) > mad_threshold * mad and I_here > median:
            contamination_flag[ih, ik, il] = True

            # Identify clean equivalents (below flag threshold)
            clean_vals = []
            clean_vars = []
            for v, var, (ji, jk, jl) in zip(equiv_vals, equiv_vars, equiv_pos):
                if not contamination_flag[ji, jk, jl]:
                    excess = (v - median) / mad
                    if excess <= mad_threshold:  # not itself contaminated
                        clean_vals.append(v)
                        clean_vars.append(var)

            if len(clean_vals) >= min_clean_equivalents:
                w = 1.0 / (np.array(clean_vars) + 1e-30)
                data_out[ih, ik, il] = float(np.average(clean_vals, weights=w))
                sigma_out[ih, ik, il] = float(np.sqrt(1.0 / w.sum()))
            else:
                # Not enough clean equivalents; mark as invalid for downstream inpainting
                mask_out[ih, ik, il] = False

    import dataclasses
    vol_cleaned = dataclasses.replace(vol, data=data_out, sigma=sigma_out, mask=mask_out)
    return vol_cleaned, contamination_flag
