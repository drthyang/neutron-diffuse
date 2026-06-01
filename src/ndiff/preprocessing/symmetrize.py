"""Symmetrize a 3D HKL volume under the crystal Laue symmetry.

Symmetrization is the **first** preprocessing step. It:
1. Averages all symmetry-equivalent voxels (inverse-variance weighted).
2. Updates per-voxel σ to reflect the averaged uncertainty.
3. Flags voxels with high inter-equivalent variance — likely contaminated.

This step is run *before* Al removal so that the cleaned data is already
symmetry-consistent, making subsequent filling with symmetry equivalents exact.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from ndiff.core import HKLVolume
from ndiff.inpainting.symmetry import LAUE_CLASSES


def symmetrize(
    vol: HKLVolume,
    symmetry_ops: Optional[Sequence[NDArray]] = None,
    laue_class: str = "m3m",
    variance_flag_sigma: float = 5.0,
) -> tuple[HKLVolume, NDArray[np.bool_]]:
    """Average symmetry-equivalent voxels in *vol*.

    Parameters
    ----------
    vol:
        Input volume. Only valid (mask == True) voxels contribute to averages.
    symmetry_ops:
        List of (3,3) integer rotation matrices. If None, *laue_class* is used.
    laue_class:
        Laue class key (``"m3m"``, ``"4/mmm"``, ``"mmm"``).
    variance_flag_sigma:
        Voxels whose inter-equivalent standard deviation exceeds this multiple
        of the Poisson σ are flagged in the returned *outlier_flag* array.
        Set to np.inf to disable.

    Returns
    -------
    vol_sym:
        New HKLVolume with symmetry-averaged data and updated σ.
    outlier_flag:
        Boolean array (same shape); True = inter-equivalent variance is
        anomalously high, suggesting residual contamination or bad pixels.
    """
    ops = list(symmetry_ops) if symmetry_ops is not None else LAUE_CLASSES[laue_class]

    h_arr, k_arr, l_arr = vol.h_axis, vol.k_axis, vol.l_axis
    dh = (h_arr[-1] - h_arr[0]) / max(len(h_arr) - 1, 1)
    dk = (k_arr[-1] - k_arr[0]) / max(len(k_arr) - 1, 1)
    dl = (l_arr[-1] - l_arr[0]) / max(len(l_arr) - 1, 1)

    data_sum = np.zeros(vol.shape, dtype=np.float64)
    weight_sum = np.zeros(vol.shape, dtype=np.float64)
    # second moment for variance flagging
    data_sq_sum = np.zeros(vol.shape, dtype=np.float64)
    count = np.zeros(vol.shape, dtype=np.int32)

    for op in ops:
        # For every voxel (ih, ik, il), find where its equivalent lands
        H, K, L = vol.hkl_grid()
        hkl_eq = np.einsum("ij,...j->...i", op.astype(float), np.stack([H, K, L], axis=-1))

        ih_eq = np.round((hkl_eq[..., 0] - h_arr[0]) / (dh + 1e-15)).astype(int)
        ik_eq = np.round((hkl_eq[..., 1] - k_arr[0]) / (dk + 1e-15)).astype(int)
        il_eq = np.round((hkl_eq[..., 2] - l_arr[0]) / (dl + 1e-15)).astype(int)

        in_bounds = (
            (ih_eq >= 0) & (ih_eq < vol.shape[0]) &
            (ik_eq >= 0) & (ik_eq < vol.shape[1]) &
            (il_eq >= 0) & (il_eq < vol.shape[2])
        )

        # mask: both original and equivalent voxel must be valid
        src_mask = vol.mask & in_bounds
        ih_s = np.where(src_mask, ih_eq, 0)
        ik_s = np.where(src_mask, ik_eq, 0)
        il_s = np.where(src_mask, il_eq, 0)

        eq_data = vol.data[ih_s, ik_s, il_s]
        eq_var = vol.sigma[ih_s, ik_s, il_s] ** 2 + 1e-30
        w = np.where(src_mask, 1.0 / eq_var, 0.0)

        data_sum += w * eq_data * src_mask
        weight_sum += w * src_mask
        data_sq_sum += w * eq_data**2 * src_mask
        count += src_mask.astype(np.int32)

    valid = weight_sum > 0
    data_avg = np.where(valid, data_sum / weight_sum, vol.data)
    sigma_avg = np.where(valid, 1.0 / np.sqrt(weight_sum + 1e-30), vol.sigma)

    # Inter-equivalent variance for outlier detection
    mean_sq = np.where(valid, data_sq_sum / weight_sum, 0.0)
    inter_var = np.maximum(mean_sq - data_avg**2, 0.0)
    expected_var = sigma_avg**2
    outlier_flag = (np.sqrt(inter_var) > variance_flag_sigma * np.sqrt(expected_var)) & valid

    import dataclasses
    vol_sym = dataclasses.replace(vol, data=data_avg, sigma=sigma_avg)
    return vol_sym, outlier_flag
