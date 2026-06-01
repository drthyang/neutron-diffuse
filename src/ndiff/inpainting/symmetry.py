"""Symmetry-based inpainting: fill masked voxels from crystallographic equivalents.

For a crystal with point group G, voxels related by G-operations in reciprocal
space should have equal intensity (up to noise). Masked voxels can be filled
by averaging their symmetry equivalents that are unmasked.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from ndiff.core import HKLVolume


# Each symmetry operation is a (3,3) integer matrix acting on (h,k,l)^T.
# A minimal library of point-group generators is provided; pass your own ops
# if needed.
CUBIC_M3M = [
    # identity
    np.eye(3, dtype=int),
    # 4-fold rotations around a, b, c axes
    np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
    np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    # inversion
    -np.eye(3, dtype=int),
]

TETRAGONAL_4_MMM = [
    np.eye(3, dtype=int),
    np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
    np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
    np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
    np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
    np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
]

ORTHORHOMBIC_MMM = [
    np.eye(3, dtype=int),
    np.diag([-1, 1, 1]),
    np.diag([1, -1, 1]),
    np.diag([1, 1, -1]),
    -np.eye(3, dtype=int),
    np.diag([1, -1, -1]),
    np.diag([-1, 1, -1]),
    np.diag([-1, -1, 1]),
]

LAUE_CLASSES: dict[str, list[NDArray]] = {
    "m3m": CUBIC_M3M,
    "4/mmm": TETRAGONAL_4_MMM,
    "mmm": ORTHORHOMBIC_MMM,
}


def symmetry_fill(
    vol: HKLVolume,
    symmetry_ops: Optional[Sequence[NDArray]] = None,
    laue_class: str = "m3m",
    min_equivalents: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Fill masked voxels using crystallographic symmetry equivalents.

    Parameters
    ----------
    vol:
        Volume to fill. Masked voxels (vol.mask == False) will be reconstructed.
    symmetry_ops:
        List of (3,3) integer matrices. If None, *laue_class* is used.
    laue_class:
        Key into LAUE_CLASSES if symmetry_ops is None. Default ``"m3m"``.
    min_equivalents:
        Minimum number of unmasked equivalents required to fill a voxel.
        Voxels with fewer equivalents are left unchanged and flagged.

    Returns
    -------
    data_filled:
        Intensity array with masked voxels filled where possible.
    sigma_filled:
        Uncertainty array (propagated from equivalents).
    filled_flag:
        Boolean array, True where a voxel was successfully filled.
    """
    ops = list(symmetry_ops) if symmetry_ops is not None else LAUE_CLASSES[laue_class]

    # Build lookup: fractional HKL → array index
    H, K, L = vol.hkl_grid()
    data_out = vol.data.copy()
    sigma_out = vol.sigma.copy()
    filled_flag = np.zeros(vol.shape, dtype=bool)

    masked_idx = np.argwhere(~vol.mask)

    h_arr, k_arr, l_arr = vol.h_axis, vol.k_axis, vol.l_axis
    dh = h_arr[1] - h_arr[0] if len(h_arr) > 1 else 1.0
    dk = k_arr[1] - k_arr[0] if len(k_arr) > 1 else 1.0
    dl = l_arr[1] - l_arr[0] if len(l_arr) > 1 else 1.0

    for ih, ik, il in masked_idx:
        hkl = np.array([h_arr[ih], k_arr[ik], l_arr[il]])
        vals: list[float] = []
        vars_: list[float] = []
        for op in ops:
            hkl_eq = op @ hkl
            # find nearest grid point
            ji = int(round((hkl_eq[0] - h_arr[0]) / dh))
            jk = int(round((hkl_eq[1] - k_arr[0]) / dk))
            jl = int(round((hkl_eq[2] - l_arr[0]) / dl))
            if (0 <= ji < vol.shape[0] and 0 <= jk < vol.shape[1]
                    and 0 <= jl < vol.shape[2]):
                if vol.mask[ji, jk, jl]:
                    vals.append(float(vol.data[ji, jk, jl]))
                    vars_.append(float(vol.sigma[ji, jk, jl]) ** 2)
        if len(vals) >= min_equivalents:
            w = 1.0 / (np.array(vars_) + 1e-30)
            data_out[ih, ik, il] = float(np.average(vals, weights=w))
            sigma_out[ih, ik, il] = float(np.sqrt(1.0 / w.sum()))
            filled_flag[ih, ik, il] = True

    return data_out, sigma_out, filled_flag
