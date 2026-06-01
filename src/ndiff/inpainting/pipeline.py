"""High-level fill() API: orchestrates symmetry → TV → RBF fallback."""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from ndiff.core import HKLVolume

Method = Literal["symmetry", "tv", "rbf", "biharmonic", "symmetry+tv", "symmetry+rbf"]


def fill(
    vol: HKLVolume,
    mask: Optional[NDArray[np.bool_]] = None,
    method: Method = "symmetry+tv",
    symmetry_ops: Optional[Sequence[NDArray]] = None,
    laue_class: str = "m3m",
    tv_lam: float = 0.1,
    tv_iter: int = 300,
    rbf_kernel: str = "thin_plate_spline",
    rbf_neighbors: int = 64,
) -> HKLVolume:
    """Fill masked voxels in *vol* and return a new HKLVolume.

    Parameters
    ----------
    vol:
        Source volume. ``vol.mask`` indicates valid voxels.
    mask:
        If provided, overrides ``vol.mask`` to select voxels to fill.
    method:
        Inpainting strategy:

        - ``"symmetry"``: use crystal symmetry equivalents only.
        - ``"tv"``: total-variation inpainting (entire masked region).
        - ``"rbf"``: radial-basis-function interpolation.
        - ``"biharmonic"``: iterative biharmonic relaxation.
        - ``"symmetry+tv"`` *(default)*: symmetry first, then TV for remainder.
        - ``"symmetry+rbf"``: symmetry first, then RBF for remainder.
    laue_class:
        Crystal Laue class for symmetry-based filling (``"m3m"``, ``"4/mmm"``,
        ``"mmm"``).
    tv_lam:
        TV regularisation weight (see :func:`tv_inpainting.tv_inpaint`).
    tv_iter:
        Maximum TV iterations.

    Returns
    -------
    HKLVolume
        New volume with filled data, updated sigma, and mask reset to all-True.
    """
    from ndiff.inpainting.symmetry import symmetry_fill
    from ndiff.inpainting.interpolation import rbf_fill, biharmonic_fill
    from ndiff.inpainting.tv_inpainting import tv_inpaint

    work_mask = (mask if mask is not None else vol.mask).copy()
    data = vol.data.copy()
    sigma = vol.sigma.copy()
    filled_flag = np.zeros(vol.shape, dtype=bool)

    if method in ("symmetry", "symmetry+tv", "symmetry+rbf"):
        data, sigma, sym_filled = symmetry_fill(
            vol, symmetry_ops=symmetry_ops, laue_class=laue_class
        )
        filled_flag |= sym_filled
        # update working mask: symmetry-filled voxels are now valid
        work_mask = work_mask | sym_filled

    remaining = ~work_mask

    if remaining.any():
        if method in ("tv", "symmetry+tv"):
            data = tv_inpaint(data, work_mask, lam=tv_lam, max_iter=tv_iter)
            filled_flag |= remaining
        elif method in ("rbf", "symmetry+rbf"):
            data = rbf_fill(data, work_mask, kernel=rbf_kernel, neighbors=rbf_neighbors)
            filled_flag |= remaining
        elif method == "biharmonic":
            data = biharmonic_fill(data, work_mask)
            filled_flag |= remaining

    import dataclasses
    out = dataclasses.replace(
        vol,
        data=data,
        sigma=sigma,
        mask=np.ones(vol.shape, dtype=bool),
    )
    out.mask[~filled_flag & ~vol.mask] = False  # voxels still unfilled stay masked
    return out
