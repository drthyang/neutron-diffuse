"""Backfill Bragg-punched holes — step 5 of the further analysis pipeline.

The preferred real-data fill is local-background replacement: each punched
Bragg/satellite hole is filled from the nearby unpunched shell around that hole.
Generic TV inpainting remains available as an option, but it can introduce
slice-scale staircase / smoothing artefacts in structured diffuse scattering.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ndiff.core import HKLVolume
from ndiff.inpainting.pipeline import fill, Method

BraggFillMethod = Method | Literal["local"]


def backfill_bragg(
    vol: HKLVolume,
    method: BraggFillMethod = "local",
    laue_class: str = "m3m",
    symmetry_ops: Optional[Sequence[NDArray]] = None,
    tv_lam: float = 0.2,
    tv_iter: int = 300,
    local_radius: int = 2,
    local_min_count: int = 8,
) -> HKLVolume:
    """Fill Bragg-punched voxels in *vol*.

    ``method="local"`` fills each connected punched region with the median of
    nearby valid voxels in a dilated shell around that region.  This estimates
    the local background level near the Bragg peak and avoids inventing a smooth
    TV surface through real diffuse texture.  TV/symmetry methods are retained
    for explicit comparisons.

    Parameters
    ----------
    vol:
        Volume after Bragg punching (``vol.mask`` marks valid voxels).
    method:
        Inpainting strategy. Default ``"symmetry+tv"``.
    laue_class:
        Crystal Laue class for symmetry filling.
    tv_lam:
        TV regularisation weight (higher than for Al backfill).
    tv_iter:
        Maximum TV iterations.
    local_radius:
        Number of binary-dilation iterations used to form the local shell around
        each punched component.
    local_min_count:
        Minimum valid shell voxels required before using the local shell median.
        Components with fewer fall back to the global valid-data median.

    Returns
    -------
    HKLVolume with Bragg holes filled.
    """
    if method == "local":
        return _local_background_fill(
            vol, radius=local_radius, min_count=local_min_count
        )
    return fill(
        vol,
        method=method,
        laue_class=laue_class,
        symmetry_ops=list(symmetry_ops) if symmetry_ops else None,
        tv_lam=tv_lam,
        tv_iter=tv_iter,
    )


def _local_background_fill(
    vol: HKLVolume,
    radius: int = 2,
    min_count: int = 8,
) -> HKLVolume:
    """Fill each punched connected component from its local valid shell."""
    data = vol.data.copy()
    sigma = vol.sigma.copy()
    holes = (~vol.mask) & np.isfinite(vol.data)
    if not holes.any():
        return dataclasses.replace(vol, mask=np.ones(vol.shape, dtype=bool))

    valid = vol.mask & np.isfinite(vol.data)
    global_vals = data[valid]
    global_fill = float(np.median(global_vals)) if global_vals.size else 0.0
    global_sigma = float(np.median(sigma[valid])) if global_vals.size else 1.0

    labels, n_label = ndimage.label(holes, structure=np.ones((3, 3, 3), dtype=bool))
    objects = ndimage.find_objects(labels)
    structure = np.ones((3, 3, 3), dtype=bool)
    pad = max(int(radius) + 1, 1)

    for lbl, obj in enumerate(objects, start=1):
        if obj is None:
            continue
        slices = []
        for s, n in zip(obj, vol.shape):
            slices.append(slice(max(0, s.start - pad), min(n, s.stop + pad)))
        region = tuple(slices)
        comp = labels[region] == lbl
        shell = ndimage.binary_dilation(
            comp, structure=structure, iterations=max(int(radius), 1)
        ) & ~comp
        shell_valid = shell & valid[region]
        if int(shell_valid.sum()) >= min_count:
            vals = data[region][shell_valid]
            fill_val = float(np.median(vals))
            fill_sig = float(np.std(vals)) if vals.size > 1 else global_sigma
        else:
            fill_val = global_fill
            fill_sig = global_sigma

        data_region = data[region]
        sigma_region = sigma[region]
        data_region[comp] = fill_val
        sigma_region[comp] = max(fill_sig, global_sigma)
        data[region] = data_region
        sigma[region] = sigma_region

    return dataclasses.replace(vol, data=data, sigma=sigma,
                               mask=np.ones(vol.shape, dtype=bool))
