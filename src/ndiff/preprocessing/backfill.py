"""Backfill Al-masked holes — step 3 of the data processing pipeline.

Thin wrapper around ndiff.inpainting.pipeline.fill() with defaults
appropriate for Al holes (typically narrow ring-shaped regions).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from ndiff.core import HKLVolume
from ndiff.inpainting.pipeline import fill, Method


def backfill_al(
    vol: HKLVolume,
    method: Method = "symmetry+tv",
    laue_class: str = "m3m",
    symmetry_ops: Optional[Sequence[NDArray]] = None,
    tv_lam: float = 0.05,
) -> HKLVolume:
    """Fill Al-masked voxels in *vol*.

    Al holes are narrow powder rings, so a small TV regularisation weight
    (0.05) is appropriate — we want to preserve sharp diffuse features
    while bridging the gap smoothly.

    Parameters
    ----------
    vol:
        Volume after Al masking (``vol.mask`` marks valid voxels).
    method:
        Inpainting strategy. Default ``"symmetry+tv"``.
    laue_class:
        Crystal Laue class for symmetry filling.
    tv_lam:
        TV regularisation weight.

    Returns
    -------
    HKLVolume with Al holes filled.
    """
    return fill(
        vol,
        method=method,
        laue_class=laue_class,
        symmetry_ops=list(symmetry_ops) if symmetry_ops else None,
        tv_lam=tv_lam,
    )
