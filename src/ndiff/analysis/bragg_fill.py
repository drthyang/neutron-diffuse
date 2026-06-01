"""Backfill Bragg-punched holes — step 5 of the further analysis pipeline.

Bragg holes are typically larger and more numerous than Al holes.
TV inpainting with a slightly higher λ is the default; symmetry fill
is applied first when crystal equivalents are available.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from ndiff.core import HKLVolume
from ndiff.inpainting.pipeline import fill, Method


def backfill_bragg(
    vol: HKLVolume,
    method: Method = "symmetry+tv",
    laue_class: str = "m3m",
    symmetry_ops: Optional[Sequence[NDArray]] = None,
    tv_lam: float = 0.2,
) -> HKLVolume:
    """Fill Bragg-punched voxels in *vol*.

    Bragg holes are larger than Al holes, so a moderate TV weight (0.2)
    is used — enough to produce smooth fills without over-smoothing
    nearby diffuse structure.

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

    Returns
    -------
    HKLVolume with Bragg holes filled.
    """
    return fill(
        vol,
        method=method,
        laue_class=laue_class,
        symmetry_ops=list(symmetry_ops) if symmetry_ops else None,
        tv_lam=tv_lam,
    )
