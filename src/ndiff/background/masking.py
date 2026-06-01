"""General masking utilities for 3D HKL volumes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation

from ndiff.core import HKLVolume


class MaskBuilder:
    """Combine multiple mask sources into a final keep-mask."""

    def __init__(self, vol: HKLVolume) -> None:
        self._shape = vol.shape
        self._keep = vol.mask.copy()

    def add_q_shell(
        self,
        q_mag: NDArray[np.float64],
        q0: float,
        half_width: float,
        taper: float = 0.0,
    ) -> "MaskBuilder":
        """Mask a |Q|-shell at q0 with given half-width."""
        dq = np.abs(q_mag - q0)
        if taper > 0:
            from ndiff.background.aluminum import _sigmoid_taper
            weight = _sigmoid_taper(dq, half_width, taper)
            self._keep &= weight > 0.5
        else:
            self._keep &= dq > half_width
        return self

    def add_hkl_mask(self, bad: NDArray[np.bool_]) -> "MaskBuilder":
        """Mark voxels where *bad* is True as invalid."""
        self._keep &= ~bad
        return self

    def dilate(self, iterations: int = 1) -> "MaskBuilder":
        """Dilate the masked (False) region by *iterations* voxels.

        Expands mask boundary outward, protecting clean data near Al peaks
        from subtle wing contamination.
        """
        masked = ~self._keep
        masked = binary_dilation(masked, iterations=iterations)
        self._keep = ~masked
        return self

    def build(self) -> NDArray[np.bool_]:
        return self._keep.copy()

    def apply_to(self, vol: HKLVolume) -> None:
        """Apply accumulated mask in place to *vol*."""
        vol.mask &= self._keep


def count_masked(mask: NDArray[np.bool_]) -> dict[str, float]:
    """Return mask statistics."""
    total = mask.size
    n_masked = int((~mask).sum())
    return {
        "total_voxels": total,
        "masked_voxels": n_masked,
        "fraction_masked": n_masked / total,
    }
