# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tsung-han Yang

"""Step 1 of powder ring removal: subtract an empty-environment scan.

The empty scan captures the ring contribution from the sample environment
(cryostat, furnace, pressure cell walls, etc.) without the sample.
Subtracting it removes the bulk of the powder ring signal.

Residual rings from the **sample holder** remain after this step because
the sample holder is present during the sample scan but absent during
the empty scan.  Those are handled in step 2 (see residual_rings.py).

Scale factor
------------
The empty scan must be normalised to the same incident flux as the sample
scan (typically done by monitor normalisation in Mantid before export).
If additional scaling is needed (e.g. different counting times or slightly
different flux), :meth:`EmptySubtractor.estimate_scale` fits the scale
factor by minimising the residual in ring-dominated |Q| shells.
"""

from __future__ import annotations

import numpy as np

from nebula3d.core import HKLVolume


class EmptySubtractor:
    """Subtract a monitor-normalised empty-environment scan.

    Parameters
    ----------
    empty : HKLVolume
        Empty-environment scan on the **same HKL grid** as the sample data.
        Must be monitor-normalised before passing in (done in Mantid).
    scale : float or None
        Multiplicative scale applied to the empty before subtraction.
        If None, :meth:`estimate_scale` is called automatically.
    scale_q_range : tuple[float, float]
        |Q| range (Å^-1) used for scale estimation; should cover a
        prominent ring and avoid regions dominated by diffuse signal.
    """

    def __init__(
        self,
        empty: HKLVolume,
        scale: float | None = None,
        scale_q_range: tuple[float, float] = (2.0, 3.5),
        clip_percentile: float = 99.0,
    ) -> None:
        self.empty = empty
        self._scale = scale
        self.scale_q_range = scale_q_range
        self.clip_percentile = clip_percentile

    def estimate_scale(self, sample: HKLVolume) -> float:
        """Robust least-squares scale in *scale_q_range* where the ring dominates.

        Minimises  ||I_sample(Q) - s * I_empty(Q)||  over valid voxels in the
        specified |Q| window, solving analytically:

            s = sum(I_sample * I_empty) / sum(I_empty^2)

        Empty scans can contain a few extreme-intensity voxels (sample-
        environment Bragg peaks, ring hot-spots) whose I_empty² would dominate
        the denominator and collapse the scale toward zero.  Voxels whose
        I_empty exceeds the ``clip_percentile`` percentile within the window
        are therefore excluded from the fit (set ``clip_percentile=100`` to
        disable clipping and recover the plain least-squares estimate).
        """
        q_s = sample.q_magnitude()

        in_range = (
            (q_s >= self.scale_q_range[0]) & (q_s <= self.scale_q_range[1]) &
            sample.mask & self.empty.mask
        )
        if in_range.sum() < 10:
            return 1.0

        I_s = sample.data[in_range]
        I_e = self.empty.data[in_range]

        if self.clip_percentile < 100.0:
            cut = float(np.percentile(I_e, self.clip_percentile))
            keep = I_e <= cut
            if keep.sum() >= 10:
                I_s, I_e = I_s[keep], I_e[keep]

        denom = float((I_e ** 2).sum())
        if denom < 1e-12:
            return 1.0
        return float((I_s * I_e).sum() / denom)

    @property
    def scale(self) -> float:
        return self._scale if self._scale is not None else 1.0

    def subtract(self, sample: HKLVolume) -> HKLVolume:
        """Subtract the scaled empty from *sample*.

        The scale factor is estimated from the data if not set at construction.

        Parameters
        ----------
        sample : HKLVolume
            Sample scan (monitor-normalised).

        Returns
        -------
        HKLVolume
            New volume with empty subtracted; σ propagated in quadrature.
            Mask: valid only where both sample and empty voxels are valid.
        """
        import dataclasses

        if self._scale is None:
            self._scale = self.estimate_scale(sample)

        data_sub = sample.data - self._scale * self.empty.data
        sigma_sub = np.sqrt(sample.sigma ** 2 + (self._scale * self.empty.sigma) ** 2)
        combined_mask = sample.mask & self.empty.mask

        return dataclasses.replace(
            sample,
            data=data_sub,
            sigma=sigma_sub,
            mask=combined_mask,
        )
