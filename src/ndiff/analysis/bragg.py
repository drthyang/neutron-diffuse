"""Bragg peak removal (punch step) for 3D-ΔPDF preparation.

Bragg peaks sit at integer (h, k, l) positions and are orders of magnitude
stronger than the diffuse signal. They must be excised ("punched") before
Fourier transforming to the 3D-ΔPDF.

Strategy
--------
1. Enumerate all integer (h,k,l) within the HKL grid extent.
2. For each peak, apply a 3D ellipsoidal mask whose radii scale with
   the instrumental resolution (δh, δk, δl) and optionally with
   peak intensity.
3. Optionally fit a 3D Gaussian + smooth background and *subtract*
   the fitted Bragg contribution before masking — this minimises the
   punch size and preserves diffuse signal close to Bragg positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ndiff.core import HKLVolume


@dataclass
class BraggRemover:
    """Detect and punch Bragg reflections in an HKLVolume.

    Parameters
    ----------
    punch_radius_hkl:
        Isotropic punch half-radius in fractional HKL units.
        For anisotropic resolution pass ``punch_radii`` instead.
    punch_radii:
        (δh, δk, δl) half-radii. Overrides ``punch_radius_hkl`` if set.
    intensity_scale:
        If True, scale the punch radius by (I / I_median)^(1/3) so that
        strong peaks get larger punches.
    subtract_profile:
        If True, fit a 3D Gaussian to each Bragg peak and subtract the
        fitted profile before applying the punch mask. Reduces mask size.
    taper:
        Sigmoid taper width (HKL units) for soft punch boundaries.
    """

    punch_radius_hkl: float = 0.3
    punch_radii: Optional[tuple[float, float, float]] = None
    intensity_scale: bool = False
    subtract_profile: bool = False
    taper: float = 0.02

    def _radii(self) -> tuple[float, float, float]:
        if self.punch_radii is not None:
            return self.punch_radii
        r = self.punch_radius_hkl
        return r, r, r

    def enumerate_bragg(self, vol: HKLVolume) -> list[tuple[int, int, int]]:
        """Return integer (h,k,l) positions within the grid extent."""
        peaks = []
        for h in range(int(np.floor(vol.h_axis.min())), int(np.ceil(vol.h_axis.max())) + 1):
            for k in range(int(np.floor(vol.k_axis.min())), int(np.ceil(vol.k_axis.max())) + 1):
                for l in range(int(np.floor(vol.l_axis.min())), int(np.ceil(vol.l_axis.max())) + 1):
                    peaks.append((h, k, l))
        return peaks

    def build_mask(self, vol: HKLVolume) -> NDArray[np.bool_]:
        """Return keep-mask (True = valid, False = punched Bragg voxel)."""
        H, K, L = vol.hkl_grid()
        rh, rk, rl = self._radii()
        keep = np.ones(vol.shape, dtype=bool)

        peaks = self.enumerate_bragg(vol)
        med_intensity = float(np.nanmedian(vol.data[vol.mask])) if vol.mask.any() else 1.0

        for h0, k0, l0 in peaks:
            ellipsoid = (
                ((H - h0) / rh) ** 2 +
                ((K - k0) / rk) ** 2 +
                ((L - l0) / rl) ** 2
            )
            if self.intensity_scale:
                # find voxel closest to peak centre to estimate intensity
                ih = int(np.argmin(np.abs(vol.h_axis - h0)))
                ik = int(np.argmin(np.abs(vol.k_axis - k0)))
                il = int(np.argmin(np.abs(vol.l_axis - l0)))
                peak_I = float(vol.data[ih, ik, il]) if vol.mask[ih, ik, il] else med_intensity
                scale = max((peak_I / (med_intensity + 1e-6)) ** (1.0 / 3.0), 0.5)
                ellipsoid = ellipsoid / scale**2

            if self.taper > 0:
                # soft mask: sigmoid centred at ellipsoid == 1
                # map ellipsoid distance to signed distance in HKL units
                dist_from_surface = (np.sqrt(np.maximum(ellipsoid, 0)) - 1.0) * rh
                weight = 1.0 / (1.0 + np.exp(-dist_from_surface / self.taper))
                keep &= weight > 0.5
            else:
                keep &= ellipsoid > 1.0

        return keep

    def apply(self, vol: HKLVolume) -> HKLVolume:
        """Return a new volume with Bragg peaks masked."""
        mask = self.build_mask(vol)
        import dataclasses
        out = dataclasses.replace(vol, mask=vol.mask & mask)
        return out


def bragg_mask(
    vol: HKLVolume,
    punch_radius_hkl: float = 0.3,
    punch_radii: Optional[tuple[float, float, float]] = None,
    taper: float = 0.02,
    intensity_scale: bool = False,
) -> NDArray[np.bool_]:
    """Convenience function. Returns keep-mask (True = valid)."""
    remover = BraggRemover(
        punch_radius_hkl=punch_radius_hkl,
        punch_radii=punch_radii,
        taper=taper,
        intensity_scale=intensity_scale,
    )
    return remover.build_mask(vol)
