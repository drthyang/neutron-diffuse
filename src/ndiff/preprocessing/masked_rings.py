"""Mask-and-replace cleanup for residual powder-ring marks.

The radial background model is useful for estimating where the rings are, but a
perfect visual cleanup often wants a different operation: identify the pixels
that are ring-contaminated, then replace only those pixels by the smooth
background level.  This module keeps that experiment separate from the
subtractive model so the two can be compared directly.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_closing, binary_dilation

from ndiff.core import HKLVolume
from ndiff.preprocessing.radial_background import (
    PatchedRadialRingModel,
    RadialRingProfiles,
    _azimuthal_angle,
    _offset_q_magnitude,
)


@dataclass
class MaskedRingReplacement:
    """Result of replacing only detected ring-contaminated pixels."""

    clean: HKLVolume
    mask: NDArray[np.bool_]
    background: NDArray[np.float64]
    ring_model: NDArray[np.float64]


def replace_masked_ring_regions(
    vol: HKLVolume,
    model: PatchedRadialRingModel,
    profiles: Optional[RadialRingProfiles] = None,
    *,
    model_threshold_frac: float = 0.18,
    excess_sigma: float = 2.5,
    dilation_iter: int = 1,
    closing_iter: int = 1,
    fill_method: str = "sideband",
    n_phi_bins: int = 180,
) -> MaskedRingReplacement:
    """Replace detected ring pixels by a local non-ring background estimate.

    The mask is deliberately evidence-based rather than a hard circular shell:
    a voxel must have both a non-negligible fitted ring contribution and a
    positive excess above the fitted smooth background.  The candidate mask is
    then closed/dilated, allowing slightly elliptical or distorted ring marks to
    be selected as coherent image regions.

    ``fill_method='sideband'`` then fills the masked pixels by interpolating
    through nearby unmasked data along |Q| within the same azimuth bin.  That is
    intentionally less smooth than the model baseline: it bridges over the ring
    mark while preserving diffuse texture that is continuous through the ring.
    """
    prof = profiles or model.profiles or model.fit(vol)

    q_mag = _offset_q_magnitude(
        vol, model.plane, model.center_offset, model.center_offset_h_slope
    )
    phi = _azimuthal_angle(
        vol, model.plane, model.center_offset, model.center_offset_h_slope
    )
    ring_model = prof.evaluate(q_mag, phi)
    background = _evaluate_patch_background(prof, q_mag, phi)

    valid = vol.mask & np.isfinite(vol.data) & np.isfinite(background)
    excess = vol.data - background
    quiet = valid & (ring_model <= np.percentile(ring_model[valid], 60))
    noise = _robust_sigma(excess[quiet] if np.any(quiet) else excess[valid])
    min_model = model_threshold_frac * float(np.nanpercentile(ring_model[valid], 99))
    raw_mask = valid & (ring_model > min_model) & (excess > excess_sigma * noise)

    ring_mask = raw_mask
    if closing_iter > 0:
        ring_mask = binary_closing(ring_mask, structure=_plane_structure(vol.shape), iterations=closing_iter)
    if dilation_iter > 0:
        ring_mask = binary_dilation(ring_mask, structure=_plane_structure(vol.shape), iterations=dilation_iter)
    ring_mask &= valid

    if fill_method == "sideband":
        fill = _sideband_background(vol, ring_mask, q_mag, phi, background, n_phi_bins)
    elif fill_method == "model":
        fill = background
    else:
        raise ValueError(f"Unknown fill_method: {fill_method!r}")

    data = vol.data.copy()
    data[ring_mask] = fill[ring_mask]
    sigma = vol.sigma.copy()
    sigma[ring_mask] = np.sqrt(sigma[ring_mask] ** 2 + (0.1 * np.abs(fill[ring_mask])) ** 2)
    clean = dataclasses.replace(vol, data=data, sigma=sigma)
    return MaskedRingReplacement(clean=clean, mask=ring_mask, background=fill, ring_model=ring_model)


def _evaluate_patch_background(
    prof: RadialRingProfiles,
    q_mag: NDArray[np.float64],
    phi: NDArray[np.float64],
) -> NDArray[np.float64]:
    bg_prof = dataclasses.replace(
        prof,
        ring_profile=prof.baseline,
        texture_coeffs=np.array([]),
        texture_values=np.array([]),
    )
    return bg_prof.evaluate(q_mag, phi)


def _robust_sigma(vals: NDArray[np.float64]) -> float:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return max(1.4826 * mad, 1e-12)


def _sideband_background(
    vol: HKLVolume,
    ring_mask: NDArray[np.bool_],
    q_mag: NDArray[np.float64],
    phi: NDArray[np.float64],
    fallback: NDArray[np.float64],
    n_phi_bins: int,
) -> NDArray[np.float64]:
    """Interpolate masked ring pixels from unmasked radial sidebands.

    For each azimuth bin, sort the unmasked pixels by |Q| and interpolate the
    masked pixels from their nearest lower/higher-|Q| neighbours.  This is the
    backfill operation we want for diffuse preservation: it does not force the
    image onto a globally smooth radial baseline.
    """
    out = fallback.copy()
    valid = vol.mask & np.isfinite(vol.data) & np.isfinite(q_mag) & np.isfinite(phi)
    target_all = ring_mask & valid
    if not np.any(target_all):
        return out

    phi_bin = np.floor((phi % (2.0 * np.pi)) / (2.0 * np.pi) * n_phi_bins).astype(int)
    phi_bin = np.clip(phi_bin, 0, n_phi_bins - 1)

    for b in range(n_phi_bins):
        in_bin = valid & (phi_bin == b)
        target = target_all & in_bin
        if not np.any(target):
            continue
        clean = in_bin & ~ring_mask
        if int(clean.sum()) < 2:
            continue

        q_clean = q_mag[clean]
        i_clean = vol.data[clean]
        order = np.argsort(q_clean)
        q_sorted = q_clean[order]
        i_sorted = i_clean[order]
        q_unique, keep = np.unique(q_sorted, return_index=True)
        if q_unique.size < 2:
            continue
        i_unique = i_sorted[keep]
        out[target] = np.interp(q_mag[target], q_unique, i_unique)
    return out


def _plane_structure(shape: tuple[int, int, int]) -> NDArray[np.bool_]:
    """Morphological structure that stays within a single 2D slice when needed."""
    structure = np.ones((3, 3, 3), dtype=bool)
    for axis, n in enumerate(shape):
        if n == 1:
            slicer = [slice(None)] * 3
            slicer[axis] = [0, 2]
            structure[tuple(slicer)] = False
    return structure
