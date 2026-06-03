"""Non-parametric powder-ring removal by per-patch radial background subtraction.

Motivation
----------
The parametric :class:`~ndiff.preprocessing.ring_model.PatchedRingModel` models
each ring as a Gaussian in |Q| with a shared azimuthal texture T(φ).  On real
data that model under-subtracts badly because (a) a single shared T(φ) cannot
represent rings with different azimuthal texture, (b) the ring centres drift
from the nominal hints, and (c) the rings are far narrower in |Q| than the
collection shell, so a shell-averaged amplitude washes the peak away.

This estimator makes **no parametric assumption** about ring position, width,
or shape.  A powder ring is, by definition, the part of the signal that is

    * peaked and narrow in |Q|, and
    * azimuthally smooth (slowly varying with φ),

sitting on top of the broad diffuse background and the sharp, localised Bragg
peaks.  We isolate it directly:

1. Divide φ into overlapping (Hann-blended) patches.
2. In each patch, build a **robust radial profile** ``prof(|Q|)`` — a per-|Q|-bin
   trimmed mean.  The trim rejects the high tail (Bragg peaks) and the low tail
   (detector gaps/shadows), so the profile tracks the smooth ring+diffuse level.
3. Estimate the smooth **diffuse baseline** ``base(|Q|)`` under the rings by a
   morphological *opening* (rolling minimum then maximum) wider than the rings,
   followed by light smoothing.  Peaks narrower than ``ring_width`` are removed;
   the broad diffuse survives.
4. The ring component in that patch is ``ring(|Q|) = max(0, prof − base)``.
5. Subtract: each voxel gets the Hann-weighted blend of its neighbouring
   patches' ``ring(|Q|)`` interpolated at the voxel's |Q|.

Because the profile is built from a *trimmed* statistic, Bragg peaks never enter
``ring`` and are therefore left untouched in the residual for the Bragg punch.
Because the baseline is estimated per patch, each ring's azimuthal texture is
captured directly — no rank-1 / shared-T(φ) assumption.

This is a drop-in alternative to ``PatchedRingModel`` for Step 2 of the
pipeline; the two are independently swappable.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d, grey_opening

from ndiff.core import HKLVolume


# ---------------------------------------------------------------------------
# Fitted result
# ---------------------------------------------------------------------------

@dataclass
class RadialRingProfiles:
    """Per-patch radial ring profiles produced by :meth:`PatchedRadialRingModel.fit`.

    Attributes
    ----------
    plane : str
        Reference plane defining the azimuthal angle φ.
    patch_centers : (P,)
        Azimuthal centre (rad) of each patch.
    half_width : float
        Angular half-width (rad) of each patch's Hann window.
    q_grid : (Q,)
        Radial |Q| grid (bin centres, Å⁻¹).
    ring_profile : (P, Q)
        Smooth ring component ``max(0, prof − base)`` per patch — the quantity
        subtracted from the data.
    raw_profile : (P, Q)
        Robust (trimmed) radial profile per patch, before baseline removal.
    baseline : (P, Q)
        Estimated diffuse baseline per patch (diagnostic).
    """
    plane: str
    patch_centers: NDArray[np.float64]
    half_width: float
    q_grid: NDArray[np.float64]
    ring_profile: NDArray[np.float64]
    raw_profile: NDArray[np.float64]
    baseline: NDArray[np.float64]

    def evaluate(
        self,
        q_mag: NDArray[np.float64],
        phi: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Ring intensity at voxels with given |Q| and φ (Hann-blended over patches)."""
        flat_q = q_mag.ravel()
        flat_phi = phi.ravel()
        I = np.zeros(flat_q.shape, dtype=np.float64)
        wsum = np.zeros(flat_q.shape, dtype=np.float64)

        for p, pc in enumerate(self.patch_centers):
            d = _angular_distance(flat_phi, float(pc))
            in_p = np.abs(d) <= self.half_width
            if not np.any(in_p):
                continue
            # Hann weight: 1 at patch centre, 0 at the edge of the window.
            w = 0.5 * (1.0 + np.cos(np.pi * d[in_p] / self.half_width))
            ring_at = np.interp(
                flat_q[in_p], self.q_grid, self.ring_profile[p],
                left=0.0, right=0.0,
            )
            I[in_p] += w * ring_at
            wsum[in_p] += w

        good = wsum > 0
        I[good] /= wsum[good]
        return I.reshape(q_mag.shape)


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class PatchedRadialRingModel:
    """Remove powder rings by per-patch radial background subtraction.

    Parameters
    ----------
    n_patches : int
        Number of azimuthal patches spanning [0, 2π).
    overlap_frac : float
        Hann-overlap fraction on each side of a patch (0–0.5).  The window
        half-width is ``0.5 · (2π/n_patches) · (1 + 2·overlap_frac)``, so
        neighbouring patches overlap and blend smoothly (C¹ across boundaries).
    plane : str
        Reference plane for φ: ``'hk0'`` → atan2(k_Q, h_Q); ``'h0l'`` →
        atan2(l_Q, h_Q); ``'0kl'`` → atan2(l_Q, k_Q).
    q_step : float
        Radial bin width (Å⁻¹) of the per-patch profile.  Should be a few times
        finer than the ring width to resolve the peak (default 0.02).
    ring_width : float
        Approximate maximum full width (Å⁻¹) of a powder ring in |Q|.  Sets the
        morphological-opening element: peaks narrower than this are treated as
        rings; broader structure is kept as diffuse baseline (default 0.18).
    baseline_smooth : float
        σ (Å⁻¹) of the Gaussian applied to the baseline after opening, to avoid
        kinks (default 0.06).  Set 0 to disable.
    profile_percentiles : tuple[float, float]
        Low/high percentile band kept per |Q| bin when forming the robust
        radial profile (default 10–80).  Low-trim drops gaps/shadows; high-trim
        drops Bragg peaks.
    min_voxels_per_patch : int
        Patches with fewer valid voxels are skipped (contribute no ring).
    min_voxels_per_bin : int
        |Q| bins with fewer voxels fall back to the bin median (or NaN, then
        interpolated) rather than a trimmed mean.
    snr_mask_threshold : float or None
        After subtraction, mask voxels where ``I_ring / σ_data`` exceeds this,
        flagging them for the downstream backfill.  ``None`` leaves the mask
        unchanged (subtract only).
    """

    def __init__(
        self,
        n_patches: int = 36,
        overlap_frac: float = 0.3,
        plane: str = "hk0",
        q_step: float = 0.02,
        ring_width: float = 0.18,
        baseline_smooth: float = 0.06,
        profile_percentiles: tuple[float, float] = (10.0, 80.0),
        min_voxels_per_patch: int = 200,
        min_voxels_per_bin: int = 4,
        snr_mask_threshold: Optional[float] = None,
    ) -> None:
        self.n_patches = n_patches
        self.overlap_frac = overlap_frac
        self.plane = plane
        self.q_step = q_step
        self.ring_width = ring_width
        self.baseline_smooth = baseline_smooth
        self.profile_percentiles = profile_percentiles
        self.min_voxels_per_patch = min_voxels_per_patch
        self.min_voxels_per_bin = min_voxels_per_bin
        self.snr_mask_threshold = snr_mask_threshold
        self._profiles: Optional[RadialRingProfiles] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        vol: HKLVolume,
        q_range: Optional[tuple[float, float]] = None,
    ) -> RadialRingProfiles:
        """Estimate per-patch ring profiles from *vol*."""
        q_mag = vol.q_magnitude()
        phi = _azimuthal_angle(vol, self.plane)
        valid = vol.mask & np.isfinite(vol.data)

        if q_range is None:
            q_range = (float(q_mag[valid].min()), float(q_mag[valid].max()))
        q0, q1 = q_range

        edges = np.arange(q0, q1 + self.q_step, self.q_step)
        q_grid = 0.5 * (edges[:-1] + edges[1:])

        patch_width = 2 * np.pi / self.n_patches
        half_width = 0.5 * patch_width * (1.0 + 2.0 * self.overlap_frac)
        patch_centers = np.linspace(0, 2 * np.pi, self.n_patches, endpoint=False)

        qv_all = q_mag[valid]
        Iv_all = vol.data[valid]
        phiv_all = phi[valid]

        n_q = len(q_grid)
        raw = np.zeros((self.n_patches, n_q))
        base = np.zeros((self.n_patches, n_q))
        ring = np.zeros((self.n_patches, n_q))

        for p, pc in enumerate(patch_centers):
            d = _angular_distance(phiv_all, float(pc))
            in_p = np.abs(d) <= half_width
            if int(in_p.sum()) < self.min_voxels_per_patch:
                continue

            prof = _robust_radial_profile(
                qv_all[in_p], Iv_all[in_p], edges,
                self.profile_percentiles, self.min_voxels_per_bin,
            )
            prof = _fill_nan_1d(prof)
            b = _estimate_baseline(
                prof, self.q_step, self.ring_width, self.baseline_smooth,
            )
            raw[p] = prof
            base[p] = b
            ring[p] = np.maximum(0.0, prof - b)

        self._profiles = RadialRingProfiles(
            plane=self.plane,
            patch_centers=patch_centers,
            half_width=half_width,
            q_grid=q_grid,
            ring_profile=ring,
            raw_profile=raw,
            baseline=base,
        )
        return self._profiles

    def subtract(
        self,
        vol: HKLVolume,
        profiles: Optional[RadialRingProfiles] = None,
    ) -> tuple[HKLVolume, NDArray[np.float64]]:
        """Subtract the fitted ring profiles from *vol*.

        Returns ``(vol_sub, I_ring)``.  ``vol_sub.data = vol.data − I_ring``;
        if ``snr_mask_threshold`` is set, voxels where the ring dominates are
        masked for the downstream backfill.
        """
        prof = profiles or self._profiles
        if prof is None:
            raise RuntimeError("Call fit() before subtract().")

        q_mag = vol.q_magnitude()
        phi = _azimuthal_angle(vol, self.plane)
        I_ring = prof.evaluate(q_mag, phi)

        data_sub = vol.data - I_ring
        sigma_sub = np.sqrt(vol.sigma**2 + (0.1 * np.abs(I_ring)) ** 2)

        mask = vol.mask
        if self.snr_mask_threshold is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                snr = np.where(vol.sigma > 0, I_ring / vol.sigma, 0.0)
            mask = vol.mask & (snr < self.snr_mask_threshold)

        vol_sub = dataclasses.replace(vol, data=data_sub, sigma=sigma_sub, mask=mask)
        return vol_sub, I_ring

    @property
    def profiles(self) -> Optional[RadialRingProfiles]:
        return self._profiles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _azimuthal_angle(vol: HKLVolume, plane: str) -> NDArray[np.float64]:
    """Azimuthal angle φ (radians) for every voxel, in the given plane."""
    H, K, L = vol.hkl_grid()
    Q = np.stack([H, K, L], axis=-1) @ vol.ub_matrix.T  # (..., 3) Å⁻¹
    if plane == "hk0":
        return np.arctan2(Q[..., 1], Q[..., 0])
    if plane == "h0l":
        return np.arctan2(Q[..., 2], Q[..., 0])
    if plane == "0kl":
        return np.arctan2(Q[..., 2], Q[..., 1])
    raise ValueError(f"Unknown plane: {plane!r}")


def _angular_distance(phi: NDArray, phi_c: float) -> NDArray:
    """Signed angular distance in (-π, π], accounting for wrap-around."""
    d = phi - phi_c
    return (d + np.pi) % (2 * np.pi) - np.pi


def _robust_radial_profile(
    q: NDArray,
    I: NDArray,
    edges: NDArray,
    percentiles: tuple[float, float],
    min_per_bin: int,
) -> NDArray[np.float64]:
    """Per-|Q|-bin trimmed mean (rejects Bragg high tail and gap low tail)."""
    n_bins = len(edges) - 1
    out = np.full(n_bins, np.nan)
    lo_p, hi_p = percentiles
    bin_idx = np.digitize(q, edges) - 1
    for b in range(n_bins):
        sel = I[bin_idx == b]
        if sel.size >= min_per_bin:
            lo, hi = np.percentile(sel, (lo_p, hi_p))
            keep = sel[(sel >= lo) & (sel <= hi)]
            out[b] = keep.mean() if keep.size else float(np.median(sel))
        elif sel.size > 0:
            out[b] = float(np.median(sel))
    return out


def _fill_nan_1d(prof: NDArray) -> NDArray[np.float64]:
    """Linearly interpolate NaNs in a 1D profile (edge-extended)."""
    prof = prof.astype(np.float64).copy()
    nan = np.isnan(prof)
    if nan.all():
        return np.zeros_like(prof)
    if nan.any():
        idx = np.arange(prof.size)
        prof[nan] = np.interp(idx[nan], idx[~nan], prof[~nan])
    return prof


def _estimate_baseline(
    prof: NDArray,
    q_step: float,
    ring_width: float,
    smooth: float,
) -> NDArray[np.float64]:
    """Smooth diffuse baseline under the rings via morphological opening.

    Opening (erosion → dilation) with a flat element wider than the rings
    removes positive peaks narrower than ``ring_width`` while preserving the
    broad diffuse background.  A light Gaussian smooth removes kinks.  The
    result is clamped not to exceed the profile, so ``prof − base ≥ 0``.
    """
    size = max(3, int(round(ring_width / q_step)))
    if size % 2 == 0:
        size += 1
    base = grey_opening(prof, size=size, mode="nearest")
    if smooth > 0:
        base = gaussian_filter1d(base, smooth / q_step, mode="nearest")
    return np.minimum(base, prof)
