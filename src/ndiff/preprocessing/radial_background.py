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
from dataclasses import dataclass, field
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
        Smooth ring component ``max(0, prof − base)`` per patch — the per-patch
        ring amplitude the azimuthal texture model is fit to.
    raw_profile : (P, Q)
        Robust (trimmed) radial profile per patch, before baseline removal.
    baseline : (P, Q)
        Estimated diffuse baseline per patch (diagnostic).
    counts : (P, Q)
        Number of valid voxels behind each (patch, |Q|) cell — the weights used
        for the azimuthal Fourier fit (sparse patches count for little).
    texture_coeffs : (Q, M)
        Per-|Q| azimuthal-texture Fourier coefficients Tᵩ(φ).  ``M`` is
        ``1 + n_fourier`` (symmetric, even-cosine basis) or ``1 + 2·n_fourier``
        (full).  Empty when the texture model is the discrete patch blend.
    n_fourier : int
        Number of azimuthal harmonics in ``texture_coeffs``.
    symmetric : bool
        If True the texture basis is the even-cosine series {1, cos2φ, cos4φ, …}
        (orthorhombic *mmm* in the kl/hl/hk plane); otherwise a full Fourier
        series {1, cosφ, sinφ, …}.
    """
    plane: str
    patch_centers: NDArray[np.float64]
    half_width: float
    q_grid: NDArray[np.float64]
    ring_profile: NDArray[np.float64]
    raw_profile: NDArray[np.float64]
    baseline: NDArray[np.float64]
    counts: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    texture_coeffs: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    n_fourier: int = 0
    symmetric: bool = True

    def evaluate(
        self,
        q_mag: NDArray[np.float64],
        phi: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Ring intensity at voxels with given |Q| and φ.

        Uses the per-|Q| low-order azimuthal Fourier texture when available
        (smooth, Bragg-immune, defined over all φ including the unmeasured
        azimuths); otherwise falls back to the discrete Hann patch blend.
        """
        if self.texture_coeffs.size:
            return self._evaluate_fourier(q_mag, phi)
        return self._evaluate_patches(q_mag, phi)

    def _evaluate_fourier(
        self, q_mag: NDArray, phi: NDArray
    ) -> NDArray[np.float64]:
        flat_q = q_mag.ravel()
        flat_phi = phi.ravel()
        # Interpolate each Fourier coefficient across |Q|, then evaluate the
        # azimuthal basis at each voxel's φ.
        basis = _azimuthal_basis(flat_phi, self.n_fourier, self.symmetric)  # (N, M)
        coeff_at = np.empty((flat_q.size, self.texture_coeffs.shape[1]))
        for m in range(self.texture_coeffs.shape[1]):
            coeff_at[:, m] = np.interp(
                flat_q, self.q_grid, self.texture_coeffs[:, m], left=0.0, right=0.0
            )
        I = np.einsum("nm,nm->n", basis, coeff_at)
        np.maximum(I, 0.0, out=I)
        return I.reshape(q_mag.shape)

    def _evaluate_patches(
        self, q_mag: NDArray, phi: NDArray
    ) -> NDArray[np.float64]:
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

    def texture(self, q0: float, phi: NDArray) -> NDArray[np.float64]:
        """Evaluate the azimuthal ring texture Tᵩ(φ) at a single |Q|=q0 (diagnostic)."""
        basis = _azimuthal_basis(np.asarray(phi, float), self.n_fourier, self.symmetric)
        coeffs = np.array([
            np.interp(q0, self.q_grid, self.texture_coeffs[:, m], left=0.0, right=0.0)
            for m in range(self.texture_coeffs.shape[1])
        ])
        return np.maximum(0.0, basis @ coeffs)


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
        rings; broader structure is kept as diffuse baseline (default 0.24,
        validated on the 28K 0kl slice — larger removes broader rings more
        completely at negligible diffuse cost; too large risks eating broad
        diffuse features).
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
    texture_model : {'fourier', 'patch'}
        ``'fourier'`` (default): model the ring's azimuthal amplitude Tᵩ(φ) as a
        low-order Fourier series fit per |Q| to the robust per-patch amplitudes,
        weighted by voxel count.  Smooth, immune to Bragg (low order + trimmed),
        and defined over all φ — so it interpolates/extrapolates the texture
        across azimuths that are sparsely sampled or unmeasured.  ``'patch'``
        keeps the discrete Hann patch blend (no extrapolation).
    n_fourier : int
        Number of azimuthal harmonics for the Fourier texture (default 1 — the
        single lowest *mmm* harmonic cos2φ; only long-wavelength texture, well
        below the angular scale of Bragg peaks).  With the ring measured over
        only narrow arcs, higher orders are poorly constrained — raise with care.
    texture_symmetric : bool
        Use the even-cosine basis {1, cos2φ, cos4φ, …} appropriate for a
        symmetrised orthorhombic (*mmm*) volume in the kl/hl/hk plane (default
        True).  The two symmetry-equivalent ring arcs then constrain one
        texture, which is what makes the extrapolation well-posed.
    texture_ridge : float
        Dimensionless smoothness prior on the harmonic coefficients (∝ order²),
        scaled to the weighted normal-matrix magnitude, to stabilise the fit
        where the measured arcs are narrow (default 0.3 — keeps the texture
        gentle without flattening it to a constant).
    texture_min_count_frac : float
        Per-|Q|, only patches sampled to at least this fraction of the
        best-sampled patch's count enter the texture fit (default 0.15).
        Under-sampled patches bias the ring amplitude low (too few voxels miss
        the radial peak); excluding them keeps the texture on the well-measured
        arcs so the ring is fully subtracted there.
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
        ring_width: float = 0.24,
        baseline_smooth: float = 0.06,
        profile_percentiles: tuple[float, float] = (10.0, 80.0),
        min_voxels_per_patch: int = 200,
        min_voxels_per_bin: int = 4,
        texture_model: str = "fourier",
        n_fourier: int = 1,
        texture_symmetric: bool = True,
        texture_ridge: float = 0.3,
        texture_min_count_frac: float = 0.15,
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
        self.texture_model = texture_model
        self.n_fourier = n_fourier
        self.texture_symmetric = texture_symmetric
        self.texture_ridge = texture_ridge
        self.texture_min_count_frac = texture_min_count_frac
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
        counts = np.zeros((self.n_patches, n_q))

        for p, pc in enumerate(patch_centers):
            d = _angular_distance(phiv_all, float(pc))
            in_p = np.abs(d) <= half_width
            if int(in_p.sum()) < self.min_voxels_per_patch:
                continue

            prof, cnt = _robust_radial_profile(
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
            counts[p] = cnt

        # Per-|Q| low-order azimuthal Fourier texture, weighted by voxel count.
        texture_coeffs = np.array([])
        if self.texture_model == "fourier":
            texture_coeffs = _fit_azimuthal_texture(
                patch_centers, ring, counts, self.n_fourier,
                self.texture_symmetric, self.texture_ridge,
                self.texture_min_count_frac,
            )

        self._profiles = RadialRingProfiles(
            plane=self.plane,
            patch_centers=patch_centers,
            half_width=half_width,
            q_grid=q_grid,
            ring_profile=ring,
            raw_profile=raw,
            baseline=base,
            counts=counts,
            texture_coeffs=texture_coeffs,
            n_fourier=self.n_fourier,
            symmetric=self.texture_symmetric,
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
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-|Q|-bin trimmed mean (rejects Bragg high tail and gap low tail).

    Returns ``(profile, counts)`` where ``counts`` is the number of voxels in
    each |Q| bin (used as the azimuthal-fit weight).
    """
    n_bins = len(edges) - 1
    out = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins)
    lo_p, hi_p = percentiles
    bin_idx = np.digitize(q, edges) - 1
    for b in range(n_bins):
        sel = I[bin_idx == b]
        counts[b] = sel.size
        if sel.size >= min_per_bin:
            lo, hi = np.percentile(sel, (lo_p, hi_p))
            keep = sel[(sel >= lo) & (sel <= hi)]
            out[b] = keep.mean() if keep.size else float(np.median(sel))
        elif sel.size > 0:
            out[b] = float(np.median(sel))
    return out, counts


def _azimuthal_basis(phi: NDArray, n_fourier: int, symmetric: bool) -> NDArray[np.float64]:
    """Design matrix (N, M) of the azimuthal texture basis evaluated at *phi*.

    symmetric : {1, cos2φ, cos4φ, …}  (orthorhombic *mmm* in the plane)
    full      : {1, cosφ, sinφ, cos2φ, sin2φ, …}
    """
    cols = [np.ones_like(phi)]
    if symmetric:
        for j in range(1, n_fourier + 1):
            cols.append(np.cos(2 * j * phi))
    else:
        for k in range(1, n_fourier + 1):
            cols.append(np.cos(k * phi))
            cols.append(np.sin(k * phi))
    return np.column_stack(cols)


def _fit_azimuthal_texture(
    patch_centers: NDArray,
    ring: NDArray,          # (P, Q) per-patch ring amplitude
    counts: NDArray,        # (P, Q) per-patch voxel counts (weights)
    n_fourier: int,
    symmetric: bool,
    ridge: float,
    min_count_frac: float,
) -> NDArray[np.float64]:
    """Fit a low-order azimuthal Fourier series per |Q| bin (weighted ridge LS).

    Returns coefficients of shape (Q, M).  Bins with no measured ring amplitude
    yield all-zero coefficients (no ring → nothing to subtract).

    Only patches with at least ``min_count_frac`` of the best-sampled patch's
    count at that |Q| enter the fit: under-sampled patches bias the ring
    amplitude *low* (too few voxels miss the radial peak), so including them
    would drag the texture below the well-measured arcs and leave a residual
    ring.
    """
    B = _azimuthal_basis(patch_centers, n_fourier, symmetric)   # (P, M)
    P, M = B.shape
    n_q = ring.shape[1]
    coeffs = np.zeros((n_q, M))

    # Smoothness prior: damp the non-constant harmonics, more strongly at higher
    # order (∝ harmonic *index*², so the first/primary harmonic is only lightly
    # penalised), leaving c₀ (the mean amplitude) free.  Scaled to the weighted
    # normal matrix below so `ridge` is dimensionless.
    if symmetric:
        order = np.array([0] + [j ** 2 for j in range(1, n_fourier + 1)], float)
    else:
        order = np.array([0] + [k ** 2 for k in range(1, n_fourier + 1)
                                for _ in (0, 1)], float)
    reg = np.diag(order)

    for b in range(n_q):
        w = counts[:, b].copy()
        y = ring[:, b]
        if w.max() <= 0 or not np.any(y > 0):
            continue
        w[w < min_count_frac * w.max()] = 0.0   # trust only well-sampled patches
        if w.sum() <= 0:
            continue
        wn = w / w[w > 0].mean()                 # normalise weight scale
        AtA = B.T @ (B * wn[:, None])
        # Scale the prior to the data term so `ridge` is relative, not absolute.
        scale = np.trace(AtA) / M
        Aty = B.T @ (wn * y)
        try:
            coeffs[b] = np.linalg.solve(AtA + ridge * scale * reg, Aty)
        except np.linalg.LinAlgError:
            coeffs[b], *_ = np.linalg.lstsq(AtA + ridge * scale * reg, Aty, rcond=None)
    return coeffs


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
