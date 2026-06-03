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
from scipy.optimize import minimize, nnls

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
    texture_values : (P, Q)
        Per-patch smooth texture values from the minimizer-based texture model.
        Empty unless ``texture_model='smooth'``.
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
    texture_values: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    n_fourier: int = 0
    symmetric: bool = False

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
        if self.texture_values.size:
            return self._evaluate_smooth(q_mag, phi)
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

    def _evaluate_smooth(
        self, q_mag: NDArray, phi: NDArray
    ) -> NDArray[np.float64]:
        flat_q = q_mag.ravel()
        flat_phi = phi.ravel()

        # Interpolate the optimized patch values across |Q|, then periodic
        # linear-interpolate along φ.  This keeps the smooth-minimizer texture
        # independent of the Hann patch windows used to build the observations.
        vals_at_q = np.empty((self.patch_centers.size, flat_q.size))
        for p in range(self.patch_centers.size):
            vals_at_q[p] = np.interp(
                flat_q, self.q_grid, self.texture_values[p], left=0.0, right=0.0
            )

        period = 2 * np.pi
        phi_mod = flat_phi % period
        centers = self.patch_centers
        step = period / centers.size
        idx_float = phi_mod / step
        i0 = np.floor(idx_float).astype(int) % centers.size
        i1 = (i0 + 1) % centers.size
        t = idx_float - np.floor(idx_float)

        n = np.arange(flat_q.size)
        I = (1.0 - t) * vals_at_q[i0, n] + t * vals_at_q[i1, n]
        np.maximum(I, 0.0, out=I)
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
    ring_smooth : float
        σ (Å⁻¹) of an optional Gaussian applied to the fitted ring excess along
        |Q| after baseline subtraction/template projection.  This suppresses
        radial bin-to-bin noise, which becomes visible when ``q_step`` is very
        fine, and helps keep the residual background continuous through ring
        shells.  Set 0 to disable.
    profile_percentiles : tuple[float, float]
        Low/high percentile band kept per |Q| bin when forming the robust
        radial profile (default 10–80).  Low-trim drops gaps/shadows; high-trim
        drops Bragg peaks.
    profile_method : {'trimmed_mean', 'winsorized_mean', 'median', 'huber'}
        Robust statistic used after the per-bin signal distribution is
        collected.  ``'trimmed_mean'`` is the current default: drop values
        outside ``profile_percentiles`` and average the rest.  ``'winsorized_mean'``
        clips values to the percentile interval before averaging.  ``'median'``
        uses the median only.  ``'huber'`` clips values to median ± 3·MAD before
        averaging, which is a symmetric outlier rejection not tied to fixed
        percentile cutoffs.
    min_voxels_per_patch : int
        Patches with fewer valid voxels are skipped (contribute no ring).
    min_voxels_per_bin : int
        |Q| bins with fewer voxels fall back to the bin median (or NaN, then
        interpolated) rather than a trimmed mean.
    texture_model : {'fourier', 'smooth', 'patch'}
        ``'fourier'`` (default): model the ring's azimuthal amplitude Tᵩ(φ) as a
        low-order Fourier series fit per |Q| to the robust per-patch amplitudes,
        weighted by voxel count.  Smooth, immune to Bragg (low order + trimmed),
        and defined over all φ — so it interpolates/extrapolates the texture
        across azimuths that are sparsely sampled or unmeasured.  ``'smooth'``
        fits one nonnegative texture value per patch with a minimizer and a
        cyclic second-difference penalty, so the texture is smooth without
        choosing a Fourier basis.  ``'patch'`` keeps the discrete Hann patch
        blend (no extrapolation).
    n_fourier : int
        Number of azimuthal harmonics for the Fourier texture (default 3).  Low
        order captures only long-wavelength texture, well below the angular scale
        of (point-like) Bragg peaks, so they cannot leak into the texture.
    texture_symmetric : bool
        If True, restrict the texture to the even-cosine basis {1, cos2φ, cos4φ,
        …} (a symmetrised orthorhombic *mmm* volume in the plane).  Default
        False — fit a **general** Fourier series {1, cosφ, sinφ, cos2φ, sin2φ,
        …} that makes no symmetry assumption.  With full azimuthal coverage the
        general fit is well-posed; impose symmetry only when coverage is
        one-sided and the point group justifies it.
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
    texture_smoothness : float
        Smoothness penalty for ``texture_model='smooth'``.  Larger values
        suppress fine azimuthal texture more strongly by penalizing cyclic
        second differences between neighboring patch amplitudes.
    ring_templates : sequence, optional
        Fixed radial ring shapes, each an object with ``q_center`` and ``sigma``
        attributes (e.g. :class:`~ndiff.preprocessing.powder_rings.RingProfile`)
        or a ``(q_center, sigma)`` tuple — typically from a Bragg-free linecut
        (:func:`~ndiff.preprocessing.powder_rings.fit_ring_profiles`).  When
        given, each patch's ring profile is rebuilt as Σ aᵢ·Gᵢ(|Q|) with the
        amplitudes ``aᵢ`` projected from the patch's baseline-subtracted profile,
        so the subtracted radial *shape* matches the measured ring width exactly
        (the trimmed per-patch profile otherwise slightly under-fills the peak
        and leaves a faint residual ring).  ``None`` keeps the fully
        non-parametric profile.
    center_offset : (float, float)
        Experimental in-plane ring-center offset in Å⁻¹, in the same
        orthonormal plane frame used for φ. ``(0, 0)`` means rings are centered
        on Q=0.
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
        ring_smooth: float = 0.0,
        profile_percentiles: tuple[float, float] = (10.0, 80.0),
        profile_method: str = "trimmed_mean",
        min_voxels_per_patch: int = 200,
        min_voxels_per_bin: int = 4,
        texture_model: str = "fourier",
        n_fourier: int = 3,
        texture_symmetric: bool = False,
        texture_ridge: float = 0.3,
        texture_min_count_frac: float = 0.15,
        texture_smoothness: float = 10.0,
        ring_templates: Optional[object] = None,
        center_offset: tuple[float, float] = (0.0, 0.0),
        snr_mask_threshold: Optional[float] = None,
    ) -> None:
        self.n_patches = n_patches
        self.overlap_frac = overlap_frac
        self.plane = plane
        self.q_step = q_step
        self.ring_width = ring_width
        self.baseline_smooth = baseline_smooth
        self.ring_smooth = ring_smooth
        self.profile_percentiles = profile_percentiles
        self.profile_method = profile_method
        self.min_voxels_per_patch = min_voxels_per_patch
        self.min_voxels_per_bin = min_voxels_per_bin
        self.texture_model = texture_model
        self.n_fourier = n_fourier
        self.texture_symmetric = texture_symmetric
        self.texture_ridge = texture_ridge
        self.texture_min_count_frac = texture_min_count_frac
        self.texture_smoothness = texture_smoothness
        self.ring_templates = ring_templates
        self.center_offset = center_offset
        self.snr_mask_threshold = snr_mask_threshold
        self._profiles: Optional[RadialRingProfiles] = None

    def _template_gaussians(self, q_grid: NDArray) -> list[NDArray]:
        """Unit-height Gaussians Gᵢ(|Q|) on *q_grid* from ``ring_templates``."""
        if not self.ring_templates:
            return []
        gauss = []
        for t in self.ring_templates:
            c = getattr(t, "q_center", None)
            s = getattr(t, "sigma", None)
            if c is None:                      # (center, sigma) tuple
                c, s = t
            gauss.append(np.exp(-0.5 * ((q_grid - float(c)) / float(s)) ** 2))
        return gauss

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        vol: HKLVolume,
        q_range: Optional[tuple[float, float]] = None,
    ) -> RadialRingProfiles:
        """Estimate per-patch ring profiles from *vol*."""
        q_mag = _offset_q_magnitude(vol, self.plane, self.center_offset)
        phi = _azimuthal_angle(vol, self.plane, self.center_offset)
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

        # Optional fixed radial templates Gᵢ(|Q|) (Gaussian, from a Bragg-free
        # linecut fit).  When given, each patch's ring profile is rebuilt as
        # Σ aᵢ·Gᵢ so the subtracted radial *shape* matches the measured ring
        # width exactly (no washed-out peak / baseline shoulder).
        templates = self._template_gaussians(q_grid)

        for p, pc in enumerate(patch_centers):
            d = _angular_distance(phiv_all, float(pc))
            in_p = np.abs(d) <= half_width
            if int(in_p.sum()) < self.min_voxels_per_patch:
                continue

            prof, cnt = _robust_radial_profile(
                qv_all[in_p], Iv_all[in_p], edges,
                self.profile_percentiles, self.min_voxels_per_bin,
                self.profile_method,
            )
            prof = _fill_nan_1d(prof)
            b = _estimate_baseline(
                prof, self.q_step, self.ring_width, self.baseline_smooth,
            )
            excess = np.maximum(0.0, prof - b)
            raw[p] = prof
            base[p] = b
            counts[p] = cnt
            ring[p] = _project_templates(excess, templates) if templates else excess
            if self.ring_smooth > 0:
                ring[p] = gaussian_filter1d(
                    ring[p], self.ring_smooth / self.q_step, mode="nearest"
                )

        # Per-|Q| low-order azimuthal Fourier texture, weighted by voxel count.
        texture_coeffs = np.array([])
        texture_values = np.array([])
        if self.texture_model == "fourier":
            texture_coeffs = _fit_azimuthal_texture(
                patch_centers, ring, counts, self.n_fourier,
                self.texture_symmetric, self.texture_ridge,
                self.texture_min_count_frac,
            )
        elif self.texture_model == "smooth":
            texture_values = _fit_smooth_texture(
                ring, counts, self.texture_smoothness, self.texture_min_count_frac,
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
            texture_values=texture_values,
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

        q_mag = _offset_q_magnitude(vol, self.plane, self.center_offset)
        phi = _azimuthal_angle(vol, self.plane, self.center_offset)
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

def _plane_components(
    vol: HKLVolume,
    plane: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    axes = {"hk0": (0, 1), "h0l": (0, 2), "0kl": (1, 2)}
    if plane not in axes:
        raise ValueError(f"Unknown plane: {plane!r}")
    i, j = axes[plane]

    H, K, L = vol.hkl_grid()
    Q = np.stack([H, K, L], axis=-1) @ vol.ub_matrix.T  # (..., 3) Å⁻¹

    # Orthonormal in-plane basis from the two reciprocal axis vectors
    # (Gram–Schmidt, so it is correct even if the axes are not orthogonal).
    a1 = vol.ub_matrix[:, i].astype(np.float64)
    a2 = vol.ub_matrix[:, j].astype(np.float64)
    e1 = a1 / (np.linalg.norm(a1) + 1e-12)
    a2_perp = a2 - (a2 @ e1) * e1
    e2 = a2_perp / (np.linalg.norm(a2_perp) + 1e-12)

    return Q, Q @ e1, Q @ e2


def _offset_q_magnitude(
    vol: HKLVolume,
    plane: str,
    center_offset: tuple[float, float] = (0.0, 0.0),
) -> NDArray[np.float64]:
    Q, x, y = _plane_components(vol, plane)
    cx, cy = center_offset
    if cx == 0.0 and cy == 0.0:
        return np.linalg.norm(Q, axis=-1)
    q2 = np.einsum("...i,...i->...", Q, Q)
    q2 = q2 - x * x - y * y + (x - cx) ** 2 + (y - cy) ** 2
    return np.sqrt(np.maximum(q2, 0.0))


def _azimuthal_angle(
    vol: HKLVolume,
    plane: str,
    center_offset: tuple[float, float] = (0.0, 0.0),
) -> NDArray[np.float64]:
    """Azimuthal angle φ (radians) for every voxel, within the given plane.

    The angle is measured in the plane spanned by the two in-plane reciprocal
    axes (e.g. b*, c* for ``'0kl'``), NOT from fixed lab-frame Q components.
    This matters because the UB matrix carries the crystal orientation: when the
    crystal is rotated, a reciprocal-lattice plane does not lie in a lab
    coordinate plane, so ``atan2`` of raw lab components collapses every voxel
    to ≈±90° and destroys the azimuth.  Projecting Q onto an orthonormal basis
    built from the in-plane reciprocal axes gives the correct angle for any
    orientation (and any lattice).
    """
    _, x, y = _plane_components(vol, plane)
    cx, cy = center_offset
    return np.arctan2(y - cy, x - cx)


def _angular_distance(phi: NDArray, phi_c: float) -> NDArray:
    """Signed angular distance in (-π, π], accounting for wrap-around."""
    d = phi - phi_c
    return (d + np.pi) % (2 * np.pi) - np.pi


def _project_templates(excess: NDArray, templates: list[NDArray]) -> NDArray[np.float64]:
    """Rebuild a radial ring profile as Σ aᵢ·Gᵢ from fixed Gaussian templates.

    The template amplitudes are fit jointly with non-negative least squares, so
    close/overlapping rings compete for the shared intensity instead of each
    independently claiming the same tail.  The result has exactly the
    linecut-measured radial shapes while preserving the local per-patch
    amplitudes.
    """
    if not templates:
        return np.zeros_like(excess)
    G = np.column_stack(templates)
    good = np.isfinite(excess) & np.isfinite(G).all(axis=1)
    if int(good.sum()) < len(templates):
        return np.zeros_like(excess)
    amps, _ = nnls(G[good], excess[good])
    return G @ amps


def _robust_radial_profile(
    q: NDArray,
    I: NDArray,
    edges: NDArray,
    percentiles: tuple[float, float],
    min_per_bin: int,
    method: str = "trimmed_mean",
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
            out[b] = _robust_bin_stat(sel, lo_p, hi_p, method)
        elif sel.size > 0:
            out[b] = float(np.median(sel))
    return out, counts


def _robust_bin_stat(
    vals: NDArray,
    lo_p: float,
    hi_p: float,
    method: str,
) -> float:
    vals = np.asarray(vals, dtype=np.float64)
    if method == "median":
        return float(np.median(vals))

    if method in {"trimmed_mean", "winsorized_mean"}:
        lo, hi = np.percentile(vals, (lo_p, hi_p))
        if method == "winsorized_mean":
            return float(np.clip(vals, lo, hi).mean())
        keep = vals[(vals >= lo) & (vals <= hi)]
        return float(keep.mean()) if keep.size else float(np.median(vals))

    if method == "huber":
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        if mad <= 0:
            return med
        scale = 1.4826 * mad
        clipped = np.clip(vals, med - 3.0 * scale, med + 3.0 * scale)
        return float(clipped.mean())

    raise ValueError(f"Unknown profile_method: {method!r}")


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


def _fit_smooth_texture(
    ring: NDArray,          # (P, Q) per-patch ring amplitude
    counts: NDArray,        # (P, Q) per-patch voxel counts
    smoothness: float,
    min_count_frac: float,
) -> NDArray[np.float64]:
    """Fit smooth nonnegative per-patch texture values with a minimizer.

    For each |Q| bin, solve

        Σ_p w_p (t_p - y_p)^2 + λ Σ_p (t_{p-1} - 2t_p + t_{p+1})^2

    with cyclic boundary conditions and ``t_p >= 0``.  This captures broad
    azimuthal texture without choosing a Fourier basis and without allowing
    high-frequency patch-to-patch structure.
    """
    P, n_q = ring.shape
    out = np.zeros_like(ring, dtype=np.float64)
    prev: Optional[NDArray[np.float64]] = None

    for b in range(n_q):
        w = counts[:, b].astype(np.float64).copy()
        y = ring[:, b].astype(np.float64)
        if w.max() <= 0 or not np.any(y > 0):
            prev = None
            continue

        w[w < min_count_frac * w.max()] = 0.0
        active = w > 0
        if not np.any(active):
            prev = None
            continue

        # Fill missing/under-sampled patches with a circular interpolation so
        # the smoothness penalty has a reasonable starting point everywhere.
        x0 = _fill_circular_patches(y, active)
        if prev is not None and prev.shape == x0.shape:
            x0 = 0.5 * x0 + 0.5 * prev
        x0 = np.maximum(x0, 0.0)

        wn = np.zeros_like(w)
        wn[active] = w[active] / w[active].mean()
        scale = float(np.mean(wn[active]))
        lam = max(0.0, smoothness) * max(scale, 1e-12)

        def objective(t: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            data_resid = t - y
            curv = np.roll(t, 1) - 2.0 * t + np.roll(t, -1)
            value = float(np.sum(wn * data_resid**2) + lam * np.sum(curv**2))

            grad = 2.0 * wn * data_resid
            # D2 is symmetric with cyclic boundaries, so grad of ||D2 t||² is
            # 2 D2(D2 t).
            grad += 2.0 * lam * (
                np.roll(curv, 1) - 2.0 * curv + np.roll(curv, -1)
            )
            return value, grad

        res = minimize(
            lambda t: objective(t),
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, None)] * P,
            options={"maxiter": 200, "ftol": 1e-9},
        )
        out[:, b] = np.maximum(0.0, res.x if res.success else x0)
        prev = out[:, b]

    return out


def _fill_circular_patches(y: NDArray, active: NDArray) -> NDArray[np.float64]:
    if active.all():
        return y.copy()
    P = y.size
    idx = np.arange(P)
    active_idx = idx[active]
    if active_idx.size == 0:
        return np.zeros_like(y, dtype=np.float64)
    if active_idx.size == 1:
        return np.full_like(y, float(y[active_idx[0]]), dtype=np.float64)

    xp = np.concatenate([active_idx - P, active_idx, active_idx + P])
    fp = np.concatenate([y[active], y[active], y[active]])
    return np.interp(idx, xp, fp)


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
