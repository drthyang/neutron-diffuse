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
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize, nnls

from ndiff.core import HKLVolume


class _RingTemplateLike(Protocol):
    q_center: float
    sigma: float


_RingTemplate = _RingTemplateLike | tuple[float, float]

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
        baseline peak-removal window: peaks narrower than this are treated as
        rings; broader structure is kept as diffuse baseline (default 0.24).
        Used directly when ``adaptive_ring_width=False``; when adaptive it is the
        fallback width for |Q| regions with no detected ring.
    adaptive_ring_width : bool
        If True (default), the baseline window is set **per ring** from the
        thickness of each detected ring instead of one global ``ring_width``.
        Real powder rings vary several-fold in width, so a single window either
        under-captures the broad rings (residual) or eats diffuse / bridges
        close ring pairs (over-subtraction).  The rings are detected in the
        azimuthally-pooled radial profile and each gets a window of
        ``ring_width_scale × FWHM``, capped to ``ring_width_cap_frac`` of the
        distance to its nearest neighbour ring.  Requires ``baseline_method=
        'snip'`` (the per-bin window is a SNIP feature).
    ring_width_scale : float
        Window = this multiple of each ring's measured FWHM (default 3.0 — wide
        enough to reach the diffuse baseline on both flanks of the peak).
    ring_width_cap_frac : float
        Cap each ring's window at this fraction of the distance to the nearest
        neighbouring ring (default 0.9), so the clip never bridges into an
        adjacent ring and over-subtracts the valley between them.
    baseline_method : {'snip', 'opening'}
        Algorithm used to estimate the smooth diffuse baseline under the rings.
        ``'snip'`` (default): Statistics-sensitive Non-linear Iterative
        Peak-clipping — iteratively clips peaks to the midpoint of their
        neighbors at increasing distances.  Unlike morphological opening, SNIP
        is **slope-aware**: it uses the average of left and right neighbors,
        not the minimum, so it correctly tracks a sloping background at the
        ring position and avoids the systematic over-subtraction that opening
        produces when the diffuse signal decreases with |Q|.  ``'opening'``:
        the original grey_opening (erosion → dilation) — kept for comparison.
    baseline_smooth : float
        σ (Å⁻¹) of the Gaussian applied to the baseline after the opening/SNIP
        step, to remove kinks (default 0.06).  Set 0 to disable.
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
        collected.  ``'median'`` is the default: the symmetric, unbiased robust
        centre.  ``'trimmed_mean'`` (drop values outside ``profile_percentiles``
        and average the rest) is **asymmetric** with the default 10–80 band — it
        trims 20% off the top to reject Bragg but only 10% off the bottom, so on
        a right-skewed cell it sits *below* the true ring level and
        under-subtracts the bright ring arcs; the median avoids that bias (Bragg
        is a small fraction of each cell so it cannot move the median).
        ``'winsorized_mean'`` clips values to the percentile interval before
        averaging.  ``'huber'`` clips values to median ± 3·MAD before averaging,
        a symmetric outlier rejection not tied to fixed percentile cutoffs.
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
        Number of azimuthal harmonics for the Fourier texture (default 8).  Real
        powder rings carry genuine multi-lobed azimuthal texture that a low order
        (≤3) cannot follow, leaving uneven over-/under-subtraction.  A higher
        order resolves it — but only safely in combination with ``texture_q_smooth``
        (pooling the texture across the ring's |Q| width), which suppresses the
        per-bin noise and sparse-azimuth ringing that an unpooled high-order fit
        would otherwise produce.  Still well below the angular scale of
        (point-like) Bragg peaks, so the trimmed-profile fit keeps them out of
        the texture.
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
        where the measured arcs are narrow (default 0.05 — low because the
        per-|Q| ringing is now controlled by ``texture_q_smooth`` rather than by
        flattening the harmonics; with the old unpooled fit a much larger ridge
        ~0.3 was needed and it suppressed real texture).
    texture_min_count_frac : float
        Per-|Q|, only patches sampled to at least this fraction of the
        best-sampled patch's count enter the texture fit (default 0.15).
        Under-sampled patches bias the ring amplitude low (too few voxels miss
        the radial peak); excluding them keeps the texture on the well-measured
        arcs so the ring is fully subtracted there.
    texture_q_smooth : float
        σ (Å⁻¹) for pooling the azimuthal texture *shape* across |Q| (default
        0.0 = disabled; > 0 enables).  Pooling assumes the ring's azimuthal
        pattern is identical at the peak and the wings, which holds only if the
        ring's radial WIDTH is azimuthally uniform.  On real data the width
        varies with φ (strongest at H≠0: the powder ring is broad at some
        azimuths, narrow at others), and pooling across |Q| then forces one
        shared pattern → it *homogenises* the width, under-subtracting the broad
        arcs and over-subtracting the narrow ones.  Disabled by default so each
        |Q| bin keeps its own azimuthal pattern (the low-order Fourier basis
        still smooths in φ), which captures the inhomogeneous width.  A small
        value (~0.02) is a useful compromise only when coverage is one-sided and
        high harmonics ring into unmeasured azimuths.  When enabled the per-|Q|
        texture fit is otherwise noisy (each bin sees only a thin radial slice of
        voxels); physically a ring's azimuthal texture comes from detector
        geometry / absorption and is coherent across the ring's narrow radial
        width, and this pools that information:
        each bin's coefficients are split into a radial amplitude ``A(q)`` (the
        constant term — kept sharp so the radial peak is not broadened) and a
        normalized texture shape ``t(q,·) = coeff(q,·)/A(q)``; the shape is
        smoothed along |Q| with an **amplitude-weighted** Gaussian (so it pools
        within a ring and tapers to nothing off-ring) and recombined as
        ``A(q)·t_smoothed(q,·)``.  This raises the texture SNR by ≈√(ring_width
        /q_step), letting a higher ``n_fourier`` resolve genuine azimuthal
        inhomogeneity without per-bin ringing.  Only used by
        ``texture_model='fourier'``.
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
    allowed_ring_centers, allowed_ring_halfwidths : array, optional
        Restrict ring subtraction to these confirmed |Q| shells (Å⁻¹).  When set,
        the per-patch ring excess is multiplied by a smooth [0, 1] envelope that
        is 1 within ``±halfwidth`` of a centre and tapers to 0 over the next
        half-width; excess outside every shell is dropped.  This is how a ring set
        confirmed *across the stack axis* (see
        :func:`confirm_ring_shells_across_h`) is enforced on each plane: it
        rejects single-plane phantom rings (Bragg peaks that fill a |Q| shell at
        several azimuths on integer-index planes) and makes the subtracted shells
        identical plane-to-plane, so the cleaned volume is continuous along the
        stack axis (no FFT-corrupting discontinuity for the ΔPDF).
        ``halfwidths`` defaults to ``ring_width`` when omitted.  ``None`` (both)
        disables the envelope — the default single-plane behaviour.
    allowed_ring_ceilings : array, optional
        Per-shell upper bound (intensity units, same length as
        ``allowed_ring_centers``) on the per-patch ring excess inside each shell.
        Catches what the |Q|-envelope cannot: a Bragg peak landing *on* a real
        ring inflates that ring's per-patch amplitude on one plane, spiking the
        subtraction (an over-subtraction trough *inside* a legitimate shell).
        Setting the ceiling from the across-H typical ring amplitude (see
        :func:`confirm_ring_shells_across_h`, × a margin) caps the spike back to
        the cross-plane norm — keeping the ring amplitude continuous in H while
        leaving normal planes (amplitude below the ceiling) untouched.  ``None``
        disables the cap.
    center_offset : (float, float)
        Experimental in-plane ring-center offset in Å⁻¹, in the same
        orthonormal plane frame used for φ. ``(0, 0)`` means rings are centered
        on Q=0.
    center_offset_h_slope : (float, float)
        Experimental H-dependent in-plane center drift in Å⁻¹ per H r.l.u.
        The effective center is ``center_offset + H * center_offset_h_slope``.
        This is intended for diagnosing whether nonzero-H 0kl slices have an
        apparent ring center that drifts with H.
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
        adaptive_ring_width: bool = True,
        ring_width_scale: float = 3.0,
        ring_width_cap_frac: float = 0.9,
        baseline_method: str = "snip",
        baseline_smooth: float = 0.06,
        ring_smooth: float = 0.0,
        profile_percentiles: tuple[float, float] = (10.0, 80.0),
        profile_method: str = "median",
        min_voxels_per_patch: int = 200,
        min_voxels_per_bin: int = 4,
        texture_model: str = "fourier",
        n_fourier: int = 8,
        texture_symmetric: bool = False,
        texture_ridge: float = 0.05,
        texture_min_count_frac: float = 0.15,
        texture_q_smooth: float = 0.0,
        texture_smoothness: float = 10.0,
        ring_templates: Sequence[_RingTemplate] | None = None,
        allowed_ring_centers: NDArray[np.float64] | None = None,
        allowed_ring_halfwidths: NDArray[np.float64] | None = None,
        allowed_ring_ceilings: NDArray[np.float64] | None = None,
        center_offset: tuple[float, float] = (0.0, 0.0),
        center_offset_h_slope: tuple[float, float] = (0.0, 0.0),
        snr_mask_threshold: float | None = None,
    ) -> None:
        self.n_patches = n_patches
        self.overlap_frac = overlap_frac
        self.plane = plane
        self.q_step = q_step
        self.ring_width = ring_width
        self.adaptive_ring_width = adaptive_ring_width
        self.ring_width_scale = ring_width_scale
        self.ring_width_cap_frac = ring_width_cap_frac
        self.baseline_method = baseline_method
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
        self.texture_q_smooth = texture_q_smooth
        self.texture_smoothness = texture_smoothness
        self.ring_templates = ring_templates
        self.allowed_ring_centers = (
            None if allowed_ring_centers is None
            else np.asarray(allowed_ring_centers, dtype=np.float64)
        )
        self.allowed_ring_halfwidths = (
            None if allowed_ring_halfwidths is None
            else np.asarray(allowed_ring_halfwidths, dtype=np.float64)
        )
        self.allowed_ring_ceilings = (
            None if allowed_ring_ceilings is None
            else np.asarray(allowed_ring_ceilings, dtype=np.float64)
        )
        self.center_offset = center_offset
        self.center_offset_h_slope = center_offset_h_slope
        self.snr_mask_threshold = snr_mask_threshold
        self._profiles: RadialRingProfiles | None = None

    def _template_gaussians(self, q_grid: NDArray) -> list[NDArray]:
        """Unit-height Gaussians Gᵢ(|Q|) on *q_grid* from ``ring_templates``."""
        if not self.ring_templates:
            return []
        gauss = []
        for t in self.ring_templates:
            if isinstance(t, tuple):
                c, s = t
            else:
                c = t.q_center
                s = t.sigma
            gauss.append(np.exp(-0.5 * ((q_grid - float(c)) / float(s)) ** 2))
        return gauss

    def _ring_q_envelope(self, q_grid: NDArray) -> NDArray[np.float64] | None:
        """A [0, 1] per-|Q|-bin weight that is 1 inside the confirmed ring shells
        and tapers to 0 between them, or ``None`` when no shells are configured.

        Each confirmed ring (centre cᵢ, half-width wᵢ) contributes a flat-topped,
        raised-cosine-tapered window: 1 for ``|q − cᵢ| ≤ wᵢ`` then a half-cosine
        roll-off to 0 over the next ``wᵢ``.  The taper (rather than a hard box)
        avoids stamping a new sharp |Q| edge into the subtracted profile, which
        would itself ring radially.  The envelope is the max over rings, so
        overlapping shells merge smoothly.
        """
        centers = self.allowed_ring_centers
        if centers is None or centers.size == 0:
            return None
        half = self.allowed_ring_halfwidths
        if half is None:
            half = np.full(centers.size, max(self.ring_width, 4.0 * self.q_step))
        env = np.zeros_like(q_grid, dtype=np.float64)
        for c, w in zip(centers, np.atleast_1d(half)):
            w = max(float(w), self.q_step)
            d = np.abs(q_grid - float(c))
            bump = np.where(
                d <= w, 1.0,
                np.where(d <= 2.0 * w, 0.5 * (1.0 + np.cos(np.pi * (d - w) / w)), 0.0),
            )
            env = np.maximum(env, bump)
        return env

    def _ring_q_ceiling(self, q_grid: NDArray) -> NDArray[np.float64] | None:
        """Per-|Q|-bin upper bound on the per-patch ring excess, or ``None``.

        Built from ``allowed_ring_centers`` + ``allowed_ring_ceilings``: within
        each confirmed shell (``±2·half-width``, matching the envelope support)
        the excess is capped to that shell's ceiling; elsewhere the cap is
        infinite (no effect — the envelope has already zeroed it).  This catches
        the case the |Q|-envelope cannot: a Bragg peak landing *on* a real ring
        inflates that ring's per-patch amplitude on one plane, so the subtraction
        spikes and over-subtracts.  A ceiling set from the across-H typical ring
        amplitude (× a margin) caps the spike back to the cross-plane norm,
        keeping the subtracted amplitude continuous in H while leaving the real
        ring (whose amplitude is below the ceiling on normal planes) untouched.
        """
        centers = self.allowed_ring_centers
        ceilings = self.allowed_ring_ceilings
        if centers is None or ceilings is None or centers.size == 0:
            return None
        half = self.allowed_ring_halfwidths
        if half is None:
            half = np.full(centers.size, max(self.ring_width, 4.0 * self.q_step))
        half = np.atleast_1d(half)
        out = np.full(q_grid.shape, np.inf, dtype=np.float64)
        for c, w, cap in zip(centers, half, np.atleast_1d(ceilings)):
            in_shell = np.abs(q_grid - float(c)) <= 2.0 * max(float(w), self.q_step)
            out[in_shell] = np.minimum(out[in_shell], float(cap))
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        vol: HKLVolume,
        q_range: tuple[float, float] | None = None,
    ) -> RadialRingProfiles:
        """Estimate per-patch ring profiles from *vol*."""
        q_mag = _offset_q_magnitude(
            vol, self.plane, self.center_offset, self.center_offset_h_slope
        )
        phi = _azimuthal_angle(
            vol, self.plane, self.center_offset, self.center_offset_h_slope
        )
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

        # Pass 1 — robust radial profile per patch (no baseline yet).
        filled = np.zeros(self.n_patches, dtype=bool)
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
            raw[p] = _fill_nan_1d(prof)
            counts[p] = cnt
            filled[p] = True

        # Adaptive per-|Q| baseline window: size each ring's window to its own
        # thickness (see :func:`_adaptive_ring_width_profile`).  Detect the rings
        # on the **cross-patch median** of the per-patch profiles — a ring sits
        # in (almost) every patch so it survives, while Bragg peaks (isolated in
        # azimuth, in only a patch or two) are rejected.  Falls back to the
        # scalar ``ring_width`` when disabled or unsupported.
        ring_width = self.ring_width
        if self.adaptive_ring_width and self.baseline_method == "snip" and filled.any():
            pooled = np.median(raw[filled], axis=0)
            pooled_cnt = np.median(counts[filled], axis=0)
            ring_width = _adaptive_ring_width_profile(
                q_grid, pooled, self.q_step, self.ring_width,
                self.ring_width_scale, self.ring_width_cap_frac, pooled_cnt,
            )
        self._ring_width_profile = ring_width

        # Optional |Q|-envelope restricting ring excess to confirmed shells.  On
        # a single 2D plane the per-patch profiles cannot tell a real powder ring
        # from symmetry-replicated Bragg peaks that happen to populate a |Q|
        # shell at several azimuths (integer-H planes) — the latter masquerade as
        # a ring and get a phantom subtraction.  A ring set confirmed *across H*
        # (see :func:`confirm_ring_shells_across_h`) rejects those phantoms and
        # makes the subtracted shells identical on every plane, so the cleaned
        # volume is continuous in H (no FFT-corrupting discontinuity for ΔPDF).
        ring_envelope = self._ring_q_envelope(q_grid)
        ring_ceiling = self._ring_q_ceiling(q_grid)

        # Pass 2 — baseline + ring excess per patch with the (adaptive) window.
        for p in np.nonzero(filled)[0]:
            prof = raw[p]
            b = _estimate_baseline(
                prof, self.q_step, ring_width, self.baseline_smooth,
                self.baseline_method,
            )
            excess = np.maximum(0.0, prof - b)
            base[p] = b
            ring[p] = _project_templates(excess, templates) if templates else excess
            if ring_envelope is not None:
                ring[p] = ring[p] * ring_envelope
            if ring_ceiling is not None:
                ring[p] = np.minimum(ring[p], ring_ceiling)
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
                self.texture_q_smooth / self.q_step,
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
        profiles: RadialRingProfiles | None = None,
    ) -> tuple[HKLVolume, NDArray[np.float64]]:
        """Subtract the fitted ring profiles from *vol*.

        Returns ``(vol_sub, I_ring)``.  ``vol_sub.data = vol.data − I_ring``;
        if ``snr_mask_threshold`` is set, voxels where the ring dominates are
        masked for the downstream backfill.
        """
        prof = profiles or self._profiles
        if prof is None:
            raise RuntimeError("Call fit() before subtract().")

        q_mag = _offset_q_magnitude(
            vol, self.plane, self.center_offset, self.center_offset_h_slope
        )
        phi = _azimuthal_angle(
            vol, self.plane, self.center_offset, self.center_offset_h_slope
        )
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
    def profiles(self) -> RadialRingProfiles | None:
        return self._profiles


def confirm_ring_shells_across_h(
    vol: HKLVolume,
    plane: str = "0kl",
    q_range: tuple[float, float] = (1.5, 10.5),
    q_step: float = 0.02,
    profile_percentiles: tuple[float, float] = (10.0, 80.0),
    profile_method: str = "median",
    min_voxels_per_bin: int = 8,
    ring_width: float = 0.24,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Detect the powder-ring |Q| shells that are present *across the stack axis*.

    A real powder ring is a sphere at constant 3D |Q|, so it sits at the same |Q|
    on **every** plane that geometrically samples that shell.  A Bragg-fed phantom
    (symmetry-replicated peaks filling a |Q| shell at several azimuths) appears on
    only a few integer-index planes.  This computes one **all-azimuth** robust
    radial profile per plane — Bragg-robust because each |Q| bin pools every
    azimuth, so the few Bragg voxels are a small fraction the median/trim rejects
    — then pools across planes (median over only the planes that sample each |Q|
    bin).  Real rings survive the across-plane median; phantoms wash out.

    Returns ``(centers, halfwidths, amplitudes)`` in Å⁻¹ (amplitudes in intensity
    units) — the confirmed ring centres, their FWHM, and the **across-H typical
    ring excess** at each centre (the pooled profile peak above its SNIP
    baseline).  The first two go to ``PatchedRadialRingModel(allowed_ring_centers=
    …, allowed_ring_halfwidths=…)``; the amplitude sets the cross-H scale a
    per-plane ``allowed_ring_ceilings`` can cap a Bragg-inflated plane back to.
    Empty arrays when no rings are confirmed.
    """
    stack_axis = {"0kl": 0, "h0l": 1, "hk0": 2}[plane]
    q_mag = _offset_q_magnitude(vol, plane)            # full 3D |Q|
    valid = vol.mask & np.isfinite(vol.data)

    q0, q1 = q_range
    edges = np.arange(q0, q1 + q_step, q_step)
    q_grid = 0.5 * (edges[:-1] + edges[1:])
    n_planes = vol.data.shape[stack_axis]
    n_q = q_grid.size

    prof_all = np.full((n_planes, n_q), np.nan)
    samp_all = np.zeros((n_planes, n_q))
    for ip in range(n_planes):
        vv = np.take(valid, ip, axis=stack_axis)
        if not vv.any():
            continue
        qm = np.take(q_mag, ip, axis=stack_axis)[vv]
        dv = np.take(vol.data, ip, axis=stack_axis)[vv]
        prof, cnt = _robust_radial_profile(
            qm, dv, edges, profile_percentiles, min_voxels_per_bin, profile_method,
        )
        prof_all[ip] = prof
        samp_all[ip] = cnt

    # Pool across planes per |Q| bin, using only planes that actually sample it.
    sampled = samp_all >= min_voxels_per_bin
    n_sampled = sampled.sum(axis=0).astype(np.float64)
    pooled = np.full(n_q, np.nan)
    for b in range(n_q):
        if n_sampled[b] > 0:
            pooled[b] = np.nanmedian(prof_all[sampled[:, b], b])

    centers, fwhm = _detect_rings(q_grid, pooled, q_step, ring_width, counts=n_sampled)
    if centers.size == 0:
        empty = np.array([])
        return empty, empty, empty

    # Cross-H typical ring excess at each centre: the pooled profile peak above a
    # SNIP baseline (the same excess _detect_rings keys on).  Sets the amplitude
    # scale a per-plane ceiling caps a Bragg-inflated plane back to.
    g = _fill_nan_1d(pooled)
    rough = _snip_baseline(g, max(3, int(round(ring_width / q_step))))
    exc = np.maximum(0.0, g - rough)
    amplitudes = np.interp(centers, q_grid, exc)
    return centers, fwhm, amplitudes


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
    center_offset_h_slope: tuple[float, float] = (0.0, 0.0),
) -> NDArray[np.float64]:
    Q, x, y = _plane_components(vol, plane)
    cx0, cy0 = center_offset
    sx, sy = center_offset_h_slope
    if cx0 == 0.0 and cy0 == 0.0 and sx == 0.0 and sy == 0.0:
        return np.linalg.norm(Q, axis=-1)
    H, _, _ = vol.hkl_grid()
    cx = cx0 + sx * H
    cy = cy0 + sy * H
    q2 = np.einsum("...i,...i->...", Q, Q)
    q2 = q2 - x * x - y * y + (x - cx) ** 2 + (y - cy) ** 2
    return np.sqrt(np.maximum(q2, 0.0))


def _azimuthal_angle(
    vol: HKLVolume,
    plane: str,
    center_offset: tuple[float, float] = (0.0, 0.0),
    center_offset_h_slope: tuple[float, float] = (0.0, 0.0),
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
    cx0, cy0 = center_offset
    sx, sy = center_offset_h_slope
    if sx == 0.0 and sy == 0.0:
        cx, cy = cx0, cy0
    else:
        H, _, _ = vol.hkl_grid()
        cx = cx0 + sx * H
        cy = cy0 + sy * H
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


def _smooth_texture_shape_along_q(
    coeffs: NDArray, sigma_bins: float
) -> NDArray[np.float64]:
    """Pool the azimuthal texture *shape* across |Q|, keeping the radial peak sharp.

    ``coeffs`` is (Q, M): column 0 is the radial amplitude ``A(q)`` (mean ring
    level), columns 1… are the azimuthal harmonics.  Splitting each row as
    ``A(q)·t(q,·)`` (with ``t(q,0)=1``) and smoothing only the shape ``t`` along
    |Q| — weighted by ``A(q)`` so the average pools within a ring and ignores
    off-ring bins — gives a texture that is coherent across the ring's narrow
    width without broadening the radial amplitude.
    """
    A = coeffs[:, 0]
    if sigma_bins <= 0 or coeffs.shape[1] < 2 or not np.any(A > 0):
        return coeffs
    w = np.maximum(A, 0.0)
    wsm = gaussian_filter1d(w, sigma_bins, mode="nearest")
    out = coeffs.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(A[:, None] > 0, coeffs / A[:, None], 0.0)  # t(q,·)
        for m in range(1, coeffs.shape[1]):
            num = gaussian_filter1d(w * ratio[:, m], sigma_bins, mode="nearest")
            t_sm = np.where(wsm > 0, num / wsm, 0.0)
            out[:, m] = A * t_sm        # recombine with the sharp amplitude
    return out


def _fit_azimuthal_texture(
    patch_centers: NDArray,
    ring: NDArray,          # (P, Q) per-patch ring amplitude
    counts: NDArray,        # (P, Q) per-patch voxel counts (weights)
    n_fourier: int,
    symmetric: bool,
    ridge: float,
    min_count_frac: float,
    q_smooth_bins: float = 0.0,
) -> NDArray[np.float64]:
    """Fit a low-order azimuthal Fourier series per |Q| bin (weighted ridge LS).

    Returns coefficients of shape (Q, M).  Bins with no measured ring amplitude
    yield all-zero coefficients (no ring → nothing to subtract).

    Only patches with at least ``min_count_frac`` of the best-sampled patch's
    count at that |Q| enter the fit: under-sampled patches bias the ring
    amplitude *low* (too few voxels miss the radial peak), so including them
    would drag the texture below the well-measured arcs and leave a residual
    ring.

    When ``q_smooth_bins > 0`` the fitted texture *shape* is pooled across |Q|
    (see :func:`_smooth_texture_shape_along_q`) so a higher ``n_fourier`` can
    resolve real azimuthal structure without per-bin ringing.
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
    return _smooth_texture_shape_along_q(coeffs, q_smooth_bins)


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
    prev: NDArray[np.float64] | None = None

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


def _detect_rings(
    q_grid: NDArray,
    pooled: NDArray,
    q_step: float,
    base_width: float,
    counts: NDArray | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Detect powder rings in a pooled radial profile.

    Returns ``(centers_q, fwhm_q)`` — the |Q| centre and FWHM (both Å⁻¹) of each
    detected ring, sorted by |Q|.  A ring is a *narrow* positive peak (FWHM ≤
    ``base_width``) above the SNIP baseline; broad bumps are diffuse structure
    and are rejected.  Empty arrays are returned when no rings are found.
    """
    from scipy.signal import find_peaks, peak_widths

    empty = (np.array([]), np.array([]))
    g = _fill_nan_1d(pooled)
    # Drop sparsely-sampled |Q| bins (e.g. a thin shell that, on a coarse grid,
    # contains only Bragg voxels) before detection — interpolate across them so
    # they can't masquerade as a giant narrow "ring".
    if counts is not None and np.any(counts > 0):
        sparse = counts < 0.1 * np.median(counts[counts > 0])
        if np.any(sparse) and not np.all(sparse):
            idx = np.arange(g.size)
            g = g.copy()
            g[sparse] = np.interp(idx[sparse], idx[~sparse], g[~sparse])
    rough = _snip_baseline(g, max(3, int(round(base_width / q_step))))
    exc = np.maximum(0.0, g - rough)
    if not np.any(exc > 0):
        return empty
    prom = 0.06 * float(np.max(exc))
    min_sep = max(1, int(round(0.05 / q_step)))
    peaks, _ = find_peaks(exc, prominence=max(prom, 1e-9), distance=min_sep)
    if peaks.size == 0:
        return empty

    fwhm_bins = peak_widths(exc, peaks, rel_height=0.5)[0]
    # Keep only genuine *rings* — narrow peaks.  Broad bumps (≳ the fallback
    # window) are diffuse structure, not rings.
    narrow = fwhm_bins * q_step <= base_width
    peaks, fwhm_bins = peaks[narrow], fwhm_bins[narrow]
    if peaks.size == 0:
        return empty
    fwhm_q = np.maximum(fwhm_bins * q_step, q_step)
    centers = q_grid[peaks]
    return centers, fwhm_q


def _adaptive_ring_width_profile(
    q_grid: NDArray,
    pooled: NDArray,
    q_step: float,
    base_width: float,
    scale: float,
    cap_frac: float,
    counts: NDArray | None = None,
) -> NDArray[np.float64]:
    """Per-|Q| baseline window matched to each ring's own thickness.

    A single global ``ring_width`` cannot fit rings whose widths vary several-
    fold: too narrow under-captures broad rings (leaves residual), too wide eats
    diffuse and bridges close ring pairs (over-subtracts the valley between
    them).  This detects the rings in the azimuthally-pooled radial profile,
    measures each one's FWHM, and returns a window of ``scale × FWHM`` around it
    — capped to ``cap_frac`` of the distance to the nearest neighbouring ring so
    the clip never reaches into an adjacent ring.  Each |Q| bin takes the window
    of its nearest detected ring; bins far from any ring keep ``base_width`` (it
    does not matter — there is no peak to clip there).

    Returns an array of widths (Å⁻¹) the same length as ``q_grid``.  Falls back
    to a constant ``base_width`` array when no rings are detected.
    """
    out = np.full(q_grid.size, float(base_width))
    centers, fwhm_q = _detect_rings(q_grid, pooled, q_step, base_width, counts)
    if centers.size == 0:
        return out

    # Cap each ring's window so it cannot bridge to its nearest neighbour ring.
    win = scale * fwhm_q
    if centers.size > 1:
        nbr = np.full(centers.size, np.inf)
        nbr[:-1] = np.minimum(nbr[:-1], np.diff(centers))
        nbr[1:] = np.minimum(nbr[1:], np.diff(centers))
        win = np.minimum(win, cap_frac * nbr)
    # Floor: never narrower than enough to capture the ring itself (~1.5·FWHM)
    # nor below half the fallback width — a spurious nearby detection must not
    # starve a real ring's window.
    floor = np.maximum(1.5 * fwhm_q, 0.5 * base_width)
    win = np.maximum(win, np.maximum(floor, 4.0 * q_step))

    nearest = np.argmin(np.abs(q_grid[:, None] - centers[None, :]), axis=1)
    return win[nearest]


def _snip_baseline(prof: NDArray, n_iter: int | NDArray) -> NDArray[np.float64]:
    """SNIP: Statistics-sensitive Non-linear Iterative Peak-clipping.

    For i = 1 … n_iter, each INTERIOR bin b (i ≤ b < n−i) is clipped to the
    midpoint of base[b−i] and base[b+i].  Edge bins are never modified.

    Using the *average* (not the minimum as in morphological opening) makes the
    algorithm slope-aware: for a purely linear background the midpoint equals
    the current bin exactly, so no clipping occurs.  For a ring on a slope the
    baseline correctly interpolates the diffuse level at the ring position
    without the systematic downward bias of morphological opening.

    ``n_iter`` may be a scalar (uniform peak-removal half-width of ``n_iter``
    bins everywhere) or a per-bin integer array — bin ``b`` then stops being
    clipped once the iteration index exceeds ``n_iter[b]``, giving a |Q|-varying
    window so each ring is clipped over a width matched to its own thickness
    (narrow rings don't reach into neighbouring diffuse; broad rings are fully
    captured).
    """
    n = len(prof)
    base = prof.astype(np.float64).copy()
    w = np.asarray(n_iter)
    if w.ndim == 0:
        per_bin = None
        max_iter = int(w)
    else:
        per_bin = np.asarray(w, dtype=int)
        max_iter = int(per_bin.max()) if per_bin.size else 0
    for i in range(1, max_iter + 1):
        if 2 * i >= n:
            break
        mid = 0.5 * (base[: n - 2 * i] + base[2 * i :])   # interior bins i…n-i-1
        clipped = np.minimum(base[i : n - i], mid)
        if per_bin is None:
            base[i : n - i] = clipped
        else:
            active = per_bin[i : n - i] >= i              # this bin still clipping
            base[i : n - i] = np.where(active, clipped, base[i : n - i])
    return base


def _estimate_baseline(
    prof: NDArray,
    q_step: float,
    ring_width: float | NDArray,
    smooth: float,
    method: str = "snip",
) -> NDArray[np.float64]:
    """Smooth diffuse baseline under the rings.

    Parameters
    ----------
    ring_width : float or array
        Peak-removal width (Å⁻¹).  A scalar applies one width everywhere; a
        per-|Q|-bin array gives an adaptive window matched to each ring's
        thickness (only honoured by ``method='snip'``).
    method : {'snip', 'opening'}
        ``'snip'`` (default): SNIP peak-clipping — slope-aware, avoids the
        systematic over-subtraction of morphological opening on sloping
        backgrounds, and supports a per-bin adaptive window.  ``'opening'``:
        original grey_opening (erosion → dilation), scalar width only.
    """
    if method == "snip":
        if np.ndim(ring_width) == 0:
            n_iter = max(3, int(round(ring_width / (2.0 * q_step))))
        else:
            n_iter = np.maximum(
                3, np.round(np.asarray(ring_width) / (2.0 * q_step)).astype(int)
            )
        base = _snip_baseline(prof, n_iter)
    else:
        from scipy.ndimage import grey_opening
        width = float(np.mean(ring_width)) if np.ndim(ring_width) else float(ring_width)
        size = max(3, int(round(width / q_step)))
        if size % 2 == 0:
            size += 1
        base = grey_opening(prof, size=size, mode="nearest")
    if smooth > 0:
        base = gaussian_filter1d(base, smooth / q_step, mode="nearest")
    return np.minimum(base, prof)
