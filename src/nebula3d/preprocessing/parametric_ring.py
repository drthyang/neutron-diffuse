# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tsung-han Yang

"""Parametric powder-ring model: separable per-ring pseudo-Voigt × azimuthal texture.

Model
-----
    I_ring(|Q|, φ) = Σᵢ Tᵢ(φ) · PVᵢ(|Q|)

where ``PVᵢ`` is a **unit-peak pseudo-Voigt** in |Q| (a Gaussian instrumental
profile mixed with a Lorentzian size/strain tail — the physically correct
powder-line shape) and ``Tᵢ(φ)`` is ring *i*'s own non-negative Fourier
azimuthal amplitude.  ``Tᵢ`` carries the amplitude, so ``PVᵢ`` is normalised to
unit peak height.

Why this exists (vs ``PatchedRadialRingModel``)
-----------------------------------------------
The non-parametric :class:`~nebula3d.preprocessing.radial_background.PatchedRadialRingModel`
builds a 2-D **(azimuthal-patch × |Q|-bin)** grid.  The number of voxels behind
each cell scales with the arc length ``∝ |Q|``, so the low-|Q| patches are
starved (few voxels, noisy texture — and patches below a voxel floor are dropped
entirely) while the high-|Q| patches are angularly coarse.  The statistics are
*|Q|-dependent*, which is the wrong place to spend the data.

This model never bins the 2-D grid.  It is **separable per ring**:

1. ``PVᵢ(|Q|)`` is fit from **thin radial shells pooled over all φ** — every
   azimuthal voxel at a radius contributes to the radial shape (maximum radial
   SNR), and a true pseudo-Voigt is fit rather than a shell-averaged amplitude.
2. ``Tᵢ(φ)`` is fit **binning-free** from each ring's shell voxels by a robust,
   template-weighted ridge least-squares directly on the scattered
   ``(φ, intensity)`` samples — no azimuthal patches at all, so no low-|Q|
   starvation.

Per-ring ``Tᵢ`` (rather than one shared ``T(φ)``) keeps the model robust to
rings whose azimuthal texture differs — the documented failure mode of the
older shared-texture :class:`~nebula3d.preprocessing.ring_model.PatchedRingModel`.

This is a drop-in alternative to ``PatchedRadialRingModel`` for stage 2 of the
pipeline; the two expose the same ``fit`` / ``subtract`` interface and the same
``min_voxels_per_patch`` driver guard, and both honour the across-stack
confirmed ring shells (``allowed_ring_centers`` / ``…_halfwidths`` /
``…_ceilings``) so the per-slice 3-D driver stays continuous along the stack
axis.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from nebula3d.core import HKLVolume
from nebula3d.preprocessing.radial_background import (
    _adaptive_ring_width_profile,
    _azimuthal_angle,
    _azimuthal_basis,
    _detect_rings,
    _estimate_baseline,
    _fill_nan_1d,
    _offset_q_magnitude,
    _robust_radial_profile,
)

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


# ---------------------------------------------------------------------------
# Radial line shape
# ---------------------------------------------------------------------------
def _pseudo_voigt(
    q: NDArray[np.float64], q0: float, fwhm: float, eta: float
) -> NDArray[np.float64]:
    """Unit-peak pseudo-Voigt: ``η·Lorentzian + (1−η)·Gaussian``.

    Both components share the same FWHM and are normalised to unit *peak height*
    (not unit area), so the mixing parameter ``η`` does not rescale the peak and
    the amplitude lives entirely in the azimuthal texture ``Tᵢ(φ)``.
    """
    fwhm = max(float(fwhm), 1e-6)
    eta = float(np.clip(eta, 0.0, 1.0))
    dq = (np.asarray(q, dtype=np.float64) - float(q0))
    sigma = fwhm * _FWHM_TO_SIGMA
    gamma = 0.5 * fwhm
    gauss = np.exp(-0.5 * (dq / sigma) ** 2)
    lorentz = 1.0 / (1.0 + (dq / gamma) ** 2)
    return eta * lorentz + (1.0 - eta) * gauss


def _pseudo_voigt_phi(
    q: NDArray[np.float64],
    q0: NDArray[np.float64],
    fwhm: NDArray[np.float64],
    eta: float,
) -> NDArray[np.float64]:
    """Unit-peak pseudo-Voigt with **per-voxel** centre ``q0`` and width ``fwhm``
    arrays (``eta`` scalar) — the non-separable ``PV(|Q|; q0(φ), fwhm(φ))``."""
    q = np.asarray(q, dtype=np.float64)
    q0 = np.asarray(q0, dtype=np.float64)
    fwhm = np.maximum(np.asarray(fwhm, dtype=np.float64), 1e-6)
    eta = float(np.clip(eta, 0.0, 1.0))
    dq = q - q0
    sigma = fwhm * _FWHM_TO_SIGMA
    gamma = 0.5 * fwhm
    gauss = np.exp(-0.5 * (dq / sigma) ** 2)
    lorentz = 1.0 / (1.0 + (dq / gamma) ** 2)
    return eta * lorentz + (1.0 - eta) * gauss


# ---------------------------------------------------------------------------
# Fitted result
# ---------------------------------------------------------------------------
@dataclass
class ParametricRing:
    """One fitted ring: a pseudo-Voigt radial shape + its azimuthal texture.

    Attributes
    ----------
    q_center : float   Ring |Q| (Å⁻¹).
    fwhm     : float   Pseudo-Voigt full width at half maximum (Å⁻¹).
    eta      : float   Lorentzian fraction (0 = pure Gaussian, 1 = pure Lorentzian).
    texture_coeffs : (M,) Fourier coefficients of the non-negative azimuthal
        amplitude ``Tᵢ(φ)`` — this carries the ring's amplitude, so the radial
        shape is unit-peak.  ``M = 1 + n_fourier`` (symmetric) or ``1 + 2·n_fourier``.
    q0_coeffs, fwhm_coeffs : (1 + 2·radial_n_fourier,) or None
        Full-Fourier coefficients of the **azimuth-dependent** centre ``q0(φ)``
        and width ``fwhm(φ)`` — the *non-separable* extension, set only when the
        adaptive fit accepted them for this ring (else ``None`` → the ring is
        separable, using the scalar ``q_center`` / ``fwhm``).  Evaluated against
        the model's ``radial_n_fourier`` basis.
    """

    q_center: float
    fwhm: float
    eta: float
    texture_coeffs: NDArray[np.float64]
    q0_coeffs: NDArray[np.float64] | None = None
    fwhm_coeffs: NDArray[np.float64] | None = None


@dataclass
class FittedParametricRingModel:
    """Result of :meth:`ParametricRingModel.fit` — evaluable ring contribution.

    Two radial representations share one azimuthal idea (a per-shell Fourier
    texture); ``mode`` selects which the :meth:`evaluate` uses.

    Attributes
    ----------
    plane : str        Reference plane defining φ.
    n_fourier : int    Azimuthal harmonics in the texture.
    symmetric : bool   Even-cosine basis (mmm) vs full Fourier series.
    mode : {'peaks', 'rolling'}
        ``'peaks'`` — discrete ``rings`` (a pseudo-Voigt × per-ring texture each).
        ``'rolling'`` — a continuous ``Ring(|Q|)·T(|Q|,φ)`` carried by
        ``roll_centers`` + ``roll_coeffs`` (column 0 is the radial amplitude
        ``A(|Q|)``; columns 1… are the azimuthal harmonics at that shell).
    rings : list[ParametricRing]    (peaks mode)
    roll_centers : (R,)             rolling-window |Q| centres (Å⁻¹)
    roll_coeffs : (R, M)            per-shell azimuthal Fourier coefficients
    q_grid, pooled_profile, baseline : 1-D diagnostics of the radial fit.
    ceilings : (n_rings,) or None   (peaks mode) per-ring upper bound on Tᵢ(φ).
    """

    plane: str
    n_fourier: int
    symmetric: bool
    mode: str = "peaks"
    rings: list[ParametricRing] = field(default_factory=list)
    roll_centers: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    roll_coeffs: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    q_grid: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    pooled_profile: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    baseline: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    ceilings: NDArray[np.float64] | None = None
    radial_n_fourier: int = 0

    def ring_shape(
        self, i: int, phi: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Per-voxel ``(q0(φ), fwhm(φ))`` for peak *i*.

        Constant ``q_center`` / ``fwhm`` when the ring is separable; the accepted
        Fourier series (clipped to a sane band around the centre values) when the
        non-separable radial harmonics were fit.
        """
        r = self.rings[i]
        shp = np.asarray(phi).shape
        if r.q0_coeffs is None and r.fwhm_coeffs is None:
            return np.full(shp, r.q_center), np.full(shp, r.fwhm)
        flat = np.asarray(phi, dtype=np.float64).ravel()
        basis = _azimuthal_basis(flat, self.radial_n_fourier, False)
        if r.q0_coeffs is not None:
            q0 = np.clip(basis @ r.q0_coeffs,
                         r.q_center - 2.0 * r.fwhm, r.q_center + 2.0 * r.fwhm)
        else:
            q0 = np.full(flat.size, r.q_center)
        if r.fwhm_coeffs is not None:
            fwhm = np.clip(basis @ r.fwhm_coeffs, 0.3 * r.fwhm, 3.0 * r.fwhm)
        else:
            fwhm = np.full(flat.size, r.fwhm)
        return q0.reshape(shp), fwhm.reshape(shp)

    def ring_texture(self, i: int, phi: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate peak *i*'s non-negative azimuthal amplitude ``Tᵢ(φ)``."""
        flat = np.asarray(phi, dtype=np.float64).ravel()
        basis = _azimuthal_basis(flat, self.n_fourier, self.symmetric)
        t = basis @ self.rings[i].texture_coeffs
        if self.ceilings is not None and i < self.ceilings.size:
            t = np.minimum(t, float(self.ceilings[i]))
        return np.maximum(0.0, t).reshape(np.asarray(phi).shape)

    def radial_amplitude(self) -> NDArray[np.float64]:
        """The azimuthally-averaged ring level A(|Q|) on ``roll_centers``
        (rolling mode) — the continuous ``Ring(|Q|)``."""
        if self.mode != "rolling" or self.roll_coeffs.size == 0:
            return np.array([])
        return self.roll_coeffs[:, 0].copy()

    def evaluate(
        self, q_mag: NDArray[np.float64], phi: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Ring contribution at every voxel (dispatch on ``mode``)."""
        if self.mode == "rolling":
            return self._evaluate_rolling(q_mag, phi)
        out = np.zeros_like(np.asarray(q_mag, dtype=np.float64))
        for i, r in enumerate(self.rings):
            if r.q0_coeffs is None and r.fwhm_coeffs is None:
                shape = _pseudo_voigt(q_mag, r.q_center, r.fwhm, r.eta)
            else:
                q0, fwhm = self.ring_shape(i, phi)
                shape = _pseudo_voigt_phi(q_mag, q0, fwhm, r.eta)
            out += self.ring_texture(i, phi) * shape
        return np.maximum(out, 0.0)

    def _evaluate_rolling(
        self, q_mag: NDArray[np.float64], phi: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Continuous ``Ring(|Q|)·T(|Q|,φ)``: interpolate each Fourier
        coefficient along |Q| (zero outside the swept range), evaluate the
        azimuthal basis at φ, and clamp to non-negative."""
        shape_in = np.asarray(q_mag).shape
        if self.roll_coeffs.size == 0:
            return np.zeros(shape_in, dtype=np.float64)
        flat_q = np.asarray(q_mag, dtype=np.float64).ravel()
        flat_phi = np.asarray(phi, dtype=np.float64).ravel()
        basis = _azimuthal_basis(flat_phi, self.n_fourier, self.symmetric)  # (N, M)
        coeff_at = np.empty((flat_q.size, self.roll_coeffs.shape[1]))
        for m in range(self.roll_coeffs.shape[1]):
            coeff_at[:, m] = np.interp(
                flat_q, self.roll_centers, self.roll_coeffs[:, m], left=0.0, right=0.0)
        out = np.einsum("nm,nm->n", basis, coeff_at)
        return np.maximum(out, 0.0).reshape(shape_in)


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------
class ParametricRingModel:
    """Fit and subtract a separable pseudo-Voigt × per-ring-texture ring model.

    Parameters
    ----------
    plane : str
        Reference plane for φ and |Q|: ``'hk0'`` / ``'h0l'`` / ``'0kl'``.
    q_step : float
        Radial bin width (Å⁻¹) of the pooled profile used to detect rings and
        fit the radial shapes (default 0.02).
    ring_width : float
        Maximum full width (Å⁻¹) for a feature to count as a ring during
        detection / the SNIP baseline window (default 0.24).  Broader bumps are
        treated as diffuse and kept.
    baseline_method : {'snip', 'opening'}
        Diffuse-baseline estimator under the rings (default ``'snip'``).
    baseline_smooth : float
        σ (Å⁻¹) of the Gaussian applied to the baseline (default 0.06).
    profile_method : {'median', 'trimmed_mean', 'winsorized_mean', 'huber'}
        Robust per-|Q|-bin statistic for the pooled radial profile (default
        ``'median'`` — Bragg is a small fraction of each shell so it cannot move
        the median).
    profile_percentiles : (float, float)
        Low/high band for the trimming-based statistics (default 10–80).
    eta0 : float
        Initial pseudo-Voigt Lorentzian fraction for the radial fit (default 0.5).
    n_fourier : int
        Azimuthal harmonics for each ring's texture ``Tᵢ(φ)`` (default 8).
    symmetric : bool
        Restrict the texture to the even-cosine basis {1, cos2φ, cos4φ, …}
        (orthorhombic *mmm* in the plane).  Default False — a full Fourier
        series, no symmetry assumption.
    texture_ridge : float
        Dimensionless smoothness prior on the texture harmonics (∝ order²),
        scaled to the data term (default 0.05).
    texture_shell_scale : float
        Half-width of the |Q| shell collected for each ring's texture fit, in
        multiples of that ring's FWHM (default 1.5).
    texture_min_template : float
        Only shell voxels whose unit-peak template value is at least this enter
        the texture fit (default 0.2) — near-peak voxels carry the texture; the
        far wings are noise-dominated.
    texture_irls_iter : int
        Robust iteratively-reweighted iterations in the texture fit (default 3).
    texture_spike_reject : bool
        How the IRLS separates Bragg from the ring's azimuthal texture (default
        True).  **True** — φ-shape-aware: down-weight only *azimuthally-narrow*
        positive excursions (a sample far above its φ-neighbourhood = a Bragg
        spike) and otherwise iterate a *symmetric* robust reweight, so broad
        bright arcs (real texture) keep full weight and are captured.  **False**
        — legacy: reject every positive residual on the high side, which cannot
        tell a bright arc from a Bragg spike and so under-subtracts textured
        rings (the documented ~⅔-amplitude failure).
    min_ring_snr : float
        Detected rings whose peak excess is below this multiple of the pooled
        profile's robust noise are rejected as spurious (default 5.0).  The
        underlying detector uses a *relative* prominence, so on a ring-free
        profile it would otherwise pick noise spikes; this absolute floor leaves
        a flat profile with zero rings.  Bypassed when ``allowed_ring_centers``
        are supplied (those shells are already confirmed across the stack axis).
    min_voxels_per_patch : int
        Plane-level voxel floor used by the per-slice driver guard (kept under
        the ``PatchedRadialRingModel`` name so the two are interchangeable).
    min_voxels_per_bin : int
        |Q| bins with fewer voxels fall back to the bin median.
    allowed_ring_centers, allowed_ring_halfwidths : array, optional
        Restrict ring fitting to these confirmed |Q| shells (Å⁻¹) — the rings
        confirmed across the stack axis by
        :func:`~nebula3d.preprocessing.radial_background.confirm_ring_shells_across_h`.
        When given, detection is skipped and a ring is fit at each centre with an
        initial FWHM of ``2·halfwidth``.
    allowed_ring_ceilings : array, optional
        Per-shell upper bound on the azimuthal amplitude ``Tᵢ(φ)`` (same length
        and order as ``allowed_ring_centers``), capping a Bragg-inflated arc back
        to the cross-plane norm.  ``None`` disables the cap.
    center_offset, center_offset_h_slope : (float, float)
        Optional in-plane ring-centre offset (and H-slope), in the φ-plane frame.
    snr_mask_threshold : float or None
        After subtraction, mask voxels where ``I_ring / σ_data`` exceeds this
        (flagging them for backfill).  ``None`` = subtract only.
    radial_mode : {'rolling', 'peaks'}
        ``'rolling'`` (default) — sweep a thick overlapping |Q| window from Qmin
        to Qmax and fit one binning-free azimuthal Fourier texture per shell,
        giving a *continuous* ``Ring(|Q|)·T(|Q|,φ)`` with no discrete-peak
        detection (broad/overlapping rings handled uniformly; thicker
        ``ring_width`` = smoother).  ``'peaks'`` — detect discrete rings and fit a
        pseudo-Voigt × per-ring texture each.
    roll_step : float
        Spacing (Å⁻¹) of the rolling-window centres (default 0.04).  The window
        half-width is ``ring_width``; centres overlap so the texture varies
        smoothly along |Q|.
    radial_harmonics : bool
        **Peaks mode only.**  When True, also fit the azimuth-dependent radial
        *shape* of each ring — ``q0(φ)`` and ``fwhm(φ)`` as low-order Fourier
        series — giving the non-separable ``I_ring = Tᵢ(φ)·PV(|Q|; q0ᵢ(φ),
        fwhmᵢ(φ))``.  This is **adaptive per ring**: the harmonics are accepted
        only where they actually reduce that ring's on-shell residual
        inhomogeneity AND the azimuthal sampling is adequate; otherwise the ring
        falls back to a constant ``q0``/``fwhm`` (separable).  Default False —
        the separable model is the validated default; this is opt-in pending A/B.
        Diagnostics show non-separability helps the well-sampled outer rings but
        over-fits the innermost / sparsest shells, hence the adaptive guard.
    radial_n_fourier : int
        Harmonics for ``q0(φ)``/``fwhm(φ)`` (default 2 — a centre offset is the
        1st harmonic; keep low, the radial shape varies slowly with φ).
    radial_harmonic_sectors : int
        Azimuthal sectors used to estimate the per-φ radial shape (default 16).
    radial_harmonic_min_voxels_per_sector : int
        A ring needs at least this many shell voxels in every sector (on average)
        to even attempt the radial harmonics (default 40) — the sparse-sampling
        guard that keeps the innermost/outermost shells separable.
    radial_harmonic_accept_margin : float
        Accept the harmonics only if they cut the on-shell residual
        inhomogeneity by at least this fraction vs the constant shape
        (default 0.05).
    """

    def __init__(
        self,
        plane: str = "0kl",
        q_step: float = 0.02,
        ring_width: float = 0.24,
        baseline_method: str = "snip",
        baseline_smooth: float = 0.06,
        profile_method: str = "median",
        profile_percentiles: tuple[float, float] = (10.0, 80.0),
        eta0: float = 0.5,
        n_fourier: int = 8,
        symmetric: bool = False,
        texture_ridge: float = 0.05,
        texture_shell_scale: float = 1.5,
        texture_min_template: float = 0.2,
        texture_irls_iter: int = 3,
        texture_spike_reject: bool = True,
        min_ring_snr: float = 5.0,
        min_voxels_per_patch: int = 200,
        min_voxels_per_bin: int = 4,
        allowed_ring_centers: NDArray[np.float64] | None = None,
        allowed_ring_halfwidths: NDArray[np.float64] | None = None,
        allowed_ring_ceilings: NDArray[np.float64] | None = None,
        center_offset: tuple[float, float] = (0.0, 0.0),
        center_offset_h_slope: tuple[float, float] = (0.0, 0.0),
        snr_mask_threshold: float | None = None,
        radial_mode: str = "rolling",
        roll_step: float = 0.04,
        radial_harmonics: bool = False,
        radial_n_fourier: int = 2,
        radial_harmonic_sectors: int = 16,
        radial_harmonic_min_voxels_per_sector: int = 40,
        radial_harmonic_accept_margin: float = 0.05,
    ) -> None:
        self.plane = plane
        self.q_step = q_step
        self.ring_width = ring_width
        self.radial_mode = radial_mode
        self.roll_step = roll_step
        self.baseline_method = baseline_method
        self.baseline_smooth = baseline_smooth
        self.profile_method = profile_method
        self.profile_percentiles = profile_percentiles
        self.eta0 = eta0
        self.n_fourier = n_fourier
        self.symmetric = symmetric
        self.texture_ridge = texture_ridge
        self.texture_shell_scale = texture_shell_scale
        self.texture_min_template = texture_min_template
        self.texture_irls_iter = texture_irls_iter
        self.texture_spike_reject = texture_spike_reject
        self.radial_harmonics = radial_harmonics
        self.radial_n_fourier = radial_n_fourier
        self.radial_harmonic_sectors = radial_harmonic_sectors
        self.radial_harmonic_min_voxels_per_sector = radial_harmonic_min_voxels_per_sector
        self.radial_harmonic_accept_margin = radial_harmonic_accept_margin
        self.min_ring_snr = min_ring_snr
        self.min_voxels_per_patch = min_voxels_per_patch
        self.min_voxels_per_bin = min_voxels_per_bin
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
        self._model: FittedParametricRingModel | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _plane_coords(
        self,
        vol: HKLVolume,
        q_mag: NDArray[np.float64] | None,
        phi: NDArray[np.float64] | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Per-voxel |Q| and azimuth for this model's plane, computing only the
        ones the caller did not already supply (identical values either way)."""
        if q_mag is None:
            q_mag = _offset_q_magnitude(
                vol, self.plane, self.center_offset, self.center_offset_h_slope)
        if phi is None:
            phi = _azimuthal_angle(
                vol, self.plane, self.center_offset, self.center_offset_h_slope)
        return q_mag, phi

    def fit(
        self,
        vol: HKLVolume,
        q_range: tuple[float, float] | None = None,
        bragg_keep_mask: NDArray[np.bool_] | None = None,
        q_mag: NDArray[np.float64] | None = None,
        phi: NDArray[np.float64] | None = None,
    ) -> FittedParametricRingModel:
        """Fit the separable ring model to *vol*.

        Parameters
        ----------
        bragg_keep_mask : array of bool, optional
            A keep-mask (``True`` = keep, ``False`` = Bragg) the same shape as
            ``vol.data``, e.g. from
            :meth:`~nebula3d.analysis.BraggRemover.build_mask`.  Bragg voxels are
            excluded **from the fit only** (``subtract`` still models the ring at
            every voxel).  Removing the sharp single-crystal reflections that sit
            *on* the powder ring lets the texture fit use a gentle IRLS
            (``texture_irls_iter`` low) and so capture the bright-arc amplitude —
            without the high-side rejection conflating those arcs with Bragg and
            under-subtracting the ring.

        ``q_mag`` / ``phi`` may be supplied to reuse already-computed coordinate
        grids (pure functions of the plane geometry); see
        :func:`~nebula3d.pipeline.remove_rings`, which computes them once per
        volume and passes the plane slice to both ``fit`` and ``subtract``.
        """
        q_mag, phi = self._plane_coords(vol, q_mag, phi)

        valid = vol.mask & np.isfinite(vol.data)
        if bragg_keep_mask is not None:
            valid &= np.asarray(bragg_keep_mask, dtype=bool)
        if q_range is not None:
            valid &= (q_mag >= q_range[0]) & (q_mag <= q_range[1])
        if int(valid.sum()) < max(self.min_voxels_per_bin * 4, 16):
            raise RuntimeError("Too few valid voxels for the parametric ring fit.")

        q = q_mag[valid]
        intensity = vol.data[valid]
        ph = phi[valid]

        q_lo = float(q_range[0]) if q_range else float(q.min())
        q_hi = float(q_range[1]) if q_range else float(q.max())

        # --- pooled robust radial profile (all φ together) --------------------
        edges = np.arange(q_lo, q_hi + self.q_step, self.q_step)
        if edges.size < 3:
            raise RuntimeError("q_range too narrow for the parametric ring fit.")
        q_grid = 0.5 * (edges[:-1] + edges[1:])
        pooled, counts = _robust_radial_profile(
            q, intensity, edges, self.profile_percentiles,
            self.min_voxels_per_bin, self.profile_method)
        pooled_f = _fill_nan_1d(pooled)

        # --- smooth diffuse baseline ------------------------------------------
        width = _adaptive_ring_width_profile(
            q_grid, pooled_f, self.q_step, self.ring_width, 3.0, 0.9, counts)
        baseline = _estimate_baseline(
            pooled_f, self.q_step, width, self.baseline_smooth, self.baseline_method)
        excess = np.maximum(0.0, pooled_f - baseline)

        # --- rolling-window continuous mode -----------------------------------
        if self.radial_mode == "rolling":
            self._model = self._fit_rolling(
                q, intensity, ph, q_grid, pooled_f, baseline, q_lo, q_hi)
            return self._model

        # --- peaks mode: ring centres (confirmed shells, else detection) ------
        if self.allowed_ring_centers is not None and self.allowed_ring_centers.size:
            centers = self.allowed_ring_centers
            if self.allowed_ring_halfwidths is not None:
                fwhm0 = 2.0 * np.atleast_1d(self.allowed_ring_halfwidths)
            else:
                fwhm0 = np.full(centers.size, max(self.ring_width / 2.0, self.q_step))
            ceilings = self.allowed_ring_ceilings
        else:
            centers, fwhm0 = _detect_rings(
                q_grid, pooled_f, self.q_step, self.ring_width, counts)
            ceilings = None
            # absolute SNR gate — the detector's prominence is relative, so a
            # ring-free profile would otherwise yield noise-spike "rings".
            if centers.size:
                noise = _profile_noise(pooled_f)
                peak_exc = np.interp(centers, q_grid, excess)
                keep = peak_exc >= self.min_ring_snr * noise
                centers, fwhm0 = centers[keep], fwhm0[keep]

        if centers.size == 0:
            self._model = FittedParametricRingModel(
                plane=self.plane, rings=[], n_fourier=self.n_fourier,
                symmetric=self.symmetric, q_grid=q_grid,
                pooled_profile=pooled_f, baseline=baseline)
            return self._model

        # --- radial shape: multi-pseudo-Voigt fit to the excess ---------------
        pv = _fit_multi_pseudo_voigt(
            q_grid, excess, centers, fwhm0, self.eta0,
            self.q_step, self.ring_width)

        # --- per-ring binning-free azimuthal texture --------------------------
        rings: list[ParametricRing] = []
        for q0, fwhm, eta, amp in pv:
            coeffs = self._fit_ring_texture(
                q, intensity, ph, q_grid, baseline, q0, fwhm, eta, amp)
            q0_coeffs = fwhm_coeffs = None
            if self.radial_harmonics:
                q0_coeffs, fwhm_coeffs = self._fit_radial_harmonics(
                    q, intensity, ph, q_grid, baseline, q0, fwhm, eta, coeffs)
            rings.append(ParametricRing(q0, fwhm, eta, coeffs,
                                        q0_coeffs=q0_coeffs, fwhm_coeffs=fwhm_coeffs))

        self._model = FittedParametricRingModel(
            plane=self.plane, rings=rings, n_fourier=self.n_fourier,
            symmetric=self.symmetric, q_grid=q_grid, pooled_profile=pooled_f,
            baseline=baseline, radial_n_fourier=self.radial_n_fourier,
            ceilings=None if ceilings is None else np.asarray(ceilings, float))
        return self._model

    def subtract(
        self,
        vol: HKLVolume,
        model: FittedParametricRingModel | None = None,
        q_mag: NDArray[np.float64] | None = None,
        phi: NDArray[np.float64] | None = None,
    ) -> tuple[HKLVolume, NDArray[np.float64]]:
        """Subtract the fitted ring model; return ``(vol_sub, I_ring)``.

        ``q_mag`` / ``phi`` may be supplied to reuse already-computed coordinate
        grids (see :meth:`fit`).
        """
        m = model or self._model
        if m is None:
            raise RuntimeError("Call fit() before subtract().")

        q_mag, phi = self._plane_coords(vol, q_mag, phi)
        I_ring = m.evaluate(q_mag, phi)

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
    def model(self) -> FittedParametricRingModel | None:
        return self._model

    # ------------------------------------------------------------------
    # Texture fit
    # ------------------------------------------------------------------
    def _fit_ring_texture(
        self,
        q: NDArray[np.float64],
        intensity: NDArray[np.float64],
        phi: NDArray[np.float64],
        q_grid: NDArray[np.float64],
        baseline: NDArray[np.float64],
        q0: float,
        fwhm: float,
        eta: float,
        amp: float,
    ) -> NDArray[np.float64]:
        """Robust, binning-free Fourier fit of one ring's azimuthal amplitude.

        Each shell voxel contributes ``(φ, target/template)`` where ``target =
        I − baseline(|Q|)`` and ``template = PVᵢ(|Q|)`` (unit peak).  The
        intended relation is ``target ≈ Tᵢ(φ)·template`` so the ratio estimates
        ``Tᵢ(φ)`` directly; weighting by ``template²`` trusts the near-peak
        voxels, and the robust solve down-weights the φ-narrow Bragg spikes
        (see :meth:`_robust_texture_solve`).  Falls back to the flat radial
        amplitude ``amp`` if the shell is too sparse.
        """
        n_basis = (1 + self.n_fourier) if self.symmetric else (1 + 2 * self.n_fourier)
        flat = np.zeros(n_basis, dtype=np.float64)
        flat[0] = max(amp, 0.0)

        half = self.texture_shell_scale * max(fwhm, self.q_step)
        shell = np.abs(q - q0) <= half
        if int(shell.sum()) < 3 * n_basis:
            return flat

        q_s = q[shell]
        template = _pseudo_voigt(q_s, q0, fwhm, eta)
        near = template >= self.texture_min_template
        if int(near.sum()) < 3 * n_basis:
            return flat

        q_s = q_s[near]
        template = template[near]
        base_at = np.interp(q_s, q_grid, baseline, left=baseline[0], right=baseline[-1])
        target = intensity[shell][near] - base_at
        ratio = target / template
        phi_s = phi[shell][near]

        B = _azimuthal_basis(phi_s, self.n_fourier, self.symmetric)  # (N, M)
        # base weight: near-peak voxels (template ≈ 1) dominate
        w0 = template**2

        # smoothness prior ∝ harmonic order², scaled to the data term, c₀ free
        if self.symmetric:
            order = np.array([0] + [j**2 for j in range(1, self.n_fourier + 1)], float)
        else:
            order = np.array(
                [0] + [k**2 for k in range(1, self.n_fourier + 1) for _ in (0, 1)],
                float)
        reg = np.diag(order)

        return self._robust_texture_solve(B, ratio, w0, phi_s, n_basis, reg)

    def _robust_texture_solve(
        self,
        B: NDArray[np.float64],
        ratio: NDArray[np.float64],
        w0: NDArray[np.float64],
        phi_s: NDArray[np.float64],
        n_basis: int,
        reg: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Ridge-regularised, robust IRLS solve of one azimuthal texture.

        With ``texture_spike_reject`` (default) the φ-narrow Bragg spikes are
        suppressed *once* by :func:`_phi_spike_weight` (a model-independent
        weight from each sample's φ-neighbourhood) and the texture is then a
        single weighted ridge solve — no model-residual reweighting, because the
        ridge under-fits the arc peaks and an IRLS on those residuals would
        re-suppress the very bright arcs we want to keep.  Otherwise the legacy
        high-side IRLS is used (down-weights every positive residual — bright
        arcs included — and so under-subtracts textured rings).
        """
        def _solve(w: NDArray[np.float64]) -> NDArray[np.float64]:
            AtA = B.T @ (B * w[:, None])
            scale = np.trace(AtA) / n_basis
            Aty = B.T @ (w * ratio)
            try:
                return np.linalg.solve(AtA + self.texture_ridge * scale * reg, Aty)
            except np.linalg.LinAlgError:
                c, *_ = np.linalg.lstsq(
                    AtA + self.texture_ridge * scale * reg, Aty, rcond=None)
                return c

        if self.texture_spike_reject:
            # Bragg handled up front by the φ-shape weight; one ridge solve keeps
            # the broad arcs (no residual-based reweighting to undo it).
            return _solve(w0 * _phi_spike_weight(phi_s, ratio, n_basis))

        # legacy: high-side-only IRLS (conflates bright arcs with Bragg)
        w = w0.copy()
        c = np.zeros(n_basis, dtype=np.float64)
        for _ in range(max(1, self.texture_irls_iter)):
            c = _solve(w)  # type: ignore[assignment]
            resid = ratio - B @ c
            mad = float(np.median(np.abs(resid - np.median(resid)))) + 1e-12
            scale_r = 1.4826 * mad
            tukey = np.clip(1.0 - (resid / (4.0 * scale_r)) ** 2, 0.0, 1.0) ** 2
            tukey = np.where(resid > 0, tukey, 1.0)
            w = w0 * tukey
        return c

    # ------------------------------------------------------------------
    # Adaptive non-separable radial shape: q0(φ), fwhm(φ)
    # ------------------------------------------------------------------
    def _fit_radial_harmonics(
        self,
        q: NDArray[np.float64],
        intensity: NDArray[np.float64],
        phi: NDArray[np.float64],
        q_grid: NDArray[np.float64],
        baseline: NDArray[np.float64],
        q0: float,
        fwhm: float,
        eta: float,
        texture_coeffs: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
        """Fit ``q0(φ)`` and ``fwhm(φ)`` for one ring and accept them only if they
        help (adaptive).

        Per azimuthal sector, fit a pseudo-Voigt to the shell voxels to get that
        sector's centre and width, then a low-order Fourier series across φ.
        Guards: the shell must average at least
        ``radial_harmonic_min_voxels_per_sector`` voxels/sector, and enough
        sectors must yield a fit.  Accept the harmonics only if they cut the
        on-shell residual inhomogeneity (std over φ of the per-sector median
        residual, using the ring's texture so the amplitude is matched) by at
        least ``radial_harmonic_accept_margin`` vs the constant shape — otherwise
        the ring stays separable (returns ``None, None``).
        """
        from scipy.optimize import curve_fit

        w = max(fwhm, self.q_step)
        shell = np.abs(q - q0) <= 3.0 * w
        nsec = self.radial_harmonic_sectors
        minv = self.radial_harmonic_min_voxels_per_sector
        if int(shell.sum()) < nsec * minv:
            return None, None

        qs, ints, phs = q[shell], intensity[shell], phi[shell]
        base_at = np.interp(qs, q_grid, baseline, left=baseline[0], right=baseline[-1])
        exc = ints - base_at

        edges = np.linspace(-np.pi, np.pi, nsec + 1)
        ctr = 0.5 * (edges[:-1] + edges[1:])
        sec = np.clip(np.digitize(phs, edges) - 1, 0, nsec - 1)
        q0_s = np.full(nsec, np.nan)
        fw_s = np.full(nsec, np.nan)
        for s in range(nsec):
            m = sec == s
            if int(m.sum()) < minv:
                continue
            try:
                popt, _ = curve_fit(
                    lambda qq, a, c, f: a * _pseudo_voigt(qq, c, f, eta),
                    qs[m], exc[m],
                    p0=[max(float(np.nanmax(exc[m])), 1e-3), q0, w],
                    bounds=([0.0, q0 - 1.5 * w, 0.3 * w],
                            [np.inf, q0 + 1.5 * w, 3.0 * w]),
                    maxfev=4000)
                q0_s[s], fw_s[s] = popt[1], popt[2]
            except (RuntimeError, ValueError):
                continue

        valid = np.isfinite(q0_s) & np.isfinite(fw_s)
        nh = self.radial_n_fourier
        if int(valid.sum()) < 2 * nh + 2:
            return None, None

        B = _azimuthal_basis(ctr[valid], nh, False)
        q0_coeffs, *_ = np.linalg.lstsq(B, q0_s[valid], rcond=None)
        fw_coeffs, *_ = np.linalg.lstsq(B, fw_s[valid], rcond=None)

        # accept test: residual inhomogeneity, harmonic vs its OWN DC constant
        # (same texture, same estimation source) — so we credit the azimuthal
        # VARIATION only, not a slightly better mean width than the global fit.
        Bv = _azimuthal_basis(phs, nh, False)
        q0_phi = np.clip(Bv @ q0_coeffs, q0 - 2.0 * fwhm, q0 + 2.0 * fwhm)
        fw_phi = np.clip(Bv @ fw_coeffs, 0.3 * fwhm, 3.0 * fwhm)
        q0_dc = float(np.median(q0_s[valid]))
        fw_dc = float(np.clip(np.median(fw_s[valid]), 0.3 * fwhm, 3.0 * fwhm))
        tex = np.maximum(
            0.0, _azimuthal_basis(phs, self.n_fourier, self.symmetric) @ texture_coeffs)
        res_h = exc - tex * _pseudo_voigt_phi(qs, q0_phi, fw_phi, eta)
        res_c = exc - tex * _pseudo_voigt(qs, q0_dc, fw_dc, eta)
        inh_h = self._shell_inhom(res_h, sec, nsec)
        inh_c = self._shell_inhom(res_c, sec, nsec)
        if inh_c > 0 and inh_h < inh_c * (1.0 - self.radial_harmonic_accept_margin):
            return q0_coeffs, fw_coeffs
        return None, None

    @staticmethod
    def _shell_inhom(
        resid: NDArray[np.float64], sec: NDArray[np.int_], nsec: int
    ) -> float:
        """Azimuthal inhomogeneity = std over sectors of the per-sector median
        residual (lower = cleaner, more uniform ring removal)."""
        prof = np.array([
            np.median(resid[sec == s]) if np.any(sec == s) else np.nan
            for s in range(nsec)])
        return float(np.nanstd(prof))

    # ------------------------------------------------------------------
    # Rolling-window continuous fit
    # ------------------------------------------------------------------
    def _fit_rolling(
        self,
        q: NDArray[np.float64],
        intensity: NDArray[np.float64],
        phi: NDArray[np.float64],
        q_grid: NDArray[np.float64],
        pooled_f: NDArray[np.float64],
        baseline: NDArray[np.float64],
        q_lo: float,
        q_hi: float,
    ) -> FittedParametricRingModel:
        """Sweep a thick |Q| window and fit one azimuthal Fourier texture per
        shell, binning-free.

        At each rolling centre ``q0`` the voxels in ``|q − q0| ≤ ring_width``
        contribute ``(φ, (I − baseline)/r(q))`` where ``r(q)`` is the continuous
        radial excess ``pooled − baseline`` normalised to 1 at ``q0``.  Dividing
        by the local radial template means the thick window supplies azimuthal
        SNR **without smearing the radial amplitude** — the fitted constant is
        the value *at* ``q0`` (≈ ``Ring(q0)``), not the window average, and the
        harmonics are the azimuthal texture.  A Hann weight in ``|q − q0|`` (×
        the template) localises the estimate, a ridge damps high harmonics, and
        the robust solve rejects the φ-narrow Bragg spikes while keeping the
        broad bright arcs (see :meth:`_robust_texture_solve`).  Off-ring shells (where
        the radial excess is ~0) contribute nothing, so the diffuse is preserved.
        The result is a *continuous* ``Ring(|Q|)·T(|Q|,φ)`` — no discrete-peak
        detection.

        When confirmed shells are configured the per-shell coefficients are
        multiplied by the same [0, 1] |Q|-envelope (and the amplitude capped to
        the per-shell ceiling) used by the peaks mode, so the rolling fit stays
        compatible with the across-stack 3-D driver; with no confirmed shells it
        subtracts the azimuthally-structured excess at every |Q|.
        """
        n_basis = (1 + self.n_fourier) if self.symmetric else (1 + 2 * self.n_fourier)
        centers = np.arange(q_lo, q_hi + self.roll_step, self.roll_step)
        coeffs = np.zeros((centers.size, n_basis), dtype=np.float64)

        W = max(self.ring_width, 2.0 * self.roll_step)

        # smoothness prior ∝ harmonic order², scaled to the data term, c₀ free
        if self.symmetric:
            order = np.array([0] + [j**2 for j in range(1, self.n_fourier + 1)], float)
        else:
            order = np.array(
                [0] + [k**2 for k in range(1, self.n_fourier + 1) for _ in (0, 1)],
                float)
        reg = np.diag(order)

        # continuous radial excess r(|Q|) = pooled − baseline (the Ring(|Q|) shape)
        radial_excess = np.maximum(0.0, pooled_f - baseline)
        excess_at = np.interp(q, q_grid, radial_excess, left=0.0, right=0.0)
        base_at = np.interp(q, q_grid, baseline, left=baseline[0], right=baseline[-1])
        a_ref = np.interp(centers, q_grid, radial_excess)
        floor = self.min_ring_snr * _profile_noise(pooled_f)

        for j, q0 in enumerate(centers):
            a0 = float(a_ref[j])
            if a0 <= max(floor, 1e-9):
                continue                       # off-ring: nothing to subtract
            dq = q - q0
            shell = np.abs(dq) <= W
            if int(shell.sum()) < 3 * n_basis:
                continue
            template = excess_at[shell] / a0   # local radial shape, 1 at q0
            near = template >= self.texture_min_template
            if int(near.sum()) < 3 * n_basis:
                continue
            template = template[near]
            phi_s = phi[shell][near]
            ratio = (intensity[shell][near] - base_at[shell][near]) / template
            hann = 0.5 * (1.0 + np.cos(np.pi * np.clip(dq[shell][near] / W, -1.0, 1.0)))
            w0 = template**2 * hann            # near-peak + radially-local voxels

            B = _azimuthal_basis(phi_s, self.n_fourier, self.symmetric)
            coeffs[j] = self._robust_texture_solve(
                B, ratio, w0, phi_s, n_basis, reg)

        # the continuous radial amplitude is the constant (φ-mean) term; never < 0
        coeffs[:, 0] = np.maximum(coeffs[:, 0], 0.0)

        # restrict to confirmed shells (envelope + per-shell ceiling) when given
        if self.allowed_ring_centers is not None and self.allowed_ring_centers.size:
            env = self._shell_envelope(centers)  # type: ignore[arg-type]
            coeffs *= env[:, None]
            if self.allowed_ring_ceilings is not None:
                cap = self._shell_ceiling(centers)  # type: ignore[arg-type]
                np.minimum(coeffs[:, 0], cap, out=coeffs[:, 0])

        return FittedParametricRingModel(
            plane=self.plane, n_fourier=self.n_fourier, symmetric=self.symmetric,
            mode="rolling", roll_centers=centers, roll_coeffs=coeffs,  # type: ignore[arg-type]
            q_grid=q_grid, pooled_profile=pooled_f, baseline=baseline)

    def _shell_envelope(self, centers: NDArray[np.float64]) -> NDArray[np.float64]:
        """[0, 1] weight: 1 within ±halfwidth of a confirmed shell, raised-cosine
        roll-off to 0 over the next halfwidth, max over shells."""
        cc = self.allowed_ring_centers
        assert cc is not None
        half = self.allowed_ring_halfwidths
        if half is None:
            half = np.full(cc.size, max(self.ring_width, 4.0 * self.roll_step))
        env = np.zeros_like(centers)
        for c, w in zip(cc, np.atleast_1d(half)):
            w = max(float(w), self.roll_step)
            d = np.abs(centers - float(c))
            bump = np.where(
                d <= w, 1.0,
                np.where(d <= 2.0 * w,
                         0.5 * (1.0 + np.cos(np.pi * (d - w) / w)), 0.0))
            env = np.maximum(env, bump)
        return env

    def _shell_ceiling(self, centers: NDArray[np.float64]) -> NDArray[np.float64]:
        """Per-rolling-centre amplitude ceiling from the nearest confirmed shell
        (within its 2·halfwidth support); +inf elsewhere."""
        cc = self.allowed_ring_centers
        ceil = self.allowed_ring_ceilings
        assert cc is not None and ceil is not None
        half = self.allowed_ring_halfwidths
        if half is None:
            half = np.full(cc.size, max(self.ring_width, 4.0 * self.roll_step))
        out = np.full(centers.size, np.inf)
        for c, w, cap in zip(cc, np.atleast_1d(half), np.atleast_1d(ceil)):
            within = np.abs(centers - float(c)) <= 2.0 * max(float(w), self.roll_step)
            out[within] = np.minimum(out[within], float(cap))
        return out


def _phi_spike_weight(
    phi: NDArray[np.float64],
    ratio: NDArray[np.float64],
    n_basis: int,
    cut: float = 4.0,
) -> NDArray[np.float64]:
    """Down-weight φ-bins that stick up above their NEIGHBOURS (Bragg spikes).

    A Bragg reflection on the ring is sharp in φ — its φ-bin level sits far
    above the bins on either side.  A bright ring arc is broad — its bin sits at
    the level of its neighbours because they are elevated too.  So we bin φ,
    take each bin's robust level, subtract a *neighbour-median baseline* (a
    rolling median over a window wider than a Bragg peak but narrower than an
    arc), and down-weight bins whose excess is a positive outlier.  Comparing
    bin-to-neighbours (not sample-to-own-bin) is what keeps the upper half of the
    in-bin noise — and the bright arcs — at full weight.  Returns a per-sample
    multiplicative weight in [0, 1].
    """
    phi = np.asarray(phi, dtype=np.float64)
    ratio = np.asarray(ratio, dtype=np.float64)
    n = phi.size
    if n < 3:
        return np.ones(n, dtype=np.float64)
    lo, hi = float(phi.min()), float(phi.max())
    if hi <= lo:
        return np.ones(n, dtype=np.float64)
    nb = int(np.clip(n // (4 * max(n_basis, 1)), 16, 72))
    edges = np.linspace(lo, hi, nb + 1)
    b = np.clip(np.digitize(phi, edges) - 1, 0, nb - 1)

    level = np.full(nb, np.nan)
    for k in range(nb):
        m = b == k
        if np.any(m):
            level[k] = np.median(ratio[m])
    valid = np.isfinite(level)
    if valid.sum() < 5:
        return np.ones(n, dtype=np.float64)

    # neighbour-median baseline: rolling median over ±rad bins (excludes nothing;
    # a lone Bragg bin can't move a median of many neighbours, a broad arc can)
    rad = max(2, nb // 12)
    base = np.copy(level)
    for k in range(nb):
        if not valid[k]:
            continue
        win = level[max(0, k - rad): k + rad + 1]
        win = win[np.isfinite(win)]
        if win.size:
            base[k] = np.median(win)
    excess = level - base
    ex = excess[valid]
    mad = float(np.median(np.abs(ex - np.median(ex)))) + 1e-12
    scale = 1.4826 * mad
    bin_w = np.ones(nb, dtype=np.float64)
    pos = valid & (excess > 0.0)
    bin_w[pos] = np.clip(1.0 - (excess[pos] / (cut * scale)) ** 2, 0.0, 1.0) ** 2
    return bin_w[b]


def _profile_noise(prof: NDArray[np.float64]) -> float:
    """Robust noise level of a 1-D profile from its first differences.

    ``MAD(Δprof) / √2`` estimates the per-bin noise σ even on a sloping or
    structured background, because a smooth signal contributes near-zero
    differences and a few sharp rings cannot move the median.
    """
    d = np.diff(np.asarray(prof, dtype=np.float64))
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0
    mad = float(np.median(np.abs(d - np.median(d))))
    return 1.4826 * mad / np.sqrt(2.0)


# ---------------------------------------------------------------------------
# Multi-pseudo-Voigt radial fit
# ---------------------------------------------------------------------------
def _fit_multi_pseudo_voigt(
    q_grid: NDArray[np.float64],
    excess: NDArray[np.float64],
    centers: NDArray[np.float64],
    fwhm0: NDArray[np.float64],
    eta0: float,
    q_step: float,
    ring_width: float,
) -> list[tuple[float, float, float, float]]:
    """Fit Σ ampᵢ·PV(|Q|; q0ᵢ, fwhmᵢ, etaᵢ) to the baseline-subtracted excess.

    Overlapping rings (centres within ``3·max(fwhm)``) are fit jointly per
    cluster; isolated rings are fit one at a time.  Returns one
    ``(q_center, fwhm, eta, amplitude)`` per input centre, sorted by |Q|.  A
    cluster whose bounded least-squares fails falls back to the seed values
    (amplitude read from ``excess`` at the centre).
    """
    from scipy.optimize import curve_fit

    centers = np.atleast_1d(np.asarray(centers, float))
    fwhm0 = np.atleast_1d(np.asarray(fwhm0, float))
    order = np.argsort(centers)
    centers, fwhm0 = centers[order], fwhm0[order]

    def _seed_amp(c: float) -> float:
        return max(float(np.interp(c, q_grid, excess)), 1e-6)

    # cluster nearby centres for joint fitting
    clusters: list[list[int]] = [[0]] if centers.size else []
    for i in range(1, centers.size):
        gap = centers[i] - centers[i - 1]
        if gap < 3.0 * max(fwhm0[i], fwhm0[i - 1]):
            clusters[-1].append(i)
        else:
            clusters.append([i])

    results: dict[int, tuple[float, float, float, float]] = {}
    for cl in clusters:
        lo = centers[cl[0]] - 3.0 * ring_width
        hi = centers[cl[-1]] + 3.0 * ring_width
        sel = (q_grid >= lo) & (q_grid <= hi) & np.isfinite(excess)
        x, y = q_grid[sel], excess[sel]

        def _model(xx: NDArray[np.float64], *p: float, _n: int = len(cl)) -> NDArray[np.float64]:
            out = np.zeros_like(xx)
            for j in range(_n):
                amp, c, fw, et = p[4 * j: 4 * j + 4]
                out = out + amp * _pseudo_voigt(xx, c, fw, et)
            return out

        p0: list[float] = []
        lb: list[float] = []
        ub: list[float] = []
        for idx in cl:
            c0 = float(centers[idx])
            f0 = float(np.clip(fwhm0[idx], 2.0 * q_step, ring_width))
            p0 += [_seed_amp(c0), c0, f0, float(np.clip(eta0, 0.0, 1.0))]
            lb += [0.0, c0 - 3.0 * q_step, q_step, 0.0]
            ub += [np.inf, c0 + 3.0 * q_step, ring_width, 1.0]

        ok = x.size >= len(p0)
        popt = np.array(p0)
        if ok:
            try:
                popt, _ = curve_fit(_model, x, y, p0=p0, bounds=(lb, ub), maxfev=20000)
            except (RuntimeError, ValueError):
                popt = np.array(p0)

        for j, idx in enumerate(cl):
            amp, c, fw, et = popt[4 * j: 4 * j + 4]
            results[idx] = (float(c), float(fw), float(et), float(amp))

    return [results[i] for i in range(centers.size)]
