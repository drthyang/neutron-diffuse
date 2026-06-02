"""Factored powder ring model with azimuthal texture.

Model
-----
The ring contribution at any voxel is:

    I_ring(Q, φ) = T(φ) × Σᵢ Aᵢ × G(|Q| − qᵢ, σᵢ)

where:
    G(x, σ)  : Gaussian profile for ring i in |Q|               (same for all φ)
    Aᵢ       : per-ring amplitude scale
    T(φ)     : one shared azimuthal texture function

T(φ) captures detector coverage, absorption, and normalisation artefacts
that modulate the ring amplitude around the azimuthal angle.  All rings
from the same polycrystalline material share the same T(φ) because they
all come from the same detector geometry.

Fitting strategy
----------------
1. Project voxels onto a reference 2D plane (default: hk0).
   φ = atan2(k_Q, h_Q) in that plane.
2. Divide φ ∈ [0, 2π) into N overlapping patches (Hann-weighted).
3. Per patch: fit Gaussians to the 1D radial profile I(|Q|).
   Returns an amplitude matrix  A[n_rings × n_patches].
4. Rank-1 SVD factorisation of A → per-ring amplitudes Aᵢ and
   per-patch texture values T[P].
5. Fit a Fourier series to (φ_P, T[P]) → smooth, periodic T(φ).

For 3D data the Gaussian is evaluated at the full 3D |Q| of each voxel;
T(φ) uses the projected azimuthal angle from the reference plane.

Subtraction
-----------
After fitting, subtract I_ring at each voxel.  Voxels where the ring
clearly dominates (I_ring / σ_data > threshold) are masked and filled
by the radial-interpolation backfill (backfill.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ndiff.core import HKLVolume


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RingParams:
    """Radial parameters of a single powder ring.

    Attributes
    ----------
    q_center : float   Ring |Q| position in Å^-1.
    q_sigma  : float   Gaussian σ in Å^-1.
    amplitude: float   Global amplitude (before texture weighting).
    """
    q_center: float
    q_sigma: float
    amplitude: float = 0.0


@dataclass
class FittedRingModel:
    """Result of :meth:`PatchedRingModel.fit`.

    Attributes
    ----------
    rings        : fitted radial parameters for each ring.
    texture_coeffs : Fourier coefficients for T(φ) as a (2K+1,) array
                     [c₀, a₁, b₁, a₂, b₂, …, aK, bK].
    n_patches    : number of azimuthal patches used.
    patch_centers: φ value (rad) at each patch centre.
    A_matrix     : raw per-patch amplitude matrix, shape (n_rings, n_patches).
    texture_values: per-patch T[P] values before Fourier smoothing.
    singular_values : full singular value spectrum of A_matrix.
        rank1_variance gives the fraction explained by the shared-T model.
        A second significant singular value signals that higher-|Q| rings
        have a different azimuthal texture and may need per-ring T_i(φ).
    """
    rings: list[RingParams]
    texture_coeffs: NDArray[np.float64]
    n_patches: int
    patch_centers: NDArray[np.float64]
    A_matrix: NDArray[np.float64]
    texture_values: NDArray[np.float64]
    singular_values: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    def texture(self, phi: NDArray) -> NDArray[np.float64]:
        """Evaluate T(φ) at arbitrary angles (radians)."""
        return _eval_fourier(self.texture_coeffs, phi)

    @property
    def rank1_variance(self) -> float:
        """Fraction of amplitude-matrix variance explained by the shared T(φ).

        Close to 1.0 → shared texture is a good approximation for all rings.
        Significantly below 1.0 (e.g. < 0.90) → higher-|Q| rings likely have
        a different azimuthal texture; consider per-ring T_i(φ) fitting.
        """
        if len(self.singular_values) == 0:
            return float("nan")
        s2 = self.singular_values ** 2
        return float(s2[0] / s2.sum()) if s2.sum() > 0 else float("nan")

    def per_ring_texture_residual(self) -> NDArray[np.float64]:
        """RMS residual of each ring's amplitude from the shared T(φ) model.

        Shape: (n_rings,).  Large values for a specific ring indicate its
        azimuthal texture deviates from the shared T(φ) — typically the
        outer (higher-|Q|) rings.
        """
        if self.A_matrix.size == 0:
            return np.array([])
        T_norm = self.texture_values / (self.texture_values.mean() + 1e-12)
        residuals = []
        for i, ring in enumerate(self.rings):
            row = self.A_matrix[i]
            scale = row.mean() / (T_norm.mean() + 1e-12)
            predicted = scale * T_norm
            rms = float(np.sqrt(np.mean((row - predicted) ** 2)))
            rms_rel = rms / (row.mean() + 1e-12)
            residuals.append(rms_rel)
        return np.array(residuals)

    def evaluate(self, q_mag: NDArray, phi: NDArray) -> NDArray[np.float64]:
        """Ring contribution at voxels with given |Q| and φ arrays."""
        T = self.texture(phi)
        I_ring = np.zeros_like(q_mag, dtype=np.float64)
        for ring in self.rings:
            I_ring += ring.amplitude * _gaussian(q_mag, 1.0, ring.q_center, ring.q_sigma)
        return T * I_ring


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PatchedRingModel:
    """Fit and subtract a factored ring model from a 3D HKL volume.

    Parameters
    ----------
    n_patches : int
        Number of azimuthal patches in [0, 2π).  Each patch spans
        2π/n_patches radians.  Typical: 24–72.
    overlap_frac : float
        Fraction of patch width used as overlap on each side (0–0.5).
        Hann-window weighting within the overlap region ensures C¹
        continuity of the texture function across patch boundaries.
    n_fourier : int
        Number of Fourier harmonics for T(φ).  n_fourier = 6 gives a
        period of 2π/6 ≈ 60°, which resolves most detector-geometry
        variations.
    plane : str
        Reference plane for the azimuthal angle.  ``'hk0'`` uses
        φ = atan2(k_Q, h_Q);  ``'h0l'`` uses φ = atan2(l_Q, h_Q);
        ``'0kl'`` uses φ = atan2(l_Q, k_Q).
    n_radial_bins : int
        Number of |Q| bins used by the auto ring-detection pass (when no
        ``ring_hints`` are given).
    snr_mask_threshold : float
        After subtraction, mask voxels where I_ring / σ_data exceeds
        this value.
    ring_shell_halfwidth : float
        Half-width (Å⁻¹) of the |Q| shell collected around each ring centre
        for the per-patch intensity estimate.
    ring_percentile_range : tuple[float, float]
        Intensity percentile window kept within each ring's shell when
        estimating its intensity and texture (default the 20th–80th
        percentile).  Trimming the **low** tail rejects detector
        gaps/shadows; trimming the **high** tail rejects Bragg peaks; the
        surviving central band is averaged to give a clean ring intensity
        per azimuthal patch — no Bragg punching required first.
    ring_flank_halfwidth : float
        Outer half-width (Å⁻¹) of the flanking |Q| annulus used to estimate
        the local diffuse **baseline** under each ring.  Voxels with
        ``ring_shell_halfwidth < |q − qᵢ| ≤ ring_flank_halfwidth`` (just
        inside and outside the shell) are trimmed the same way and averaged;
        the ring amplitude is the shell level **minus** this baseline, so the
        diffuse is lowered to the baseline rather than removed entirely.
    flatness_cv : float or None
        Flatness gate.  A ring is only subtracted in patches where its
        trimmed-shell coefficient of variation ``std / level`` is at or below
        this value — i.e. the shell is a *clean, flat* ring sitting on the
        baseline (your "small std" criterion).  Where the shell is rough
        (Bragg-overlapping → large std), no ring is subtracted and the voxels
        are left for the Bragg punch.  ``None`` disables the gate (always
        subtract the baseline-referenced excess).
    """

    def __init__(
        self,
        n_patches: int = 36,
        overlap_frac: float = 0.3,
        n_fourier: int = 6,
        plane: str = "hk0",
        n_radial_bins: int = 200,
        snr_mask_threshold: float = 3.0,
        ring_shell_halfwidth: float = 0.12,
        ring_percentile_range: tuple[float, float] = (20.0, 80.0),
        ring_flank_halfwidth: float = 0.24,
        flatness_cv: Optional[float] = None,
    ) -> None:
        self.n_patches = n_patches
        self.overlap_frac = overlap_frac
        self.n_fourier = n_fourier
        self.plane = plane
        self.n_radial_bins = n_radial_bins
        self.snr_mask_threshold = snr_mask_threshold
        self.ring_shell_halfwidth = ring_shell_halfwidth
        self.ring_percentile_range = ring_percentile_range
        self.ring_flank_halfwidth = ring_flank_halfwidth
        self.flatness_cv = flatness_cv
        self._model: Optional[FittedRingModel] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        vol: HKLVolume,
        ring_hints: Optional[list[float]] = None,
        q_range: Optional[tuple[float, float]] = None,
    ) -> FittedRingModel:
        """Fit the ring model to *vol*.

        Parameters
        ----------
        vol : HKLVolume
        ring_hints : list of |Q| values (Å^-1), optional
            Approximate positions of known rings.  If provided, used as
            initial guesses for the Gaussian fitting.  If None, an
            auto-detection pass is run first.
        q_range : (q_min, q_max), optional
            Restrict fitting to this |Q| range.

        Returns
        -------
        FittedRingModel
        """
        q_mag = vol.q_magnitude()
        phi = self._azimuthal_angle(vol)

        if q_range is None:
            q_range = (float(q_mag[vol.mask].min()), float(q_mag[vol.mask].max()))

        # Step 1–3: per-patch Gaussian fits → amplitude matrix
        patch_centers, A_matrix, ring_params = self._fit_patches(
            vol, q_mag, phi, ring_hints, q_range
        )

        if A_matrix.shape[1] < 2:
            raise RuntimeError("Too few usable patches for texture fitting.")

        # Step 4: rank-1 SVD → T[P] and Aᵢ
        texture_values, ring_amplitudes, singular_values = _rank1_factorize(A_matrix)

        for ring, amp in zip(ring_params, ring_amplitudes):
            ring.amplitude = float(amp)

        # Step 5: Fourier fit to T[P]
        coeffs = _fit_fourier(patch_centers, texture_values, self.n_fourier)

        self._model = FittedRingModel(
            rings=ring_params,
            texture_coeffs=coeffs,
            n_patches=self.n_patches,
            patch_centers=patch_centers,
            A_matrix=A_matrix,
            texture_values=texture_values,
            singular_values=singular_values,
        )
        return self._model

    def subtract(
        self,
        vol: HKLVolume,
        model: Optional[FittedRingModel] = None,
    ) -> tuple[HKLVolume, NDArray[np.float64]]:
        """Subtract the fitted ring model from *vol*.

        Parameters
        ----------
        vol : HKLVolume
        model : FittedRingModel, optional
            If None, uses the result of the last :meth:`fit` call.

        Returns
        -------
        vol_sub : HKLVolume
            Volume with ring contribution subtracted.  Voxels where the
            ring clearly dominates are masked (for backfill downstream).
        I_ring : NDArray
            The subtracted ring contribution at each voxel (diagnostic).
        """
        import dataclasses

        m = model or self._model
        if m is None:
            raise RuntimeError("Call fit() before subtract().")

        q_mag = vol.q_magnitude()
        phi = self._azimuthal_angle(vol)
        I_ring = m.evaluate(q_mag, phi)

        data_sub = vol.data - I_ring
        sigma_sub = np.sqrt(vol.sigma**2 + (0.1 * np.abs(I_ring))**2)

        # Mask where ring dominates
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.where(vol.sigma > 0, I_ring / vol.sigma, 0.0)
        keep = vol.mask & (snr < self.snr_mask_threshold)

        vol_sub = dataclasses.replace(vol, data=data_sub, sigma=sigma_sub, mask=keep)
        return vol_sub, I_ring

    @property
    def model(self) -> Optional[FittedRingModel]:
        return self._model

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _azimuthal_angle(self, vol: HKLVolume) -> NDArray[np.float64]:
        """Return azimuthal angle φ (radians) for every voxel."""
        H, K, L = vol.hkl_grid()
        # Convert to Cartesian Q via UB matrix
        hkl = np.stack([H, K, L], axis=-1)          # (..., 3)
        Q = hkl @ vol.ub_matrix.T                    # (..., 3) in Å^-1

        if self.plane == "hk0":
            return np.arctan2(Q[..., 1], Q[..., 0])  # atan2(k_Q, h_Q)
        elif self.plane == "h0l":
            return np.arctan2(Q[..., 2], Q[..., 0])  # atan2(l_Q, h_Q)
        elif self.plane == "0kl":
            return np.arctan2(Q[..., 2], Q[..., 1])  # atan2(l_Q, k_Q)
        else:
            raise ValueError(f"Unknown plane: {self.plane!r}")

    # ------------------------------------------------------------------
    # Patch fitting
    # ------------------------------------------------------------------

    def _fit_patches(
        self,
        vol: HKLVolume,
        q_mag: NDArray,
        phi: NDArray,
        ring_hints: Optional[list[float]],
        q_range: tuple[float, float],
    ) -> tuple[NDArray, NDArray, list[RingParams]]:
        """Divide φ into patches, fit Gaussians per patch.

        Returns
        -------
        patch_centers : (n_good_patches,)
        A_matrix      : (n_rings, n_good_patches)
        ring_params   : list[RingParams] with shared qᵢ and σᵢ
        """
        patch_width = 2 * np.pi / self.n_patches
        half_overlap = self.overlap_frac * patch_width
        phi_centers = np.linspace(0, 2 * np.pi, self.n_patches, endpoint=False)

        # First pass: detect ring positions from the full radial profile
        # (all patches combined) so qᵢ and σᵢ are shared
        ring_params = self._detect_rings_global(vol, q_mag, ring_hints, q_range)
        if not ring_params:
            return np.array([]), np.empty((0, 0)), []

        n_rings = len(ring_params)
        A_list: list[NDArray] = []
        centers_list: list[float] = []

        for phi_c in phi_centers:
            # Select voxels in this patch (with wrap-around at ±π)
            dphi = _angular_distance(phi, phi_c)
            half_w = 0.5 * patch_width + half_overlap
            in_patch = vol.mask & (np.abs(dphi) <= half_w)

            if in_patch.sum() < 20:
                continue  # not enough voxels; skip patch

            q_p = q_mag[in_patch]
            I_p = vol.data[in_patch]

            # Per-ring intensity from the interquantile-trimmed shell voxels.
            amps = self._fit_shell_amplitudes(q_p, I_p, ring_params)
            if amps is not None:
                A_list.append(amps)
                centers_list.append(float(phi_c))

        if not A_list:
            return np.array([]), np.empty((n_rings, 0)), ring_params

        A_matrix = np.column_stack(A_list)          # (n_rings, n_good_patches)
        patch_centers = np.array(centers_list)
        return patch_centers, A_matrix, ring_params

    def _detect_rings_global(
        self,
        vol: HKLVolume,
        q_mag: NDArray,
        ring_hints: Optional[list[float]],
        q_range: tuple[float, float],
    ) -> list[RingParams]:
        """Detect ring positions from the full radial profile."""
        from ndiff.preprocessing.powder_rings import detect_ring_shells, RingShell

        if ring_hints is not None:
            # Use provided positions; the radial profile shape used for
            # subtraction is a Gaussian whose σ spans the collection shell
            # (σ ≈ half-width / 2, so the shell is ~±2σ).
            default_sigma = self.ring_shell_halfwidth / 2.0
            return [RingParams(q_center=q0, q_sigma=default_sigma) for q0 in ring_hints]

        # Auto-detect
        rings_auto, *_ = detect_ring_shells(
            vol, n_bins=self.n_radial_bins,
            baseline_window=min(40, self.n_radial_bins // 5),
            sigma_threshold=4.0,
            min_q=q_range[0],
        )
        return [RingParams(q_center=r.q_center, q_sigma=r.q_halfwidth / 3.0)
                for r in rings_auto]

    def _trimmed_mean(self, vals: NDArray) -> tuple[float, float]:
        """Mean and std of the central ``ring_percentile_range`` band of *vals*."""
        lo_p, hi_p = self.ring_percentile_range
        p_lo, p_hi = np.percentile(vals, (lo_p, hi_p))
        keep = vals[(vals >= p_lo) & (vals <= p_hi)]
        if keep.size == 0:
            return float(np.median(vals)), float(vals.std())
        return float(keep.mean()), float(keep.std())

    def _fit_shell_amplitudes(
        self,
        q: NDArray,
        I: NDArray,
        ring_params: list[RingParams],
    ) -> Optional[NDArray]:
        """Per-ring amplitude in one azimuthal patch, referenced to a baseline.

        For each ring:

        1. **Shell** voxels (``|q − qᵢ| ≤ ring_shell_halfwidth``) give the ring
           *level* — the trimmed (``ring_percentile_range``) mean, which drops
           the low tail (detector gaps/shadows) and the high tail (Bragg peaks).
        2. **Flank** voxels (``ring_shell_halfwidth < |q − qᵢ| ≤
           ring_flank_halfwidth``) give the local diffuse *baseline*, trimmed
           the same way.
        3. The amplitude is ``max(0, level − baseline)`` — the ring is lowered
           *to* the baseline, so the underlying diffuse is preserved.

        Flatness gate (``flatness_cv``): where the trimmed shell is rough
        (``std / level`` above the gate → Bragg-overlapping, not a clean ring),
        the amplitude is set to 0 and those voxels are left for the Bragg punch.

        Returns an ``(n_rings,)`` amplitude array, or ``None`` if no ring is
        usable in this patch.
        """
        amps = np.full(len(ring_params), np.nan, dtype=float)

        for i, r in enumerate(ring_params):
            dq = np.abs(q - r.q_center)
            shell = I[dq <= self.ring_shell_halfwidth]
            if shell.size < 5:
                continue

            level, std = self._trimmed_mean(shell)

            # flatness gate: only subtract clean (flat) ring shells
            if self.flatness_cv is not None and level > 0 and (std / level) > self.flatness_cv:
                amps[i] = 0.0
                continue

            flank = I[(dq > self.ring_shell_halfwidth) & (dq <= self.ring_flank_halfwidth)]
            baseline = self._trimmed_mean(flank)[0] if flank.size >= 5 else level
            amps[i] = max(0.0, level - baseline)

        if np.all(np.isnan(amps)):
            return None
        return np.nan_to_num(amps, nan=0.0)


# ---------------------------------------------------------------------------
# Fourier texture
# ---------------------------------------------------------------------------

def _fit_fourier(
    phi: NDArray,
    values: NDArray,
    n_harmonics: int,
) -> NDArray[np.float64]:
    """Fit a Fourier series to (φ, values) pairs.

    T(φ) = c₀ + Σₖ₌₁ᴷ (aₖ cos kφ + bₖ sin kφ)

    Returns coefficient array [c₀, a₁, b₁, …, aK, bK].
    """
    K = n_harmonics
    cols = [np.ones(len(phi))]
    for k in range(1, K + 1):
        cols.append(np.cos(k * phi))
        cols.append(np.sin(k * phi))
    A = np.column_stack(cols)
    coeffs, *_ = np.linalg.lstsq(A, values, rcond=None)
    return coeffs


def _eval_fourier(coeffs: NDArray, phi: NDArray) -> NDArray[np.float64]:
    """Evaluate Fourier series at angles *phi*."""
    result = np.full_like(phi, coeffs[0], dtype=np.float64)
    K = (len(coeffs) - 1) // 2
    for k in range(1, K + 1):
        a = coeffs[2 * k - 1]
        b = coeffs[2 * k]
        result = result + a * np.cos(k * phi) + b * np.sin(k * phi)
    return result


# ---------------------------------------------------------------------------
# SVD rank-1 factorisation
# ---------------------------------------------------------------------------

def _rank1_factorize(
    A: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Factorise A[n_rings × n_patches] ≈ ring_amps × texture^T.

    Uses the first singular vector pair (rank-1 approximation).

    The full singular value spectrum S is returned so the caller can
    assess whether the shared-texture assumption holds:
      - rank1_variance = S[0]² / sum(S²)
      - Values well below 1.0 suggest rings differ in azimuthal texture,
        with higher-|Q| rings typically being the worst offenders.

    Returns
    -------
    texture   : (n_patches,)  — T[P], normalised so mean = 1
    ring_amps : (n_rings,)    — Aᵢ
    singular_values : (min(n_rings, n_patches),) — full spectrum for diagnostics
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    left = U[:, 0] * S[0]
    right = Vt[0, :]

    if left.mean() < 0:
        left, right = -left, -right

    t_mean = right.mean() if right.mean() != 0 else 1.0
    texture = right / t_mean
    ring_amps = left * t_mean

    return texture, ring_amps, S


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian(x: NDArray, amp: float, x0: float, sigma: float) -> NDArray:
    return amp * np.exp(-0.5 * ((x - x0) / (sigma + 1e-12)) ** 2)


def _angular_distance(phi: NDArray, phi_c: float) -> NDArray:
    """Signed angular distance in (-π, π], accounting for wrap-around."""
    d = phi - phi_c
    return (d + np.pi) % (2 * np.pi) - np.pi
