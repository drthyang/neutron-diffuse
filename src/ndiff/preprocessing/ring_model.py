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
from scipy.optimize import curve_fit

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
    """
    rings: list[RingParams]
    texture_coeffs: NDArray[np.float64]
    n_patches: int
    patch_centers: NDArray[np.float64]
    A_matrix: NDArray[np.float64]
    texture_values: NDArray[np.float64]

    def texture(self, phi: NDArray) -> NDArray[np.float64]:
        """Evaluate T(φ) at arbitrary angles (radians)."""
        return _eval_fourier(self.texture_coeffs, phi)

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
        Number of |Q| bins in each patch for the Gaussian fitting.
    snr_mask_threshold : float
        After subtraction, mask voxels where I_ring / σ_data exceeds
        this value.
    """

    def __init__(
        self,
        n_patches: int = 36,
        overlap_frac: float = 0.3,
        n_fourier: int = 6,
        plane: str = "hk0",
        n_radial_bins: int = 200,
        snr_mask_threshold: float = 3.0,
    ) -> None:
        self.n_patches = n_patches
        self.overlap_frac = overlap_frac
        self.n_fourier = n_fourier
        self.plane = plane
        self.n_radial_bins = n_radial_bins
        self.snr_mask_threshold = snr_mask_threshold
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
        texture_values, ring_amplitudes = _rank1_factorize(A_matrix)

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
            dphi_p = dphi[in_patch]

            # Hann weight within patch
            w_p = np.cos(0.5 * np.pi * dphi_p / half_w) ** 2

            # Bin into radial profile (weighted)
            amps = self._fit_gaussians_in_patch(q_p, I_p, w_p, ring_params, q_range)
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
            # Use provided positions; estimate σ from bin spacing
            q_all = q_mag[vol.mask]
            q_span = q_all.max() - q_all.min()
            default_sigma = q_span / (self.n_radial_bins * 4)
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

    def _fit_gaussians_in_patch(
        self,
        q: NDArray,
        I: NDArray,
        weights: NDArray,
        ring_params: list[RingParams],
        q_range: tuple[float, float],
    ) -> Optional[NDArray]:
        """Fit Gaussian amplitudes in a single angular patch.

        Ring positions (qᵢ) and widths (σᵢ) are fixed from the global fit;
        only amplitudes Aᵢ are free.  Returns amplitude array or None on
        failure.
        """
        # Weighted binning
        n_bins = max(30, self.n_radial_bins // 4)
        q_edges = np.linspace(q_range[0], q_range[1], n_bins + 1)
        q_c = 0.5 * (q_edges[:-1] + q_edges[1:])
        bin_idx = np.clip(np.digitize(q, q_edges) - 1, 0, n_bins - 1)

        profile = np.zeros(n_bins)
        w_sum = np.zeros(n_bins)
        for b in range(n_bins):
            m = bin_idx == b
            if m.sum() > 0:
                profile[b] = np.average(I[m], weights=weights[m])
                w_sum[b] = weights[m].sum()

        valid = w_sum > 0
        if valid.sum() < len(ring_params) * 3:
            return None

        # Linear least-squares: profile ≈ Σᵢ Aᵢ × G(q_c - qᵢ, σᵢ)
        # Build design matrix
        G = np.column_stack([
            _gaussian(q_c[valid], 1.0, r.q_center, r.q_sigma)
            for r in ring_params
        ])
        # Non-negative least squares
        try:
            from scipy.optimize import nnls
            amps, _ = nnls(G, profile[valid])
        except Exception:
            return None

        return amps


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
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Factorise A[n_rings × n_patches] ≈ ring_amps × texture^T.

    Uses the first singular vector pair.

    Returns
    -------
    texture : (n_patches,)  — T[P], normalised so mean = 1
    ring_amps : (n_rings,)  — Aᵢ
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    # First mode
    left = U[:, 0] * S[0]   # ring amplitudes direction (n_rings,)
    right = Vt[0, :]         # texture direction (n_patches,)

    # Ensure both left and right are positive (sign ambiguity)
    if left.mean() < 0:
        left, right = -left, -right

    # Normalise texture so its mean = 1; absorb scale into ring_amps
    t_mean = right.mean() if right.mean() != 0 else 1.0
    texture = right / t_mean
    ring_amps = left * t_mean

    return texture, ring_amps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian(x: NDArray, amp: float, x0: float, sigma: float) -> NDArray:
    return amp * np.exp(-0.5 * ((x - x0) / (sigma + 1e-12)) ** 2)


def _angular_distance(phi: NDArray, phi_c: float) -> NDArray:
    """Signed angular distance in (-π, π], accounting for wrap-around."""
    d = phi - phi_c
    return (d + np.pi) % (2 * np.pi) - np.pi
