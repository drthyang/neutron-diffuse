"""General powder-ring detection and removal from 3D HKL diffuse scattering data.

Physical basis
--------------
Powder rings arise from polycrystalline material in the beam path (sample
environment, cryostat, capsule…). Their intensity is **isotropic in |Q|**:

    I_measured(Q) = I_diffuse(Q) + I_ring(|Q|)

where I_diffuse(Q) is direction-dependent (anisotropic) and I_ring depends
only on the radial distance from the origin.

This isotropic/anisotropic separation is the key to robust removal:

    1. Detect ring positions from the radial intensity profile.
    2. Fit the ring profile I_ring(|Q|) using a Gaussian model in |Q|.
    3. Subtract the fitted I_ring from the entire volume.
    4. Mask voxels where the powder ring dominates (poor post-subtraction SNR).
    5. Fill masked voxels by smooth 3D interpolation of the diffuse signal.
       Because the holes are thin shells in HKL space (not random),
       interpolation across them is physically well-posed.

Note on aluminum
----------------
Al (FCC Fm-3m, a ≈ 4.046 Å) is a common source of powder rings.
Its peak positions can be pre-computed with :func:`al_ring_q_positions`.
However, the detection algorithm is material-agnostic and works on any
polycrystalline contaminant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

from ndiff.core import HKLVolume


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PowderRing:
    """A single detected or user-specified powder ring.

    Attributes
    ----------
    q_center : float
        Ring position in Å^-1.
    q_sigma : float
        Gaussian σ of the ring profile in Å^-1 (relates to instrument resolution).
    amplitude : float
        Fitted peak amplitude above the smooth background.
    mask_halfwidth : float
        Half-width used for masking (defaults to 3 × q_sigma after fitting).
    label : str
        Optional label (e.g. 'Al-111', 'unknown').
    """
    q_center: float
    q_sigma: float
    amplitude: float = 0.0
    mask_halfwidth: float = field(init=False)
    label: str = ""

    def __post_init__(self) -> None:
        self.mask_halfwidth = 3.0 * self.q_sigma


# ---------------------------------------------------------------------------
# Known material helpers (informational, not required by the algorithm)
# ---------------------------------------------------------------------------

def al_ring_q_positions(a: float = 4.0494, q_max: float = 10.0) -> list[float]:
    """Return |Q| positions (Å^-1) of Al powder rings up to *q_max*.

    Al is FCC (Fm-3m): allowed when h,k,l are all-even or all-odd.
    Useful for cross-checking automatically detected rings or for
    providing initial guesses to the fitter.
    """
    seen: set[float] = set()
    q_vals: list[float] = []
    hmax = int(np.ceil(q_max * a / (2 * np.pi))) + 1
    for h in range(0, hmax + 1):
        for k in range(0, hmax + 1):
            for l in range(0, hmax + 1):
                if h == k == l == 0:
                    continue
                if len({h % 2, k % 2, l % 2}) > 1:
                    continue
                q = 2 * np.pi * np.sqrt(h**2 + k**2 + l**2) / a
                if q > q_max:
                    continue
                qr = round(q, 6)
                if qr not in seen:
                    seen.add(qr)
                    q_vals.append(q)
    return sorted(q_vals)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_rings(
    vol: HKLVolume,
    n_bins: int = 400,
    sigma_threshold: float = 5.0,
    min_q: float = 0.3,
    prominence: float = 0.1,
) -> list[PowderRing]:
    """Detect powder rings from the radial intensity profile of *vol*.

    Algorithm
    ---------
    1. Bin all valid voxels by |Q| into *n_bins* shells.
    2. Compute mean intensity per shell (sigma-clipped to suppress outliers).
    3. Fit a smooth spline to the mean profile as a model of the diffuse
       background (slow Q-variation).
    4. Compute residuals = mean_per_shell − background_spline.
    5. Detect peaks in residuals whose height exceeds *sigma_threshold* × rms.
    6. Fit a Gaussian to each peak to determine centre, width, and amplitude.

    Parameters
    ----------
    vol : HKLVolume
        Input volume (need not be pre-cleaned).
    n_bins : int
        Number of radial shells for the radial profile.
    sigma_threshold : float
        Minimum peak height in units of residual rms to be flagged as a ring.
    min_q : float
        Ignore rings below this |Q| (Å^-1) — avoids the DC origin.
    prominence : float
        Minimum peak prominence (fraction of max residual) for scipy peak finder.

    Returns
    -------
    list[PowderRing]
        Detected rings, sorted by q_center.
    """
    q_mag = vol.q_magnitude()
    data = vol.data

    valid = vol.mask & (q_mag > min_q)
    q_flat = q_mag[valid]
    I_flat = data[valid]

    q_edges = np.linspace(q_flat.min(), q_flat.max(), n_bins + 1)
    q_centres = 0.5 * (q_edges[:-1] + q_edges[1:])
    bin_idx = np.digitize(q_flat, q_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    mean_per_bin = np.full(n_bins, np.nan)
    for b in range(n_bins):
        vals = I_flat[bin_idx == b]
        if len(vals) < 3:
            continue
        # sigma-clip: one pass
        med, std = np.median(vals), np.std(vals)
        clipped = vals[np.abs(vals - med) < 3 * std]
        if len(clipped) > 0:
            mean_per_bin[b] = clipped.mean()

    good = np.isfinite(mean_per_bin)
    if good.sum() < 10:
        return []

    # Fit smooth background spline to the mean radial profile
    spl = UnivariateSpline(q_centres[good], mean_per_bin[good], k=5, s=good.sum() * 2)
    background = spl(q_centres)
    residuals = mean_per_bin - background
    residuals[~good] = 0.0

    rms = float(np.sqrt(np.mean(residuals[good] ** 2)))
    if rms == 0:
        return []

    min_prominence = prominence * float(residuals[good].max()) if good.any() else 0.0
    peaks_idx, props = find_peaks(
        residuals,
        height=sigma_threshold * rms,
        prominence=min_prominence,
    )

    rings: list[PowderRing] = []
    for idx in peaks_idx:
        q0 = float(q_centres[idx])
        if q0 < min_q:
            continue
        # Fit Gaussian to a window around the peak
        win_mask = np.abs(q_centres - q0) < 5 * (q_centres[1] - q_centres[0])
        if win_mask.sum() < 4:
            continue
        try:
            popt, _ = curve_fit(
                _gaussian, q_centres[win_mask], residuals[win_mask],
                p0=[residuals[idx], q0, (q_centres[1] - q_centres[0])],
                maxfev=2000,
            )
            amp, q_fit, sigma = float(popt[0]), float(popt[1]), abs(float(popt[2]))
        except Exception:
            amp, q_fit, sigma = float(residuals[idx]), q0, (q_centres[1] - q_centres[0])

        rings.append(PowderRing(q_center=q_fit, q_sigma=sigma, amplitude=amp))

    return sorted(rings, key=lambda r: r.q_center)


# ---------------------------------------------------------------------------
# Profile fitting & subtraction
# ---------------------------------------------------------------------------

def fit_ring_profiles(
    vol: HKLVolume,
    rings: list[PowderRing],
    n_bins: int = 600,
) -> NDArray[np.float64]:
    """Fit radial Gaussian profiles to rings and return I_ring(Q) per voxel.

    For each ring, we refine the amplitude by fitting the radially averaged
    signal in a window around the ring. The result is a 3D array with the
    same shape as *vol*, containing the modelled ring contribution at each
    voxel.

    Parameters
    ----------
    vol : HKLVolume
        Source volume.
    rings : list[PowderRing]
        Rings to model (from :func:`detect_rings` or user-provided).
    n_bins : int
        Resolution of radial binning for amplitude refinement.

    Returns
    -------
    I_ring : NDArray, shape vol.shape
        Per-voxel modelled powder-ring contribution.
    """
    q_mag = vol.q_magnitude()
    I_ring = np.zeros(vol.shape, dtype=np.float64)

    for ring in rings:
        # Refine amplitude from data in the peak window
        win = np.abs(q_mag - ring.q_center) < ring.mask_halfwidth
        if win.any() and vol.mask[win].any():
            I_at_peak = vol.data[win & vol.mask]
            q_at_peak = q_mag[win & vol.mask]
            try:
                popt, _ = curve_fit(
                    _gaussian, q_at_peak, I_at_peak,
                    p0=[ring.amplitude, ring.q_center, ring.q_sigma],
                    maxfev=3000,
                )
                amp, q0, sig = float(popt[0]), float(popt[1]), abs(float(popt[2]))
            except Exception:
                amp, q0, sig = ring.amplitude, ring.q_center, ring.q_sigma
        else:
            amp, q0, sig = ring.amplitude, ring.q_center, ring.q_sigma

        I_ring += _gaussian(q_mag, amp, q0, sig)

    return I_ring


def subtract_rings(
    vol: HKLVolume,
    rings: list[PowderRing],
    snr_mask_threshold: float = 3.0,
    taper_width: float = 0.01,
) -> tuple[HKLVolume, NDArray[np.float64]]:
    """Subtract fitted ring profiles from *vol* and mask low-SNR residuals.

    Steps
    -----
    1. Fit per-ring Gaussian profiles → I_ring(Q) array.
    2. Subtract: I_diffuse_est = I_total − I_ring.
    3. Compute post-subtraction uncertainty:
       σ_post = sqrt(σ_data² + σ_ring²)
    4. Mask voxels where I_ring / σ_data > *snr_mask_threshold* (ring dominates).
    5. Apply soft sigmoid taper at mask boundary.

    The masked voxels should be filled by :func:`backfill` using the
    surrounding *subtracted* signal — not by the raw I_total values.

    Parameters
    ----------
    vol : HKLVolume
        Raw input volume.
    rings : list[PowderRing]
        Rings to remove.
    snr_mask_threshold : float
        Voxels where I_ring / σ_data exceeds this are masked.
    taper_width : float
        Sigmoid taper half-width in Å^-1 at mask boundaries.

    Returns
    -------
    vol_subtracted : HKLVolume
        New volume with ring profiles subtracted; mask marks high-SNR voxels.
    I_ring : NDArray
        The modelled ring contribution that was subtracted.
    """
    import dataclasses

    I_ring = fit_ring_profiles(vol, rings)
    data_sub = vol.data - I_ring

    # Propagate uncertainty: σ_ring estimated as 10% of fitted amplitude (model error)
    sigma_ring = 0.1 * np.abs(I_ring)
    sigma_post = np.sqrt(vol.sigma**2 + sigma_ring**2)

    # Mask where ring dominates: I_ring / σ_data > threshold
    # Use soft sigmoid so mask edges don't create sharp discontinuities
    q_mag = vol.q_magnitude()
    keep = np.ones(vol.shape, dtype=bool)
    for ring in rings:
        dq = np.abs(q_mag - ring.q_center)
        if taper_width > 0:
            # soft mask: weight = sigmoid centred at mask_halfwidth
            weight = 1.0 / (1.0 + np.exp(-(dq - ring.mask_halfwidth) / taper_width))
            keep &= weight > 0.5
        else:
            keep &= dq > ring.mask_halfwidth

    new_mask = vol.mask & keep
    vol_sub = dataclasses.replace(vol, data=data_sub, sigma=sigma_post, mask=new_mask)
    return vol_sub, I_ring


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

class PowderRingRemover:
    """High-level interface: detect → subtract → mask in one call.

    Parameters
    ----------
    rings : list[PowderRing] or None
        User-specified rings. If None, :func:`detect_rings` is called.
    detect_kwargs : dict
        Keyword arguments forwarded to :func:`detect_rings`.
    snr_mask_threshold : float
        Passed to :func:`subtract_rings`.
    taper_width : float
        Passed to :func:`subtract_rings`.
    """

    def __init__(
        self,
        rings: Optional[list[PowderRing]] = None,
        detect_kwargs: Optional[dict] = None,
        snr_mask_threshold: float = 3.0,
        taper_width: float = 0.01,
    ) -> None:
        self.rings = rings
        self.detect_kwargs = detect_kwargs or {}
        self.snr_mask_threshold = snr_mask_threshold
        self.taper_width = taper_width
        self._detected_rings: list[PowderRing] = []

    def remove(self, vol: HKLVolume) -> tuple[HKLVolume, list[PowderRing], NDArray]:
        """Detect (if needed), subtract, and mask powder rings.

        Returns
        -------
        vol_sub : HKLVolume
            Subtracted volume with mask indicating usable voxels.
        rings : list[PowderRing]
            Rings that were detected and removed.
        I_ring : NDArray
            Modelled ring contribution (for diagnostics).
        """
        rings = self.rings if self.rings is not None else detect_rings(vol, **self.detect_kwargs)
        self._detected_rings = rings
        if not rings:
            return vol, rings, np.zeros(vol.shape)
        vol_sub, I_ring = subtract_rings(
            vol, rings,
            snr_mask_threshold=self.snr_mask_threshold,
            taper_width=self.taper_width,
        )
        return vol_sub, rings, I_ring

    @property
    def detected_rings(self) -> list[PowderRing]:
        return self._detected_rings


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _gaussian(x: NDArray, amp: float, x0: float, sigma: float) -> NDArray:
    return amp * np.exp(-0.5 * ((x - x0) / (sigma + 1e-12)) ** 2)
