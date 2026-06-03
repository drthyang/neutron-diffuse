"""Powder ring detection and masking in 1D |Q| space.

Design principles
-----------------
* Work entirely in |Q| (the ring is a feature of the radial intensity profile).
* Make **no assumption** about the diffuse signal — only assume it varies
  slowly in |Q| compared to a ring peak.
* The baseline is estimated by a rolling median of the radial profile.
  The median is robust to narrow peaks (ring contributions) as long as the
  window is wider than the ring (~5–10× ring FWHM is typical).
* Detection returns |Q| ranges; masking applies a soft (sigmoid-tapered)
  boundary so that the transition into the masked shell is C¹.
* Filling of the masked shell is handled separately in backfill.py.

Note on aluminium
-----------------
Al (FCC, Fm-3m, a ≈ 4.046 Å) is the most common source of powder rings.
``al_ring_q_positions()`` returns its known peak |Q| values and can be used
to cross-check or seed the detection, but the algorithm is material-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from ndiff.core import HKLVolume


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class RingShell:
    """A detected or user-specified powder ring shell.

    Attributes
    ----------
    q_center : float
        Ring peak position in Å^-1.
    q_lo, q_hi : float
        |Q| range to mask (where the ring excess is significant).
    amplitude : float
        Peak excess above the rolling-median baseline (diagnostic).
    """
    q_center: float
    q_lo: float
    q_hi: float
    amplitude: float = 0.0

    @property
    def q_halfwidth(self) -> float:
        return 0.5 * (self.q_hi - self.q_lo)


@dataclass
class RingProfile:
    """Radial shape of one powder ring fit on a (Bragg-free) linecut.

    Attributes
    ----------
    q_center : float   Ring peak |Q| (Å⁻¹).
    sigma    : float   Gaussian σ in |Q| (Å⁻¹).
    amplitude: float   Peak height above the local baseline.
    baseline : float   Local diffuse level under the ring.
    """
    q_center: float
    sigma: float
    amplitude: float
    baseline: float = 0.0

    @property
    def fwhm(self) -> float:
        return 2.3548 * self.sigma


# ---------------------------------------------------------------------------
# Known material helper
# ---------------------------------------------------------------------------

def al_ring_q_positions(a: float = 4.0494, q_max: float = 10.0) -> list[float]:
    """Return |Q| positions (Å^-1) of Al powder rings up to *q_max*.

    Al is FCC (Fm-3m): allowed reflections have h, k, l all-even or all-odd.
    Useful for cross-checking automatically detected rings.
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
# 1D radial profile
# ---------------------------------------------------------------------------

def radial_profile(
    vol: HKLVolume,
    n_bins: int = 500,
    min_q: float = 0.3,
    stat: str = "mean",
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute the 1D radial intensity profile.

    Parameters
    ----------
    vol : HKLVolume
    n_bins : int
        Number of |Q| bins.
    min_q : float
        Ignore voxels below this |Q| (avoids origin artefacts).
    stat : {"mean", "median"}
        Per-bin statistic.  ``"mean"`` is faster; ``"median"`` is more
        robust to remaining Bragg tails.

    Returns
    -------
    q_centres : (n_bins,)
    profile   : (n_bins,)  — NaN where a bin is empty
    counts    : (n_bins,)  — number of valid voxels per bin
    """
    q_mag = vol.q_magnitude()
    valid = vol.mask & (q_mag >= min_q)

    q_flat = q_mag[valid]
    I_flat = vol.data[valid]

    q_edges = np.linspace(q_flat.min(), q_flat.max(), n_bins + 1)
    q_centres = 0.5 * (q_edges[:-1] + q_edges[1:])

    bin_idx = np.clip(np.digitize(q_flat, q_edges) - 1, 0, n_bins - 1)

    profile = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask_b = bin_idx == b
        counts[b] = int(mask_b.sum())
        if counts[b] < 3:
            continue
        vals = I_flat[mask_b]
        if stat == "median":
            profile[b] = float(np.median(vals))
        else:
            profile[b] = float(vals.mean())

    return q_centres, profile, counts


def line_profile(
    vol: HKLVolume,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    n_points: int = 600,
) -> tuple[NDArray, NDArray, NDArray]:
    """Interpolate intensity along a straight line in (h, k, l) space.

    A linecut that threads *between* the crystal Bragg peaks (e.g. along
    ``(0, ±1, l)`` when the ``0kl`` reflections with odd k are systematically
    absent) gives a clean radial profile of the powder rings alone, with no
    Bragg contamination — ideal for reading off ring |Q| positions.

    Parameters
    ----------
    vol : HKLVolume
    start, end : (h, k, l)
        Endpoints of the line in reciprocal-lattice units.
    n_points : int
        Number of samples along the line.

    Returns
    -------
    q_mag : (n_points,)
        |Q| (Å⁻¹) at each sample.
    intensity : (n_points,)
        Trilinearly interpolated intensity; NaN where the line leaves the
        measured (masked) region.
    hkl : (n_points, 3)
        The sampled (h, k, l) coordinates.
    """
    from scipy.interpolate import RegularGridInterpolator

    hkl = np.linspace(np.asarray(start, float), np.asarray(end, float), n_points)
    interp = RegularGridInterpolator(
        (vol.h_axis, vol.k_axis, vol.l_axis), vol.masked_data(),
        bounds_error=False, fill_value=np.nan,
    )
    intensity = interp(hkl)
    q_mag = np.linalg.norm(hkl @ vol.ub_matrix.T, axis=1)
    return q_mag, intensity, hkl


def fit_ring_profiles(
    q: NDArray,
    intensity: NDArray,
    centers: Optional[list[float]] = None,
    prominence: float = 0.04,
    min_distance: int = 8,
    cluster_gap: float = 0.3,
    half_window: float = 0.22,
    sigma0: float = 0.04,
) -> list["RingProfile"]:
    """Fit each powder ring's radial shape (center, σ, amplitude, baseline).

    Intended for a *Bragg-free* radial profile (see :func:`line_profile` along a
    systematically-absent line such as ``(0, ±1, l)``), where each ring is a
    clean peak.  Overlapping rings are fit jointly as a sum of Gaussians on a
    shared linear baseline.

    Parameters
    ----------
    q, intensity : 1D arrays
        Radial |Q| (Å⁻¹) and intensity of the linecut (NaNs allowed).
    centers : list of float, optional
        Ring |Q| positions.  If None, detected via :func:`scipy.signal.find_peaks`
        (``prominence``, ``min_distance``).
    cluster_gap : float
        Rings closer than this (Å⁻¹) are fit jointly.
    half_window : float
        Fit window padding on each side of a cluster (Å⁻¹).
    sigma0 : float
        Initial Gaussian σ guess (Å⁻¹).

    Returns
    -------
    list[RingProfile]  (sorted by q_center)
    """
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks

    q = np.asarray(q, float)
    intensity = np.asarray(intensity, float)
    finite = np.isfinite(intensity)

    if centers is None:
        peaks, _ = find_peaks(np.where(finite, intensity, 0.0),
                              prominence=prominence, distance=min_distance)
        centers = list(q[peaks])
    centers = sorted(centers)
    if not centers:
        return []

    # Cluster nearby centers for joint fitting.
    clusters: list[list[float]] = [[centers[0]]]
    for c in centers[1:]:
        if c - clusters[-1][-1] < cluster_gap:
            clusters[-1].append(c)
        else:
            clusters.append([c])

    def _model(x: NDArray, *p: float) -> NDArray:
        n = (len(p) - 2) // 3
        x0_ref = x.mean()
        y = p[-2] + p[-1] * (x - x0_ref)
        for i in range(n):
            amp, x0, sig = p[3 * i:3 * i + 3]
            y = y + amp * np.exp(-0.5 * ((x - x0) / sig) ** 2)
        return y

    out: list[RingProfile] = []
    for cl in clusters:
        lo, hi = min(cl) - half_window, max(cl) + half_window
        sel = finite & (q >= lo) & (q <= hi)
        if int(sel.sum()) < 3 * len(cl) + 2:
            continue
        qq, ii = q[sel], intensity[sel]
        base0 = float(np.percentile(ii, 20))
        p0: list[float] = []
        lb: list[float] = []
        ub: list[float] = []
        for c in cl:
            p0 += [max(1e-3, float(ii.max()) - base0), float(c), sigma0]
            lb += [0.0, c - 0.1, 0.005]
            ub += [np.inf, c + 0.1, 0.2]
        p0 += [base0, 0.0]
        lb += [0.0, -np.inf]
        ub += [np.inf, np.inf]
        try:
            popt, _ = curve_fit(_model, qq, ii, p0=p0, bounds=(lb, ub), maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        baseline = float(popt[-2])
        for i in range(len(cl)):
            amp, x0, sig = popt[3 * i:3 * i + 3]
            out.append(RingProfile(q_center=float(x0), sigma=abs(float(sig)),
                                   amplitude=float(amp), baseline=baseline))
    out.sort(key=lambda r: r.q_center)
    return out


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_ring_shells(
    vol: HKLVolume,
    n_bins: int = 500,
    baseline_window: int = 50,
    sigma_threshold: float = 5.0,
    lower_threshold_fraction: float = 0.2,
    min_q: float = 0.3,
    min_peak_bins: int = 2,
    merge_gap_bins: int = 3,
    profile_stat: str = "mean",
) -> tuple[list[RingShell], NDArray, NDArray, NDArray]:
    """Detect powder ring shells from the 1D radial intensity profile.

    Algorithm
    ---------
    1. Compute the radial profile I(|Q|).
    2. Estimate baseline B(|Q|) = rolling median with window ``baseline_window``.
       The median is robust to ring peaks as long as the window is wider than
       the ring.  No assumption is made about the diffuse signal shape.
    3. Residual R = I − B.
    4. Noise level σ = MAD of R over ring-free bins.
    5. Detect bins where R > ``sigma_threshold`` × σ.
    6. Cluster contiguous detected bins → ring shells.
       The outer edge of each shell is where R falls below
       ``lower_threshold_fraction`` × peak_residual.

    Parameters
    ----------
    baseline_window : int
        Rolling-median window size in bins.  Should be >> ring width in bins.
        For Al rings (~0.05–0.15 Å^-1 wide) and n_bins = 500 over 10 Å^-1
        range (bin size ≈ 0.02 Å^-1), a window of 30–60 bins is appropriate.
    sigma_threshold : float
        Detection threshold in units of residual noise.
    lower_threshold_fraction : float
        Ring edge is where residual drops below this fraction of the peak
        residual.  Controls how wide the masked shell is.
    merge_gap_bins : int
        Detected clusters separated by fewer than this many bins are merged.

    Returns
    -------
    rings : list[RingShell]
    q_centres, profile, baseline : 1D arrays for diagnostics / plotting
    """
    q_centres, profile, counts = radial_profile(vol, n_bins=n_bins, min_q=min_q, stat=profile_stat)

    good = np.isfinite(profile)
    if good.sum() < baseline_window:
        return [], q_centres, profile, np.full_like(profile, np.nan)

    # Rolling-median baseline — handles gaps with a fill-forward pass first
    profile_filled = _fill_nans(profile)
    baseline = median_filter(profile_filled, size=baseline_window, mode="nearest")
    baseline = np.where(good, baseline, np.nan)

    residual = np.where(good, profile - baseline, 0.0)

    # Noise: MAD of residual (ring-free bins dominate the MAD)
    noise = float(np.median(np.abs(residual[good]))) + 1e-12

    # Detect bins above threshold
    above_high = residual > sigma_threshold * noise

    # Cluster contiguous detections, with gap merging
    labels = _cluster_with_merge(above_high, merge_gap_bins)

    rings: list[RingShell] = []
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        cluster_bins = np.where(labels == lbl)[0]
        if len(cluster_bins) < min_peak_bins:
            continue

        peak_bin = cluster_bins[np.argmax(residual[cluster_bins])]
        peak_amp = float(residual[peak_bin])
        lower = lower_threshold_fraction * peak_amp

        # Expand shell boundary outward until residual drops below lower threshold
        lo = int(cluster_bins[0])
        while lo > 0 and residual[lo - 1] > lower:
            lo -= 1

        hi = int(cluster_bins[-1])
        while hi < n_bins - 1 and residual[hi + 1] > lower:
            hi += 1

        rings.append(RingShell(
            q_center=float(q_centres[peak_bin]),
            q_lo=float(q_centres[lo]),
            q_hi=float(q_centres[hi]),
            amplitude=peak_amp,
        ))

    return rings, q_centres, profile, baseline


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def mask_ring_shells(
    vol: HKLVolume,
    rings: list[RingShell],
    taper_width: float = 0.005,
) -> NDArray[np.bool_]:
    """Build a keep-mask (True = valid) for all detected ring shells.

    A sigmoid taper is applied at each shell boundary so that the mask
    transitions smoothly (C¹) rather than as a hard step.  This ensures
    the boundary between masked and unmasked voxels does not introduce a
    discontinuity that would appear as ringing in the 3D-ΔPDF.

    Parameters
    ----------
    vol : HKLVolume
    rings : list[RingShell]
    taper_width : float
        Controls the steepness of the mask edge in Å^-1.
        Smaller → sharper edge.  ~0.005–0.02 Å^-1 is typical.

    Returns
    -------
    keep : bool array, shape = vol.shape
    """
    q_mag = vol.q_magnitude()
    keep = vol.mask.copy()

    for ring in rings:
        q_lo, q_hi = ring.q_lo, ring.q_hi
        q_mid = 0.5 * (q_lo + q_hi)

        # Signed distance from the shell boundary (positive = outside shell)
        # Use the nearer edge for each voxel
        dist_lo = q_mag - q_lo        # positive outside (q < q_lo side)
        dist_hi = q_hi - q_mag        # positive outside (q > q_hi side)
        # Inside the shell: both dist_lo < 0 or dist_hi < 0
        # Outside the shell: the relevant dist is positive
        signed_dist = np.where(q_mag < q_mid, -dist_lo, -dist_hi)
        # signed_dist < 0 → inside shell (to be masked)

        weight = _sigmoid(signed_dist, taper_width)
        keep &= weight > 0.5

    return keep


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: NDArray, width: float) -> NDArray:
    """Smooth step: 0 at x << 0, 1 at x >> 0, transition width ≈ width."""
    return 1.0 / (1.0 + np.exp(-x / (width + 1e-12)))


def _fill_nans(arr: NDArray) -> NDArray:
    """Forward/backward fill of NaN values."""
    out = arr.copy()
    idx = np.arange(len(out))
    good = np.isfinite(out)
    if not good.any():
        return out
    out[~good] = np.interp(idx[~good], idx[good], out[good])
    return out


def _cluster_with_merge(mask: NDArray[np.bool_], gap: int) -> NDArray[np.int32]:
    """Label contiguous True-runs in *mask*, merging runs separated by <= gap."""
    labels = np.zeros(len(mask), dtype=np.int32)
    lbl = 0
    i = 0
    while i < len(mask):
        if mask[i]:
            lbl += 1
            j = i
            while j < len(mask):
                if mask[j]:
                    labels[j] = lbl
                    j += 1
                elif j + gap < len(mask) and mask[j + gap:j + gap + 1].any():
                    # gap region — merge
                    labels[j] = lbl
                    j += 1
                else:
                    break
            i = j
        else:
            i += 1
    return labels
