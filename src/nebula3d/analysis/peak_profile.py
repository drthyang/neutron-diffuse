"""Peak-shape decomposition for Bragg / diffuse separation.

When diffuse scattering sits *at* a Bragg position (e.g. magnetic diffuse around
the magnetic satellites), the punch-and-backfill workflow in
:mod:`nebula3d.analysis.bragg` / :mod:`nebula3d.analysis.bragg_fill` destroys exactly
the signal of interest.  The two contributions can instead be separated by their
**width**: a true Bragg reflection is resolution-limited (a fixed instrument
width), while spin-correlation diffuse is broader (finite correlation length
ξ → measurable wings).

This module provides the *diagnostic* primitives that measure that width contrast
on real line cuts, before any subtraction is attempted:

- line-shape models (:func:`gaussian`, :func:`lorentzian`,
  :func:`squared_lorentzian`, :func:`pseudo_voigt`),
- a single-Gaussian fit (:func:`fit_single_gaussian`) used to **calibrate the
  instrument resolution** on resolution-limited nuclear Bragg peaks,
- a two-component sharp+broad fit (:func:`fit_two_component`) that, with the
  sharp width fixed to the calibrated resolution, extracts the broad diffuse
  component and its correlation length,
- :func:`extract_orthogonal_cuts` / :func:`decompose_peak` that drive the fit on
  the three orthogonal cuts through a peak, and
- :func:`calibrate_resolution` / :func:`magnetic_satellite_centers` helpers that
  find the reference and target peaks.

The fit machinery reuses the ``curve_fit`` model/p0/bounds pattern from
:func:`nebula3d.preprocessing.powder_rings.fit_ring_profiles`; peak finding reuses
:class:`nebula3d.analysis.bragg.BraggRemover`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from nebula3d.analysis.bragg import BraggRemover
from nebula3d.core import HKLVolume

if TYPE_CHECKING:
    from scipy.interpolate import RegularGridInterpolator

_GAUSS_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))           # σ → FWHM
_SQLOR_HWHM = np.sqrt(np.sqrt(2.0) - 1.0)                 # γ → HWHM (squared Lor.)
AXIS_NAMES = ("H", "K", "L")


# ---------------------------------------------------------------------------
# Line-shape models (unit peak height, centred at x0)
# ---------------------------------------------------------------------------

def gaussian(x: NDArray, x0: float, sigma: float) -> NDArray:
    """Unit-height Gaussian; FWHM = 2.3548·σ."""
    sigma = max(abs(sigma), 1e-12)
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def lorentzian(x: NDArray, x0: float, gamma: float) -> NDArray:
    """Unit-height Lorentzian; γ = HWHM, FWHM = 2·γ."""
    gamma = max(abs(gamma), 1e-12)
    return gamma**2 / ((x - x0) ** 2 + gamma**2)


def squared_lorentzian(x: NDArray, x0: float, gamma: float) -> NDArray:
    """Unit-height squared Lorentzian; HWHM = γ·√(√2−1)."""
    gamma = max(abs(gamma), 1e-12)
    return (gamma**2 / ((x - x0) ** 2 + gamma**2)) ** 2


def pseudo_voigt(x: NDArray, x0: float, sigma: float, eta: float) -> NDArray:
    """Unit-height pseudo-Voigt: η·Lorentzian + (1−η)·Gaussian (matched FWHM)."""
    eta = float(np.clip(eta, 0.0, 1.0))
    fwhm = _GAUSS_FWHM * abs(sigma)
    return eta * lorentzian(x, x0, 0.5 * fwhm) + (1.0 - eta) * gaussian(x, x0, sigma)


# Broad-component shapes selectable by name, with their unit-height integral and
# HWHM expressed as multiples of the width parameter γ.
_BROAD_SHAPES = {
    "lorentzian": (lorentzian, np.pi, 1.0),            # ∫ = π·γ, HWHM = γ
    "squared_lorentzian": (squared_lorentzian, 0.5 * np.pi, _SQLOR_HWHM),
}


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class PeakDecomposition:
    """Sharp (Bragg) + broad (diffuse) decomposition of one 1D line cut.

    All widths are in the cut's own axis units (fractional rlu) unless noted.
    Integrated intensities are ∫(component) d(rlu) along the cut, so
    ``diffuse_fraction`` is the broad share of the line-cut area.
    """

    axis: str
    center: float                       # fitted peak centre (rlu, along axis)
    q_center: float                     # |Q| at the peak (Å⁻¹)
    sharp_amp: float
    sharp_sigma: float                  # Gaussian σ (rlu)
    broad_amp: float
    broad_gamma: float                  # broad width parameter γ (rlu)
    broad_shape: str
    baseline: float
    slope: float
    x_ref: float                        # reference point for the linear baseline
    sharp_integral: float
    broad_integral: float
    diffuse_fraction: float
    sharp_fwhm: float                   # rlu
    broad_fwhm: float                   # rlu
    xi_angstrom: float                  # correlation length 1/κ (Å) from broad HWHM
    points_across_fwhm: float           # sharp FWHM / axis step — resolved if ≳3–4
    rms_residual: float
    aic: float
    n_points: int
    success: bool = True

    @property
    def is_diffuse(self) -> bool:
        """True when a broad component is meaningfully present and physical.

        Conservative on purpose: the sharp/broad split is partly degenerate (a
        Lorentzian can mimic some of a Gaussian core, or soak up background
        curvature near an intense peak), so a co-located diffuse component is only
        flagged when it (i) carries a substantial integrated share, (ii) is
        distinctly broader than the resolution core, and (iii) has a physical
        correlation length (ξ ≳ 2 Å — anything shorter is a background-like wing,
        not real short-range order).
        """
        return (
            self.success
            and self.diffuse_fraction > 0.2
            and self.broad_fwhm > 2.0 * self.sharp_fwhm
            and self.xi_angstrom > 2.0
        )


@dataclass
class Resolution:
    """Per-axis instrument resolution σ(|Q|) calibrated on nuclear Bragg peaks.

    ``sigma(axis, q)`` returns the resolution-limited Gaussian σ (rlu) for an
    axis (0=H, 1=K, 2=L) at scattering-vector magnitude ``q`` (Å⁻¹), from a
    linear regression σ = intercept + slope·|Q| over the reference peaks.  Axes
    with too few references fall back to a constant median σ.
    """

    slope: NDArray[np.float64]          # (3,)
    intercept: NDArray[np.float64]      # (3,)
    q_ref: list[NDArray[np.float64]] = field(default_factory=list)
    sigma_ref: list[NDArray[np.float64]] = field(default_factory=list)
    n_ref: tuple[int, int, int] = (0, 0, 0)

    def sigma(self, axis: int, q: float) -> float:
        s = float(self.intercept[axis] + self.slope[axis] * float(q))
        return max(s, 1e-6)

    def fwhm(self, axis: int, q: float) -> float:
        return _GAUSS_FWHM * self.sigma(axis, q)


def evaluate_components(
    dec: PeakDecomposition, x: NDArray
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Reconstruct ``(sharp, broad, baseline, total)`` curves of a fit on grid x."""
    base = dec.baseline + dec.slope * (x - dec.x_ref)
    sharp = dec.sharp_amp * gaussian(x, dec.center, dec.sharp_sigma)
    fn = {"lorentzian": lorentzian, "squared_lorentzian": squared_lorentzian}.get(
        dec.broad_shape
    )
    broad = (dec.broad_amp * fn(x, dec.center, dec.broad_gamma)
             if fn is not None else np.zeros_like(x))
    return sharp, broad, base, base + sharp + broad


# ---------------------------------------------------------------------------
# Axis / |Q| helpers
# ---------------------------------------------------------------------------

def _axis_arrays(vol: HKLVolume) -> tuple[NDArray, NDArray, NDArray]:
    return vol.h_axis, vol.k_axis, vol.l_axis


def _axis_step(vol: HKLVolume, axis: int) -> float:
    a = _axis_arrays(vol)[axis]
    return float(abs(a[1] - a[0])) if a.size > 1 else 1.0


def _axis_q_scale(vol: HKLVolume, axis: int) -> float:
    """Å⁻¹ per unit rlu along a pure ``axis`` step: |UB column|."""
    return float(np.linalg.norm(vol.ub_matrix[:, axis]))


def _q_at(vol: HKLVolume, hkl: tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.asarray(hkl, float) @ vol.ub_matrix.T))


def _nearest_index(axis: NDArray, value: float) -> int:
    return int(np.argmin(np.abs(axis - value)))


def refine_center(
    vol: HKLVolume,
    hkl: tuple[float, float, float],
    window_hkl: float = 0.15,
) -> tuple[float, float, float]:
    """Snap an approximate (h,k,l) onto the local intensity maximum."""
    axes = _axis_arrays(vol)
    idx = [_nearest_index(axes[a], hkl[a]) for a in range(3)]
    half = [max(1, int(round(window_hkl / _axis_step(vol, a)))) for a in range(3)]
    sl = tuple(
        slice(max(0, idx[a] - half[a]), min(vol.shape[a], idx[a] + half[a] + 1))
        for a in range(3)
    )
    win = vol.data[sl]
    valid = vol.mask[sl] & np.isfinite(win)
    if not valid.any():
        return tuple(float(axes[a][idx[a]]) for a in range(3))  # type: ignore[return-value]
    wv = np.where(valid, win, -np.inf)
    off = np.unravel_index(int(np.argmax(wv)), wv.shape)
    return tuple(
        float(axes[a][sl[a].start + int(off[a])]) for a in range(3)
    )  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Line-cut extraction
# ---------------------------------------------------------------------------

def build_interpolator(vol: HKLVolume) -> RegularGridInterpolator:
    """Trilinear interpolator over the masked volume (masked voxels → NaN).

    Build this **once per volume** and pass it to :func:`extract_orthogonal_cuts`
    / :func:`decompose_peak` / :func:`calibrate_resolution`; otherwise each line
    cut re-copies the whole (~0.5 GB for 401³) volume.
    """
    from scipy.interpolate import RegularGridInterpolator

    return RegularGridInterpolator(
        (vol.h_axis, vol.k_axis, vol.l_axis), vol.masked_data(),
        bounds_error=False, fill_value=np.nan,
    )


def extract_orthogonal_cuts(
    vol: HKLVolume,
    center_hkl: tuple[float, float, float],
    half_window: float | tuple[float, float, float] = 0.6,
    n_points: int = 161,
    interp: RegularGridInterpolator | None = None,
) -> dict[int, tuple[NDArray, NDArray]]:
    """Three orthogonal line cuts through ``center_hkl``.

    Returns ``{axis: (coord_rlu, intensity)}`` for axis 0=H, 1=K, 2=L, where
    ``coord_rlu`` is the value along that axis (the other two held fixed at the
    centre).  Pass a shared ``interp`` from :func:`build_interpolator` to avoid
    re-copying the volume per cut.
    """
    if interp is None:
        interp = build_interpolator(vol)
    if isinstance(half_window, (int, float)):
        hw = (float(half_window), float(half_window), float(half_window))
    else:
        hw = (float(half_window[0]), float(half_window[1]), float(half_window[2]))
    cuts: dict[int, tuple[NDArray, NDArray]] = {}
    for axis in range(3):
        coord = np.linspace(
            center_hkl[axis] - hw[axis],
            center_hkl[axis] + hw[axis],
            n_points,
        )
        hkl = np.tile(np.asarray(center_hkl, float), (n_points, 1))
        hkl[:, axis] = coord
        cuts[axis] = (coord, interp(hkl))
    return cuts


# ---------------------------------------------------------------------------
# 1D fits
# ---------------------------------------------------------------------------

def _aic(residual: NDArray, n_params: int) -> float:
    n = int(residual.size)
    if n <= n_params + 1:
        return np.inf
    rss = float(np.sum(residual**2))
    if rss <= 0:
        return -np.inf
    return n * np.log(rss / n) + 2 * n_params


def fit_single_gaussian(
    x: NDArray,
    y: NDArray,
    *,
    sigma0: float | None = None,
) -> tuple[float, float, float, float, float] | None:
    """Fit ``bg0 + bg1·(x−x̄) + A·G(x;x0,σ)``; return (x0, A, σ, bg0, bg1) or None.

    Used to calibrate the resolution-limited width on nuclear Bragg peaks (a
    single resolution Gaussian on a sloped baseline).
    """
    from scipy.optimize import curve_fit

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 6:
        return None
    xref = float(x.mean())
    span = float(x.max() - x.min()) or 1.0
    x0_guess = float(x[int(np.argmax(y))])
    base0 = float(np.percentile(y, 20))
    amp0 = max(float(y.max()) - base0, 1e-6)
    sig0 = float(sigma0) if sigma0 else 0.05 * span

    def model(xx: NDArray, x0: float, amp: float, sig: float, b0: float, b1: float) -> NDArray:
        return b0 + b1 * (xx - xref) + amp * gaussian(xx, x0, sig)

    p0 = [x0_guess, amp0, sig0, base0, 0.0]
    lb = [x.min(), 0.0, 1e-3 * span, -np.inf, -np.inf]
    ub = [x.max(), np.inf, span, np.inf, np.inf]
    try:
        popt, _ = curve_fit(model, x, y, p0=p0, bounds=(lb, ub), maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    return float(popt[0]), float(popt[1]), abs(float(popt[2])), float(popt[3]), float(popt[4])


def fit_two_component(
    x: NDArray,
    y: NDArray,
    *,
    axis: int = 0,
    sharp_sigma: float | None = None,
    q_center: float = 0.0,
    q_scale: float = 1.0,
    axis_step: float = 1.0,
    broad: str | None = None,
) -> PeakDecomposition:
    """Decompose a line cut into a sharp Gaussian core + a broad diffuse wing.

    The model is ``bg0 + bg1·(x−x̄) + A_s·G(x;x0,σ_s) + A_b·B(x;x0,γ)`` with a
    shared centre ``x0``, ``A_s, A_b ≥ 0`` and ``γ ≥ σ_s`` (the broad component
    must be broader than the sharp core).  When ``sharp_sigma`` is given the core
    width is fixed to the calibrated resolution; otherwise it is a free parameter
    bounded above by ``γ``.

    ``broad`` selects the diffuse line shape (``"lorentzian"`` or
    ``"squared_lorentzian"``); ``None`` fits both and keeps the lower-AIC one.
    """
    from scipy.optimize import curve_fit

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]

    failed = PeakDecomposition(
        axis=AXIS_NAMES[axis], center=float("nan"), q_center=q_center,
        sharp_amp=0.0, sharp_sigma=float(sharp_sigma or 0.0), broad_amp=0.0,
        broad_gamma=0.0, broad_shape="", baseline=0.0, slope=0.0, x_ref=float("nan"),
        sharp_integral=0.0, broad_integral=0.0, diffuse_fraction=0.0,
        sharp_fwhm=0.0, broad_fwhm=0.0, xi_angstrom=float("nan"),
        points_across_fwhm=0.0, rms_residual=float("nan"), aic=np.inf,
        n_points=int(x.size), success=False,
    )
    if x.size < 8:
        return failed

    if broad is None:
        candidates = [
            fit_two_component(
                x, y, axis=axis, sharp_sigma=sharp_sigma, q_center=q_center,
                q_scale=q_scale, axis_step=axis_step, broad=name,
            )
            for name in _BROAD_SHAPES
        ]
        ok = [c for c in candidates if c.success]
        return min(ok, key=lambda c: c.aic) if ok else failed

    broad_fn, broad_area_k, broad_hwhm_k = _BROAD_SHAPES[broad]
    xref = float(x.mean())
    span = float(x.max() - x.min()) or 1.0
    x0_guess = float(x[int(np.argmax(y))])
    base0 = float(np.percentile(y, 20))
    peak0 = max(float(y.max()) - base0, 1e-6)
    sig_fixed = float(sharp_sigma) if sharp_sigma else None
    sig_seed = sig_fixed if sig_fixed is not None else 0.04 * span
    gam_seed = max(3.0 * sig_seed, 0.1 * span)

    # One model with a single, fixed signature.  When the resolution width is
    # supplied, σ is pinned by tight bounds rather than dropped from the parameter
    # vector — this keeps one signature for both the fitter and the type checker.
    def model(xx: NDArray, x0: float, a_s: float, sig: float, a_b: float,
              gam: float, b0: float, b1: float) -> NDArray:
        return (b0 + b1 * (xx - xref)
                + a_s * gaussian(xx, x0, sig)
                + a_b * broad_fn(xx, x0, gam))

    if sig_fixed is not None:
        sig_lo, sig_hi = sig_fixed * (1.0 - 1e-6), sig_fixed * (1.0 + 1e-6)
        gam_lo = sig_fixed                 # the diffuse must be broader than the core
    else:
        sig_lo, sig_hi = 1e-3 * span, span
        gam_lo = 1e-3 * span
    p0 = [x0_guess, 0.6 * peak0, sig_seed, 0.4 * peak0, gam_seed, base0, 0.0]
    lb = [x.min(), 0.0, sig_lo, 0.0, gam_lo, -np.inf, -np.inf]
    ub = [x.max(), np.inf, sig_hi, np.inf, span, np.inf, np.inf]

    try:
        popt, _ = curve_fit(model, x, y, p0=p0, bounds=(lb, ub), maxfev=30000)
    except (RuntimeError, ValueError):
        return failed

    x0, a_s, sig, a_b, gam, b0, b1 = (float(v) for v in popt)
    sig, gam = abs(sig), abs(gam)

    resid = y - model(x, *popt)
    n_params = len(popt)
    sharp_int = a_s * sig * np.sqrt(2.0 * np.pi)
    broad_int = a_b * broad_area_k * gam
    total = sharp_int + broad_int
    diffuse_frac = float(broad_int / total) if total > 0 else 0.0
    broad_hwhm_rlu = broad_hwhm_k * gam
    kappa_q = broad_hwhm_rlu * q_scale                        # Å⁻¹
    xi = float(1.0 / kappa_q) if kappa_q > 0 else float("inf")

    return PeakDecomposition(
        axis=AXIS_NAMES[axis], center=x0, q_center=q_center,
        sharp_amp=a_s, sharp_sigma=sig, broad_amp=a_b, broad_gamma=gam,
        broad_shape=broad, baseline=b0, slope=b1, x_ref=xref,
        sharp_integral=float(sharp_int), broad_integral=float(broad_int),
        diffuse_fraction=diffuse_frac, sharp_fwhm=_GAUSS_FWHM * sig,
        broad_fwhm=2.0 * broad_hwhm_rlu, xi_angstrom=xi,
        points_across_fwhm=float(_GAUSS_FWHM * sig / axis_step) if axis_step else 0.0,
        rms_residual=float(np.sqrt(np.mean(resid**2))), aic=_aic(resid, n_params),
        n_points=int(x.size), success=True,
    )


# ---------------------------------------------------------------------------
# Peak-level decomposition
# ---------------------------------------------------------------------------

def decompose_peak(
    vol: HKLVolume,
    center_hkl: tuple[float, float, float],
    resolution: Resolution | None = None,
    *,
    half_window: float | tuple[float, float, float] = 0.6,
    n_points: int = 161,
    broad: str | None = None,
    refine: bool = True,
    interp: RegularGridInterpolator | None = None,
) -> dict[int, PeakDecomposition]:
    """Two-component decomposition on the three orthogonal cuts through a peak.

    With ``resolution`` supplied the sharp core width on each axis is fixed to
    the calibrated instrument σ(|Q|); otherwise the core width is free.  Pass a
    shared ``interp`` from :func:`build_interpolator` to avoid re-copying the
    volume per peak.
    """
    center = refine_center(vol, center_hkl) if refine else center_hkl
    q_center = _q_at(vol, center)
    cuts = extract_orthogonal_cuts(
        vol, center, half_window=half_window, n_points=n_points, interp=interp)
    out: dict[int, PeakDecomposition] = {}
    for axis, (coord, intensity) in cuts.items():
        sharp_sigma = resolution.sigma(axis, q_center) if resolution else None
        out[axis] = fit_two_component(
            coord, intensity, axis=axis, sharp_sigma=sharp_sigma,
            q_center=q_center, q_scale=_axis_q_scale(vol, axis),
            axis_step=_axis_step(vol, axis), broad=broad,
        )
    return out


# ---------------------------------------------------------------------------
# Resolution calibration on nuclear Bragg peaks
# ---------------------------------------------------------------------------

def calibrate_resolution(
    vol: HKLVolume,
    nuclear_centers: list[tuple[float, float, float]] | None = None,
    *,
    min_intensity: float | None = None,
    max_peaks: int = 60,
    half_window: float = 0.5,
    n_points: int = 121,
    interp: RegularGridInterpolator | None = None,
) -> Resolution:
    """Calibrate per-axis resolution σ(|Q|) from resolution-limited nuclear Bragg.

    Nuclear (integer-node) Bragg peaks are the truest resolution-limited
    references.  Each is cut along H, K, L and fit with a single Gaussian
    (:func:`fit_single_gaussian`); the fitted σ vs |Q| is regressed linearly per
    axis.  Axes with too few clean fits fall back to a constant median σ.  Pass a
    shared ``interp`` from :func:`build_interpolator` to avoid re-copying the
    volume per peak.
    """
    if interp is None:
        interp = build_interpolator(vol)
    if nuclear_centers is None:
        thr = min_intensity
        if thr is None:
            valid = vol.mask & np.isfinite(vol.data)
            med = float(np.median(vol.data[valid])) if valid.any() else 0.0
            mad = float(np.median(np.abs(vol.data[valid] - med))) if valid.any() else 1.0
            thr = med + 20.0 * 1.4826 * max(mad, 1e-9)
        remover = BraggRemover(mode="integer", min_intensity=thr, min_prominence=thr * 0.1)
        peaks = remover.detect_peaks(vol)
        peaks.sort(key=lambda p: p[3], reverse=True)
        axes = _axis_arrays(vol)
        nuclear_centers = [
            (float(axes[0][p[0]]), float(axes[1][p[1]]), float(axes[2][p[2]]))
            for p in peaks[:max_peaks]
        ]

    q_pts: list[list[float]] = [[], [], []]
    s_pts: list[list[float]] = [[], [], []]
    for center in nuclear_centers:
        c = refine_center(vol, center)
        qc = _q_at(vol, c)
        cuts = extract_orthogonal_cuts(
            vol, c, half_window=half_window, n_points=n_points, interp=interp)
        for axis, (coord, intensity) in cuts.items():
            fit = fit_single_gaussian(coord, intensity)
            if fit is None:
                continue
            _, amp, sig, _, _ = fit
            step = _axis_step(vol, axis)
            # keep physically-sane resolution fits: resolved, not a spurious blob
            if amp <= 0 or sig < 0.25 * step or sig > 0.5:
                continue
            q_pts[axis].append(qc)
            s_pts[axis].append(sig)

    slope = np.zeros(3)
    intercept = np.zeros(3)
    q_ref: list[NDArray[np.float64]] = []
    sigma_ref: list[NDArray[np.float64]] = []
    n_ref = [0, 0, 0]
    for axis in range(3):
        qa = np.asarray(q_pts[axis], float)
        sa = np.asarray(s_pts[axis], float)
        q_ref.append(qa)
        sigma_ref.append(sa)
        n_ref[axis] = int(qa.size)
        if qa.size >= 4 and float(qa.max() - qa.min()) > 1e-3:
            b, a = np.polyfit(qa, sa, 1)
            slope[axis], intercept[axis] = float(b), float(a)
        elif qa.size >= 1:
            intercept[axis] = float(np.median(sa))
        else:
            intercept[axis] = max(0.5 * _axis_step(vol, axis), 1e-3)

    return Resolution(
        slope=slope, intercept=intercept, q_ref=q_ref, sigma_ref=sigma_ref,
        n_ref=(n_ref[0], n_ref[1], n_ref[2]),
    )


# ---------------------------------------------------------------------------
# Target finding: the q=1/3 magnetic-satellite family
# ---------------------------------------------------------------------------

def _fraction_h_mask(h_axis: NDArray, fractions: tuple[float, ...], half_width: float) -> NDArray:
    """Boolean over ``h_axis`` selecting H within ``half_width`` of any fraction.

    Mirror of :meth:`BraggRemover._search_excluded_h_mask`'s periodic logic, but
    used here to *select* (not exclude) the satellite planes.
    """
    frac = np.mod(h_axis, 1.0)
    sel = np.zeros(h_axis.shape, dtype=bool)
    for f in fractions:
        f0 = float(f) % 1.0
        d = np.abs(frac - f0)
        d = np.minimum(d, 1.0 - d)
        sel |= d <= half_width
    return sel


def magnetic_satellite_centers(
    vol: HKLVolume,
    *,
    fractions: tuple[float, ...] = (1.0 / 3.0, 2.0 / 3.0),
    h_half_width: float = 0.08,
    q_step: float = 0.05,
    n_mad: float = 8.0,
    min_intensity: float = 1.0,
    max_peaks: int = 40,
) -> list[tuple[float, float, float]]:
    """Locate strong peaks on the integer±fraction H-planes (magnetic satellites).

    Restricts the per-|Q|-shell outlier search of
    :class:`~nebula3d.analysis.bragg.BraggRemover` to the fractional-H planes, then
    returns the local-maximum centres sorted by intensity.
    """
    from scipy import ndimage

    valid = vol.mask & np.isfinite(vol.data)
    sel_h = _fraction_h_mask(vol.h_axis, fractions, h_half_width)
    valid &= sel_h[:, None, None]
    if not valid.any():
        return []

    _, bin_idx, thr = BraggRemover._q_shell_thresholds(
        vol, q_step=q_step, n_mad=n_mad, min_intensity=min_intensity,
    )
    cand = valid & (vol.data > thr[bin_idx])
    if not cand.any():
        return []
    scored = np.where(valid, vol.data, -np.inf)
    local_max = ndimage.maximum_filter(scored, size=3, mode="nearest")
    peaks = np.argwhere(cand & (scored >= local_max))
    order = np.argsort([vol.data[ih, ik, il] for ih, ik, il in peaks])[::-1]
    axes = _axis_arrays(vol)
    return [
        (float(axes[0][ih]), float(axes[1][ik]), float(axes[2][il]))
        for ih, ik, il in peaks[order][:max_peaks]
    ]
