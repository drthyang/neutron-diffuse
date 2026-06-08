"""Tests for Bragg/diffuse peak-shape decomposition (ndiff.analysis.peak_profile)."""

import numpy as np

from ndiff.analysis.peak_profile import (
    Resolution,
    calibrate_resolution,
    decompose_peak,
    fit_two_component,
    gaussian,
    lorentzian,
    magnetic_satellite_centers,
)
from ndiff.core import HKLVolume

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# 1D two-component fit
# ---------------------------------------------------------------------------

def _two_comp_cut(x, a_s, sig_s, a_b, gam_b, bg=1.0, slope=0.3, noise=0.0):
    y = bg + slope * (x - x.mean())
    y = y + a_s * gaussian(x, 0.0, sig_s) + a_b * lorentzian(x, 0.0, gam_b)
    if noise:
        y = y + RNG.normal(0.0, noise, x.shape)
    return y


def test_two_component_recovers_sharp_and_broad():
    x = np.linspace(-0.6, 0.6, 201)
    y = _two_comp_cut(x, a_s=10.0, sig_s=0.03, a_b=3.0, gam_b=0.15, noise=0.02)

    dec = fit_two_component(
        x, y, axis=2, sharp_sigma=0.03, broad="lorentzian",
        q_scale=1.0, axis_step=0.01,
    )
    assert dec.success
    # broad amplitude / width recovered to within ~25 %
    assert abs(dec.broad_amp - 3.0) / 3.0 < 0.25
    assert abs(dec.broad_gamma - 0.15) / 0.15 < 0.25
    # broad component is genuinely broader than the sharp core
    assert dec.broad_fwhm > 1.5 * dec.sharp_fwhm
    # analytic truth: broad area 3·π·0.15 / (3·π·0.15 + 10·0.03·√2π) ≈ 0.653
    assert 0.4 < dec.diffuse_fraction < 0.8
    # ξ = 1/HWHM(Å⁻¹); here q_scale=1 so ξ ≈ 1/0.15 ≈ 6.7 Å
    assert 4.0 < dec.xi_angstrom < 10.0
    assert dec.is_diffuse


def test_two_component_auto_selects_lorentzian():
    x = np.linspace(-0.6, 0.6, 201)
    y = _two_comp_cut(x, a_s=8.0, sig_s=0.03, a_b=4.0, gam_b=0.12, noise=0.01)
    dec = fit_two_component(x, y, axis=2, sharp_sigma=0.03, broad=None,
                            q_scale=1.0, axis_step=0.01)
    assert dec.success
    assert dec.broad_shape == "lorentzian"  # data is Lorentzian → lower AIC


def test_pure_sharp_has_no_diffuse():
    x = np.linspace(-0.6, 0.6, 201)
    y = _two_comp_cut(x, a_s=10.0, sig_s=0.03, a_b=0.0, gam_b=0.15, noise=0.005)
    dec = fit_two_component(x, y, axis=2, sharp_sigma=0.03, broad="lorentzian",
                            q_scale=1.0, axis_step=0.01)
    assert dec.success
    assert dec.diffuse_fraction < 0.2
    assert not dec.is_diffuse


def test_pure_broad_is_mostly_diffuse():
    x = np.linspace(-0.6, 0.6, 201)
    y = _two_comp_cut(x, a_s=0.0, sig_s=0.03, a_b=5.0, gam_b=0.15, noise=0.01)
    dec = fit_two_component(x, y, axis=2, sharp_sigma=0.03, broad="lorentzian",
                            q_scale=1.0, axis_step=0.01)
    assert dec.success
    assert dec.diffuse_fraction > 0.7
    assert dec.is_diffuse


# ---------------------------------------------------------------------------
# 3D peak decomposition
# ---------------------------------------------------------------------------

def _separable_peak_vol(shape, hkl_range, center, sig, gam, a_s, a_b, bg=1.0):
    """Volume with one sharp-Gaussian + broad-Lorentzian peak (separable per axis).

    Along any axis cut through ``center`` the other two factors are 1, so each
    orthogonal cut is exactly ``bg + a_s·G + a_b·L`` with the per-axis widths.
    """
    h = np.linspace(hkl_range[0], hkl_range[1], shape[0])
    k = np.linspace(hkl_range[0], hkl_range[1], shape[1])
    l = np.linspace(hkl_range[0], hkl_range[1], shape[2])
    H, K, L = np.meshgrid(h, k, l, indexing="ij")
    sharp = a_s * np.exp(
        -0.5 * (((H - center[0]) / sig[0]) ** 2
                + ((K - center[1]) / sig[1]) ** 2
                + ((L - center[2]) / sig[2]) ** 2)
    )
    def lor(c, g, axis):
        return g**2 / ((axis - c) ** 2 + g**2)
    broad = a_b * lor(center[0], gam[0], H) * lor(center[1], gam[1], K) * lor(center[2], gam[2], L)
    data = bg + sharp + broad
    ub = 2 * np.pi * np.eye(3) / 4.0
    return HKLVolume.from_arrays(
        data, (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub
    )


def test_decompose_peak_per_axis_and_undersampling_flag():
    # steps: H 0.05, K 0.20, L 0.025 over [-2.5, 2.5]
    shape = (201, 51, 401)
    sig = (0.08, 0.04, 0.08)      # core σ per axis (rlu)
    gam = (0.25, 0.30, 0.25)      # broad HWHM per axis (rlu) — broader than core
    vol = _separable_peak_vol(shape, (-2.5, 2.5), (1.0, 0.0, 0.0),
                              sig=sig, gam=gam, a_s=20.0, a_b=5.0)
    res = Resolution(slope=np.zeros(3), intercept=np.array(sig))
    dec = decompose_peak(vol, (1.0, 0.0, 0.0), res, half_window=0.8, broad="lorentzian")

    for axis in range(3):
        d = dec[axis]
        assert d.success
        assert d.broad_fwhm > 1.5 * d.sharp_fwhm
        assert d.diffuse_fraction > 0.1

    # points-across-FWHM tracks FWHM/step: L well-resolved, K badly undersampled
    assert dec[2].points_across_fwhm > 4.0          # L (fine grid)
    assert dec[1].points_across_fwhm < 1.0          # K (coarse grid)


# ---------------------------------------------------------------------------
# Resolution calibration
# ---------------------------------------------------------------------------

def _nuclear_vol(shape=(101, 101, 101), hkl_range=(-2.5, 2.5)):
    """Sharp Gaussians at integer nodes with σ(|Q|) = 0.06 + 0.02·|Q|."""
    h = np.linspace(hkl_range[0], hkl_range[1], shape[0])
    k = np.linspace(hkl_range[0], hkl_range[1], shape[1])
    l = np.linspace(hkl_range[0], hkl_range[1], shape[2])
    H, K, L = np.meshgrid(h, k, l, indexing="ij")
    ub = 2 * np.pi * np.eye(3) / 4.0
    nodes = [(1, 0, 0), (2, 0, 0), (0, 1, 0), (0, 2, 0),
             (0, 0, 1), (0, 0, 2), (1, 1, 0), (1, 0, 1)]
    data = np.full(shape, 1.0)
    for n in nodes:
        q = np.linalg.norm(np.asarray(n, float) @ ub.T)
        s = 0.06 + 0.02 * q
        data = data + 40.0 * np.exp(
            -0.5 * (((H - n[0]) / s) ** 2 + ((K - n[1]) / s) ** 2 + ((L - n[2]) / s) ** 2)
        )
    vol = HKLVolume.from_arrays(
        data, (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub
    )
    return vol, nodes


def test_calibrate_resolution_recovers_linear_law():
    vol, nodes = _nuclear_vol()
    centers = [(float(a), float(b), float(c)) for a, b, c in nodes]
    res = calibrate_resolution(vol, nuclear_centers=centers, half_window=0.6, n_points=121)
    for axis in range(3):
        assert res.n_ref[axis] >= 4
        # σ(|Q|) = 0.06 + 0.02·|Q| recovered within tolerance
        assert 0.01 < res.slope[axis] < 0.03
        assert 0.03 < res.intercept[axis] < 0.09
    # σ increases with |Q|
    assert res.sigma(0, 4.0) > res.sigma(0, 1.0)


def test_calibrate_resolution_detection_path_runs():
    vol, _ = _nuclear_vol()
    res = calibrate_resolution(vol, max_peaks=10)  # nuclear_centers=None → auto-detect
    assert isinstance(res, Resolution)
    assert sum(res.n_ref) > 0


# ---------------------------------------------------------------------------
# Magnetic-satellite targeting (q = 1/3 family)
# ---------------------------------------------------------------------------

def test_magnetic_satellite_centers_finds_third_plane_only():
    h = np.linspace(-1.0, 1.0, 61)     # step ≈ 0.0333 → 1/3 ≈ index 40
    k = np.linspace(-1.0, 1.0, 31)
    l = np.linspace(-1.0, 1.0, 31)
    H, K, L = np.meshgrid(h, k, l, indexing="ij")
    ub = 2 * np.pi * np.eye(3) / 4.0
    data = np.full(H.shape, 1.0)
    # magnetic satellite at H=1/3 (should be found)
    data += 60.0 * np.exp(-0.5 * (((H - 1/3) / 0.04) ** 2 + (K / 0.04) ** 2 + (L / 0.04) ** 2))
    # nuclear peak at H=1 (frac 0 — should be ignored)
    data += 80.0 * np.exp(-0.5 * (((H - 1.0) / 0.04) ** 2 + (K / 0.04) ** 2 + (L / 0.04) ** 2))
    vol = HKLVolume.from_arrays(data, (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub)

    centers = magnetic_satellite_centers(vol, h_half_width=0.08, n_mad=6.0, min_intensity=2.0)
    assert centers, "expected at least the H=1/3 satellite"
    hs = np.array([c[0] for c in centers])
    assert np.any(np.abs(hs - 1/3) < 0.05)          # found the 1/3 satellite
    assert not np.any(np.abs(hs - 1.0) < 0.05)      # ignored the integer node
