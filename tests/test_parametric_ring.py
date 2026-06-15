"""Tests for the separable parametric ring model (pseudo-Voigt × per-ring texture)."""

import numpy as np

from ndiff.core import HKLVolume
from ndiff.preprocessing import ParametricRingModel
from ndiff.preprocessing.parametric_ring import _fit_multi_pseudo_voigt, _pseudo_voigt


def _ring_vol(
    shape=(81, 81, 1),
    ring_q=2.6,
    ring_fwhm=0.12,
    eta=0.5,
    seed=0,
):
    """2-D hk slice (l=0) with one pseudo-Voigt ring carrying a known azimuthal
    texture 1 + 0.4·cos(2φ), on a diffuse floor, plus sharp Bragg spikes."""
    rng = np.random.default_rng(seed)
    h = np.linspace(-4, 4, shape[0])
    k = np.linspace(-4, 4, shape[1])
    lc = np.linspace(0, 0, shape[2])
    ub = 2 * np.pi * np.eye(3) / 4.0
    vol = HKLVolume.from_arrays(
        np.ones(shape), (h[0], h[-1]), (k[0], k[-1]), (lc[0], lc[-1]), ub_matrix=ub
    )
    q = vol.q_magnitude()
    H, K, _ = vol.hkl_grid()
    Q = np.stack([H, K, np.zeros_like(H)], axis=-1) @ ub.T
    phi = np.arctan2(Q[..., 1], Q[..., 0])

    diffuse = 1.0 + 0.3 * np.cos(np.pi * H) * np.cos(np.pi * K)
    texture = 1.0 + 0.4 * np.cos(2 * phi)
    ring = texture * 3.0 * _pseudo_voigt(q, ring_q, ring_fwhm, eta)
    bragg = np.zeros(shape)
    for hb, kb in [(1, 0), (0, 1), (-1, 0), (0, -1), (2, 2)]:
        ih = int(np.argmin(np.abs(h - hb)))
        ik = int(np.argmin(np.abs(k - kb)))
        bragg[ih, ik, 0] = 200.0
    data = diffuse + ring + bragg + rng.normal(0, 0.02, shape)
    return (
        HKLVolume.from_arrays(
            data, (h[0], h[-1]), (k[0], k[-1]), (lc[0], lc[-1]), ub_matrix=ub
        ),
        ring,
        bragg,
    )


def _radial_median(vol, q, qlo, qhi):
    sel = vol.mask & np.isfinite(vol.data) & (q >= qlo) & (q <= qhi)
    return float(np.median(vol.data[sel]))


def test_pseudo_voigt_is_unit_peak():
    q = np.linspace(2.0, 3.0, 401)
    for eta in (0.0, 0.5, 1.0):
        pv = _pseudo_voigt(q, 2.5, 0.1, eta)
        assert abs(pv.max() - 1.0) < 1e-6
        assert pv.min() >= 0.0


def test_multi_pseudo_voigt_recovers_overlapping_rings():
    q = np.linspace(2.6, 3.6, 600)
    excess = 0.7 * _pseudo_voigt(q, 3.05, 0.08, 0.3) + 0.4 * _pseudo_voigt(
        q, 3.20, 0.10, 0.7
    )
    pv = _fit_multi_pseudo_voigt(
        q, excess, np.array([3.05, 3.20]), np.array([0.08, 0.10]),
        eta0=0.5, q_step=0.02, ring_width=0.4,
    )
    (c0, f0, _e0, a0), (c1, f1, _e1, a1) = pv
    assert abs(c0 - 3.05) < 0.02 and abs(c1 - 3.20) < 0.02
    assert abs(f0 - 0.08) < 0.04 and abs(f1 - 0.10) < 0.04
    assert abs(a0 - 0.7) < 0.15 and abs(a1 - 0.4) < 0.15


def test_parametric_ring_recovers_center_and_width():
    vol, *_ = _ring_vol(ring_q=2.6, ring_fwhm=0.12, eta=0.4)
    model = ParametricRingModel(plane="hk0", q_step=0.03, ring_width=0.4,
                                radial_mode="peaks")
    fitted = model.fit(vol, q_range=(1.0, 4.0))

    assert len(fitted.rings) >= 1
    # the dominant fitted ring sits at the planted |Q| with the planted width
    main = max(fitted.rings, key=lambda r: float(r.texture_coeffs[0]))
    assert abs(main.q_center - 2.6) < 0.05
    assert abs(main.fwhm - 0.12) < 0.06


def test_parametric_ring_suppresses_ring_preserves_diffuse_and_bragg():
    vol, ring, bragg = _ring_vol(ring_q=2.6, ring_fwhm=0.12, eta=0.5)
    q = vol.q_magnitude()

    model = ParametricRingModel(plane="hk0", q_step=0.03, ring_width=0.4,
                                radial_mode="peaks")
    model.fit(vol, q_range=(1.0, 4.0))
    out, I_ring = model.subtract(vol)

    assert np.isfinite(out.data).all()
    assert np.isfinite(I_ring).all()

    # the ring excess over the off-ring baseline is strongly suppressed
    baseline = _radial_median(vol, q, 1.2, 1.8)
    before = _radial_median(vol, q, 2.55, 2.65) - baseline
    after = _radial_median(out, q, 2.55, 2.65) - baseline
    assert before > 0.3
    assert after < 0.2 * before

    # diffuse away from the ring is preserved
    diffuse_removed = np.median(np.abs(I_ring[(q > 1.2) & (q < 1.8)]))
    assert diffuse_removed < 0.1

    # sharp Bragg peaks are left in the residual (rejected by the robust texture fit)
    bragg_loc = bragg > 100
    assert np.all(I_ring[bragg_loc] < 5.0)
    assert np.all(out.data[bragg_loc] > 100.0)


def test_parametric_ring_texture_follows_planted_anisotropy():
    # planted texture 1 + 0.4 cos(2φ) → max at φ=0/π, min at ±π/2
    vol, *_ = _ring_vol(ring_q=2.6, ring_fwhm=0.12, eta=0.4)
    model = ParametricRingModel(plane="hk0", q_step=0.03, ring_width=0.4, n_fourier=4,
                                radial_mode="peaks")
    fitted = model.fit(vol, q_range=(1.0, 4.0))
    main_i = int(np.argmax([float(r.texture_coeffs[0]) for r in fitted.rings]))

    phi = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2])
    t = fitted.ring_texture(main_i, phi)
    # max at 0 and π, min at ±π/2 (cos 2φ pattern), with positive contrast
    assert t[0] > t[1] and t[2] > t[3]
    assert (t[0] - t[1]) / (t[0] + t[1] + 1e-9) > 0.1


def test_parametric_ring_empty_when_no_rings():
    # flat diffuse + noise, no ring → empty model, zero subtraction
    rng = np.random.default_rng(1)
    shape = (61, 61, 1)
    ub = 2 * np.pi * np.eye(3) / 4.0
    data = 1.0 + rng.normal(0, 0.02, shape)
    vol = HKLVolume.from_arrays(data, (-4, 4), (-4, 4), (0, 0), ub_matrix=ub)
    model = ParametricRingModel(plane="hk0", q_step=0.03, ring_width=0.4,
                                radial_mode="peaks")
    fitted = model.fit(vol, q_range=(1.0, 4.0))
    _, I_ring = model.subtract(vol)
    assert len(fitted.rings) == 0
    assert np.allclose(I_ring, 0.0)


# ---------------------------------------------------------------------------
# rolling-window continuous mode
# ---------------------------------------------------------------------------
def test_rolling_is_the_default_mode():
    model = ParametricRingModel()
    assert model.radial_mode == "rolling"


def test_rolling_suppresses_ring_preserves_diffuse_and_bragg():
    vol, ring, bragg = _ring_vol(ring_q=2.6, ring_fwhm=0.12, eta=0.5)
    q = vol.q_magnitude()

    model = ParametricRingModel(plane="hk0", q_step=0.03, ring_width=0.3,
                                roll_step=0.05, radial_mode="rolling")
    fitted = model.fit(vol, q_range=(1.0, 4.0))
    out, I_ring = model.subtract(vol)

    assert fitted.mode == "rolling"
    assert fitted.roll_coeffs.shape[0] == fitted.roll_centers.size
    assert np.isfinite(out.data).all()
    assert np.isfinite(I_ring).all()

    baseline = _radial_median(vol, q, 1.2, 1.8)
    before = _radial_median(vol, q, 2.55, 2.65) - baseline
    after = _radial_median(out, q, 2.55, 2.65) - baseline
    assert before > 0.3
    assert after < 0.25 * before

    # diffuse far from the ring is preserved (continuous A(|Q|) ≈ 0 there)
    diffuse_removed = np.median(np.abs(I_ring[(q > 1.2) & (q < 1.8)]))
    assert diffuse_removed < 0.1

    # sharp Bragg peaks survive (rejected by the IRLS)
    bragg_loc = bragg > 100
    assert np.all(out.data[bragg_loc] > 100.0)


def test_rolling_radial_amplitude_is_continuous_and_peaks_at_ring():
    vol, *_ = _ring_vol(ring_q=2.6, ring_fwhm=0.12, eta=0.4)
    model = ParametricRingModel(plane="hk0", q_step=0.03, ring_width=0.3,
                                roll_step=0.05, radial_mode="rolling")
    fitted = model.fit(vol, q_range=(1.0, 4.0))
    a = fitted.radial_amplitude()
    centers = fitted.roll_centers
    assert a.size == centers.size and a.size > 5
    assert np.all(a >= 0.0)
    # the continuous radial amplitude peaks at the planted ring |Q|, not off-ring
    on = a[np.abs(centers - 2.6) <= 0.15].max()
    off = a[np.abs(centers - 1.6) <= 0.15].max()
    assert on > 3.0 * off


def test_spike_reject_recovers_bright_arc_better_than_legacy():
    """φ-shape rejection (default) captures the broad bright-arc texture that the
    legacy high-side rejection under-subtracts, while both still spare Bragg.

    The planted texture ``1 + 0.4·cos(2φ)`` peaks at φ=0 (the +H direction, since
    Q ∥ a*).  The legacy IRLS treats that bright arc like a Bragg outlier and
    leaves more of it behind; the φ-shape detector keeps it (broad in φ) and
    subtracts more, so less ring survives on the bright arc."""
    vol, _ring, bragg = _ring_vol(ring_q=2.6, ring_fwhm=0.12, eta=0.5)
    q = vol.q_magnitude()
    H, K, _ = vol.hkl_grid()
    phi = np.arctan2(K, H)
    bright = (q > 2.5) & (q < 2.7) & (np.abs(phi) < 0.4) & (bragg < 1)
    assert bright.sum() > 4

    resid = {}
    for sr in (True, False):
        model = ParametricRingModel(plane="hk0", q_step=0.03, ring_width=0.3,
                                    roll_step=0.05, radial_mode="rolling",
                                    texture_spike_reject=sr)
        model.fit(vol, q_range=(1.0, 4.0))
        out, _I = model.subtract(vol)
        resid[sr] = float(np.median(out.data[bright]))
        # both modes leave the sharp Bragg peaks in the residual
        assert np.all(out.data[bragg > 100] > 100.0)

    # φ-shape rejection leaves less of the bright arc behind (captures more of it)
    assert resid[True] < resid[False]


def test_rolling_near_zero_on_flat_noise():
    rng = np.random.default_rng(2)
    shape = (61, 61, 1)
    ub = 2 * np.pi * np.eye(3) / 4.0
    data = 1.0 + rng.normal(0, 0.02, shape)
    vol = HKLVolume.from_arrays(data, (-4, 4), (-4, 4), (0, 0), ub_matrix=ub)
    model = ParametricRingModel(plane="hk0", q_step=0.03, ring_width=0.3,
                                roll_step=0.05, radial_mode="rolling")
    model.fit(vol, q_range=(1.0, 4.0))
    _, I_ring = model.subtract(vol)
    # no real ring → the continuous fit removes only noise-level intensity
    assert float(np.median(np.abs(I_ring))) < 0.05


# ---------------------------------------------------------------------------
# adaptive non-separable radial shape: q0(φ), fwhm(φ)
# ---------------------------------------------------------------------------
def _phi_width_ring_vol(N=121, q0=2.6, f0=0.12, f1=0.06, eta=0.4, seed=0,
                        const=False):
    """hk slice with one ring at fixed |Q| whose WIDTH varies azimuthally:
    fwhm(φ) = f0 + f1·cos(2φ)  (or constant f0 when ``const``)."""
    from ndiff.preprocessing.parametric_ring import _pseudo_voigt_phi
    rng = np.random.default_rng(seed)
    h = np.linspace(-4, 4, N)
    ub = 2 * np.pi * np.eye(3) / 4.0
    shape = (N, N, 1)
    base = HKLVolume.from_arrays(np.ones(shape), (h[0], h[-1]), (h[0], h[-1]),
                                 (0, 0), ub_matrix=ub)
    q = base.q_magnitude()
    H, K, _ = base.hkl_grid()
    phi = np.arctan2(K, H)
    fwhm = np.full_like(q, f0) if const else (f0 + f1 * np.cos(2 * phi))
    ring = 3.0 * _pseudo_voigt_phi(q, np.full_like(q, q0), fwhm, eta)
    data = 1.0 + ring + rng.normal(0, 0.02, shape)
    return HKLVolume.from_arrays(data, (h[0], h[-1]), (h[0], h[-1]), (0, 0),
                                 ub_matrix=ub)


def _shell_inhomogeneity(vol_out, q0=2.6, w=0.12, nsec=8):
    q = vol_out.q_magnitude()
    H, K, _ = vol_out.hkl_grid()
    phi = np.arctan2(K, H)
    on = np.abs(q - q0) <= w
    edges = np.linspace(-np.pi, np.pi, nsec + 1)
    sec = np.clip(np.digitize(phi[on], edges) - 1, 0, nsec - 1)
    rv = vol_out.data[on]
    prof = [np.median(rv[sec == s]) for s in range(nsec) if np.any(sec == s)]
    return float(np.std(prof))


def test_adaptive_radial_harmonics_accepted_for_phi_dependent_width():
    """A ring whose width varies with φ: the adaptive model accepts q0(φ)/fwhm(φ)
    and leaves a more uniform (less azimuthally-structured) residual than the
    separable model with a single shared width."""
    vol = _phi_width_ring_vol(const=False)
    common = dict(plane="hk0", q_step=0.03, radial_mode="peaks",
                  allowed_ring_centers=np.array([2.6]),
                  allowed_ring_halfwidths=np.array([0.12]),
                  radial_n_fourier=2, radial_harmonic_sectors=8,
                  radial_harmonic_min_voxels_per_sector=8)
    sep = ParametricRingModel(radial_harmonics=False, **common)
    sep.fit(vol, q_range=(1.0, 4.0))
    out_sep, _ = sep.subtract(vol)

    ns = ParametricRingModel(radial_harmonics=True,
                             radial_harmonic_accept_margin=0.02, **common)
    fitted = ns.fit(vol, q_range=(1.0, 4.0))
    out_ns, _ = ns.subtract(vol)

    # the harmonics were accepted for this ring
    assert fitted.rings[0].fwhm_coeffs is not None
    # and they reduce the azimuthal residual structure
    assert _shell_inhomogeneity(out_ns) < _shell_inhomogeneity(out_sep)


def test_adaptive_radial_harmonics_rejected_for_constant_width():
    """A constant-width ring: the harmonics give no improvement, so the adaptive
    test rejects them and the ring stays separable (q0/fwhm coeffs are None)."""
    vol = _phi_width_ring_vol(const=True)
    ns = ParametricRingModel(plane="hk0", q_step=0.03, radial_mode="peaks",
                             allowed_ring_centers=np.array([2.6]),
                             allowed_ring_halfwidths=np.array([0.12]),
                             radial_harmonics=True, radial_n_fourier=2,
                             radial_harmonic_sectors=8,
                             radial_harmonic_min_voxels_per_sector=8,
                             radial_harmonic_accept_margin=0.05)
    fitted = ns.fit(vol, q_range=(1.0, 4.0))
    assert fitted.rings[0].q0_coeffs is None
    assert fitted.rings[0].fwhm_coeffs is None


def test_radial_harmonics_off_by_default():
    vol = _phi_width_ring_vol(const=False)
    model = ParametricRingModel(plane="hk0", q_step=0.03, radial_mode="peaks",
                                allowed_ring_centers=np.array([2.6]),
                                allowed_ring_halfwidths=np.array([0.12]))
    assert model.radial_harmonics is False
    fitted = model.fit(vol, q_range=(1.0, 4.0))
    assert fitted.rings[0].fwhm_coeffs is None
