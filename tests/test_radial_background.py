"""Tests for the non-parametric per-patch radial-background ring removal."""

import numpy as np

from ndiff.core import HKLVolume
from ndiff.preprocessing import PatchedRadialRingModel, azimuthal_sampling_mask
from ndiff.preprocessing.radial_background import (
    _project_templates,
    _snip_baseline,
    _estimate_baseline,
    _adaptive_ring_width_profile,
)
from ndiff.preprocessing.ring_model import _gaussian


def _ring_vol(shape=(61, 61, 1), ring_q=2.6, ring_sigma=0.05, seed=0):
    """2D hk slice (l=0) with a textured ring + diffuse + Bragg peaks.

    Both h and k vary so the hk0 azimuth φ=atan2(k_Q, h_Q) spans the full
    circle — needed to exercise the azimuthal texture.
    """
    rng = np.random.default_rng(seed)
    h = np.linspace(-4, 4, shape[0])
    k = np.linspace(-4, 4, shape[1])
    l = np.linspace(0, 0, shape[2])
    ub = 2 * np.pi * np.eye(3) / 4.0
    vol = HKLVolume.from_arrays(
        np.ones(shape), (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub
    )
    q = vol.q_magnitude()
    H, K, L = vol.hkl_grid()
    Q = np.stack([H, K, L], axis=-1) @ ub.T
    phi = np.arctan2(Q[..., 1], Q[..., 0])

    diffuse = 1.0 + 0.3 * np.cos(np.pi * H) * np.cos(np.pi * K)
    texture = 1.0 + 0.4 * np.cos(2 * phi)
    ring = texture * _gaussian(q, 3.0, ring_q, ring_sigma)
    bragg = np.zeros(shape)
    for hb, kb in [(1, 0), (0, 1), (-1, 0), (0, -1), (2, 2)]:
        ih = int(np.argmin(np.abs(h - hb)))
        ik = int(np.argmin(np.abs(k - kb)))
        bragg[ih, ik, 0] = 200.0
    data = diffuse + ring + bragg + rng.normal(0, 0.02, shape)
    return HKLVolume.from_arrays(
        data, (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub
    ), ring, bragg


def _radial_median(vol, q, qlo, qhi):
    valid = vol.mask & np.isfinite(vol.data)
    sel = valid & (q >= qlo) & (q <= qhi)
    return float(np.median(vol.data[sel]))


def test_radial_background_suppresses_ring_preserves_diffuse_and_bragg():
    vol, ring, bragg = _ring_vol()
    q = vol.q_magnitude()

    model = PatchedRadialRingModel(
        n_patches=24, plane="hk0", q_step=0.04, ring_width=0.3,
        baseline_smooth=0.08,
    )
    model.fit(vol, q_range=(1.0, 4.0))
    out, I_ring = model.subtract(vol)

    assert np.isfinite(out.data).all()
    assert np.isfinite(I_ring).all()

    # Ring excess over the diffuse baseline is strongly suppressed (the median
    # is robust to the Bragg peaks).  Measure relative to an off-ring baseline
    # so the preserved diffuse floor doesn't mask the suppression.
    baseline = _radial_median(vol, q, 1.2, 1.8)
    before = _radial_median(vol, q, 2.55, 2.65) - baseline
    after = _radial_median(out, q, 2.55, 2.65) - baseline
    assert before > 0.3                       # there really was a ring
    assert after < 0.2 * before               # >80% of the ring excess removed

    # Diffuse away from the ring is preserved (little removed there).
    diffuse_removed = np.median(np.abs(I_ring[(q > 1.2) & (q < 1.8)]))
    assert diffuse_removed < 0.1

    # Bragg peaks (sharp outliers) are NOT removed — rejected by the trim.
    bragg_loc = bragg > 100
    assert np.all(I_ring[bragg_loc] < 1.0)        # ring model leaves Bragg
    assert np.all(out.data[bragg_loc] > 100.0)    # Bragg still in the residual


def test_radial_background_no_negative_ring_estimate():
    vol, *_ = _ring_vol()
    model = PatchedRadialRingModel(n_patches=24, plane="hk0", q_step=0.04)
    prof = model.fit(vol, q_range=(1.0, 4.0))
    assert (prof.ring_profile >= 0).all()


def test_template_projection_fits_overlapping_rings_jointly():
    q = np.linspace(2.8, 3.5, 300)
    g0 = np.exp(-0.5 * ((q - 3.10) / 0.055) ** 2)
    g1 = np.exp(-0.5 * ((q - 3.18) / 0.055) ** 2)
    excess = 0.7 * g0 + 0.4 * g1

    projected = _project_templates(excess, [g0, g1])

    assert np.max(np.abs(projected - excess)) < 1e-10
    assert projected.max() < 1.15 * excess.max()


def test_fourier_texture_recovers_anisotropy_with_correct_phase():
    # Injected texture is 1 + 0.4 cos(2φ): max at φ=0/π, min at ±π/2.
    # General (non-symmetric) basis must recover it given n_fourier>=2.
    # texture_q_smooth=0 isolates the per-|Q|-bin phase/amplitude recovery
    # (the synthetic ring is only a few bins wide, so |Q|-pooling would average
    # over its noisy edges — exercised separately below).
    vol, *_ = _ring_vol()
    model = PatchedRadialRingModel(n_patches=24, plane="hk0", q_step=0.04,
                                   ring_width=0.3, baseline_smooth=0.08,
                                   texture_model="fourier", n_fourier=2,
                                   texture_symmetric=False, texture_ridge=0.3,
                                   texture_q_smooth=0.0)
    prof = model.fit(vol, q_range=(1.0, 4.0))
    assert prof.texture_coeffs.size                    # Fourier model populated
    qpk = float(prof.q_grid[np.argmax(prof.ring_profile.max(axis=0))])
    t0 = float(prof.texture(qpk, np.array([0.0]))[0])          # cos2φ = +1
    t90 = float(prof.texture(qpk, np.array([np.pi / 2]))[0])   # cos2φ = -1
    assert t0 > t90                                    # correct phase
    assert t0 / t90 > 1.4                              # substantial anisotropy captured


def test_texture_q_pooling_keeps_radial_peak_sharp_and_preserves_phase():
    # |Q|-pooling the azimuthal texture shape must (a) preserve the cos2φ phase
    # and a substantial fraction of the anisotropy, and (b) NOT broaden the
    # radial ring amplitude (the constant term must be untouched by the shape
    # smoothing).  Compare pooled vs unpooled on the same volume.
    vol, *_ = _ring_vol()
    base = dict(n_patches=24, plane="hk0", q_step=0.04, ring_width=0.3,
                baseline_smooth=0.08, texture_model="fourier", n_fourier=4,
                texture_symmetric=False, texture_ridge=0.1)
    unpooled = PatchedRadialRingModel(texture_q_smooth=0.0, **base).fit(vol, q_range=(1.0, 4.0))
    pooled = PatchedRadialRingModel(texture_q_smooth=0.12, **base).fit(vol, q_range=(1.0, 4.0))

    qpk = float(pooled.q_grid[np.argmax(pooled.ring_profile.max(axis=0))])
    t0 = float(pooled.texture(qpk, np.array([0.0]))[0])
    t90 = float(pooled.texture(qpk, np.array([np.pi / 2]))[0])
    assert t0 > t90 and t0 / t90 > 1.3                 # phase + anisotropy kept

    # Radial amplitude (constant Fourier term, column 0) is identical: pooling
    # smooths only the shape, never the amplitude that sets the radial peak.
    assert np.allclose(pooled.texture_coeffs[:, 0], unpooled.texture_coeffs[:, 0])


def test_smooth_texture_model_runs_and_suppresses_ring():
    vol, *_ = _ring_vol()
    q = vol.q_magnitude()
    model = PatchedRadialRingModel(n_patches=24, plane="hk0", q_step=0.04,
                                   ring_width=0.3, baseline_smooth=0.08,
                                   texture_model="smooth", texture_smoothness=10.0)
    prof = model.fit(vol, q_range=(1.0, 4.0))
    out, I_ring = model.subtract(vol, prof)

    assert prof.texture_values.shape == prof.ring_profile.shape
    baseline = _radial_median(vol, q, 1.2, 1.8)
    before = _radial_median(vol, q, 2.55, 2.65) - baseline
    after = _radial_median(out, q, 2.55, 2.65) - baseline
    assert after < 0.35 * before
    assert np.isfinite(I_ring).all()


def test_fourier_texture_is_immune_to_bragg():
    # The low-order texture must not spike at the Bragg azimuths.
    vol, _, bragg = _ring_vol()
    model = PatchedRadialRingModel(n_patches=24, plane="hk0", q_step=0.04,
                                   ring_width=0.3, baseline_smooth=0.08)
    model.fit(vol, q_range=(1.0, 4.0))
    _, I_ring = model.subtract(vol)
    assert np.all(I_ring[bragg > 100] < 1.0)           # ring model ignores Bragg


def test_azimuthal_sampling_mask_drops_undersampled_sector():
    rng = np.random.default_rng(1)
    h = np.linspace(-4, 4, 101); k = np.linspace(-4, 4, 101); l = np.linspace(0, 0, 1)
    ub = 2 * np.pi * np.eye(3) / 4.0
    vol = HKLVolume.from_arrays(
        np.ones((101, 101, 1)), (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub
    )
    H, K, L = vol.hkl_grid()
    Q = np.stack([H, K, L], axis=-1) @ vol.ub_matrix.T
    phi = np.arctan2(Q[..., 1], Q[..., 0])
    sector = (phi >= 0.0) & (phi < np.pi / 2)            # a quadrant
    thin = sector & (rng.random(vol.shape) > 0.05)       # keep ~5% of it
    vol.mask[thin] = False

    keep = azimuthal_sampling_mask(vol, plane="hk0", n_phi_bins=12, n_q_bins=10,
                                   min_count_frac=0.25, min_count=3, q_range=(1.0, 4.0))
    assert (keep <= vol.mask).all()                       # never adds voxels
    # Interior of the thinned sector (clear of bin edges, within q-range) is
    # dropped; the dense opposite side is kept.
    q = vol.q_magnitude()
    in_q = (q >= 1.6) & (q <= 3.4)
    inner = (phi > 0.2) & (phi < np.pi / 2 - 0.2) & vol.mask & in_q
    outer = (phi < -0.2) & vol.mask & in_q
    assert keep[inner].mean() < 0.3
    assert keep[outer].mean() > 0.9


def test_snip_no_oversubtraction_on_slope():
    # On a pure linear slope (no ring), SNIP must return the slope exactly —
    # never dip below it.  Morphological opening (erosion → dilation) is biased
    # low on sloping backgrounds because the erosion step takes the minimum over
    # the window, which is on the lower-|Q| flank.  SNIP uses the midpoint of
    # symmetric neighbors, so it is exact for any linear background.
    q = np.linspace(0, 1, 200)
    slope = 1.0 - 0.8 * q
    base_snip = _snip_baseline(slope, n_iter=20)
    assert np.max(np.abs(base_snip - slope)) < 1e-12   # machine-precision exact


def test_snip_removes_narrow_ring_on_slope():
    # Ring on a sloping background: SNIP must detect the ring (non-zero excess)
    # and must NOT dip below the true slope anywhere (no over-subtraction).
    q = np.linspace(0, 1, 300)
    slope = 1.0 - 0.6 * q
    ring = 0.3 * np.exp(-0.5 * ((q - 0.5) / 0.025) ** 2)
    prof = slope + ring
    n_iter = 10
    base = _snip_baseline(prof, n_iter)
    excess = prof - base
    assert excess.max() > 0.2 * ring.max()      # ring is detected
    assert np.all(base >= slope - 1e-12)         # never dips below true slope


def test_baseline_method_snip_vs_opening_on_slope():
    # SNIP never creates a negative baseline dip; opening does.
    q_step = 0.02
    q = np.arange(0, 4, q_step)
    slope = 0.5 * np.exp(-0.3 * q)
    ring = 0.4 * np.exp(-0.5 * ((q - 2.0) / 0.04) ** 2)
    prof = slope + ring

    base_snip = _estimate_baseline(prof, q_step, ring_width=0.24, smooth=0.0, method="snip")
    base_open = _estimate_baseline(prof, q_step, ring_width=0.24, smooth=0.0, method="opening")

    # SNIP never dips below the true slope (no over-subtraction at any |Q|)
    assert np.max(slope - base_snip) < 1e-10
    # Opening dips below the slope by a measurable amount
    assert np.max(slope - base_open) > 0.005


def test_adaptive_ring_width_matches_each_ring_thickness():
    # Three rings of different widths: the adaptive window must scale with each
    # ring's own FWHM (narrow ring → narrow window, broad ring → broad window).
    q_step = 0.01
    q = np.arange(1.0, 6.0, q_step)
    prof = (
        1.0
        + 0.6 * np.exp(-0.5 * ((q - 2.0) / 0.03) ** 2)   # narrow ring
        + 0.6 * np.exp(-0.5 * ((q - 4.0) / 0.09) ** 2)   # broad ring
    )
    win = _adaptive_ring_width_profile(q, prof, q_step, base_width=0.24,
                                       scale=3.0, cap_frac=0.9)
    b_narrow = int(np.argmin(np.abs(q - 2.0)))
    b_broad = int(np.argmin(np.abs(q - 4.0)))
    assert win[b_broad] > win[b_narrow]            # broad ring gets a wider window
    # Each window is roughly scale·FWHM of its own ring (FWHM = 2.355σ).
    assert 0.10 < win[b_narrow] < 0.35
    assert 0.45 < win[b_broad] < 0.80


def test_adaptive_ring_width_caps_close_pair():
    # A ring in a close pair must get a NARROWER window than the same-width ring
    # in isolation, so the baseline clip is held back from bridging the pair and
    # over-subtracting the diffuse valley between them.
    q_step = 0.01
    q = np.arange(1.0, 6.0, q_step)
    sig = 0.05
    pair = (1.0 + 0.6 * np.exp(-0.5 * ((q - 3.0) / sig) ** 2)
                + 0.6 * np.exp(-0.5 * ((q - 3.18) / sig) ** 2))
    solo = 1.0 + 0.6 * np.exp(-0.5 * ((q - 3.0) / sig) ** 2)   # isolated ring

    b = int(np.argmin(np.abs(q - 3.0)))
    w_pair = _adaptive_ring_width_profile(q, pair, q_step, 0.30, 3.0, 0.9)[b]
    w_solo = _adaptive_ring_width_profile(q, solo, q_step, 0.30, 3.0, 0.9)[b]
    assert w_pair < w_solo                        # neighbour cap kicked in
    # And the clip half-width stays under the pair spacing, so it cannot bridge.
    assert 0.5 * w_pair < 0.18


def test_sampling_mask_keeps_uniform_low_q_shells():
    # Regression: a uniformly-sampled volume must NOT lose its low-|Q| annulus
    # (few points per cell there is expected, not anomalous) — the old absolute
    # threshold carved a "pixelised empty ring".
    h = np.linspace(-4, 4, 101); k = np.linspace(-4, 4, 101); l = np.linspace(0, 0, 1)
    ub = 2 * np.pi * np.eye(3) / 4.0
    vol = HKLVolume.from_arrays(
        np.ones((101, 101, 1)), (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub
    )
    keep = azimuthal_sampling_mask(vol, plane="hk0", n_phi_bins=72, n_q_bins=90,
                                   min_count_frac=0.25, min_count=1, q_range=(0.5, 4.0))
    q = vol.q_magnitude()
    low_q = vol.mask & (q > 0.6) & (q < 1.5)
    assert keep[low_q].mean() > 0.9          # low-|Q| shells preserved
