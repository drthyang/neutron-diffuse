"""Tests for the non-parametric per-patch radial-background ring removal."""

import numpy as np

from ndiff.core import HKLVolume
from ndiff.preprocessing import (
    PatchedRadialRingModel,
    azimuthal_sampling_mask,
    confirm_ring_shells_across_h,
)
from ndiff.preprocessing.radial_background import (
    _adaptive_ring_width_profile,
    _azimuthal_angle,
    _estimate_baseline,
    _offset_q_magnitude,
    _plane_components,
    _project_templates,
    _snip_baseline,
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


def test_h_dependent_center_offset_changes_q_and_phi_by_slice():
    h = np.array([0.0, 0.5, 1.0])
    k = np.linspace(-1.0, 1.0, 5)
    l = np.linspace(-1.0, 1.0, 5)
    ub = np.eye(3)
    vol = HKLVolume.from_arrays(
        np.ones((h.size, k.size, l.size)),
        (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]),
        ub_matrix=ub,
    )

    center_offset = (0.02, -0.03)
    h_slope = (0.10, 0.20)
    Q, x, y = _plane_components(vol, "0kl")
    H, _, _ = vol.hkl_grid()
    cx = center_offset[0] + h_slope[0] * H
    cy = center_offset[1] + h_slope[1] * H

    q_expected = np.sqrt(
        np.maximum(
            np.einsum("...i,...i->...", Q, Q)
            - x * x - y * y
            + (x - cx) ** 2
            + (y - cy) ** 2,
            0.0,
        )
    )
    phi_expected = np.arctan2(y - cy, x - cx)

    assert np.allclose(
        _offset_q_magnitude(vol, "0kl", center_offset, h_slope),
        q_expected,
    )
    assert np.allclose(
        _azimuthal_angle(vol, "0kl", center_offset, h_slope),
        phi_expected,
    )


# ---------------------------------------------------------------------------
# Cross-H ring confirmation / phantom-ring rejection
# ---------------------------------------------------------------------------

def _stacked_ring_vol(nh=11, nkl=41, q_real=2.6, q_phantom=1.8,
                      sigma=0.06, phantom_plane=5, seed=0):
    """3D volume stacked over H (axis 0): a real spherical ring on every plane
    plus a ring-shaped 'phantom' on a single H-plane (mimicking the symmetric
    Bragg peaks that fool single-plane detection at integer H).

    UB = 2π/(2π)·I = I, so |Q| = sqrt(H²+K²+L²) in the same units as h,k,l.
    """
    rng = np.random.default_rng(seed)
    h = np.linspace(-1.0, 1.0, nh)
    k = np.linspace(-4.0, 4.0, nkl)
    l = np.linspace(-4.0, 4.0, nkl)
    ub = np.eye(3)
    vol = HKLVolume.from_arrays(
        np.ones((nh, nkl, nkl)), (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]),
        ub_matrix=ub,
    )
    q = vol.q_magnitude()
    data = 1.0 + 3.0 * _gaussian(q, 1.0, q_real, sigma)        # real ring, all H
    # Phantom: an azimuthally-smooth ring at q_phantom on ONE plane only.
    phantom = 3.0 * _gaussian(q[phantom_plane], 1.0, q_phantom, sigma)
    data[phantom_plane] += phantom
    data += rng.normal(0, 0.01, data.shape)
    return HKLVolume.from_arrays(
        data, (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub,
    ), phantom_plane


def _permute_stack_axis(vol: HKLVolume, stack_axis: str) -> HKLVolume:
    """Move the synthetic phantom stack from H to K or L for plane tests."""
    if stack_axis == "H":
        return vol
    if stack_axis == "K":
        data = np.moveaxis(vol.data, 0, 1)
        return HKLVolume.from_arrays(
            data,
            (float(vol.k_axis[0]), float(vol.k_axis[-1])),
            (float(vol.h_axis[0]), float(vol.h_axis[-1])),
            (float(vol.l_axis[0]), float(vol.l_axis[-1])),
            ub_matrix=vol.ub_matrix,
        )
    if stack_axis == "L":
        data = np.moveaxis(vol.data, 0, 2)
        return HKLVolume.from_arrays(
            data,
            (float(vol.k_axis[0]), float(vol.k_axis[-1])),
            (float(vol.l_axis[0]), float(vol.l_axis[-1])),
            (float(vol.h_axis[0]), float(vol.h_axis[-1])),
            ub_matrix=vol.ub_matrix,
        )
    raise ValueError(stack_axis)


def test_confirm_ring_shells_keeps_real_ring_rejects_phantom():
    vol, _ = _stacked_ring_vol()
    centers, fwhm, amps = confirm_ring_shells_across_h(
        vol, plane="0kl", q_range=(1.2, 3.2), q_step=0.04, min_voxels_per_bin=4,
    )
    # The real ring at |Q|≈2.6 (present on every plane) is confirmed.
    assert centers.size >= 1
    assert np.min(np.abs(centers - 2.6)) < 0.08
    # The phantom at |Q|≈1.8 (one plane only) is washed out of the across-H
    # median and NOT confirmed.
    assert np.all(np.abs(centers - 1.8) > 0.15)
    # The reported amplitude at the real ring is a positive excess.
    assert amps.shape == centers.shape
    assert amps[np.argmin(np.abs(centers - 2.6))] > 0.3


def test_confirm_ring_shells_supports_all_principal_stack_axes():
    base, _ = _stacked_ring_vol()
    cases = [
        ("H", "0kl"),
        ("K", "h0l"),
        ("L", "hk0"),
    ]
    for stack_axis, plane in cases:
        vol = _permute_stack_axis(base, stack_axis)
        centers, _, amps = confirm_ring_shells_across_h(
            vol, plane=plane, q_range=(1.2, 3.2), q_step=0.04, min_voxels_per_bin=4,
        )
        assert centers.size >= 1, stack_axis
        assert np.min(np.abs(centers - 2.6)) < 0.08, stack_axis
        assert np.all(np.abs(centers - 1.8) > 0.15), stack_axis
        assert amps[np.argmin(np.abs(centers - 2.6))] > 0.3, stack_axis


def test_ring_q_envelope_passes_shells_and_zeros_between():
    model = PatchedRadialRingModel(
        plane="0kl", allowed_ring_centers=np.array([2.6]),
        allowed_ring_halfwidths=np.array([0.1]),
    )
    q_grid = np.linspace(1.2, 3.2, 201)
    env = model._ring_q_envelope(q_grid)
    assert env is not None
    assert env[np.argmin(np.abs(q_grid - 2.6))] == 1.0      # on the shell
    assert env[np.argmin(np.abs(q_grid - 1.8))] == 0.0      # far from any shell
    assert np.all((env >= 0.0) & (env <= 1.0))
    # No envelope when no shells configured (default single-plane behaviour).
    assert PatchedRadialRingModel(plane="0kl")._ring_q_envelope(q_grid) is None


def test_confirmed_shells_suppress_phantom_ring_subtraction():
    """On the phantom plane, the default model subtracts the phantom ring; with
    the cross-H confirmed shells it leaves it (no over-subtraction trough)."""
    import dataclasses

    vol, ip = _stacked_ring_vol()
    q_range = (1.2, 3.2)
    centers, fwhm, amps = confirm_ring_shells_across_h(
        vol, plane="0kl", q_range=q_range, q_step=0.04, min_voxels_per_bin=4)

    sl = dataclasses.replace(
        vol, data=vol.data[ip:ip + 1], sigma=vol.sigma[ip:ip + 1],
        mask=vol.mask[ip:ip + 1], h_axis=vol.h_axis[ip:ip + 1],
    )
    q2d = sl.q_magnitude()
    phantom_sel = (q2d > 1.7) & (q2d < 1.9)
    real_sel = (q2d > 2.5) & (q2d < 2.7)

    common = dict(n_patches=24, plane="0kl", q_step=0.04, ring_width=0.3,
                  baseline_smooth=0.08, min_voxels_per_patch=40)
    naive = PatchedRadialRingModel(**common)
    naive.fit(sl, q_range=q_range)
    _, I_naive = naive.subtract(sl)

    guarded = PatchedRadialRingModel(
        allowed_ring_centers=centers, allowed_ring_halfwidths=fwhm, **common)
    guarded.fit(sl, q_range=q_range)
    _, I_guarded = guarded.subtract(sl)

    # Default model subtracts a phantom ring; the guarded model essentially does
    # not (the confirmed shells exclude |Q|≈1.8).
    assert np.median(I_naive[phantom_sel]) > 0.3
    assert np.median(I_guarded[phantom_sel]) < 0.1 * np.median(I_naive[phantom_sel])
    # Both still remove the real ring at |Q|≈2.6.
    assert np.median(I_guarded[real_sel]) > 0.3


def test_amplitude_ceiling_caps_bragg_inflated_ring():
    """A Bragg-like excess landing ON a real ring inflates that ring's per-plane
    amplitude, over-subtracting elsewhere along the ring; the per-shell ceiling
    caps the spike (the |Q|-envelope alone cannot — the shell is legitimate)."""
    import dataclasses

    nh, nkl, ip = 11, 61, 5
    h = np.linspace(-1.0, 1.0, nh)
    k = np.linspace(-4.0, 4.0, nkl)
    l = np.linspace(-4.0, 4.0, nkl)
    ub = np.eye(3)
    base = HKLVolume.from_arrays(
        np.ones((nh, nkl, nkl)), (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]),
        ub_matrix=ub,
    )
    q = base.q_magnitude()
    data = 1.0 + 3.0 * _gaussian(q, 1.0, 2.6, 0.06)            # real ring, all H

    # Inflate the ring at |Q|≈2.6 in a narrow azimuthal wedge on ONE plane
    # (Bragg landing on a real ring shell).
    sl0 = dataclasses.replace(
        base, data=base.data[ip:ip + 1], sigma=base.sigma[ip:ip + 1],
        mask=base.mask[ip:ip + 1], h_axis=base.h_axis[ip:ip + 1],
    )
    q0 = sl0.q_magnitude()[0]
    phi0 = _azimuthal_angle(sl0, "0kl")[0]
    wedge = (np.abs(q0 - 2.6) < 0.1) & (np.abs(phi0) < 0.26)
    data[ip][wedge] += 40.0
    data += np.random.default_rng(1).normal(0, 0.01, data.shape)
    vol = HKLVolume.from_arrays(
        data, (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub,
    )

    centers, fwhm, amps = confirm_ring_shells_across_h(
        vol, plane="0kl", q_range=(1.5, 3.2), q_step=0.04, min_voxels_per_bin=4)

    sl = dataclasses.replace(
        vol, data=vol.data[ip:ip + 1], sigma=vol.sigma[ip:ip + 1],
        mask=vol.mask[ip:ip + 1], h_axis=vol.h_axis[ip:ip + 1],
    )
    common = dict(n_patches=24, plane="0kl", q_step=0.04, ring_width=0.3,
                  baseline_smooth=0.08, min_voxels_per_patch=40,
                  allowed_ring_centers=centers, allowed_ring_halfwidths=fwhm)

    no_cap = PatchedRadialRingModel(**common)
    no_cap.fit(sl, q_range=(1.5, 3.2))
    out_nocap, _ = no_cap.subtract(sl)

    capped = PatchedRadialRingModel(allowed_ring_ceilings=3.0 * amps, **common)
    capped.fit(sl, q_range=(1.5, 3.2))
    out_capped, _ = capped.subtract(sl)

    # The across-H amplitude reflects the NORMAL ring (the one-plane inflation
    # washes out of the median), so the ceiling sits just above the real ring.
    j = int(np.argmin(np.abs(centers - 2.6)))
    assert 2.0 < amps[j] < 6.0
    # Without the ceiling the inflated amplitude over-subtracts along the ring;
    # the ceiling makes the residual much less negative.
    assert out_nocap.data[0].min() < -1.0
    assert out_capped.data[0].min() > 0.5 * out_nocap.data[0].min()
