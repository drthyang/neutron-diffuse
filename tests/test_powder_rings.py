"""Tests for powder ring detection, masking, and backfill."""

import numpy as np
import pytest

from ndiff.core import HKLVolume
from ndiff.preprocessing.backfill import backfill_ring_shells
from ndiff.preprocessing.powder_rings import (
    RingShell,
    al_ring_q_positions,
    detect_ring_shells,
    line_profile,
    mask_ring_shells,
    radial_profile,
)


def _make_vol_with_ring(
    shape=(30, 30, 30), ring_q=2.5, ring_amp=40.0, ring_sigma=0.06, seed=0
):
    """Smooth diffuse + one Gaussian powder ring in |Q|."""
    h = np.linspace(-3, 3, shape[0])
    k = np.linspace(-3, 3, shape[1])
    l = np.linspace(-3, 3, shape[2])
    ub = 2 * np.pi * np.eye(3) / 4.0
    vol_base = HKLVolume.from_arrays(
        np.ones(shape), (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub
    )
    q_mag = vol_base.q_magnitude()
    diffuse = 1.0 + 0.3 * np.cos(np.pi * q_mag)
    ring = ring_amp * np.exp(-0.5 * ((q_mag - ring_q) / ring_sigma) ** 2)
    noise = np.random.default_rng(seed).normal(0, 0.05, shape)
    data = diffuse + ring + noise
    return HKLVolume.from_arrays(
        data, (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub
    )


# --- radial_profile ---

def test_radial_profile_shape():
    vol = _make_vol_with_ring()
    q_c, prof, counts = radial_profile(vol, n_bins=100)
    assert q_c.shape == (100,)
    assert prof.shape == (100,)
    assert counts.shape == (100,)


def test_radial_profile_shows_ring_peak():
    ring_q = 2.5
    vol = _make_vol_with_ring(ring_q=ring_q, ring_amp=60.0)
    q_c, prof, _ = radial_profile(vol, n_bins=200)
    peak_q = float(q_c[np.nanargmax(prof)])
    assert abs(peak_q - ring_q) < 0.3


# --- detect_ring_shells ---

def test_detect_finds_injected_ring():
    ring_q = 2.5
    vol = _make_vol_with_ring(ring_q=ring_q, ring_amp=60.0)
    rings, *_ = detect_ring_shells(vol, n_bins=200, baseline_window=30,
                                   sigma_threshold=3.0)
    assert len(rings) >= 1
    assert any(abs(r.q_center - ring_q) < 0.3 for r in rings)


def test_detect_returns_valid_shell_range():
    vol = _make_vol_with_ring(ring_q=2.5, ring_amp=50.0)
    rings, *_ = detect_ring_shells(vol, n_bins=200, baseline_window=30,
                                   sigma_threshold=3.0)
    for r in rings:
        assert r.q_lo < r.q_center < r.q_hi
        assert r.amplitude > 0


def test_detect_no_ring_in_clean_data():
    h = np.linspace(-3, 3, 20)
    k = np.linspace(-3, 3, 20)
    l = np.linspace(-3, 3, 20)
    ub = 2 * np.pi * np.eye(3) / 4.0
    data = np.ones((20, 20, 20)) + np.random.default_rng(7).normal(0, 0.02, (20, 20, 20))
    vol = HKLVolume.from_arrays(data, (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub)
    rings, *_ = detect_ring_shells(vol, n_bins=100, baseline_window=20, sigma_threshold=8.0)
    assert len(rings) == 0, f"False positives detected: {rings}"


# --- mask_ring_shells ---

def test_mask_removes_ring_voxels():
    ring_q = 2.5
    vol = _make_vol_with_ring(ring_q=ring_q)
    rings, *_ = detect_ring_shells(vol, n_bins=200, baseline_window=30,
                                   sigma_threshold=3.0)
    if not rings:
        pytest.skip("No rings detected")
    keep = mask_ring_shells(vol, rings, taper_width=0.005)
    # Voxels near the ring centre should be masked
    q_mag = vol.q_magnitude()
    near_peak = np.abs(q_mag - ring_q) < 0.05
    if near_peak.any():
        assert not keep[near_peak].all(), "Ring centre voxels should be masked"


def test_mask_preserves_off_ring_voxels():
    vol = _make_vol_with_ring(ring_q=2.5)
    rings = [RingShell(q_center=2.5, q_lo=2.35, q_hi=2.65)]
    keep = mask_ring_shells(vol, rings, taper_width=0.005)
    q_mag = vol.q_magnitude()
    far = q_mag < 1.5
    assert keep[far].all(), "Voxels far from ring should not be masked"


def test_mask_boundary_is_soft():
    """Check that voxels near the mask edge have intermediate weight (sigmoid)."""
    vol = _make_vol_with_ring()
    rings = [RingShell(q_center=2.5, q_lo=2.35, q_hi=2.65)]
    keep = mask_ring_shells(vol, rings, taper_width=0.02)
    # Should have some True and some False near q=2.35
    q_mag = vol.q_magnitude()
    boundary_region = (q_mag > 2.30) & (q_mag < 2.40)
    if boundary_region.any():
        assert keep[boundary_region].any() and not keep[boundary_region].all()


# --- backfill ---

def test_backfill_produces_finite_values():
    ring_q = 2.5
    vol = _make_vol_with_ring(ring_q=ring_q)
    rings, *_ = detect_ring_shells(vol, n_bins=200, baseline_window=30,
                                   sigma_threshold=3.0)
    if not rings:
        pytest.skip("No rings detected")
    keep = mask_ring_shells(vol, rings, taper_width=0.005)
    import dataclasses
    vol_masked = dataclasses.replace(vol, mask=vol.mask & keep)
    vol_filled = backfill_ring_shells(vol_masked, rings, n_neighbors=12,
                                      fallback_tv=False)
    assert np.isfinite(vol_filled.data).all()


def test_backfill_values_near_diffuse_level():
    """Filled values should be near the diffuse level (not ring level)."""
    ring_q, ring_amp = 2.5, 50.0
    vol = _make_vol_with_ring(ring_q=ring_q, ring_amp=ring_amp)
    rings = [RingShell(q_center=ring_q, q_lo=ring_q - 0.15, q_hi=ring_q + 0.15)]
    keep = mask_ring_shells(vol, rings)
    import dataclasses
    vol_masked = dataclasses.replace(vol, mask=vol.mask & keep)
    vol_filled = backfill_ring_shells(vol_masked, rings, n_neighbors=16, fallback_tv=False)

    # Filled voxels should be much closer to 1.0 (diffuse) than to 1 + ring_amp
    filled_region = ~keep
    if filled_region.any():
        filled_vals = vol_filled.data[filled_region]
        assert float(np.mean(filled_vals)) < ring_amp * 0.5


# --- al_ring_q_positions ---

def test_al_known_peaks():
    qs = al_ring_q_positions(a=4.0494, q_max=8.0)
    q_111 = 2 * np.pi * np.sqrt(3) / 4.0494
    assert any(abs(q - q_111) < 1e-3 for q in qs)
    # FCC forbidden: 100
    q_100 = 2 * np.pi / 4.0494
    assert not any(abs(q - q_100) < 1e-3 for q in qs)


# --- line_profile ---

def test_line_profile_q_and_intensity():
    # data == |Q| everywhere, so the interpolated linecut intensity must equal
    # the line's |Q| (within interpolation error), and |Q| must be monotonic
    # along (0, 1, l).
    shape = (1, 41, 81)
    h = np.linspace(0, 0, shape[0])
    k = np.linspace(-5, 5, shape[1])
    l = np.linspace(-10, 10, shape[2])
    ub = 2 * np.pi * np.eye(3) / 4.0
    vol = HKLVolume.from_arrays(
        np.ones(shape), (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub
    )
    import dataclasses
    vol = dataclasses.replace(vol, data=vol.q_magnitude())

    q, I, hkl = line_profile(vol, (0, 1, 0), (0, 1, 10), n_points=200)
    assert q.shape == I.shape == (200,)
    assert hkl.shape == (200, 3)
    assert np.all(np.diff(q) >= -1e-9)                 # monotonic in |Q|
    good = np.isfinite(I)
    assert good.mean() > 0.95
    assert np.allclose(I[good], q[good], atol=0.05)    # data==|Q| recovered
