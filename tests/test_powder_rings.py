"""Tests for powder ring detection, subtraction, and backfill."""

import numpy as np
import pytest

from ndiff.core import HKLVolume
from ndiff.preprocessing.powder_rings import (
    detect_rings, subtract_rings, PowderRing, al_ring_q_positions, _gaussian,
)
from ndiff.preprocessing.backfill import backfill


def _make_vol_with_ring(shape=(30, 30, 30), ring_q=2.5, ring_amp=50.0, ring_sigma=0.04):
    """Synthetic volume: smooth diffuse + one Gaussian powder ring."""
    h = np.linspace(-3, 3, shape[0])
    k = np.linspace(-3, 3, shape[1])
    l = np.linspace(-3, 3, shape[2])
    ub = 2 * np.pi * np.eye(3) / 4.0  # cubic, a=4 Å
    vol = HKLVolume.from_arrays(
        np.ones(shape), (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub
    )
    q_mag = vol.q_magnitude()
    # smooth diffuse background (slowly varying)
    diffuse = 1.0 + 0.3 * np.cos(np.pi * q_mag)
    # powder ring
    ring = _gaussian(q_mag, ring_amp, ring_q, ring_sigma)
    data = diffuse + ring + np.random.default_rng(0).normal(0, 0.05, shape)
    return HKLVolume.from_arrays(data, (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub)


def test_detect_finds_injected_ring():
    ring_q = 2.5
    vol = _make_vol_with_ring(ring_q=ring_q, ring_amp=100.0)
    rings = detect_rings(vol, n_bins=200, sigma_threshold=3.0)
    assert len(rings) >= 1, "Should detect at least one ring"
    detected_q = [r.q_center for r in rings]
    assert any(abs(q - ring_q) < 0.2 for q in detected_q), \
        f"Ring at {ring_q} not found; detected: {detected_q}"


def test_subtract_reduces_ring_signal():
    ring_q, ring_amp = 2.5, 80.0
    vol = _make_vol_with_ring(ring_q=ring_q, ring_amp=ring_amp)
    q_mag = vol.q_magnitude()

    # find voxels near ring centre before subtraction
    near_ring = np.abs(q_mag - ring_q) < 0.02
    mean_before = float(vol.data[near_ring].mean())

    rings = detect_rings(vol, n_bins=200, sigma_threshold=3.0)
    vol_sub, I_ring = subtract_rings(vol, rings)

    mean_after = float(vol_sub.data[near_ring & vol_sub.mask].mean()) if vol_sub.mask[near_ring].any() else 0.0
    assert mean_after < mean_before * 0.3, \
        "Subtraction should reduce intensity near ring by > 70 %"


def test_backfill_produces_finite_values():
    vol = _make_vol_with_ring()
    rings = detect_rings(vol, n_bins=200, sigma_threshold=3.0)
    if not rings:
        pytest.skip("No rings detected on this synthetic volume")
    vol_sub, _ = subtract_rings(vol, rings)
    vol_filled = backfill(vol_sub, method="tv", tv_lam=0.1, tv_iter=100)
    assert np.isfinite(vol_filled.data).all()
    assert vol_filled.mask.all(), "All voxels should be valid after backfill"


def test_backfill_rbf_finite():
    vol = _make_vol_with_ring(shape=(15, 15, 15))
    rings = detect_rings(vol, n_bins=100, sigma_threshold=3.0)
    if not rings:
        pytest.skip("No rings detected")
    vol_sub, _ = subtract_rings(vol, rings)
    vol_filled = backfill(vol_sub, method="rbf", rbf_neighbors=20)
    assert np.isfinite(vol_filled.data).all()


def test_al_ring_positions_known_peaks():
    qs = al_ring_q_positions(a=4.0494, q_max=8.0)
    # Al 111: |Q| = 2π√3/a
    q_111 = 2 * np.pi * np.sqrt(3) / 4.0494
    assert any(abs(q - q_111) < 1e-3 for q in qs), "Al 111 peak missing"
    # Al 100 is forbidden in FCC
    q_100 = 2 * np.pi / 4.0494
    assert not any(abs(q - q_100) < 1e-3 for q in qs), "Forbidden Al 100 peak found"


def test_user_specified_rings():
    """User can provide ring positions instead of auto-detecting."""
    vol = _make_vol_with_ring(ring_q=2.5, ring_amp=60.0)
    rings = [PowderRing(q_center=2.5, q_sigma=0.05, amplitude=60.0)]
    vol_sub, I_ring = subtract_rings(vol, rings)
    assert I_ring.max() > 0, "Ring profile should be non-zero"
    assert np.isfinite(vol_sub.data).all()
