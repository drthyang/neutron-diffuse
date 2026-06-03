"""Tests for the non-parametric per-patch radial-background ring removal."""

import numpy as np

from ndiff.core import HKLVolume
from ndiff.preprocessing import PatchedRadialRingModel, azimuthal_sampling_mask
from ndiff.preprocessing.ring_model import _gaussian


def _ring_vol(shape=(1, 81, 81), ring_q=2.6, ring_sigma=0.05, seed=0):
    """Single 2D slice (hk0) with a textured ring + diffuse + Bragg peaks."""
    rng = np.random.default_rng(seed)
    h = np.linspace(0, 0, shape[0])
    k = np.linspace(-4, 4, shape[1])
    l = np.linspace(-4, 4, shape[2])
    ub = 2 * np.pi * np.eye(3) / 4.0
    vol = HKLVolume.from_arrays(
        np.ones(shape), (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub
    )
    q = vol.q_magnitude()
    H, K, L = vol.hkl_grid()
    Q = np.stack([H, K, L], axis=-1) @ ub.T
    phi = np.arctan2(Q[..., 1], Q[..., 0])

    diffuse = 1.0 + 0.3 * np.cos(np.pi * K) * np.cos(np.pi * L)
    texture = 1.0 + 0.4 * np.cos(2 * phi)
    ring = texture * _gaussian(q, 3.0, ring_q, ring_sigma)
    bragg = np.zeros(shape)
    for kb, lb in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1)]:
        ik = int(np.argmin(np.abs(k - kb)))
        il = int(np.argmin(np.abs(l - lb)))
        bragg[0, ik, il] = 200.0
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


def test_azimuthal_sampling_mask_drops_undersampled_sector():
    vol, *_ = _ring_vol()
    # Carve a sparse azimuthal wedge by masking most of it (keep a few voxels).
    H, K, L = vol.hkl_grid()
    Q = np.stack([H, K, L], axis=-1) @ vol.ub_matrix.T
    phi = np.arctan2(Q[..., 1], Q[..., 0])
    wedge = np.abs(phi - 0.6) < 0.25
    rng = np.random.default_rng(1)
    thin = wedge & (rng.random(vol.shape) > 0.05)   # keep ~5% of the wedge
    vol.mask[thin] = False

    keep = azimuthal_sampling_mask(vol, plane="hk0", n_phi_bins=36,
                                   n_q_bins=40, min_count=8, q_range=(1.0, 4.0))
    # The thinned wedge should be (mostly) dropped; the dense rest retained.
    assert keep.sum() < vol.mask.sum()
    dense = vol.mask & ~wedge
    assert keep[dense].mean() > 0.9
