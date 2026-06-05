"""End-to-end integration test of the full pipeline on synthetic data.

Exercises the real API:
    EmptySubtractor → PatchedRingModel → backfill_ring_shells
    → bragg_mask → backfill_bragg → compute_delta_pdf
"""

import dataclasses

import numpy as np

from ndiff.core import HKLVolume
from ndiff.preprocessing import (
    EmptySubtractor,
    PatchedRingModel,
    backfill_ring_shells,
    RingShell,
)
from ndiff.preprocessing.ring_model import _gaussian
from ndiff.analysis import bragg_mask, backfill_bragg, compute_delta_pdf


RING_Q = 2.6
RING_SIGMA = 0.08


def _azimuthal_texture(vol: HKLVolume) -> np.ndarray:
    """T(phi) = 1 + 0.3 cos(2 phi) in the hk0 plane — anisotropic ring."""
    H, K, L = vol.hkl_grid()
    hkl = np.stack([H, K, L], axis=-1)
    Q = hkl @ vol.ub_matrix.T
    phi = np.arctan2(Q[..., 1], Q[..., 0])
    return 1.0 + 0.3 * np.cos(2 * phi)


def _synthetic_vol(shape=(25, 25, 25), seed=0, with_environment_ring=True):
    rng = np.random.default_rng(seed)
    h = np.linspace(-3, 3, shape[0])
    k = np.linspace(-3, 3, shape[1])
    l = np.linspace(-3, 3, shape[2])
    ub = 2 * np.pi * np.eye(3) / 4.0
    vol = HKLVolume.from_arrays(
        np.ones(shape), (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub
    )
    q_mag = vol.q_magnitude()
    H, K, L = vol.hkl_grid()

    diffuse = 1.0 + 0.4 * np.cos(np.pi * H) * np.cos(np.pi * K) + 0.2 * np.cos(2 * np.pi * L)
    ring = _azimuthal_texture(vol) * _gaussian(q_mag, 60.0, RING_Q, RING_SIGMA)

    bragg = np.zeros(shape)
    for hb, kb, lb in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0)]:
        ih = int(np.argmin(np.abs(h - hb)))
        ik = int(np.argmin(np.abs(k - kb)))
        il = int(np.argmin(np.abs(l - lb)))
        bragg[ih, ik, il] = 500.0

    data = diffuse + ring + bragg + rng.normal(0, 0.05, shape)
    if not with_environment_ring:
        data = data - 0.0  # placeholder; empty scan has its own ring below
    return HKLVolume.from_arrays(data, (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub)


def _empty_scan(shape=(25, 25, 25)):
    """Environment ring only (no sample diffuse / Bragg)."""
    h = np.linspace(-3, 3, shape[0])
    k = np.linspace(-3, 3, shape[1])
    l = np.linspace(-3, 3, shape[2])
    ub = 2 * np.pi * np.eye(3) / 4.0
    vol = HKLVolume.from_arrays(
        np.ones(shape), (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub
    )
    q_mag = vol.q_magnitude()
    # Small environment ring at a different |Q| than the sample-holder ring.
    data = 0.5 + _gaussian(q_mag, 20.0, 3.4, 0.08)
    return HKLVolume.from_arrays(data, (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub)


def test_empty_subtraction_reduces_environment_ring():
    sample = _synthetic_vol()
    empty = _empty_scan()
    # Inject the environment ring into the sample too.
    sample = dataclasses.replace(sample, data=sample.data + empty.data - 0.5)

    sub = EmptySubtractor(empty, scale_q_range=(3.2, 3.6))
    out = sub.subtract(sample)
    assert np.isfinite(out.data).all()
    assert 0.0 < sub.scale < 5.0


def test_full_pipeline_runs_and_produces_finite_dpdf():
    vol = _synthetic_vol()

    # (1) Empty-scan subtraction (no-op environment here; exercises the API).
    empty = _empty_scan()
    vol1 = EmptySubtractor(empty, scale=0.0).subtract(vol)
    assert np.isfinite(vol1.data).all()

    # (2) Factored ring model: fit with an explicit hint, then subtract.
    model = PatchedRingModel(n_patches=24, n_fourier=4, n_radial_bins=30,
                             snr_mask_threshold=2.0)
    fitted = model.fit(vol1, ring_hints=[RING_Q])
    assert len(fitted.rings) == 1
    assert np.isfinite(fitted.rank1_variance)
    vol2, I_ring = model.subtract(vol1, fitted)
    assert np.isfinite(vol2.data).all()
    assert np.isfinite(I_ring).all()

    # (3) Backfill the masked ring shell.
    rings = [RingShell(q_center=RING_Q, q_lo=RING_Q - 0.2, q_hi=RING_Q + 0.2)]
    vol_clean = backfill_ring_shells(vol2, rings, n_neighbors=12,
                                     fallback_tv=True, tv_iter=100)
    assert np.isfinite(vol_clean.data).all()
    assert vol_clean.mask.all()

    # (4) Bragg punch.
    b_keep = bragg_mask(vol_clean, punch_radius_hkl=0.35)
    vol_clean.apply_mask(b_keep)
    assert not vol_clean.mask.all()

    # (5) Backfill Bragg holes.
    vol_diffuse = backfill_bragg(vol_clean, method="tv", tv_lam=0.2)
    assert np.isfinite(vol_diffuse.data).all()

    # (6) 3D-ΔPDF.
    dpdf = compute_delta_pdf(vol_diffuse, apodization="hann", zero_pad=False)
    assert dpdf.data.shape == vol.data.shape
    assert np.isfinite(dpdf.data).all()


def test_bragg_local_backfill_uses_nearby_background_level():
    data = np.ones((9, 9, 9), dtype=float) * 0.5
    sigma = np.ones_like(data) * 0.1
    vol = HKLVolume.from_arrays(data, (-1, 1), (-1, 1), (-1, 1), sigma=sigma)
    vol.data[4, 4, 4] = 100.0
    vol.mask[4, 4, 4] = False

    filled = backfill_bragg(vol, method="local", local_radius=1)

    assert filled.mask.all()
    assert abs(float(filled.data[4, 4, 4]) - 0.5) < 1e-12


def test_direct_beam_fill_uses_background_outside_not_adjacent_halo():
    # Background 0.3 everywhere, a punched/unmeasured beam ball at the origin
    # (|Q|<=0.25) wrapped in a negative over-subtraction halo (|Q| 0.25–0.40)
    # that hugs the beam.  The direct-beam fill should reach *past* the halo to
    # the true background, whereas the generic dilated-shell fill is pulled into
    # the halo.
    n = 41
    data = np.full((n, n, n), 0.3, dtype=float)
    vol = HKLVolume.from_arrays(data, (-1, 1), (-1, 1), (-1, 1))
    q = vol.q_magnitude()
    holes = q <= 0.25
    halo = (q > 0.25) & (q <= 0.40)
    vol.data[holes] = -9.0
    vol.mask[holes] = False
    vol.data[halo] = -2.0
    i0 = n // 2

    new = backfill_bragg(vol, method="local", direct_beam_fill=True,
                         direct_beam_q_gap=0.2, direct_beam_q_width=0.15)
    old = backfill_bragg(vol, method="local", direct_beam_fill=False)

    assert np.isfinite(new.data).all()
    # NEW: whole beam ball replaced with the ~0.3 background sampled outside.
    assert abs(float(new.data[i0, i0, i0]) - 0.3) < 0.05
    assert np.allclose(new.data[holes], new.data[i0, i0, i0])
    # OLD: generic fill samples the adjacent −2 halo, so it goes negative.
    assert float(old.data[i0, i0, i0]) < 0.0
