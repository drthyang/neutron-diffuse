"""End-to-end integration test of the full pipeline on synthetic data."""

import numpy as np
import pytest

from ndiff.core import HKLVolume
from ndiff.preprocessing import PowderRingRemover, backfill
from ndiff.analysis import bragg_mask, backfill_bragg, compute_delta_pdf
from ndiff.preprocessing.powder_rings import _gaussian


def _synthetic_vol(shape=(25, 25, 25), seed=0):
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

    # Smooth anisotropic diffuse signal
    diffuse = 1.0 + 0.4 * np.cos(np.pi * H) * np.cos(np.pi * K) + 0.2 * np.cos(2 * np.pi * L)
    # Powder ring
    ring = _gaussian(q_mag, 60.0, 2.6, 0.05)
    # Bragg spikes at integer positions
    bragg = np.zeros(shape)
    for hb, kb, lb in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0)]:
        ih = int(np.argmin(np.abs(h - hb)))
        ik = int(np.argmin(np.abs(k - kb)))
        il = int(np.argmin(np.abs(l - lb)))
        bragg[ih, ik, il] = 500.0
    data = diffuse + ring + bragg + rng.normal(0, 0.05, shape)
    return HKLVolume.from_arrays(data, (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub)


def test_full_pipeline_runs_and_produces_finite_dpdf():
    vol = _synthetic_vol()

    # (1) Powder ring removal
    remover = PowderRingRemover(snr_mask_threshold=2.0, detect_kwargs={"sigma_threshold": 3.0})
    vol_sub, rings, I_ring = remover.remove(vol)
    assert np.isfinite(vol_sub.data).all()

    # (2) Backfill ring holes
    vol_clean = backfill(vol_sub, method="tv", tv_lam=0.1, tv_iter=150)
    assert np.isfinite(vol_clean.data).all()
    assert vol_clean.mask.all()

    # (3) Bragg punch
    b_keep = bragg_mask(vol_clean, punch_radius_hkl=0.35, taper=0.0)
    vol_clean.apply_mask(b_keep)

    # (4) Backfill Bragg holes
    vol_diffuse = backfill_bragg(vol_clean, method="tv", tv_lam=0.2)
    assert np.isfinite(vol_diffuse.data).all()

    # (5) 3D-ΔPDF
    dpdf = compute_delta_pdf(vol_diffuse, apodization="hann", zero_pad=False)
    assert dpdf.data.shape == vol.data.shape
    assert np.isfinite(dpdf.data).all()
