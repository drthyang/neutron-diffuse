"""End-to-end integration test: run the full 6-step pipeline on synthetic data."""

import numpy as np
import pytest

from ndiff.core import HKLVolume
from ndiff.preprocessing import symmetrize, aluminum_mask, backfill_al
from ndiff.analysis import bragg_mask, backfill_bragg, compute_delta_pdf


def _synthetic_vol(shape=(20, 20, 20), seed=0):
    rng = np.random.default_rng(seed)
    # Smooth diffuse signal + strong Bragg spikes at integer hkl
    h = np.linspace(-2, 2, shape[0])
    k = np.linspace(-2, 2, shape[1])
    l = np.linspace(-2, 2, shape[2])
    H, K, L = np.meshgrid(h, k, l, indexing="ij")
    diffuse = 1.0 + 0.3 * np.cos(np.pi * H) * np.cos(np.pi * K)
    bragg_spikes = np.zeros(shape)
    for hb, kb, lb in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]:
        ih = np.argmin(np.abs(h - hb))
        ik = np.argmin(np.abs(k - kb))
        il = np.argmin(np.abs(l - lb))
        bragg_spikes[ih, ik, il] = 1000.0
    data = diffuse + bragg_spikes + rng.normal(0, 0.05, shape)
    ub = 2 * np.pi * np.eye(3) / 4.0
    return HKLVolume.from_arrays(data, (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub)


def test_full_pipeline_runs():
    vol = _synthetic_vol()

    # (1) Symmetrize
    vol_sym, _ = symmetrize(vol, laue_class="mmm")
    assert vol_sym.data.shape == vol.data.shape

    # (2) Remove Al signals
    al_keep = aluminum_mask(vol_sym, width_angstrom=0.15, width_sigma=None)
    vol_sym.apply_mask(al_keep)

    # (3) Backfill Al
    vol_clean = backfill_al(vol_sym, method="tv", tv_lam=0.1)
    assert np.isfinite(vol_clean.data).all()

    # (4) Remove Bragg peaks
    b_keep = bragg_mask(vol_clean, punch_radius_hkl=0.35, taper=0.0)
    vol_clean.apply_mask(b_keep)
    assert not vol_clean.mask.all()  # some voxels should be punched

    # (5) Backfill Bragg holes
    vol_diffuse = backfill_bragg(vol_clean, method="tv", tv_lam=0.2)
    assert np.isfinite(vol_diffuse.data).all()

    # (6) 3D-ΔPDF
    dpdf = compute_delta_pdf(vol_diffuse, apodization="hann", zero_pad=False)
    assert dpdf.data.shape == vol.data.shape
    assert np.isfinite(dpdf.data).all()
