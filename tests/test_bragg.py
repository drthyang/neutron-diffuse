"""Tests for Bragg peak detection and 3D-ΔPDF."""

import numpy as np
import pytest

from ndiff.analysis.bragg import BraggRemover, bragg_mask
from ndiff.analysis.delta_pdf import compute_delta_pdf, _next_power_of_2
from ndiff.core import HKLVolume


def _make_vol(shape=(15, 15, 15), hkl_range=(-2, 2)):
    data = np.random.default_rng(42).uniform(0.5, 1.5, shape)
    h = np.linspace(hkl_range[0], hkl_range[1], shape[0])
    k = np.linspace(hkl_range[0], hkl_range[1], shape[1])
    l = np.linspace(hkl_range[0], hkl_range[1], shape[2])
    ub = 2 * np.pi * np.eye(3) / 4.0
    return HKLVolume.from_arrays(data, (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub)


def test_bragg_mask_removes_integer_positions():
    vol = _make_vol()
    mask = bragg_mask(vol, punch_radius_hkl=0.4, taper=0.0)
    # (0,0,0) should always be punched
    ih0 = np.argmin(np.abs(vol.h_axis))
    ik0 = np.argmin(np.abs(vol.k_axis))
    il0 = np.argmin(np.abs(vol.l_axis))
    assert not mask[ih0, ik0, il0], "Origin (0,0,0) should be punched"


def test_bragg_mask_preserves_non_integer():
    vol = _make_vol()
    mask = bragg_mask(vol, punch_radius_hkl=0.25, taper=0.0)
    # A voxel at hkl ≈ (0.5, 0.5, 0.5) should not be punched
    ih = np.argmin(np.abs(vol.h_axis - 0.5))
    ik = np.argmin(np.abs(vol.k_axis - 0.5))
    il = np.argmin(np.abs(vol.l_axis - 0.5))
    assert mask[ih, ik, il], "Non-integer HKL should not be punched"


def test_next_power_of_2():
    assert _next_power_of_2(1) == 1
    assert _next_power_of_2(7) == 8
    assert _next_power_of_2(16) == 16
    assert _next_power_of_2(17) == 32


def test_delta_pdf_shape_and_finite():
    vol = _make_vol(shape=(8, 8, 8))
    dpdf = compute_delta_pdf(vol, apodization="hann", zero_pad=False)
    assert dpdf.data.shape == (8, 8, 8)
    assert np.isfinite(dpdf.data).all()


def test_delta_pdf_zero_pad_increases_size():
    vol = _make_vol(shape=(10, 10, 10))
    dpdf = compute_delta_pdf(vol, apodization="none", zero_pad=True)
    assert all(s >= 10 for s in dpdf.data.shape)
    # next power of 2 after 10 is 16
    assert dpdf.data.shape == (16, 16, 16)


def test_delta_pdf_hk0_slice():
    vol = _make_vol(shape=(8, 8, 8))
    dpdf = compute_delta_pdf(vol, apodization="none", zero_pad=False)
    sl = dpdf.slice_hk0()
    assert sl.shape == (8, 8)
