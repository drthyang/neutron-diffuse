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
    mask = bragg_mask(vol, punch_radius_hkl=0.4)
    # (0,0,0) should always be punched (legacy punch-all path)
    ih0 = np.argmin(np.abs(vol.h_axis))
    ik0 = np.argmin(np.abs(vol.k_axis))
    il0 = np.argmin(np.abs(vol.l_axis))
    assert not mask[ih0, ik0, il0], "Origin (0,0,0) should be punched"


def test_bragg_mask_preserves_non_integer():
    vol = _make_vol()
    mask = bragg_mask(vol, punch_radius_hkl=0.25)
    # A voxel at hkl ≈ (0.5, 0.5, 0.5) should not be punched
    ih = np.argmin(np.abs(vol.h_axis - 0.5))
    ik = np.argmin(np.abs(vol.k_axis - 0.5))
    il = np.argmin(np.abs(vol.l_axis - 0.5))
    assert mask[ih, ik, il], "Non-integer HKL should not be punched"


def _peaky_vol(shape=(21, 21, 21), hkl_range=(-2, 2)):
    """Diffuse background with sharp Bragg peaks at SOME integer nodes only
    (mimicking systematic absences) and an off-integer peak centre."""
    rng = np.random.default_rng(7)
    data = rng.uniform(0.5, 1.5, shape)
    h = np.linspace(hkl_range[0], hkl_range[1], shape[0])
    k = np.linspace(hkl_range[0], hkl_range[1], shape[1])
    l = np.linspace(hkl_range[0], hkl_range[1], shape[2])
    ub = 2 * np.pi * np.eye(3) / 4.0
    vol = HKLVolume.from_arrays(data, (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]),
                               ub_matrix=ub)
    present = {(0, 0, 0): 200.0, (1, 0, 0): 50.0, (-1, 1, 0): 30.0}
    for (h0, k0, l0), amp in present.items():
        ih = int(np.argmin(np.abs(vol.h_axis - h0)))
        ik = int(np.argmin(np.abs(vol.k_axis - k0)))
        il = int(np.argmin(np.abs(vol.l_axis - l0)))
        vol.data[ih, ik, il] = amp
    return vol, present


def test_bragg_detect_skips_absent_nodes():
    vol, present = _peaky_vol()
    remover = BraggRemover(punch_radii=(0.25, 0.25, 0.25), min_intensity=10.0)
    detected = remover.detect_peaks(vol)
    assert len(detected) == len(present)          # only the real peaks
    # An empty node, e.g. (2,2,2), is NOT punched (preserve diffuse at absences).
    mask = remover.build_mask(vol)
    ih = int(np.argmin(np.abs(vol.h_axis - 2)))
    assert mask[ih, ih, ih]
    # ... while a present peak IS punched.
    i0 = int(np.argmin(np.abs(vol.h_axis)))
    assert not mask[i0, i0, i0]


def test_bragg_anisotropic_radii_punch_more_along_broad_axis():
    vol, _ = _peaky_vol()
    mask = bragg_mask(vol, punch_radii=(0.1, 0.1, 0.6), min_intensity=10.0)
    i0 = int(np.argmin(np.abs(vol.h_axis)))           # origin peak voxel
    punched = ~mask
    # Count punched voxels along H vs L lines through the origin: L (broad) > H.
    nh = int(punched[:, i0, i0].sum())
    nl = int(punched[i0, i0, :].sum())
    assert nl > nh


def test_bragg_intensity_scaling_enlarges_bright_peaks():
    vol, _ = _peaky_vol()
    i0 = int(np.argmin(np.abs(vol.h_axis)))
    base = BraggRemover(punch_radii=(0.2, 0.2, 0.2), min_intensity=10.0)
    scaled = BraggRemover(punch_radii=(0.2, 0.2, 0.2), min_intensity=10.0,
                          intensity_scale=True, intensity_ref=30.0)
    n_base = int((~base.build_mask(vol))[:, i0, i0].sum())
    n_scaled = int((~scaled.build_mask(vol))[:, i0, i0].sum())
    assert n_scaled > n_base                          # the bright origin grows


def test_search_mode_punches_off_integer_satellite():
    """Search mode catches a sharp peak at a NON-integer position that the
    integer mode misses (e.g. a superlattice / small-domain satellite)."""
    vol, _ = _peaky_vol()
    # add a sharp satellite at a half-integer position
    sh = int(np.argmin(np.abs(vol.h_axis - 0.5)))
    sk = int(np.argmin(np.abs(vol.k_axis - 0.5)))
    sl = int(np.argmin(np.abs(vol.l_axis - 1.0)))
    vol.data[sh, sk, sl] = 60.0

    integer = BraggRemover(mode="integer", punch_radii=(0.25, 0.25, 0.25),
                           min_intensity=10.0)
    search = BraggRemover(mode="search", punch_radii=(0.25, 0.25, 0.25),
                          search_n_mad=6.0, search_min_intensity=10.0,
                          search_q_step=0.5)

    assert integer.build_mask(vol)[sh, sk, sl]        # integer mode leaves it
    assert not search.build_mask(vol)[sh, sk, sl]     # search mode punches it
    # search also removes an off-origin integer Bragg (a sharp outlier on a
    # well-populated |Q| shell); (0,0,0) at |Q|=0 is a sparse-shell edge case.
    i1h = int(np.argmin(np.abs(vol.h_axis - 1)))
    i1 = int(np.argmin(np.abs(vol.k_axis)))
    assert not search.build_mask(vol)[i1h, i1, i1]


def test_search_mode_forces_origin_punch():
    vol, _ = _peaky_vol()
    search = BraggRemover(mode="search", punch_radii=(0.25, 0.25, 0.25),
                          search_n_mad=6.0, search_min_intensity=10.0,
                          search_q_step=0.5)
    i0 = int(np.argmin(np.abs(vol.h_axis)))
    assert not search.build_mask(vol)[i0, i0, i0]


def test_phi_tail_expands_punch_along_ring_tangent():
    vol, _ = _peaky_vol()
    ih = int(np.argmin(np.abs(vol.h_axis - 0)))
    ik = int(np.argmin(np.abs(vol.k_axis - 1)))
    il = int(np.argmin(np.abs(vol.l_axis - 0)))
    vol.data[ih, ik, il] = 100.0

    base = BraggRemover(mode="integer", punch_radii=(0.2, 0.2, 0.2),
                        min_intensity=10.0, force_origin=False)
    phi = BraggRemover(mode="integer", punch_radii=(0.2, 0.2, 0.2),
                       min_intensity=10.0, force_origin=False,
                       phi_tail_hkl=0.4)

    base_line = int((~base.build_mask(vol))[ih, ik, :].sum())
    phi_line = int((~phi.build_mask(vol))[ih, ik, :].sum())
    assert phi_line > base_line


def test_auto_mode_aliases_search_mode():
    vol, _ = _peaky_vol()
    sh = int(np.argmin(np.abs(vol.h_axis - 0.5)))
    sk = int(np.argmin(np.abs(vol.k_axis - 0.5)))
    sl = int(np.argmin(np.abs(vol.l_axis - 1.0)))
    vol.data[sh, sk, sl] = 60.0
    auto = BraggRemover(mode="auto", punch_radii=(0.25, 0.25, 0.25),
                        search_n_mad=6.0, search_min_intensity=10.0,
                        search_q_step=0.5)
    assert not auto.build_mask(vol)[sh, sk, sl]


def test_search_prominence_rejects_broad_diffuse_bump():
    vol, _ = _peaky_vol(shape=(31, 31, 31), hkl_range=(-3, 3))
    H, K, L = vol.hkl_grid()
    broad = 5.0 * np.exp(-0.5 * (((H - 0.6) / 0.45) ** 2
                                  + ((K - 0.4) / 0.45) ** 2
                                  + ((L - 0.2) / 0.45) ** 2))
    vol.data += broad
    ih = int(np.argmin(np.abs(vol.h_axis - 0.6)))
    ik = int(np.argmin(np.abs(vol.k_axis - 0.4)))
    il = int(np.argmin(np.abs(vol.l_axis - 0.2)))

    sh = int(np.argmin(np.abs(vol.h_axis + 1.4)))
    sk = int(np.argmin(np.abs(vol.k_axis - 1.2)))
    sl = int(np.argmin(np.abs(vol.l_axis + 0.8)))
    vol.data[sh, sk, sl] = 8.0

    auto = BraggRemover(mode="auto", punch_radii=(0.2, 0.2, 0.2),
                        search_n_mad=3.0, search_min_intensity=1.0,
                        search_min_prominence=1.0, search_q_step=0.5)
    keep = auto.build_mask(vol)
    assert keep[ih, ik, il]           # broad diffuse maximum survives
    assert not keep[sh, sk, sl]       # sharp satellite is punched


def test_both_mode_is_sequential_union():
    """'both' = integer punch, then search on the residual; it removes both an
    integer peak and an off-integer satellite."""
    vol, _ = _peaky_vol()
    sh = int(np.argmin(np.abs(vol.h_axis - 0.5)))
    sk = int(np.argmin(np.abs(vol.k_axis - 0.5)))
    sl = int(np.argmin(np.abs(vol.l_axis - 1.0)))
    vol.data[sh, sk, sl] = 60.0
    both = BraggRemover(mode="both", punch_radii=(0.25, 0.25, 0.25),
                        min_intensity=10.0, search_n_mad=6.0,
                        search_min_intensity=10.0, search_q_step=0.5)
    keep = both.build_mask(vol)
    i0 = int(np.argmin(np.abs(vol.h_axis)))
    assert not keep[i0, i0, i0]      # integer Bragg gone
    assert not keep[sh, sk, sl]      # satellite gone


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
