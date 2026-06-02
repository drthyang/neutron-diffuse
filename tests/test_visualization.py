"""Tests for ndiff.visualization slice extraction and colour limits."""

import matplotlib

matplotlib.use("Agg")  # headless; no display needed

import numpy as np

from ndiff.core import HKLVolume
from ndiff.visualization import extract_slice, plot_slice, plot_overview


def _ramp_volume(shape=(5, 6, 7)):
    """Volume whose intensity varies linearly along L, so the interpolated
    plane value is analytically predictable."""
    h0, h1 = -2.0, 2.0
    k0, k1 = -3.0, 3.0
    l0, l1 = 0.0, 3.0
    ub = 2 * np.pi * np.eye(3) / 4.0
    l_axis = np.linspace(l0, l1, shape[2])
    # data[h,k,l] = l-coordinate value (independent of h,k)
    data = np.broadcast_to(l_axis, shape).astype(np.float64).copy()
    return HKLVolume.from_arrays(
        data, (h0, h1), (k0, k1), (l0, l1), ub_matrix=ub
    ), l_axis


def test_extract_slice_nearest_snaps_to_grid():
    vol, l_axis = _ramp_volume()
    target = 0.3333
    sl = extract_slice(vol, "hk", target, interp=False)
    nearest = float(l_axis[np.argmin(np.abs(l_axis - target))])
    # every voxel equals the snapped L value (ramp along L)
    assert np.allclose(sl.data, nearest)
    assert f"{nearest:.4g}" in sl.cut_label


def test_extract_slice_interp_honours_offgrid_value():
    vol, l_axis = _ramp_volume()
    target = 0.3333
    assert target not in set(l_axis)  # genuinely off-grid
    sl = extract_slice(vol, "hk", target, interp=True)
    # ramp along L => interpolated plane equals the exact target everywhere
    assert np.allclose(sl.data, target, atol=1e-9)
    assert "0.3333" in sl.cut_label


def test_extract_slice_interp_is_nan_aware():
    vol, l_axis = _ramp_volume()
    # mask out one bracketing plane's voxel; interp should fall back to the other
    i1 = int(np.searchsorted(l_axis, 0.3333))
    vol.mask[0, 0, i1] = False
    sl = extract_slice(vol, "hk", 0.3333, interp=True)
    assert np.isfinite(sl.data[0, 0])  # no NaN bleed at the masked corner


def test_interp_out_of_range_clamps():
    vol, l_axis = _ramp_volume()
    below = extract_slice(vol, "hk", -100.0, interp=True)
    above = extract_slice(vol, "hk", 100.0, interp=True)
    assert np.allclose(below.data, l_axis[0])
    assert np.allclose(above.data, l_axis[-1])


def test_swapped_plane_is_transpose():
    # asymmetric shape so a transpose is unambiguous
    data = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)  # (H, K, L)
    vol = HKLVolume.from_arrays(data, (-1, 1), (-2, 2), (-3, 3))
    for a, b in [("kl", "lk"), ("hl", "lh"), ("hk", "kh")]:
        A = extract_slice(vol, a, 0.0)
        B = extract_slice(vol, b, 0.0)
        assert B.data.shape == A.data.shape[::-1]
        assert np.array_equal(B.data, A.data.T)
        # axes and labels swap too
        assert np.array_equal(B.y_axis, A.x_axis)
        assert np.array_equal(B.x_axis, A.y_axis)
        assert (B.y_label, B.x_label) == (A.x_label, A.y_label)


def test_mantid_aliases_map_to_principal_planes():
    data = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
    vol = HKLVolume.from_arrays(data, (-1, 1), (-2, 2), (-3, 3))
    for alias, principal in [("0kl", "kl"), ("h0l", "hl"), ("hk0", "hk")]:
        assert np.array_equal(
            extract_slice(vol, alias, 0.0).data,
            extract_slice(vol, principal, 0.0).data,
        )


def test_plot_slice_vmin_vmax_set_clim():
    vol, _ = _ramp_volume()
    ax = plot_slice(vol, "hk", 0.5, vmin=0.1, vmax=0.9)
    assert ax.images[0].get_clim() == (0.1, 0.9)


def test_plot_overview_shares_vmin_vmax_across_slices():
    vol, _ = _ramp_volume()
    fig = plot_overview(vol, vmin=0.2, vmax=0.8)
    slice_clims = [a.images[0].get_clim() for a in fig.axes if a.images]
    assert slice_clims  # three slice panels have images
    assert all(c == (0.2, 0.8) for c in slice_clims)
