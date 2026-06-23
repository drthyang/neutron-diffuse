"""Tests for the isotropic radial-background flatten (spherical |Q|-shell sweep)."""

import numpy as np
import pytest

from nebula3d.core import HKLVolume
from nebula3d.preprocessing import flatten_radial_background


def _base_vol(shape=(41, 41, 41), seed=0, noise=0.05):
    """Volume with a smooth, decaying, isotropic radial pedestal + small noise."""
    rng = np.random.default_rng(seed)
    ub = 2 * np.pi * np.eye(3) / 4.0
    vol = HKLVolume.from_arrays(
        np.zeros(shape, dtype=float), (-3, 3), (-3, 3), (-3, 3), ub_matrix=ub
    )
    q = vol.q_magnitude()
    bg = 5.0 * np.exp(-q / 3.0) + 0.5
    vol.data[...] = bg + rng.normal(0.0, noise, shape)
    return vol, q, bg


def _shell_medians(data, q, valid, step=0.15, min_count=10):
    """Per-shell median of *data* — an independent flatness probe for the tests."""
    edges = np.arange(float(q[valid].min()), float(q[valid].max()) + step, step)
    n = edges.size - 1
    bi = np.clip(np.digitize(q, edges) - 1, 0, n - 1)[valid]
    vv = data[valid].astype(float)
    order = np.argsort(bi, kind="stable")
    sb, sv = bi[order], vv[order]
    bounds = np.searchsorted(sb, np.arange(n + 1))
    out = np.full(n, np.nan)
    for b in range(n):
        seg = sv[bounds[b]:bounds[b + 1]]
        if seg.size >= min_count:
            out[b] = float(np.median(seg))
    return out


def test_flatten_collapses_shell_spread_and_is_continuous():
    vol, q, _ = _base_vol()
    valid = vol.mask & np.isfinite(vol.data)
    before = _shell_medians(vol.data, q, valid)

    res = flatten_radial_background(vol, q_step=0.05, smooth=0.2, min_count=15)
    after = _shell_medians(res.volume.data, q, valid)

    assert np.nanstd(before) > 0.5                       # a real radial pedestal
    assert np.nanstd(after) < 0.15 * np.nanstd(before)   # flattened across shells
    assert np.all(np.isfinite(res.bg_curve))
    # smooth + continuous: no large shell-to-shell jump in the subtracted curve
    span = float(np.nanmax(res.bg_curve) - np.nanmin(res.bg_curve))
    assert np.max(np.abs(np.diff(res.bg_curve))) < 0.1 * span


def test_preserves_anisotropic_diffuse_blob():
    vol, _, _ = _base_vol()
    H, K, L = vol.hkl_grid()
    # localized blob off-origin: it occupies one azimuth of its |Q| shell, so it
    # is signal, not background, and must survive the floor subtraction.
    blob = 4.0 * np.exp(-((H - 1.5) ** 2 + K**2 + L**2) / (2 * 0.25**2))
    vol.data[...] = vol.data + blob
    ic = np.unravel_index(int(np.argmax(blob)), blob.shape)

    res = flatten_radial_background(vol, q_step=0.05, smooth=0.2, min_count=15)

    assert res.volume.data[ic] > 0.6 * 4.0               # blob peak retained


def test_bragg_spikes_survive_and_do_not_inflate_bg():
    vol, _, _ = _base_vol()
    spikes = [(10, 20, 20), (30, 15, 25), (20, 30, 10)]
    for idx in spikes:
        vol.data[idx] += 100.0

    res = flatten_radial_background(vol, q_step=0.05, smooth=0.2, min_count=15)

    for idx in spikes:
        assert res.volume.data[idx] > 90.0               # spike stays in residual
    assert float(np.nanmax(res.bg_curve)) < 10.0         # bg not pulled to spike


def test_floor_keeps_more_diffuse_than_median():
    vol, q, _ = _base_vol()
    H, K, _ = vol.hkl_grid()
    phi = np.arctan2(K, H)
    # anisotropic diffuse: 0..2 around the azimuth of a |Q|≈4 shell.
    diffuse = 2.0 * np.exp(-((q - 4.0) ** 2) / (2 * 0.6**2)) * (0.5 + 0.5 * np.cos(2 * phi))
    vol.data[...] = vol.data + diffuse
    ic = np.unravel_index(int(np.argmax(diffuse)), diffuse.shape)

    kw = dict(q_step=0.05, smooth=0.2, min_count=15)
    floor = flatten_radial_background(vol, estimator="floor", **kw)
    med = flatten_radial_background(vol, estimator="median", **kw)

    removed_floor = float(np.nansum(vol.data - floor.volume.data))
    removed_med = float(np.nansum(vol.data - med.volume.data))
    assert removed_med > removed_floor > 0               # median is more aggressive
    assert floor.volume.data[ic] > med.volume.data[ic]   # floor keeps the diffuse lobe


def test_snip_estimator_runs_and_flattens():
    vol, q, _ = _base_vol()
    valid = vol.mask & np.isfinite(vol.data)
    before = _shell_medians(vol.data, q, valid)

    res = flatten_radial_background(vol, estimator="snip", q_step=0.05, snip_width=0.4,
                                    min_count=15)
    after = _shell_medians(res.volume.data, q, valid)

    assert np.all(np.isfinite(res.bg_curve))
    assert np.nanstd(after) < 0.3 * np.nanstd(before)


def test_clip_negative_floors_at_zero():
    vol, _, _ = _base_vol(noise=0.5)
    res = flatten_radial_background(vol, clip_negative=True, q_step=0.05, smooth=0.2,
                                    min_count=15)
    fin = np.isfinite(res.volume.data)
    assert (res.volume.data[fin] >= 0.0).all()


def test_mask_preserved_and_values_finite():
    vol, _, _ = _base_vol()
    keep = np.ones(vol.shape, dtype=bool)
    keep[:5, :, :] = False                               # mask a slab
    vol.apply_mask(keep)
    before_mask = vol.mask.copy()

    res = flatten_radial_background(vol, q_step=0.05, smooth=0.2, min_count=15)

    assert np.array_equal(res.volume.mask, before_mask)  # mask untouched
    fin = np.isfinite(vol.data)
    assert np.isfinite(res.volume.data[fin]).all()


def test_all_masked_returns_unchanged():
    vol, _, _ = _base_vol()
    vol.apply_mask(np.zeros(vol.shape, dtype=bool))

    res = flatten_radial_background(vol)

    assert res.q_grid.size == 0
    assert res.bg_curve.size == 0
    assert np.array_equal(res.volume.data, vol.data)


def test_subtraction_is_purely_radial_so_anisotropy_is_untouched():
    """The flatten subtracts a single level per |Q|, so within a thin shell every
    voxel loses the *same* amount.  That is the precise guarantee that it cannot
    distort anisotropic features: any contrast between two voxels at the same |Q|
    (a Bragg/diffuse peak vs its background) is preserved exactly — only the
    radial mean is shifted.  Validated on real 22/45/100K data (100% feature-
    contrast retention); this locks it in on synthetic data.
    """
    vol, q, _ = _base_vol()
    H, K, L = vol.hkl_grid()
    # several anisotropic blobs at different |Q| and azimuth
    for h0, k0 in [(1.5, 0.0), (0.0, 1.8), (-1.2, 1.2)]:
        vol.data[...] += 4.0 * np.exp(-((H - h0) ** 2 + (K - k0) ** 2 + L**2) / (2 * 0.25**2))

    res = flatten_radial_background(vol, q_step=0.05, smooth=0.2, min_count=15)
    delta = vol.data - res.volume.data            # the amount removed at each voxel

    bg_span = float(np.nanmax(res.bg_curve) - np.nanmin(res.bg_curve))
    # in a thin |Q| shell the removed amount varies only by the curve's slope ×
    # the shell width — a tiny fraction of the total background span.
    for q0 in (1.0, 2.0, 3.5):
        shell = np.abs(q - q0) < 0.02
        assert np.ptp(delta[shell]) < 0.03 * bg_span

    # and the blob-to-background contrast within a shell is retained to ~100%
    H1, K1, L1 = vol.hkl_grid()
    peak = (np.abs(H1 - 1.5) < 0.05) & (np.abs(K1) < 0.05) & (np.abs(L1) < 0.05)
    ref = (np.abs(q - q[peak].mean()) < 0.02) & ~peak
    c_before = float(vol.data[peak].mean() - np.median(vol.data[ref]))
    c_after = float(res.volume.data[peak].mean() - np.median(res.volume.data[ref]))
    assert abs(c_after - c_before) < 0.02 * abs(c_before)


def test_floor_is_conservative_median_centres_the_shell():
    """Why ``floor`` is the validated default: subtracting p25 leaves the shell
    bulk above zero (≈floor_pct negative), so a possibly-real isotropic-diffuse
    level is kept; ``median`` centres the shell (≈50% negative), removing it.
    On real data ``median``/``mode`` flagged as over-subtraction; ``floor`` did
    not.  This guards that conservative ordering.
    """
    vol, _, _ = _base_vol(noise=0.2)

    floor = flatten_radial_background(vol, estimator="floor", floor_percentile=25.0,
                                      q_step=0.05, smooth=0.2, min_count=15)
    med = flatten_radial_background(vol, estimator="median",
                                    q_step=0.05, smooth=0.2, min_count=15)

    fin = np.isfinite(vol.data)
    neg_floor = float(np.mean(floor.volume.data[fin] < 0.0))
    neg_med = float(np.mean(med.volume.data[fin] < 0.0))
    assert neg_floor < 0.40                       # floor keeps the bulk positive
    assert neg_med > neg_floor                    # median centres -> more negative


def test_unknown_estimator_raises():
    vol, _, _ = _base_vol()
    with pytest.raises(ValueError, match="estimator"):
        flatten_radial_background(vol, estimator="nope")
