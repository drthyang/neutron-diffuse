"""Tests for mask-based ring cleanup and sideband backfill."""

import numpy as np

from ndiff.core import HKLVolume
from ndiff.preprocessing.masked_rings import _sideband_background
from ndiff.preprocessing.radial_background import _azimuthal_angle, _offset_q_magnitude


def test_sideband_background_preserves_radial_ramp_through_mask():
    h = np.array([0.3333])
    k = np.linspace(-2.0, 2.0, 51)
    l = np.linspace(-2.0, 2.0, 51)
    vol = HKLVolume.from_arrays(
        np.ones((h.size, k.size, l.size)),
        (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]),
        ub_matrix=np.eye(3),
    )
    q = _offset_q_magnitude(vol, "0kl")
    phi = _azimuthal_angle(vol, "0kl")
    signal = 1.0 + 0.2 * q
    vol = HKLVolume.from_arrays(
        signal,
        (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]),
        sigma=np.ones_like(signal) * 0.01,
        ub_matrix=np.eye(3),
    )
    ring_mask = vol.mask & (q > 1.0) & (q < 1.2)
    fallback = np.zeros_like(signal)

    filled = _sideband_background(vol, ring_mask, q, phi, fallback, n_phi_bins=72)

    assert np.median(np.abs(filled[ring_mask] - signal[ring_mask])) < 0.03
