"""Azimuthal sampling-density mask for symmetrised volumes.

A symmetrised HKL volume is densely measured along some azimuthal directions
and sparsely along others (detector coverage).  In the under-sampled azimuthal
sectors each (|Q|, φ) cell holds only a handful of measurements, and those
values are unreliable — on real data they show up as bright radial "spokes"
well above the true powder-ring / diffuse level.  They are a data-quality
artefact, not signal, and powder-ring removal correctly leaves them untouched
(they are not azimuthally-smooth rings), so they survive into the residual.

:func:`azimuthal_sampling_mask` flags those voxels so they can be dropped and
reconstructed by the downstream backfill, the same way ring/Bragg holes are.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ndiff.core import HKLVolume
from ndiff.preprocessing.radial_background import _azimuthal_angle


def azimuthal_sampling_mask(
    vol: HKLVolume,
    plane: str = "hk0",
    n_q_bins: int = 90,
    n_phi_bins: int = 72,
    min_count: int = 15,
    q_range: Optional[tuple[float, float]] = None,
) -> NDArray[np.bool_]:
    """Keep-mask that drops voxels in under-sampled (|Q|, φ) cells.

    Parameters
    ----------
    vol : HKLVolume
    plane : str
        Reference plane defining φ (see :class:`PatchedRadialRingModel`).
    n_q_bins, n_phi_bins : int
        Resolution of the (|Q|, φ) sampling-density histogram.
    min_count : int
        A (|Q|, φ) cell with fewer than this many *valid* voxels is considered
        under-sampled; its voxels are dropped (set False in the keep-mask).
    q_range : (q_min, q_max), optional
        Restrict the histogram/mask to this |Q| range.  Voxels outside it keep
        their current mask state.

    Returns
    -------
    keep : bool array, shape ``vol.shape``
        ``vol.mask`` with under-sampled cells additionally set False.  Already
        masked / invalid voxels stay masked.
    """
    q = vol.q_magnitude()
    phi = _azimuthal_angle(vol, plane)
    valid = vol.mask & np.isfinite(vol.data)

    if q_range is None:
        q_range = (float(q[valid].min()), float(q[valid].max()))
    q0, q1 = q_range

    q_edges = np.linspace(q0, q1, n_q_bins + 1)
    phi_edges = np.linspace(-np.pi, np.pi, n_phi_bins + 1)

    # Count valid voxels per (|Q|, φ) cell.
    counts, _, _ = np.histogram2d(
        q[valid], phi[valid], bins=(q_edges, phi_edges)
    )

    # Look up each valid voxel's cell count.
    qi = np.clip(np.digitize(q[valid], q_edges) - 1, 0, n_q_bins - 1)
    pi = np.clip(np.digitize(phi[valid], phi_edges) - 1, 0, n_phi_bins - 1)
    in_range = (q[valid] >= q0) & (q[valid] <= q1)
    cell_count = counts[qi, pi]

    keep = vol.mask.copy()
    drop_flat = in_range & (cell_count < min_count)
    keep[valid] = ~drop_flat
    return keep
