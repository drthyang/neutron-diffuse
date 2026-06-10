"""Isotropic radial-background flattening by spherical |Q|-shell sweep.

Motivation
----------
After ring removal, Bragg punching, and backfilling, the volume still carries a
smooth, (nearly) isotropic radial pedestal: the average intensity drifts with
|Q| (incoherent/multiple scattering, TDS, an imperfect empty-can subtraction).
Left in, that pedestal is a broad, slowly-varying background that the ΔPDF
transform turns into a low-frequency artefact (and it makes plane-to-plane
intensity comparison harder).

This step flattens it directly.  Sweeping spherical shells from |Q|=0 to Qmax,
in each thin shell we

    * estimate a single robust **background level** from the shell's intensity
      distribution — by default the **floor** (a low percentile / mode), which
      sits *below* the diffuse and the Bragg-residual high tail, so neither
      enters the estimate, and
    * subtract that level from every voxel in the shell.

The per-shell levels are then smoothed along |Q| into one continuous curve
``bg(|Q|)`` and subtracted at each voxel's exact |Q| (linear interpolation), so
no shell-edge step is stamped into the result.  The radial pedestal flattens to
≈0 while the **anisotropic diffuse signal and Bragg residuals are preserved** —
they rise above the shell floor, so subtracting the floor leaves them standing.

This is the isotropic complement to
:class:`~ndiff.preprocessing.radial_background.PatchedRadialRingModel`, which
removes *anisotropic* powder rings per 2D plane.  The two are independent: rings
are azimuthally smooth peaks at fixed |Q|; this removes the smooth radial level
underneath everything.

Estimator
---------
``estimator='floor'`` (default) keeps diffuse: the floor is the background, and
anything above it (diffuse, Bragg) survives.  ``'mode'`` is the most-common
value (a histogram peak), similar intent.  ``'median'`` subtracts the shell
*average* and so also removes any genuinely isotropic diffuse component — more
aggressive, use only when that is intended.  ``'snip'`` builds the per-shell
median radial profile and takes its SNIP baseline (the floor under broad radial
humps) — useful when the background itself has broad bumps in |Q|.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from ndiff.core import HKLVolume
from ndiff.preprocessing.radial_background import _estimate_baseline, _fill_nan_1d

ESTIMATORS = ("floor", "mode", "median", "snip")


@dataclass
class RadialFlattenResult:
    """Result of :func:`flatten_radial_background`.

    Attributes
    ----------
    volume : HKLVolume
        The flattened volume (``data`` minus the radial background; ``mask`` and
        ``sigma`` unchanged).
    q_grid : (Q,)
        Shell centres (Å⁻¹).
    bg_curve : (Q,)
        The smooth, continuous background level subtracted at each shell — the
        curve actually interpolated and removed.
    raw_levels : (Q,)
        Per-shell level *before* the along-|Q| smoothing (diagnostic).  NaN for
        shells with fewer than ``min_count`` valid voxels.
    counts : (Q,)
        Number of valid voxels in each shell.
    estimator : str
        The estimator used (see module docstring).
    """

    volume: HKLVolume
    q_grid: NDArray[np.float64]
    bg_curve: NDArray[np.float64]
    raw_levels: NDArray[np.float64]
    counts: NDArray[np.int_]
    estimator: str


def flatten_radial_background(
    vol: HKLVolume,
    *,
    q_step: float = 0.05,
    estimator: str = "floor",
    floor_percentile: float = 25.0,
    snip_width: float = 0.3,
    smooth: float = 0.10,
    min_count: int = 20,
    q_range: tuple[float, float] | None = None,
    clip_negative: bool = False,
) -> RadialFlattenResult:
    """Subtract a smooth, continuous isotropic radial background from *vol*.

    Parameters
    ----------
    q_step : float
        Spherical-shell width (Å⁻¹).  A few times finer than the scale of the
        background drift; the along-|Q| smoothing controls noise, so a fine step
        is safe (default 0.05).
    estimator : {'floor', 'mode', 'median', 'snip'}
        How the per-shell background level is estimated (see module docstring).
        ``'floor'`` (default) preserves diffuse and Bragg.
    floor_percentile : float
        Percentile used by ``estimator='floor'`` (default 25).  Lower → a more
        conservative background that removes less and keeps more signal.
    snip_width : float
        Peak-removal width (Å⁻¹) for ``estimator='snip'`` (default 0.3).
    smooth : float
        σ (Å⁻¹) of the Gaussian smoothing the per-shell levels into a continuous
        ``bg(|Q|)`` (default 0.10).  This is what makes the subtracted background
        smooth and continuous across shells.  Set 0 to disable (not for ``snip``,
        which smooths internally).
    min_count : int
        Shells with fewer valid voxels get no level (NaN) and are filled by
        interpolation from their neighbours (default 20).
    q_range : (float, float), optional
        Restrict the swept |Q| range (Å⁻¹).  ``None`` sweeps the full data range.
    clip_negative : bool
        If True, clamp the flattened data at 0 (default False — negative
        residuals below the background are meaningful and kept).
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"Unknown estimator {estimator!r}; choose one of {ESTIMATORS}.")

    q = vol.q_magnitude()
    data = vol.data
    valid = vol.mask & np.isfinite(data)
    if not valid.any():
        empty = np.zeros(0, dtype=np.float64)
        return RadialFlattenResult(
            volume=vol, q_grid=empty, bg_curve=empty,
            raw_levels=empty, counts=np.zeros(0, dtype=int), estimator=estimator,
        )

    qs = max(float(q_step), 1e-12)
    if q_range is None:
        q0, q1 = float(q[valid].min()), float(q[valid].max())
    else:
        q0, q1 = float(q_range[0]), float(q_range[1])
    edges = np.arange(q0, q1 + qs, qs)
    if edges.size < 2:
        edges = np.array([q0, q0 + qs])
    q_grid = 0.5 * (edges[:-1] + edges[1:])
    nb = q_grid.size
    bin_idx = np.clip(np.digitize(q, edges) - 1, 0, nb - 1)

    # Per-shell level from the *valid* voxels (sorted-segment scan, like the
    # q_shell backfill lookup), so each shell is touched once.
    flat_b = bin_idx[valid]
    flat_i = data[valid].astype(np.float64)
    order = np.argsort(flat_b, kind="stable")
    sb = flat_b[order]
    si = flat_i[order]
    bounds = np.searchsorted(sb, np.arange(nb + 1))

    raw = np.full(nb, np.nan)
    counts = np.zeros(nb, dtype=int)
    median_profile = estimator == "snip"
    for b in range(nb):
        seg = si[bounds[b]:bounds[b + 1]]
        counts[b] = seg.size
        if seg.size < min_count:
            continue
        raw[b] = (
            float(np.median(seg)) if median_profile
            else _shell_level(seg, estimator, floor_percentile)
        )

    filled = _fill_nan_1d(raw)
    if estimator == "snip":
        # SNIP baseline of the median radial profile: the floor under broad humps.
        bg_curve = _estimate_baseline(filled, qs, snip_width, smooth, "snip")
    elif smooth > 0:
        bg_curve = gaussian_filter1d(filled, smooth / qs, mode="nearest")
    else:
        bg_curve = filled

    # Subtract the smooth curve at each voxel's exact |Q| (continuous, no shell
    # step), leaving NaN/masked voxels untouched.
    bg_at = np.interp(
        q, q_grid, bg_curve, left=float(bg_curve[0]), right=float(bg_curve[-1])
    )
    data_out = data.copy()
    finite = np.isfinite(data)
    data_out[finite] = data[finite] - bg_at[finite]
    if clip_negative:
        data_out[finite] = np.maximum(data_out[finite], 0.0)

    vol_out = dataclasses.replace(vol, data=data_out)
    return RadialFlattenResult(
        volume=vol_out, q_grid=q_grid, bg_curve=bg_curve,
        raw_levels=raw, counts=counts, estimator=estimator,
    )


def _shell_level(
    vals: NDArray[np.float64], estimator: str, floor_percentile: float
) -> float:
    """Robust per-shell background level for the non-profile estimators."""
    if estimator == "floor":
        return float(np.percentile(vals, floor_percentile))
    if estimator == "median":
        return float(np.median(vals))
    if estimator == "mode":
        return _shell_mode(vals)
    raise ValueError(f"Unknown estimator: {estimator!r}")


def _shell_mode(vals: NDArray[np.float64]) -> float:
    """Most-common value of a shell — peak of a Scott's-rule histogram.

    The mode is the background level when the background voxels dominate the
    shell (the usual case): diffuse and Bragg sit in the high tail and do not
    move the peak.  Falls back to the median for tiny or degenerate shells.
    """
    n = vals.size
    if n < 8:
        return float(np.median(vals))
    lo, hi = np.percentile(vals, (1.0, 99.0))
    if not (hi > lo):
        return float(np.median(vals))
    std = float(np.std(vals))
    bw = 3.49 * std * n ** (-1.0 / 3.0) if std > 0 else 0.0
    nbins = 4 if bw <= 0 else max(4, int(np.ceil((hi - lo) / bw)))
    hist, hist_edges = np.histogram(vals, bins=nbins, range=(float(lo), float(hi)))
    k = int(np.argmax(hist))
    return float(0.5 * (hist_edges[k] + hist_edges[k + 1]))
