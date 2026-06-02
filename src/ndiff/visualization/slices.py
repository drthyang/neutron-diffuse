"""2D slice views through an HKLVolume.

Each slice is defined by which two axes are displayed (the *plane*) and the
coordinate value at which the third axis is cut.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray

from ndiff.core import HKLVolume

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage

# plane key → (fixed_axis_attr, array_dim, y_axis_attr, x_axis_attr, y_label, x_label)
# After np.take(data, idx, axis=array_dim) the result has shape (n_y, n_x).
_PLANE: dict[str, tuple[str, int, str, str, str, str]] = {
    "kl": ("h_axis", 0, "k_axis", "l_axis", "K (r.l.u.)", "L (r.l.u.)"),
    "hl": ("k_axis", 1, "h_axis", "l_axis", "H (r.l.u.)", "L (r.l.u.)"),
    "hk": ("l_axis", 2, "h_axis", "k_axis", "H (r.l.u.)", "K (r.l.u.)"),
}
# Accept Mantid-style names and reversed orderings as aliases.
_ALIASES: dict[str, str] = {
    "0kl": "kl", "h0l": "hl", "hk0": "hk",
    "lk": "kl", "lh": "hl", "kh": "hk",
}


class SliceData(NamedTuple):
    """2D intensity slice extracted from an HKLVolume."""

    data:      NDArray[np.float64]  # shape (n_y, n_x); NaN where masked
    y_axis:    NDArray[np.float64]  # bin-centre coordinates along rows
    x_axis:    NDArray[np.float64]  # bin-centre coordinates along columns
    y_label:   str                  # e.g. "K (r.l.u.)"
    x_label:   str                  # e.g. "L (r.l.u.)"
    cut_label: str                  # e.g. "H = 0.020 r.l.u."


def extract_slice(
    vol: HKLVolume,
    plane: str = "kl",
    value: float = 0.0,
) -> SliceData:
    """Extract a 2D slice at the grid point nearest to *value*.

    Parameters
    ----------
    vol : HKLVolume
    plane : str
        Which two axes to show: ``'kl'``, ``'hl'``, or ``'hk'``.
        Mantid-style aliases (``'0kl'``, ``'h0l'``, ``'hk0'``) are also accepted.
    value : float
        Desired coordinate of the cut along the fixed axis (r.l.u.).
        The nearest grid point is used.
    """
    key = _ALIASES.get(plane.lower(), plane.lower())
    if key not in _PLANE:
        valid = list(_PLANE) + list(_ALIASES)
        raise ValueError(f"plane must be one of {valid}; got {plane!r}")

    fixed_attr, array_dim, y_attr, x_attr, y_label, x_label = _PLANE[key]
    fixed_axis: NDArray[np.float64] = getattr(vol, fixed_attr)
    idx = int(np.argmin(np.abs(fixed_axis - value)))
    actual = float(fixed_axis[idx])

    data_2d: NDArray[np.float64] = np.take(vol.masked_data(), idx, axis=array_dim)
    y_axis: NDArray[np.float64] = getattr(vol, y_attr)
    x_axis: NDArray[np.float64] = getattr(vol, x_attr)
    fixed_name = fixed_attr[0].upper()  # 'H', 'K', or 'L'

    return SliceData(
        data=data_2d,
        y_axis=y_axis,
        x_axis=x_axis,
        y_label=y_label,
        x_label=x_label,
        cut_label=f"{fixed_name} = {actual:.3f} r.l.u.",
    )


def plot_slice(
    vol: HKLVolume,
    plane: str = "kl",
    value: float = 0.0,
    ax: Optional["Axes"] = None,
    cmap: str = "hot",
    percentile: float = 99.5,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    log_scale: bool = False,
    title: Optional[str] = None,
) -> "Axes":
    """Plot a 2D intensity slice.

    Parameters
    ----------
    vol : HKLVolume
    plane : str
        See :func:`extract_slice`.
    value : float
        Coordinate of the cut (r.l.u.).
    ax : Axes, optional
        Existing axes to draw into; a new figure is created if *None*.
    cmap : str
        Matplotlib colormap name.
    percentile : float
        Upper percentile used to clip the colour scale (default 99.5).
        The lower clip is the symmetric percentile (``100 - percentile``).
    vmin, vmax : float, optional
        Override the percentile-derived colour limits.
    log_scale : bool
        If True, plot ``log10(max(I, floor))`` where floor is 1 % of the
        maximum valid intensity.
    title : str, optional
        Axes title; defaults to the cut label (e.g. "H = 0.000 r.l.u.").

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    sl = extract_slice(vol, plane=plane, value=value)
    display = sl.data.copy()

    if log_scale:
        valid_max = float(np.nanmax(display)) if np.any(np.isfinite(display)) else 1.0
        floor = max(valid_max * 0.01, 1e-6)
        display = np.where(np.isfinite(display), np.log10(np.maximum(display, floor)), np.nan)

    auto_vmin, auto_vmax = _percentile_clim(display, percentile)
    v0 = vmin if vmin is not None else auto_vmin
    v1 = vmax if vmax is not None else auto_vmax

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    extent = _imshow_extent(sl.x_axis, sl.y_axis)
    img = ax.imshow(
        display,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
        vmin=v0,
        vmax=v1,
        interpolation="nearest",
    )
    # grey out masked/empty voxels
    img.cmap.set_bad("0.85")

    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04,
                 label="log₁₀(I)" if log_scale else "Intensity (arb.)")
    ax.set_xlabel(sl.x_label)
    ax.set_ylabel(sl.y_label)
    ax.set_title(title or sl.cut_label)
    return ax


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _percentile_clim(
    data: NDArray[np.float64],
    pct: float,
) -> tuple[float, float]:
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(valid, 100.0 - pct))
    hi = float(np.percentile(valid, pct))
    return lo, hi


def _imshow_extent(
    x_axis: NDArray[np.float64],
    y_axis: NDArray[np.float64],
) -> list[float]:
    """Return [x_min, x_max, y_min, y_max] with half-bin margins."""
    dx = float(x_axis[1] - x_axis[0]) if len(x_axis) > 1 else 1.0
    dy = float(y_axis[1] - y_axis[0]) if len(y_axis) > 1 else 1.0
    return [
        float(x_axis[0])  - dx / 2,
        float(x_axis[-1]) + dx / 2,
        float(y_axis[0])  - dy / 2,
        float(y_axis[-1]) + dy / 2,
    ]
