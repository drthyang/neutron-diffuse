# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tsung-han Yang

"""Interactive multi-panel slice viewer with live colour-scale controls.

A thin, reusable front-end built on the :mod:`nebula3d.visualization` primitives
(:func:`extract_slice`).  It opens one ``imshow`` panel per volume, all sharing
a single pair of ``vmin`` / ``vmax`` sliders and a linear/log₁₀ mode toggle, so
you can drag the colour scale and reveal weak features (rings, diffuse) that are
otherwise swamped by Bragg peaks.

Requires an interactive Matplotlib backend (e.g. ``macosx`` on macOS, ``qt`` on
Linux/Windows).  In a plain script run it with::

    PYTHONPATH=src python -c "import matplotlib; matplotlib.use('macosx'); ..."

or launch IPython with ``--matplotlib=macosx``.  See ``docs/interactive.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.figure

import numpy as np

from nebula3d.core import HKLVolume
from nebula3d.visualization.slices import (
    _ALIASES,
    _PLANE,
    SliceData,
    _imshow_extent,
    extract_slice,
)

_AXIS_TO_PLANE = {"H": "0kl", "K": "h0l", "L": "hk0"}
_KEY_TO_AXIS = {"kl": "H", "lk": "H", "hl": "K", "lh": "K", "hk": "L", "kh": "L"}


def _take_plane(vol: HKLVolume, key: str, value: float) -> SliceData:
    """Cheap nearest-plane extraction (masks only the 2D plane, not the whole
    volume like :func:`extract_slice`) — used while scrubbing the cut slider so
    moving through a 300-plane volume stays responsive.  Returns the same
    :class:`SliceData` shape as :func:`extract_slice`."""
    fixed_attr, array_dim, y_attr, x_attr, y_label, x_label, transpose = _PLANE[key]
    fixed_axis = getattr(vol, fixed_attr)
    idx = int(np.argmin(np.abs(fixed_axis - value)))
    actual = float(fixed_axis[idx])
    data2d = np.take(vol.data, idx, axis=array_dim).astype(float)
    mask2d = np.take(vol.mask, idx, axis=array_dim)
    data2d = np.where(mask2d, data2d, np.nan)
    if transpose:
        data2d = data2d.T
    name = fixed_attr[0].upper()
    return SliceData(
        data=data2d, x_axis=getattr(vol, x_attr), y_axis=getattr(vol, y_attr),
        x_label=x_label, y_label=y_label,
        cut_label=f"{name} = {actual:.4g} r.l.u.",
    )


def interactive_slices(
    panels: Sequence[tuple[str, HKLVolume]],
    plane: str = "kl",
    value: float = 0.0,
    cmap: str = "inferno",
    log_scale: bool = False,
    interp: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    slider_min: float | None = None,
    slider_max: float | None = None,
    value_slider: bool = False,
    plane_selector: bool = False,
    show: bool = True,
) -> matplotlib.figure.Figure:
    """Open an interactive window comparing slices with live colour controls.

    Parameters
    ----------
    panels : sequence of (label, HKLVolume)
        One panel per entry, drawn left to right.  All panels share the same
        plane/value and the same live colour limits, so they are directly
        comparable (e.g. data / removed rings / residual).
    plane, value, interp :
        Passed to :func:`extract_slice` (which plane, the cut value, and
        whether to interpolate off-grid).
    cmap : str
        Matplotlib colormap.
    log_scale : bool
        Initial display mode (a radio button toggles linear/log₁₀ live).
    vmin, vmax : float, optional
        Initial colour limits (in display units — log₁₀ when ``log_scale``).
        Default: ``vmin`` at the data floor, ``vmax`` at the 99th percentile.
    slider_min, slider_max : float, optional
        Travel range (end stops) of the vmin/vmax sliders in **linear** mode.
        Default spans the full data range ``dmin..dmax``, but when the rings or
        Bragg peaks are much brighter than the diffuse signal of interest that
        leaves the useful range a tiny sliver of the pullbar.  Set a tight pair
        (e.g. ``slider_min=-0.05, slider_max=0.5``) for fine control near the
        diffuse level.  Ignored in log₁₀ mode (bounds are computed from the
        data there).
    value_slider : bool
        Add a slider over the cut (fixed-axis) position, so you can scrub through
        e.g. every H plane of a 3D volume in place.  All panels move together and
        the colour limits are held fixed across the scrub, so slices are directly
        comparable.  Scrubbing extracts only the 2D plane (not the whole volume),
        so it stays responsive on a 300-plane stack.
    plane_selector : bool
        Add an H/K/L radio selector next to the cut slider.  Switching H, K, or L
        retargets the slider to the matching fixed axis and redraws the panels as
        0kl, h0l, or hk0 slices, respectively.  Requires ``value_slider=True``.
    show : bool
        Call ``plt.show()`` before returning (set False for headless tests).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons, Slider

    items = list(panels)
    if not items:
        raise ValueError("panels must contain at least one (label, volume)")
    if plane_selector and not value_slider:
        raise ValueError("plane_selector requires value_slider=True")

    state = {
        "log": bool(log_scale),
        "key": _ALIASES.get(plane.lower(), plane.lower()),
        "plane": plane,
    }

    def panel_slice(v: HKLVolume, val: float) -> SliceData:
        if value_slider:
            return _take_plane(v, str(state["key"]), val)
        return extract_slice(v, plane=str(state["plane"]), value=val, interp=interp)

    labels = [lab for lab, _ in items]
    slices = [panel_slice(v, value) for _, v in items]
    raw = [s.data.astype(float) for s in slices]

    finite = np.concatenate([a[np.isfinite(a)].ravel() for a in raw])
    if finite.size == 0:
        finite = np.array([0.0, 1.0])
    dmin, dmax = float(finite.min()), float(finite.max())
    floor = max(abs(dmax) * 1e-4, 1e-9)

    def to_display(a: np.ndarray) -> np.ndarray:
        if state["log"]:
            return np.where(np.isfinite(a) & (a > 0),
                            np.log10(np.maximum(a, floor)), np.nan)
        return a

    def scale_bounds() -> tuple[float, float]:
        if state["log"]:
            return float(np.log10(floor)), float(np.log10(max(dmax, floor * 10)))
        lo = dmin if slider_min is None else slider_min
        hi = dmax if slider_max is None else slider_max
        return lo, hi

    def default_clim() -> tuple[float, float]:
        d = to_display(finite)
        d = d[np.isfinite(d)]
        lo, hi = scale_bounds()
        if d.size == 0:
            return lo, hi
        return (lo if vmin is None else vmin,
                float(np.percentile(d, 99)) if vmax is None else vmax)

    n = len(items)
    fig, axes_grid = plt.subplots(1, n, figsize=(5.4 * n, 6.0), squeeze=False)
    axes = list(axes_grid[0])

    v0, v1 = default_clim()
    imgs = []
    for ax, a, s, lab in zip(axes, raw, slices, labels):
        im = ax.imshow(to_display(a), origin="lower",
                       extent=_imshow_extent(s.x_axis, s.y_axis),
                       aspect="auto", cmap=cmap, vmin=v0, vmax=v1,
                       interpolation="nearest")
        im.cmap.set_bad("0.5")
        ax.set_title(lab)
        ax.set_xlabel(s.x_label)
        ax.set_ylabel(s.y_label)
        imgs.append(im)

    fig.suptitle(slices[0].cut_label)
    # Leave more room at the bottom when navigation controls are shown so they
    # sit clearly above the colour-scale controls.
    fig.subplots_adjust(left=0.16, right=0.97,
                        bottom=0.40 if plane_selector else (0.34 if value_slider else 0.22),
                        top=0.90, wspace=0.25)
    fig.colorbar(imgs[-1], ax=axes, fraction=0.046, pad=0.02,
                 label="log₁₀(I)" if state["log"] else "Intensity (arb.)")

    # Colour-scale controls share one column (sliders right of the linear/log
    # toggle); the cut slider, when present, is added a clear gap above them.
    slx, slw = 0.26, 0.56
    lo, hi = scale_bounds()
    ax_vmin = fig.add_axes((slx, 0.115, slw, 0.03))
    ax_vmax = fig.add_axes((slx, 0.065, slw, 0.03))
    s_vmin = Slider(ax_vmin, "vmin", lo, hi, valinit=min(v0, v1))
    s_vmax = Slider(ax_vmax, "vmax", lo, hi, valinit=max(v0, v1))

    ax_mode = fig.add_axes((0.045, 0.05, 0.13, 0.10))
    radio = RadioButtons(ax_mode, ("linear", "log₁₀"),
                         active=1 if state["log"] else 0)

    def apply_clim(_: object = None) -> None:
        a, b = s_vmin.val, s_vmax.val
        for im in imgs:
            im.set_clim(min(a, b), max(a, b))
        fig.canvas.draw_idle()

    def on_mode(label: str | None) -> None:
        state["log"] = (label == "log₁₀")
        for im, a in zip(imgs, raw):
            im.set_data(to_display(a))
        lo2, hi2 = scale_bounds()
        nv0, nv1 = default_clim()
        for sl_, val in ((s_vmin, min(nv0, nv1)), (s_vmax, max(nv0, nv1))):
            sl_.valmin, sl_.valmax = lo2, hi2
            sl_.ax.set_xlim(lo2, hi2)
            sl_.eventson = False
            sl_.set_val(val)
            sl_.eventson = True
        apply_clim()

    s_vmin.on_changed(apply_clim)
    s_vmax.on_changed(apply_clim)
    radio.on_clicked(on_mode)

    widgets = [s_vmin, s_vmax, radio]

    # Optional cut-position (e.g. H) slider: scrub the fixed axis in place.  The
    # colour limits are deliberately NOT recomputed per slice, so the scale stays
    # fixed and slices are directly comparable as you move through H/K/L.
    if value_slider:
        def fixed_axis_info() -> tuple[np.ndarray, str]:
            fixed_attr = _PLANE[str(state["key"])][0]
            return getattr(items[0][1], fixed_attr), fixed_attr[0].upper()

        fixed_axis, fixed_name = fixed_axis_info()
        ax_val = fig.add_axes((slx, 0.245 if plane_selector else 0.19, slw, 0.035),
                              facecolor="#e8eef7")
        s_val = Slider(ax_val, f"{fixed_name} plane", float(fixed_axis.min()),
                       float(fixed_axis.max()), valinit=float(value),
                       color="#3a6ea5")

        def on_value(val: float) -> None:
            for i, (_, v) in enumerate(items):
                s = panel_slice(v, val)
                slices[i] = s
                raw[i] = s.data.astype(float)
                imgs[i].set_data(to_display(raw[i]))
                extent = _imshow_extent(s.x_axis, s.y_axis)
                imgs[i].set_extent(extent)
                axes[i].set_xlim(extent[0], extent[1])
                axes[i].set_ylim(extent[2], extent[3])
                axes[i].set_xlabel(s.x_label)
                axes[i].set_ylabel(s.y_label)
            fig.suptitle(slices[0].cut_label)
            apply_clim()

        s_val.on_changed(on_value)
        widgets.append(s_val)

        if plane_selector:
            axes_labels = ("H", "K", "L")
            active_axis = _KEY_TO_AXIS.get(str(state["key"]), "H")
            ax_plane = fig.add_axes((0.045, 0.19, 0.13, 0.13))
            plane_radio = RadioButtons(ax_plane, axes_labels,
                                       active=axes_labels.index(active_axis))

            def reset_value_slider(prefer_zero: bool = True) -> None:
                axis, name = fixed_axis_info()
                lo2, hi2 = float(axis.min()), float(axis.max())
                cur = float(s_val.val)
                if prefer_zero and lo2 <= 0.0 <= hi2:
                    next_val = 0.0
                else:
                    next_val = min(max(cur, lo2), hi2)
                s_val.valmin, s_val.valmax = lo2, hi2
                s_val.ax.set_xlim(lo2, hi2)
                s_val.label.set_text(f"{name} plane")
                s_val.eventson = False
                s_val.set_val(next_val)
                s_val.eventson = True
                on_value(next_val)

            def on_plane(label: str | None) -> None:
                if label is None:
                    return
                axis = label.upper()
                state["plane"] = _AXIS_TO_PLANE[axis]
                state["key"] = _ALIASES[str(state["plane"])]
                reset_value_slider(prefer_zero=True)

            plane_radio.on_clicked(on_plane)
            widgets.append(plane_radio)

    # keep widget refs alive on the figure so they aren't garbage-collected
    fig._nebula3d_widgets = tuple(widgets)  # type: ignore[attr-defined]

    if show:
        plt.show()
    return fig
