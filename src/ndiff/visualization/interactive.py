"""Interactive multi-panel slice viewer with live colour-scale controls.

A thin, reusable front-end built on the :mod:`ndiff.visualization` primitives
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

from typing import Optional, Sequence

import numpy as np

from ndiff.core import HKLVolume
from ndiff.visualization.slices import extract_slice, _imshow_extent


def interactive_slices(
    panels: Sequence[tuple[str, HKLVolume]],
    plane: str = "kl",
    value: float = 0.0,
    cmap: str = "inferno",
    log_scale: bool = False,
    interp: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    slider_min: Optional[float] = None,
    slider_max: Optional[float] = None,
    show: bool = True,
):
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

    labels = [lab for lab, _ in items]
    slices = [extract_slice(v, plane=plane, value=value, interp=interp)
              for _, v in items]
    raw = [s.data.astype(float) for s in slices]

    finite = np.concatenate([a[np.isfinite(a)].ravel() for a in raw])
    if finite.size == 0:
        finite = np.array([0.0, 1.0])
    dmin, dmax = float(finite.min()), float(finite.max())
    floor = max(abs(dmax) * 1e-4, 1e-9)

    state = {"log": bool(log_scale)}

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
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 6.0), squeeze=False)
    axes = list(axes[0])

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
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.26, top=0.90, wspace=0.25)
    fig.colorbar(imgs[-1], ax=axes, fraction=0.046, pad=0.02,
                 label="log₁₀(I)" if state["log"] else "Intensity (arb.)")

    lo, hi = scale_bounds()
    ax_vmin = fig.add_axes([0.18, 0.13, 0.64, 0.03])
    ax_vmax = fig.add_axes([0.18, 0.08, 0.64, 0.03])
    s_vmin = Slider(ax_vmin, "vmin", lo, hi, valinit=min(v0, v1))
    s_vmax = Slider(ax_vmax, "vmax", lo, hi, valinit=max(v0, v1))

    ax_mode = fig.add_axes([0.015, 0.06, 0.10, 0.12])
    radio = RadioButtons(ax_mode, ("linear", "log₁₀"),
                         active=1 if state["log"] else 0)

    def apply_clim(_=None):
        a, b = s_vmin.val, s_vmax.val
        for im in imgs:
            im.set_clim(min(a, b), max(a, b))
        fig.canvas.draw_idle()

    def on_mode(label):
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

    # keep widget refs alive on the figure so they aren't garbage-collected
    fig._ndiff_widgets = (s_vmin, s_vmax, radio)  # type: ignore[attr-defined]

    if show:
        plt.show()
    return fig
