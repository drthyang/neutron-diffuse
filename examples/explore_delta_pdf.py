"""Interactive 3D-ΔPDF viewer — y_K–z_L real-space plane with an x_H slider.

Loads the full 3D-ΔPDF (real-space transform of the cleaned diffuse volume)
and lets you scrub through the **real-space x_H axis**, showing the y_K–z_L
correlation plane at each x_H.  This is the proper 3D transform (every plane
mixes all reciprocal H layers with phase), unlike the per-plane 2D-ΔPDF in
``examples/delta_pdf_plane.py``.

Source: ``examples/_delta_pdf.h5`` if present (written by ``delta_pdf.py``);
otherwise it is computed on the fly from the backfilled volume and cached.

Run (interactive, on this Mac)::

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \\
      /Users/tt9/miniforge3/envs/rmc-discord/bin/python3 \\
      examples/explore_delta_pdf.py

Controls:
    x_H slider   — scrub the real-space x_H plane (Å)
    |scale| slider — symmetric colour-scale half-range
    Close the window to exit.

Env overrides:
    PDF_FILE    precomputed ΔPDF .h5 (default: examples/_delta_pdf.h5)
    PROC_FILE   backfilled .h5 to transform if no PDF_FILE (auto-detect)
    X_VALUE     initial x_H plane in Å (default: 0.0)
    RMAX        display half-window in Å for K and L axes (default: 25)
    INTERP      imshow interpolation (default: bilinear; "nearest" for raw pixels)
    SMOKE       1 → render the initial frame to PNG and exit (no GUI); used
                to verify the script headless.
"""

import os
import sys
from pathlib import Path

import matplotlib

SMOKE = bool(int(os.environ.get("SMOKE", "0")))
matplotlib.use("Agg" if SMOKE else "macosx")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# ------------------------------------------------------------------
# load or compute the 3D-ΔPDF
# ------------------------------------------------------------------
pdf_file = Path(os.environ.get("PDF_FILE", "examples/_delta_pdf.h5"))

if pdf_file.exists():
    import h5py
    print(f"loading ΔPDF {pdf_file.name} ...", flush=True)
    with h5py.File(pdf_file, "r") as fh:
        data = fh["data"][...]
        x_axis = fh["x_axis"][...]
        y_axis = fh["y_axis"][...]
        z_axis = fh["z_axis"][...]
        apodization = fh.attrs.get("apodization", "?")
else:
    import ndiff
    from ndiff.analysis import compute_delta_pdf

    proc_file = os.environ.get("PROC_FILE")
    if proc_file:
        proc_path = Path(proc_file)
    else:
        cands = sorted(Path("data/processed").glob("*_backfilled.h5"))
        if not cands:
            sys.exit(
                f"{pdf_file} not found and no *_backfilled.h5 in data/processed/.\n"
                "Run examples/delta_pdf.py first to create the ΔPDF cache."
            )
        proc_path = cands[0]
    print(f"{pdf_file.name} not found — computing from {proc_path.name} ...", flush=True)
    vol = ndiff.load(proc_path)
    dpdf = compute_delta_pdf(vol, apodization="hann", zero_pad=True, subtract_mean=True)
    data, x_axis, y_axis, z_axis = dpdf.data, dpdf.x_axis, dpdf.y_axis, dpdf.z_axis
    apodization = dpdf.apodization

print(f"  ΔPDF shape (x_H,y_K,z_L): {data.shape}", flush=True)

x_value = float(os.environ.get("X_VALUE", "0.0"))
rmax = float(os.environ.get("RMAX", "25.0"))

# ------------------------------------------------------------------
# robust colour scale: p99 of |ΔPDF| at r>3 Å (skip near-origin spike)
# ------------------------------------------------------------------
xg, yg, zg = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
r = np.sqrt(xg**2 + yg**2 + zg**2)
del xg, yg, zg
vmax0 = float(np.percentile(np.abs(data[r > 3.0]), 99))
del r
print(f"  initial |scale| = {vmax0:.4g}  (p99 at r>3 Å)", flush=True)

# K,L display window indices
ik = np.abs(y_axis) <= rmax
il = np.abs(z_axis) <= rmax
y_win, z_win = y_axis[ik], z_axis[il]
extent = [y_win[0], y_win[-1], z_win[0], z_win[-1]]


def _plane(ix: int) -> np.ndarray:
    """y_K–z_L slice at x_H index ix, trimmed to the display window."""
    return data[ix][np.ix_(ik, il)]


def _nearest_x(val: float) -> int:
    return int(np.argmin(np.abs(x_axis - val)))


# ------------------------------------------------------------------
# figure
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 7.5))
plt.subplots_adjust(left=0.12, bottom=0.20, right=0.98, top=0.93)

ix0 = _nearest_x(x_value)
im = ax.imshow(
    _plane(ix0).T,
    origin="lower",
    extent=extent,
    cmap="RdBu_r",
    vmin=-vmax0, vmax=vmax0,
    aspect="equal",
    interpolation=os.environ.get("INTERP", "bilinear"),
)
ax.set_xlabel("y_K (Å)")
ax.set_ylabel("z_L (Å)")
title = ax.set_title(f"3D-ΔPDF  y_K–z_L plane   x_H = {x_axis[ix0]:+.2f} Å")
cbar = fig.colorbar(im, ax=ax, label="ΔPDF (arb. units)", shrink=0.85)

# sliders
ax_x = plt.axes([0.12, 0.10, 0.70, 0.03])
ax_v = plt.axes([0.12, 0.05, 0.70, 0.03])
s_x = Slider(ax_x, "x_H (Å)", float(x_axis.min()), float(x_axis.max()),
             valinit=float(x_axis[ix0]))
s_v = Slider(ax_v, "|scale|", vmax0 * 0.05, vmax0 * 5.0, valinit=vmax0)


def _update(_):
    ix = _nearest_x(s_x.val)
    im.set_data(_plane(ix).T)
    v = s_v.val
    im.set_clim(-v, v)
    title.set_text(f"3D-ΔPDF  y_K–z_L plane   x_H = {x_axis[ix]:+.2f} Å")
    fig.canvas.draw_idle()


s_x.on_changed(_update)
s_v.on_changed(_update)

if SMOKE:
    out = Path(__file__).parent / "_explore_delta_pdf_smoke.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[SMOKE] saved {out.name}", flush=True)
else:
    print("Drag x_H to scrub real-space planes; |scale| sets the colour range. "
          "Close the window to exit.", flush=True)
    plt.show()
