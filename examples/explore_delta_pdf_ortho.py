"""Interactive 3D-ΔPDF orthoslice viewer — all three real-space planes at once.

Shows the three orthogonal cuts through the real-space ΔPDF volume:

    x_H–y_K  (at z_L = cut)      a–b plane
    x_H–z_L  (at y_K = cut)      a–c plane
    y_K–z_L  (at x_H = cut)      b–c plane

with sliders to move each cut position and a global contrast control.  Each
panel auto-scales to its own robust level (so the three very different
magnitudes stay readable), and the contrast slider multiplies all three.

Source: ``examples/_delta_pdf.h5`` (written by ``delta_pdf.py``).

Run (interactive, on this Mac)::

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl RMAX=50 \\
      python3 \\
      examples/explore_delta_pdf_ortho.py

Controls:
    x_H / y_K / z_L sliders — move each orthogonal cut (Å)
    contrast slider         — multiply the per-panel colour scale
    Close the window to exit.

Env overrides:
    PDF_FILE  precomputed ΔPDF .h5 (default: examples/_delta_pdf.h5)
    RMAX      display half-window in Å for all axes (default: 50)
    PERCENTILE per-panel colour-scale percentile at r>3 Å (default: 98)
    CONTRAST_MIN / CONTRAST_MAX  range of the contrast-× slider that scales the
              per-panel colour limits (defaults 0.1 .. 20; raise CONTRAST_MAX to
              push the colour scale even larger / further de-saturate)
    SMOKE     1 → render the initial frame to PNG and exit (no GUI).
"""
import os
import sys
from pathlib import Path

import matplotlib

SMOKE = bool(int(os.environ.get("SMOKE", "0")))
matplotlib.use("Agg" if SMOKE else "macosx")

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

pdf_file = Path(os.environ.get("PDF_FILE", "examples/_delta_pdf.h5"))
if not pdf_file.exists():
    sys.exit(f"{pdf_file} not found — run examples/delta_pdf.py first.")

print(f"loading ΔPDF {pdf_file.name} ...", flush=True)
with h5py.File(pdf_file, "r") as fh:
    data = fh["data"][...]
    x = fh["x_axis"][...]      # x_H (Å)
    y = fh["y_axis"][...]      # y_K (Å)
    z = fh["z_axis"][...]      # z_L (Å)
    apod = fh.attrs.get("apodization", "?")
print(f"  shape (x_H,y_K,z_L): {data.shape}  apod={apod}", flush=True)

RMAX = float(os.environ.get("RMAX", "50.0"))
PCT = float(os.environ.get("PERCENTILE", "98.0"))
CMIN = float(os.environ.get("CONTRAST_MIN", "0.1"))
CMAX = float(os.environ.get("CONTRAST_MAX", "20.0"))

mx, my, mz = np.abs(x) <= RMAX, np.abs(y) <= RMAX, np.abs(z) <= RMAX
xw, yw, zw = x[mx], y[my], z[mz]


def nidx(ax, v):
    return int(np.argmin(np.abs(ax - v)))


def pvmax(slc, a1, a2):
    g1, g2 = np.meshgrid(a1, a2, indexing="ij")
    r = np.sqrt(g1 ** 2 + g2 ** 2)
    sel = np.abs(slc[r > 3.0])
    return float(np.percentile(sel, PCT)) if sel.size else 1.0


# central indices
ix0, iy0, iz0 = nidx(x, 0.0), nidx(y, 0.0), nidx(z, 0.0)

# slices (windowed)
def s_xy(iz):   # x_H–y_K at z=iz
    return data[np.ix_(mx, my, [iz])][:, :, 0]
def s_xz(iy):   # x_H–z_L at y=iy
    return data[np.ix_(mx, [iy], mz)][:, 0, :]
def s_yz(ix):   # y_K–z_L at x=ix
    return data[np.ix_([ix], my, mz)][0, :, :]

fig, axes = plt.subplots(1, 3, figsize=(20, 7.4))
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.24, top=0.90, wspace=0.28)

panels = []
specs = [
    (s_xy(iz0), xw, yw, "x_H–y_K  (z_L cut)", "x_H (Å)", "y_K (Å)"),
    (s_xz(iy0), xw, zw, "x_H–z_L  (y_K cut)", "x_H (Å)", "z_L (Å)"),
    (s_yz(ix0), yw, zw, "y_K–z_L  (x_H cut)", "y_K (Å)", "z_L (Å)"),
]
vmaxes = []
for ax, (img, a1, a2, ttl, xl, yl) in zip(axes, specs):
    vm = pvmax(img, a1, a2)
    vmaxes.append(vm)
    im = ax.imshow(img.T, origin="lower", aspect="equal",
                   extent=[a1[0], a1[-1], a2[0], a2[-1]],
                   cmap="RdBu_r", vmin=-vm, vmax=vm, interpolation="bilinear")
    ax.set_title(f"{ttl}", fontsize=12)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    fig.colorbar(im, ax=ax, shrink=0.8)
    panels.append(im)

# sliders
axc = "lightgoldenrodyellow"
ax_sx = plt.axes([0.08, 0.14, 0.55, 0.025], facecolor=axc)
ax_sy = plt.axes([0.08, 0.10, 0.55, 0.025], facecolor=axc)
ax_sz = plt.axes([0.08, 0.06, 0.55, 0.025], facecolor=axc)
ax_sc = plt.axes([0.74, 0.10, 0.20, 0.025], facecolor=axc)
s_x = Slider(ax_sx, "x_H cut (Å)", float(x.min()), float(x.max()), valinit=0.0)
s_y = Slider(ax_sy, "y_K cut (Å)", float(y.min()), float(y.max()), valinit=0.0)
s_z = Slider(ax_sz, "z_L cut (Å)", float(z.min()), float(z.max()), valinit=0.0)
s_c = Slider(ax_sc, "contrast ×", CMIN, CMAX, valinit=1.0)


def update(_):
    iz = nidx(z, s_z.val); iy = nidx(y, s_y.val); ix = nidx(x, s_x.val)
    imgs = [s_xy(iz), s_xz(iy), s_yz(ix)]
    a12 = [(xw, yw), (xw, zw), (yw, zw)]
    titles = [f"x_H–y_K  (z_L={z[iz]:+.1f} Å)",
              f"x_H–z_L  (y_K={y[iy]:+.1f} Å)",
              f"y_K–z_L  (x_H={x[ix]:+.1f} Å)"]
    for im, ax, img, (a1, a2), ttl in zip(panels, axes, imgs, a12, titles):
        im.set_data(img.T)
        vm = pvmax(img, a1, a2) * s_c.val
        im.set_clim(-vm, vm)
        ax.set_title(ttl, fontsize=12)
    fig.canvas.draw_idle()


for s in (s_x, s_y, s_z, s_c):
    s.on_changed(update)

fig.suptitle(f"3D-ΔPDF orthoslices  (apod={apod})  ±{RMAX:.0f} Å  "
             "— drag x_H/y_K/z_L cuts; contrast scales colour", y=0.97, fontsize=13)

if SMOKE:
    out = Path(__file__).parent / "_explore_delta_pdf_ortho_smoke.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[SMOKE] saved {out.name}", flush=True)
else:
    print("Drag the cut sliders to move each orthogonal plane; contrast scales "
          "the colour range. Close the window to exit.", flush=True)
    plt.show()
