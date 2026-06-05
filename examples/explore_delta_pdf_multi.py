"""Interactive 3D-ΔPDF temperature comparison — 22 K / 45 K / 100 K.

Shows three rows (one per temperature) × three orthogonal real-space cuts:

    col 0: x_H – y_K  (at z_L = cut)
    col 1: x_H – z_L  (at y_K = cut)
    col 2: y_K – z_L  (at x_H = cut)

All nine panels share a single colour scale (global p99 of |ΔPDF| at r > 3 Å
across all three datasets), so intensities are directly comparable across
temperatures.  The contrast slider multiplies this global scale.

Run::

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \\
      python3 examples/explore_delta_pdf_multi.py

Env:
    PDF_22K   path to 22 K ΔPDF .h5  (default: examples/_delta_pdf_22K.h5)
    PDF_45K   path to 45 K ΔPDF .h5  (default: examples/_delta_pdf_45K.h5)
    PDF_100K  path to 100 K ΔPDF .h5 (default: examples/_delta_pdf_100K.h5)
    RMAX      display half-window in Å for all axes (default: 50)
    PERCENTILE global colour-scale percentile at r > 3 Å (default: 98)
    CONTRAST_MIN / CONTRAST_MAX  contrast slider range (default: 0.1 / 20)
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
from matplotlib.widgets import CheckButtons, Slider

HERE = Path(__file__).resolve().parent

TEMPS = ["22K", "45K", "100K"]
PDF_ENVS = {"22K": "PDF_22K", "45K": "PDF_45K", "100K": "PDF_100K"}

RMAX = float(os.environ.get("RMAX", "50.0"))
PCT = float(os.environ.get("PERCENTILE", "98.0"))
CMIN = float(os.environ.get("CONTRAST_MIN", "0.1"))
CMAX = float(os.environ.get("CONTRAST_MAX", "20.0"))

# ------------------------------------------------------------------
# load all three ΔPDF files
# ------------------------------------------------------------------
datasets = {}
for t in TEMPS:
    default = HERE / f"_delta_pdf_{t}.h5"
    p = Path(os.environ.get(PDF_ENVS[t], str(default)))
    if not p.exists():
        sys.exit(f"{p} not found — run delta_pdf.py with OUT_FILE={p} first.")
    print(f"loading {t}: {p.name} ...", flush=True)
    with h5py.File(p, "r") as fh:
        data = fh["data"][...]
        x = fh["x_axis"][...]
        y = fh["y_axis"][...]
        z = fh["z_axis"][...]
        apod = str(fh.attrs.get("apodization", "?"))
        lat = None
        if all(k in fh.attrs for k in ("lat_a", "lat_b", "lat_c")):
            lat = (float(fh.attrs["lat_a"]),
                   float(fh.attrs["lat_b"]),
                   float(fh.attrs["lat_c"]))
    mx = np.abs(x) <= RMAX
    my = np.abs(y) <= RMAX
    mz = np.abs(z) <= RMAX
    datasets[t] = dict(data=data, x=x, y=y, z=z, apod=apod, lat=lat,
                       mx=mx, my=my, mz=mz,
                       xw=x[mx], yw=y[my], zw=z[mz])
    print(f"  shape={data.shape}  apod={apod}", flush=True)

# ------------------------------------------------------------------
# global colour scale: p99 of |ΔPDF| at r > 3 Å across all datasets
# ------------------------------------------------------------------
print(f"computing global colour scale (p{PCT:.0f} at r>3 Å) ...", flush=True)
_all_vals = []
for t in TEMPS:
    d = datasets[t]
    x, y, z = d["x"], d["y"], d["z"]
    r2 = x[:, None, None]**2 + y[None, :, None]**2 + z[None, None, :]**2
    _all_vals.append(np.abs(d["data"][r2 > 9.0]))
global_vmax = float(np.percentile(np.concatenate(_all_vals), PCT))
del _all_vals
print(f"  global vmax = ±{global_vmax:.4g}", flush=True)


def nidx(ax, v):
    return int(np.argmin(np.abs(ax - v)))


def _slices(d, ix, iy, iz):
    mx, my, mz, data = d["mx"], d["my"], d["mz"], d["data"]
    return [
        data[np.ix_(mx, my, [iz])][:, :, 0],
        data[np.ix_(mx, [iy], mz)][:, 0, :],
        data[np.ix_([ix], my, mz)][0, :, :],
    ]


# use first dataset's axes for the shared cut sliders
ref = datasets[TEMPS[0]]

# ------------------------------------------------------------------
# figure layout: 3 rows × 3 cols
# ------------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(21, 17))
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.16, top=0.93,
                    wspace=0.26, hspace=0.42)

COL_PLANES = ["x_H – y_K  (z_L cut)", "x_H – z_L  (y_K cut)", "y_K – z_L  (x_H cut)"]
XLABELS = ["x_H (Å)", "x_H (Å)", "y_K (Å)"]
YLABELS = ["y_K (Å)", "z_L (Å)", "z_L (Å)"]

panels = {}   # panels[temp][col] = AxesImage

for ri, t in enumerate(TEMPS):
    d = datasets[t]
    ix0 = nidx(d["x"], 0.0)
    iy0 = nidx(d["y"], 0.0)
    iz0 = nidx(d["z"], 0.0)
    imgs0 = _slices(d, ix0, iy0, iz0)
    a12 = [(d["xw"], d["yw"]), (d["xw"], d["zw"]), (d["yw"], d["zw"])]

    panels[t] = []
    for ci, (img, (a1, a2), xl, yl, plane) in enumerate(
            zip(imgs0, a12, XLABELS, YLABELS, COL_PLANES)):
        ax = axes[ri][ci]
        im = ax.imshow(img.T, origin="lower", aspect="equal",
                       extent=[a1[0], a1[-1], a2[0], a2[-1]],
                       cmap="RdBu_r", vmin=-global_vmax, vmax=global_vmax,
                       interpolation="bilinear")
        ax.set_title(f"{t}  {plane}", fontsize=10)
        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel(yl, fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.7)
        panels[t].append(im)

# ------------------------------------------------------------------
# unit-cell gridlines (one set per dataset row, toggleable)
# ------------------------------------------------------------------
gridlines_all = []
for ri, t in enumerate(TEMPS):
    d = datasets[t]
    lat = d["lat"]
    if lat is None:
        continue
    a_len, b_len, c_len = lat
    panel_axes = [(d["xw"], d["yw"]), (d["xw"], d["zw"]), (d["yw"], d["zw"])]
    spacings = [(a_len, b_len), (a_len, c_len), (b_len, c_len)]
    for ci, (ax, (a1, a2), (sx, sy)) in enumerate(
            zip(axes[ri], panel_axes, spacings)):
        row_lines = []
        for sp, axfn in [(sx, ax.axvline), (sy, ax.axhline)]:
            avals = a1 if (axfn == ax.axvline) else a2
            nmax = int(np.floor(max(abs(avals[0]), abs(avals[-1])) / sp)) if sp > 0 else 0
            for n in range(-nmax, nmax + 1):
                row_lines.append(axfn(n * sp, color="0.6", lw=0.6,
                                      alpha=0.7, zorder=3))
        gridlines_all.extend(row_lines)

# ------------------------------------------------------------------
# sliders
# ------------------------------------------------------------------
axc = "lightgoldenrodyellow"
ax_sx = plt.axes([0.07, 0.118, 0.55, 0.022], facecolor=axc)
ax_sy = plt.axes([0.07, 0.080, 0.55, 0.022], facecolor=axc)
ax_sz = plt.axes([0.07, 0.042, 0.55, 0.022], facecolor=axc)
ax_sc = plt.axes([0.76, 0.080, 0.18, 0.022], facecolor=axc)
s_x = Slider(ax_sx, "x_H cut (Å)", float(ref["x"].min()), float(ref["x"].max()), valinit=0.0)
s_y = Slider(ax_sy, "y_K cut (Å)", float(ref["y"].min()), float(ref["y"].max()), valinit=0.0)
s_z = Slider(ax_sz, "z_L cut (Å)", float(ref["z"].min()), float(ref["z"].max()), valinit=0.0)
s_c = Slider(ax_sc, "contrast ×", CMIN, CMAX, valinit=1.0)

ax_chk = plt.axes([0.76, 0.028, 0.18, 0.040])
ax_chk.set_frame_on(False)
chk = CheckButtons(ax_chk, ["unit cells"], [True])


def _toggle_grid(_label):
    vis = chk.get_status()[0]
    for ln in gridlines_all:
        ln.set_visible(vis)
    fig.canvas.draw_idle()


chk.on_clicked(_toggle_grid)


def update(_):
    vm = global_vmax * s_c.val
    for t in TEMPS:
        d = datasets[t]
        ix = nidx(d["x"], s_x.val)
        iy = nidx(d["y"], s_y.val)
        iz = nidx(d["z"], s_z.val)
        imgs = _slices(d, ix, iy, iz)
        titles = [
            f"{t}  x_H–y_K  (z_L={d['z'][iz]:+.1f} Å)",
            f"{t}  x_H–z_L  (y_K={d['y'][iy]:+.1f} Å)",
            f"{t}  y_K–z_L  (x_H={d['x'][ix]:+.1f} Å)",
        ]
        for im, img, ttl in zip(panels[t], imgs, titles):
            im.set_data(img.T)
            im.set_clim(-vm, vm)
            im.axes.set_title(ttl, fontsize=10)
    fig.canvas.draw_idle()


for s in (s_x, s_y, s_z, s_c):
    s.on_changed(update)

fig.suptitle(
    f"3D-ΔPDF: 22 K / 45 K / 100 K  ·  ±{RMAX:.0f} Å  ·  "
    f"global scale ±{global_vmax:.4g}  — drag sliders to move cuts",
    y=0.975, fontsize=13,
)

if SMOKE:
    out = HERE / "_delta_pdf_multi_smoke.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(f"[SMOKE] saved {out.name}", flush=True)
else:
    print("Drag the cut sliders to move each orthogonal plane. "
          "Close the window to exit.", flush=True)
    plt.show()
