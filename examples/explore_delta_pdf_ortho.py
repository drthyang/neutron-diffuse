"""Interactive 3D-PDF / 3D-ΔPDF orthoslice viewer — all three real-space planes at once.

The plot title labels the kind (3D-PDF when the file carries a ``kind`` attr from
``pdf_3d.py``, else 3D-ΔPDF) and the temperature parsed from the source filename.

Shows the three orthogonal cuts through the real-space ΔPDF volume:

    x_H–y_K  (at z_L = cut)      a–b plane
    x_H–z_L  (at y_K = cut)      a–c plane
    y_K–z_L  (at x_H = cut)      b–c plane

with sliders to move each cut position and a global contrast control.  Each
panel auto-scales to its own robust level (so the three very different
magnitudes stay readable), and the contrast slider multiplies all three.

Source: a ``*_delta_pdf.h5`` in ``data/processed/`` (written by
``run_pipeline.py``, or by ``delta_pdf.py`` with ``OUT_FILE`` pointed there).
With several temperatures present, set ``TEMP`` to pick one, or ``PDF_FILE`` for
an explicit file.  (A bare ``delta_pdf.py`` run defaults to
``examples/_delta_pdf.h5``, which this viewer does NOT auto-load.)

Run (interactive, on this Mac)::

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl RMAX=50 \\
      python3 \\
      examples/explore_delta_pdf_ortho.py

Controls:
    x_H / y_K / z_L sliders — move each orthogonal cut (Å)
    contrast slider         — multiply the per-panel colour scale
    "unit cells" checkbox   — toggle the light-gray unit-cell gridlines
    Close the window to exit.

Env overrides:
    PDF_FILE  explicit ΔPDF .h5 to load (overrides the data/processed glob)
    TEMP      22K | 45K | 100K — pick one when data/processed/ holds several
              *_delta_pdf.h5 (the viewer exits asking for this if it is ambiguous)
    RMAX      display half-window in Å for all axes (default: 50)
    PERCENTILE per-panel colour-scale percentile at r>3 Å (default: 98)
    CONTRAST_MIN / CONTRAST_MAX  range of the contrast-× slider that scales the
              per-panel colour limits (defaults 0.1 .. 20; raise CONTRAST_MAX to
              push the colour scale even larger / further de-saturate)
    LAT_A / LAT_B / LAT_C  direct-lattice constants in Å for the unit-cell
              gridlines (default: read from the ΔPDF file attrs, else the source
              UB matrix)
    SMOKE     1 → render the initial frame to PNG and exit (no GUI).
"""
import os
import re
import sys
from pathlib import Path

import matplotlib

SMOKE = bool(int(os.environ.get("SMOKE", "0")))
matplotlib.use("Agg" if SMOKE else "macosx")

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons, Slider

_pdf_env = os.environ.get("PDF_FILE")
_temp    = os.environ.get("TEMP", "")
if _pdf_env:
    pdf_file = Path(_pdf_env)
    if not pdf_file.exists():
        sys.exit(f"PDF_FILE={pdf_file} not found.")
else:
    _cands = sorted(Path("data/processed").glob("*_delta_pdf.h5"))
    if _temp:
        _cands = [p for p in _cands if _temp in p.name]
    if not _cands:
        sys.exit(
            "No matching *_delta_pdf.h5 in data/processed/.\n"
            "Set TEMP=22K (or 45K / 100K), or PDF_FILE=/path/to/file.h5."
        )
    if len(_cands) > 1:
        names = "\n  ".join(p.name for p in _cands)
        sys.exit(
            f"Multiple ΔPDF files — set TEMP=22K (or 45K / 100K) to pick one:\n  {names}"
        )
    pdf_file = _cands[0]

print(f"loading {pdf_file.name} ...", flush=True)
with h5py.File(pdf_file, "r") as fh:
    data = fh["data"][...]
    x = fh["x_axis"][...]      # x_H (Å)
    y = fh["y_axis"][...]      # y_K (Å)
    z = fh["z_axis"][...]      # z_L (Å)
    apod = fh.attrs.get("apodization", "?")
    _kind_attr = str(fh.attrs.get("kind", ""))
    _source = str(fh.attrs.get("source_file", ""))

# Correct label from the file: 3D-PDF (total scattering, Bragg kept; pdf_3d.py
# stamps a "kind" attr) vs 3D-ΔPDF (Bragg removed; delta_pdf.py, no such attr).
KIND = "3D-PDF" if "3D-PDF" in _kind_attr else "3D-ΔPDF"
# Temperature parsed from the source filename (…22K…/…45K…/…100K…), else "".
_m = re.search(r"(\d+)\s*K", _source or pdf_file.name)
TEMP = f"{_m.group(1)} K" if _m else ""
print(f"  {KIND}{(' — ' + TEMP) if TEMP else ''}"
      f"  shape (x_H,y_K,z_L): {data.shape}  apod={apod}", flush=True)

RMAX = float(os.environ.get("RMAX", "50.0"))
PCT = float(os.environ.get("PERCENTILE", "98.0"))
CMIN = float(os.environ.get("CONTRAST_MIN", "0.1"))
CMAX = float(os.environ.get("CONTRAST_MAX", "20.0"))


def _lattice():
    """Direct-lattice constants (a, b, c) in Å for unit-cell gridlines, or None.

    Order of precedence: ΔPDF-file attrs (lat_a/b/c) → env LAT_A/LAT_B/LAT_C →
    the source backfilled file's UB matrix (cheap h5py read).
    """
    with h5py.File(pdf_file, "r") as fh:
        if all(k in fh.attrs for k in ("lat_a", "lat_b", "lat_c")):
            return (float(fh.attrs["lat_a"]), float(fh.attrs["lat_b"]),
                    float(fh.attrs["lat_c"]))
        src = str(fh.attrs.get("source_file", ""))
    ev = [os.environ.get(k) for k in ("LAT_A", "LAT_B", "LAT_C")]
    if all(ev):
        return tuple(float(v) for v in ev)
    if src:
        sp = Path("data/processed") / src
        if sp.exists():
            try:
                with h5py.File(sp, "r") as fh:
                    ub = np.array(fh["entry/ub_matrix"], dtype=float)
                d = 2 * np.pi * np.linalg.inv(ub).T
                return tuple(float(np.linalg.norm(d[:, i])) for i in range(3))
            except Exception:
                pass
    return None

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
try:  # name the OS window so several viewers are distinguishable
    fig.canvas.manager.set_window_title(f"{KIND} {TEMP}".strip())
except Exception:
    pass
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

# --- light-gray unit-cell gridlines (toggleable) ---
# spacings per panel match the displayed axes: x_H↔a, y_K↔b, z_L↔c.
lat = _lattice()
gridlines = []
if lat is not None:
    a_len, b_len, c_len = lat
    panel_spacing = [(a_len, b_len), (a_len, c_len), (b_len, c_len)]
    panel_axes = [(xw, yw), (xw, zw), (yw, zw)]
    for ax, (a1, a2), (sx, sy) in zip(axes, panel_axes, panel_spacing):
        if sx and sx > 0:
            nmax = int(np.floor(max(abs(a1[0]), abs(a1[-1])) / sx))
            for n in range(-nmax, nmax + 1):
                gridlines.append(ax.axvline(n * sx, color="0.6", lw=0.6,
                                            alpha=0.7, zorder=3))
        if sy and sy > 0:
            mmax = int(np.floor(max(abs(a2[0]), abs(a2[-1])) / sy))
            for m in range(-mmax, mmax + 1):
                gridlines.append(ax.axhline(m * sy, color="0.6", lw=0.6,
                                            alpha=0.7, zorder=3))
    print(f"  unit-cell grid: a={a_len:.3f} b={b_len:.3f} c={c_len:.3f} Å "
          f"({len(gridlines)} lines)", flush=True)
else:
    print("  unit-cell grid: lattice unknown (set LAT_A/LAT_B/LAT_C to enable)",
          flush=True)

# controls — cut sliders (left column), contrast + unit-cell toggle (right column)
axc = "lightgoldenrodyellow"
ax_sx = plt.axes([0.09, 0.135, 0.54, 0.028], facecolor=axc)
ax_sy = plt.axes([0.09, 0.090, 0.54, 0.028], facecolor=axc)
ax_sz = plt.axes([0.09, 0.045, 0.54, 0.028], facecolor=axc)
ax_sc = plt.axes([0.76, 0.115, 0.18, 0.028], facecolor=axc)
s_x = Slider(ax_sx, "x_H cut (Å)", float(x.min()), float(x.max()), valinit=0.0)
s_y = Slider(ax_sy, "y_K cut (Å)", float(y.min()), float(y.max()), valinit=0.0)
s_z = Slider(ax_sz, "z_L cut (Å)", float(z.min()), float(z.max()), valinit=0.0)
s_c = Slider(ax_sc, "contrast ×", CMIN, CMAX, valinit=1.0)

# unit-cell gridline on/off toggle (under the contrast slider, frameless)
ax_chk = plt.axes([0.76, 0.04, 0.18, 0.055])
ax_chk.set_frame_on(False)
chk = CheckButtons(ax_chk, ["unit cells"], [True])


def _toggle_grid(_label):
    vis = chk.get_status()[0]
    for ln in gridlines:
        ln.set_visible(vis)
    fig.canvas.draw_idle()


chk.on_clicked(_toggle_grid)


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

_temp_seg = f"  {TEMP}" if TEMP else ""
fig.suptitle(f"{KIND} orthoslices{_temp_seg}  (apod={apod})  ±{RMAX:.0f} Å  "
             "— drag x_H/y_K/z_L cuts; contrast scales colour", y=0.97, fontsize=13)

if SMOKE:
    out = Path(__file__).parent / "_explore_delta_pdf_ortho_smoke.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[SMOKE] saved {out.name}", flush=True)
else:
    print("Drag the cut sliders to move each orthogonal plane; contrast scales "
          "the colour range. Close the window to exit.", flush=True)
    plt.show()
