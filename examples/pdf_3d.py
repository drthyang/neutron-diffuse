"""Compute the 3D-PDF (TOTAL scattering — Bragg peaks KEPT) and save for viewing.

This is the **3D-PDF**, not the 3D-ΔPDF.  ``examples/delta_pdf.py`` transforms the
Bragg-removed *diffuse* (punch + backfill first) to get deviations from the
average structure.  Here we instead Fourier-transform the **total** scattering
with the Bragg peaks left in — no punch, no backfill — giving a Patterson-like
3D-PDF dominated by the average-structure interatomic correlations, with the
diffuse on top.

The transform reuses the same engine, :func:`ndiff.analysis.compute_delta_pdf`.
The only deliberate difference from the ΔPDF settings is that the smooth-background
subtraction (a ΔPDF axis-cross fix that removes real low-frequency content) is
**off** by default.

Saved outputs (next to the input .h5 unless OUT_FILE is set; PNGs in this dir):
    <stem>_3dpdf.h5     3D-PDF volume + axes (viewer-compatible schema)
    _3dpdf_hk0.png      L=0 plane     _3dpdf_h0l.png  K=0     _3dpdf_0kl.png  H=0

Run::

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \\
      PROC_FILE=data/processed/<...>_ringremoved.h5 \\
      python3 examples/pdf_3d.py

Env:
    PROC_FILE       input .h5 (default: auto-detect *_ringremoved.h5 in data/processed)
    OUT_FILE        output .h5 (default: <stem>_3dpdf.h5 next to the input)
    APODIZE         hann | gaussian | none   (default: gaussian)
    GAUSSIAN_SIGMA  fraction of Q_max for the gaussian window  (default: 0.4)
    ZERO_PAD        0|1  (default: 1)
    SUBTRACT_MEAN   0|1  (default: 1 — removes the flat DC term, keeps Bragg)
    CROP_H/CROP_K/CROP_L   max |H/K/L| (rlu) fed to the FFT  (default: 4 / 8 / 15)
    SUBTRACT_BG     smooth-bg blur σ (rlu) — OFF by default for the total PDF
    VMAX            colour half-range  (default: auto p99 at r>3 Å)
    RMAX_H/RMAX_K/RMAX_L   real-space plot radius (Å)  (default: 20)
"""
import matplotlib

matplotlib.use("Agg")

import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

import ndiff
from ndiff.analysis import compute_delta_pdf

# ------------------------------------------------------------------
# locate input (the ring-removed, NON-punched volume — Bragg still present)
# ------------------------------------------------------------------
proc_file = os.environ.get("PROC_FILE")
if proc_file:
    proc_path = Path(proc_file)
else:
    cands = sorted(Path("data/processed").glob("*_ringremoved.h5"))
    cands = [c for c in cands if "braggpunched" not in c.name]
    if not cands:
        sys.exit(
            "No *_ringremoved.h5 found in data/processed/. "
            "Run remove_rings_3d.py first, or set PROC_FILE=/path/to/file.h5."
        )
    proc_path = cands[0]
    if len(cands) > 1:
        print(f"[warn] multiple ring-removed files; using {proc_path.name}")

print(f"loading {proc_path.name} ... (Bragg KEPT — total-scattering 3D-PDF)", flush=True)
vol = ndiff.load(proc_path)
print(f"  volume shape (H,K,L): {vol.data.shape}", flush=True)

# ------------------------------------------------------------------
# transform parameters (PDF defaults: smooth-bg subtraction OFF)
# ------------------------------------------------------------------
apodize = os.environ.get("APODIZE", "gaussian")
gaussian_sigma = float(os.environ.get("GAUSSIAN_SIGMA", "0.4"))
zero_pad = bool(int(os.environ.get("ZERO_PAD", "1")))
subtract_mean = bool(int(os.environ.get("SUBTRACT_MEAN", "1")))

_crop_h = os.environ.get("CROP_H", "4")
_crop_k = os.environ.get("CROP_K", "8")
_crop_l = os.environ.get("CROP_L", "15")
if _crop_h or _crop_k or _crop_l:
    _def_h = _crop_h or str(np.abs(vol.h_axis).max())
    _def_k = _crop_k or str(np.abs(vol.k_axis).max())
    _def_l = _crop_l or str(np.abs(vol.l_axis).max())
    crop_hkl = (float(_def_h), float(_def_k), float(_def_l))
else:
    crop_hkl = None

_sbg = os.environ.get("SUBTRACT_BG", "0")
if "," in _sbg:
    subtract_bg = tuple(float(x) for x in _sbg.split(","))
else:
    subtract_bg = float(_sbg) or None

rmax_h = float(os.environ.get("RMAX_H", "20.0"))
rmax_k = float(os.environ.get("RMAX_K", "20.0"))
rmax_l = float(os.environ.get("RMAX_L", "20.0"))

crop_str = f"  crop_hkl={crop_hkl}" if crop_hkl else ""
bg_str = f"  subtract_bg={subtract_bg} rlu" if subtract_bg else ""
print(
    f"computing 3D-PDF  apodize={apodize}  zero_pad={zero_pad}"
    f"  subtract_mean={subtract_mean}{crop_str}{bg_str}",
    flush=True,
)
pdf = compute_delta_pdf(
    vol,
    apodization=apodize,
    gaussian_sigma=gaussian_sigma,
    zero_pad=zero_pad,
    subtract_mean=subtract_mean,
    real_space_angstrom=True,
    crop_hkl=crop_hkl,
    subtract_smooth_bg=subtract_bg,
)
print(f"  output shape: {pdf.data.shape}", flush=True)
print(f"  |Q|_max = {pdf.q_max:.2f} Å⁻¹", flush=True)


def _param_string(value):
    if value is None:
        return ""
    if isinstance(value, tuple):
        return ",".join(f"{float(v):.12g}" for v in value)
    return f"{float(value):.12g}"


transform_config = ";".join((
    f"apodize={apodize}",
    f"gaussian_sigma={gaussian_sigma:.12g}",
    f"zero_pad={int(zero_pad)}",
    f"subtract_mean={int(subtract_mean)}",
    f"crop_hkl={_param_string(crop_hkl)}",
    f"subtract_bg={_param_string(subtract_bg)}",
    "kind=3dpdf_bragg_kept",
))

# ------------------------------------------------------------------
# save (same HDF5 schema as delta_pdf.py, so explore_delta_pdf_ortho.py works)
# ------------------------------------------------------------------
_default_out = proc_path.with_name(proc_path.stem + "_3dpdf.h5")
out_h5 = Path(os.environ.get("OUT_FILE", str(_default_out)))
with h5py.File(out_h5, "w") as fh:
    fh.create_dataset("data", data=pdf.data, compression="gzip", compression_opts=4)
    fh.create_dataset("x_axis", data=pdf.x_axis)
    fh.create_dataset("y_axis", data=pdf.y_axis)
    fh.create_dataset("z_axis", data=pdf.z_axis)
    fh.attrs["q_max"] = pdf.q_max
    fh.attrs["apodization"] = pdf.apodization
    fh.attrs["source_file"] = proc_path.name
    fh.attrs["kind"] = "3D-PDF (total scattering; Bragg kept)"
    fh.attrs["transform_config"] = transform_config
    try:
        _direct = 2 * np.pi * np.linalg.inv(vol.ub_matrix).T
        fh.attrs["lat_a"] = float(np.linalg.norm(_direct[:, 0]))
        fh.attrs["lat_b"] = float(np.linalg.norm(_direct[:, 1]))
        fh.attrs["lat_c"] = float(np.linalg.norm(_direct[:, 2]))
    except np.linalg.LinAlgError:
        pass
print(f"saved {out_h5.name}  ({out_h5.stat().st_size / 1e6:.0f} MB)", flush=True)

# ------------------------------------------------------------------
# quick-look orthoslices (distinct _3dpdf_*.png names; labelled 3D-PDF)
# ------------------------------------------------------------------
xg, yg, zg = np.meshgrid(pdf.x_axis, pdf.y_axis, pdf.z_axis, indexing="ij")
r_grid = np.sqrt(xg**2 + yg**2 + zg**2)
del xg, yg, zg
far = r_grid > 3.0
vmax = float(os.environ.get("VMAX", str(float(np.percentile(np.abs(pdf.data[far]), 99)))))
print(f"  colour scale: ±{vmax:.4g}", flush=True)


def _trim(axis, rmax):
    idx = np.where(np.abs(axis) <= rmax)[0]
    return (idx[[0, -1]] if len(idx) else np.array([0, len(axis) - 1]))


def _save_slice(arr2d, ax1, ax2, r1, r2, title, xlabel, ylabel, fname):
    i0, i1 = _trim(ax1, r1)
    j0, j1 = _trim(ax2, r2)
    img, x, y = arr2d[i0:i1 + 1, j0:j1 + 1], ax1[i0:i1 + 1], ax2[j0:j1 + 1]
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(img.T, origin="lower", extent=[x[0], x[-1], y[0], y[-1]],
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, label="3D-PDF (arb.)")
    out = Path(__file__).parent / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.name}", flush=True)


_save_slice(pdf.slice_hk0(), pdf.x_axis, pdf.y_axis, rmax_h, rmax_k,
            f"3D-PDF  H-K plane (L=0)  [±{vmax:.0f}]", "x_H (Å)", "y_K (Å)",
            "_3dpdf_hk0.png")
_save_slice(pdf.slice_h0l(), pdf.x_axis, pdf.z_axis, rmax_h, rmax_l,
            f"3D-PDF  H-L plane (K=0)  [±{vmax:.0f}]", "x_H (Å)", "z_L (Å)",
            "_3dpdf_h0l.png")
_save_slice(pdf.slice_0kl(), pdf.y_axis, pdf.z_axis, rmax_k, rmax_l,
            f"3D-PDF  K-L plane (H=0)  [±{vmax:.0f}]", "y_K (Å)", "z_L (Å)",
            "_3dpdf_0kl.png")
print("done.", flush=True)
