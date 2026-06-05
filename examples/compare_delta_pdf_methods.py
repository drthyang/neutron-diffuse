"""Compare background-removal methods for the single-plane 2D-ΔPDF.

The cleaned diffuse plane still carries a broad, slowly-varying diffuse
*envelope* after ring removal + Bragg punch + backfill.  Its (≈separable)
Fourier transform shows up as a bright **cross** on the y_K=0 / z_L=0 axes of
the ΔPDF.  This script compares three ways of dealing with it, on the same
shared colour scale, for H=0, 1/3, 2/3:

  1. baseline          — subtract the scalar mean only (current default).
  2. threshold-clip    — I_new = max(I − c, 0), c = a percentile of the plane.
                         Sparsifies the *input* but leaves the cross: it removes
                         the dim tails, not the bright central envelope that
                         actually makes the cross, and adds hard-edge ripple.
  3. smooth-bg subtract — I_new = I − GaussianBlur(I, σ).  Removes the envelope
                         *shape* while keeping the sharp modulation peaks and the
                         negative excursions.  This is the method that works.

Conclusion: smooth-bg subtraction is the clear winner; threshold-clip ≈ baseline.
See docs/algorithms/delta_pdf.md.

Run::

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \\
      python3 \\
      examples/compare_delta_pdf_methods.py

Env overrides:
    PROC_FILE   backfilled .h5 (default: auto-detect in data/processed/)
    H_PLANES    comma-separated H values (default: 0,0.3333,0.6666)
    CROP_K      max |K| in r.l.u. (default: 8)
    CROP_L      max |L| in r.l.u. (default: 20)
    THRESH_PCT  percentile for the threshold-clip constant (default: 70)
    SIGMA       Gaussian-bg sigma in r.l.u. (default: 1.5)
    RMAX        real-space plot radius in Å (default: 20)
"""
import matplotlib
matplotlib.use("Agg")

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft2, fftshift, ifftshift, fftfreq
from scipy.ndimage import gaussian_filter

import ndiff

# ------------------------------------------------------------------
# locate file and parameters
# ------------------------------------------------------------------
proc_file = os.environ.get("PROC_FILE")
if proc_file:
    proc_path = Path(proc_file)
else:
    cands = sorted(Path("data/processed").glob("*_backfilled.h5"))
    if not cands:
        sys.exit("No *_backfilled.h5 found; set PROC_FILE=/path/to/file.h5.")
    proc_path = cands[0]

h_planes = [float(x) for x in os.environ.get("H_PLANES", "0,0.3333,0.6666").split(",")]
crop_k   = float(os.environ.get("CROP_K", "8"))
crop_l   = float(os.environ.get("CROP_L", "20"))
pct      = float(os.environ.get("THRESH_PCT", "70"))
sigma    = float(os.environ.get("SIGMA", "1.5"))
rmax     = float(os.environ.get("RMAX", "20"))

print(f"loading {proc_path.name} ...", flush=True)
vol = ndiff.load(proc_path)

k_axis_full, l_axis_full = vol.k_axis.copy(), vol.l_axis.copy()
ik = np.where(np.abs(k_axis_full) <= crop_k)[0]
il = np.where(np.abs(l_axis_full) <= crop_l)[0]
k_axis = k_axis_full[ik]
l_axis = l_axis_full[il]
dk = (k_axis[-1] - k_axis[0]) / max(len(k_axis) - 1, 1)
dl = (l_axis[-1] - l_axis[0]) / max(len(l_axis) - 1, 1)

direct = 2 * np.pi * np.linalg.inv(vol.ub_matrix).T
b_len = np.linalg.norm(direct[:, 1])
c_len = np.linalg.norm(direct[:, 2])


def get_plane(h):
    ih = int(np.argmin(np.abs(vol.h_axis - h)))
    p = vol.data[ih, :, :].astype(np.float64)[np.ix_(ik, il)]
    return np.where(np.isfinite(p), p, 0.0), float(vol.h_axis[ih])


def transform(p, pad_factor=4):
    """Windowed, mean-subtracted, symmetrically zero-padded centred FT."""
    wk = np.hanning(p.shape[0])[:, None]
    wl = np.hanning(p.shape[1])[None, :]
    d = p * (wk * wl)
    d -= d.mean()
    target = tuple(pad_factor * 2 ** int(np.ceil(np.log2(s))) for s in d.shape)
    pad = [(t // 2 - s // 2, t - s - (t // 2 - s // 2)) for s, t in zip(d.shape, target)]
    d = np.pad(d, pad)
    return np.real(fftshift(fft2(ifftshift(d))))


def rs_axes(shape):
    nk, nl = shape
    return fftshift(fftfreq(nk, d=dk)) * b_len, fftshift(fftfreq(nl, d=dl)) * c_len


def vmaxof(pdf, yk, zl):
    yy, zz = np.meshgrid(yk, zl, indexing="ij")
    r = np.sqrt(yy ** 2 + zz ** 2)
    return float(np.percentile(np.abs(pdf[r > 3]), 99))


titles = [
    "baseline (subtract_mean only)",
    f"threshold-clip  (I−p{pct:.0f}, clip≥0)",
    f"smooth-bg subtract  (σ={sigma} rlu)",
]

# pass 1: build all maps + a single shared vmax across every ΔPDF panel
all_rows = []
for h in h_planes:
    plane, h_act = get_plane(h)
    c = np.percentile(plane, pct)
    thr = np.clip(plane - c, 0, None)
    bg = gaussian_filter(plane, sigma=(sigma / dk, sigma / dl), mode="nearest")
    maps = [transform(plane), transform(thr), transform(plane - bg)]
    all_rows.append((plane, h_act, maps))

yk, zl = rs_axes(all_rows[0][2][0].shape)
mk = np.abs(yk) <= rmax
ml = np.abs(zl) <= rmax
vmax = max(vmaxof(m, yk, zl) for _, _, maps in all_rows for m in maps)
print(f"shared colour scale = ±{vmax:.2f}", flush=True)

# pass 2: plot
nrow = len(all_rows)
fig, axes = plt.subplots(nrow, 4, figsize=(19, 4.7 * nrow), squeeze=False)
for row, (plane, h_act, maps) in enumerate(all_rows):
    ax = axes[row, 0]
    ax.imshow(plane.T, origin="lower", aspect="auto",
              extent=[k_axis[0], k_axis[-1], l_axis[0], l_axis[-1]],
              cmap="viridis", vmin=0, vmax=np.percentile(plane, 99))
    ax.set_title(f"input I(K,L)  H={h_act:.4f}")
    ax.set_xlabel("K (r.l.u.)")
    ax.set_ylabel("L (r.l.u.)")

    for col, (pdf, ttl) in enumerate(zip(maps, titles), start=1):
        ax = axes[row, col]
        im = ax.imshow(pdf[np.ix_(mk, ml)].T, origin="lower", aspect="equal",
                       extent=[yk[mk][0], yk[mk][-1], zl[ml][0], zl[ml][-1]],
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("y_K (Å)")
        ax.set_ylabel("z_L (Å)")

fig.suptitle(
    f"ΔPDF background removal: baseline vs threshold-clip vs smooth-bg   "
    f"shared scale ±{vmax:.0f}",
    y=1.005, fontsize=14,
)
fig.tight_layout()
out = Path(__file__).parent / "_delta_pdf_method_comparison.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"saved {out.name}", flush=True)
print("done.", flush=True)
