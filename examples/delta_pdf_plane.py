"""2D-ΔPDF of a single reciprocal-space H plane.

The full 3D-ΔPDF (``examples/delta_pdf.py``) transforms the whole volume into
real space, so its slices live at the origin (x_H=0, …).  To see the real-space
correlations carried by a *single* diffuse layer — e.g. the magnetic/structural
modulation on H=±1/3 — you instead 2D-Fourier-transform just that K-L plane.

This gives a real-space (y_K, z_L) map: the in-plane pair correlations
associated with the chosen propagation vector.  Note this is a per-plane
transform, NOT a slice of the 3D-ΔPDF (a 3D slice at fixed x_H sums all H
planes with phase factors).

Saved outputs (same directory as this script):
    _delta_pdf_plane_H<...>.png   — 2D real-space map + reciprocal input panel

Run::

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl H_PLANE=0.3333 \\
      /opt/homebrew/Caskroom/miniforge/base/envs/sci-general/bin/python3 \\
      examples/delta_pdf_plane.py

Env overrides:
    PROC_FILE   backfilled .h5 (default: auto-detect in data/processed/)
    H_PLANE     reciprocal H value to transform (default: 0.3333)
    APODIZE     hann | gaussian | none  (default: hann)
    ZERO_PAD    0|1  (default: 1)
    RMAX_K/RMAX_L  real-space plot radius in Å (default: 20)
    VMAX        colour-scale half-range (default: auto p99 at r>3 Å)
"""

import matplotlib
matplotlib.use("Agg")

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft2, fftshift, fftfreq

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

h_plane  = float(os.environ.get("H_PLANE", "0.3333"))
apodize  = os.environ.get("APODIZE", "hann")
zero_pad = bool(int(os.environ.get("ZERO_PAD", "1")))
rmax_k   = float(os.environ.get("RMAX_K", "20.0"))
rmax_l   = float(os.environ.get("RMAX_L", "20.0"))

print(f"loading {proc_path.name} ...", flush=True)
vol = ndiff.load(proc_path)

ih = int(np.argmin(np.abs(vol.h_axis - h_plane)))
h_actual = float(vol.h_axis[ih])
print(f"  H={h_plane} → index {ih} (h_axis={h_actual:.5f})", flush=True)

plane = vol.data[ih, :, :].astype(np.float64)   # (nK, nL)
plane = np.where(np.isfinite(plane), plane, 0.0)
print(f"  plane shape (K,L): {plane.shape}  "
      f"mean={plane.mean():.4g}  std={plane.std():.4g}", flush=True)

# ------------------------------------------------------------------
# 2D windowed transform
# ------------------------------------------------------------------
def _win1d(n: int) -> np.ndarray:
    if apodize == "hann":
        return np.hanning(n)
    if apodize == "gaussian":
        x = np.linspace(-1, 1, n)
        return np.exp(-0.5 * (x / 0.5) ** 2)
    return np.ones(n)

wk = _win1d(plane.shape[0])[:, None]
wl = _win1d(plane.shape[1])[None, :]
data = plane * (wk * wl)
data -= data.mean()              # window first, then zero the DC (see delta_pdf.py)

if zero_pad:
    target = tuple(1 if s == 0 else 2 ** int(np.ceil(np.log2(s))) for s in data.shape)
    data = np.pad(data, [(0, t - s) for s, t in zip(data.shape, target)])

ft = fftshift(fft2(data))
pdf2d = np.real(ft)
print(f"  output shape: {pdf2d.shape}", flush=True)

# ------------------------------------------------------------------
# real-space axes in Å (K, L directions)
# ------------------------------------------------------------------
dk = (vol.k_axis[-1] - vol.k_axis[0]) / max(len(vol.k_axis) - 1, 1)
dl = (vol.l_axis[-1] - vol.l_axis[0]) / max(len(vol.l_axis) - 1, 1)
nk, nl = pdf2d.shape
y_frac = fftshift(fftfreq(nk, d=dk))
z_frac = fftshift(fftfreq(nl, d=dl))

direct = 2 * np.pi * np.linalg.inv(vol.ub_matrix).T
b_len = np.linalg.norm(direct[:, 1])
c_len = np.linalg.norm(direct[:, 2])
y_axis = y_frac * b_len
z_axis = z_frac * c_len
print(f"  lattice b={b_len:.3f} Å  c={c_len:.3f} Å", flush=True)

# colour scale: p99 at r>3 Å to dodge the near-origin spike
yg, zg = np.meshgrid(y_axis, z_axis, indexing="ij")
r = np.sqrt(yg**2 + zg**2)
vmax_auto = float(np.percentile(np.abs(pdf2d[r > 3.0]), 99))
vmax = float(os.environ.get("VMAX", str(vmax_auto)))
print(f"  colour scale: ±{vmax:.4g}  (auto p99 at r>3Å = {vmax_auto:.4g})", flush=True)

# ------------------------------------------------------------------
# plot: input reciprocal plane (left) + real-space 2D-ΔPDF (right)
# ------------------------------------------------------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6))

# input diffuse plane
imL = axL.imshow(
    plane.T, origin="lower", aspect="auto",
    extent=[vol.k_axis[0], vol.k_axis[-1], vol.l_axis[0], vol.l_axis[-1]],
    cmap="viridis", vmin=0, vmax=np.percentile(plane, 99),
)
axL.set_title(f"input: cleaned diffuse plane H={h_actual:.4f}")
axL.set_xlabel("K (r.l.u.)")
axL.set_ylabel("L (r.l.u.)")
plt.colorbar(imL, ax=axL, label="I (arb.)")

# real-space map, trimmed to ±rmax
ik = np.abs(y_axis) <= rmax_k
il = np.abs(z_axis) <= rmax_l
img = pdf2d[np.ix_(ik, il)]
imR = axR.imshow(
    img.T, origin="lower", aspect="equal",
    extent=[y_axis[ik][0], y_axis[ik][-1], z_axis[il][0], z_axis[il][-1]],
    cmap="RdBu_r", vmin=-vmax, vmax=vmax,
)
axR.set_title(f"2D-ΔPDF of H={h_actual:.4f}  [scale ±{vmax:.0f}]")
axR.set_xlabel("y_K (Å)")
axR.set_ylabel("z_L (Å)")
for sign in (-1, 1):
    axR.axvline(sign * b_len, color="r", lw=0.7, ls=":", alpha=0.6)
    axR.axhline(sign * c_len, color="r", lw=0.7, ls=":", alpha=0.6)
plt.colorbar(imR, ax=axR, label="ΔPDF (arb.)")

fig.suptitle(f"Single-plane 2D-ΔPDF  (H={h_actual:.4f}, apodize={apodize})", y=1.02)
fig.tight_layout()

tag = f"{h_actual:+.4f}".replace(".", "p").replace("+", "p").replace("-", "m")
out = Path(__file__).parent / f"_delta_pdf_plane_H{tag}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"saved {out.name}", flush=True)
print("done.", flush=True)
