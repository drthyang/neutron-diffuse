"""Compute and visualise the 3D-ΔPDF from the backfilled diffuse volume.

Loads the fully cleaned (ring-removed → Bragg-punched → backfilled) volume,
runs the 3D-DeltaPDF Fourier transform, and writes PNG slice images.

Saved outputs (in the same directory as this script):
    _delta_pdf_hk0.png     — l=0 plane
    _delta_pdf_h0l.png     — k=0 plane
    _delta_pdf_0kl.png     — h=0 plane
    _delta_pdf_linecuts.png — 1-D line cuts (|r|>2 Å)
    _delta_pdf_radial.png  — radial RMS profile
    _delta_pdf.h5          — saved DeltaPDF arrays for future inspection

Run::

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \\
      python3 \\
      examples/delta_pdf.py

Env overrides:
    PROC_FILE       backfilled .h5 (default: auto-detect in data/processed/)
    APODIZE         hann | gaussian | none  (default: hann)
    GAUSSIAN_SIGMA  fraction of Q_max for gaussian window (default: 0.5)
    ZERO_PAD        0|1  (default: 1)
    SUBTRACT_MEAN   0|1  (default: 1)
    CROP_H          max |H| in r.l.u. to include in FFT (default: 4)
    CROP_K          max |K| in r.l.u. to include in FFT (default: full range)
    CROP_L          max |L| in r.l.u. to include in FFT (default: full range)
    SUBTRACT_BG     Gaussian-blur sigma in r.l.u. to subtract the smooth diffuse
                    background before windowing; kills the axis cross (default:
                    off; try ~1.5).  One value = isotropic 3D blur; three
                    comma-separated values = per-axis (σ_H,σ_K,σ_L).  Use
                    σ_H=0, e.g. SUBTRACT_BG=0,1.5,1.5, for a slice-wise (per-H-
                    plane) background that preserves the H-layering.
    VMAX            colour scale half-range  (default: auto 99th-percentile)
    RMAX_H          real-space plot radius along H-axis in Å (default: 20)
    RMAX_K          real-space plot radius along K-axis in Å (default: 20)
    RMAX_L          real-space plot radius along L-axis in Å (default: 20)
"""

import matplotlib
matplotlib.use("Agg")

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ndiff
from ndiff.analysis import compute_delta_pdf

# ------------------------------------------------------------------
# locate backfilled file
# ------------------------------------------------------------------
proc_file = os.environ.get("PROC_FILE")
if proc_file:
    proc_path = Path(proc_file)
else:
    proc_dir = Path("data/processed")
    cands = sorted(proc_dir.glob("*_backfilled.h5"))
    if not cands:
        sys.exit(
            "No *_backfilled.h5 found in data/processed/. "
            "Run backfill_bragg_3d.py first, or set PROC_FILE=/path/to/file.h5."
        )
    proc_path = cands[0]
    if len(cands) > 1:
        print(f"[warn] multiple backfilled files; using {proc_path.name}")

print(f"loading {proc_path.name} ...", flush=True)
vol = ndiff.load(proc_path)
print(f"  volume shape (H,K,L): {vol.data.shape}", flush=True)

finite_mask = np.isfinite(vol.data)
nan_frac = (~finite_mask).mean()
if nan_frac > 0:
    print(f"  [warn] {nan_frac*100:.2f}% NaN in backfilled volume — treating as 0", flush=True)
else:
    print("  NaN fraction: 0 (all holes filled)", flush=True)

# ------------------------------------------------------------------
# transform parameters
# ------------------------------------------------------------------
apodize        = os.environ.get("APODIZE", "hann")
gaussian_sigma = float(os.environ.get("GAUSSIAN_SIGMA", "0.5"))
zero_pad       = bool(int(os.environ.get("ZERO_PAD", "1")))
subtract_mean  = bool(int(os.environ.get("SUBTRACT_MEAN", "1")))

_crop_h = os.environ.get("CROP_H", "4")
_crop_k = os.environ.get("CROP_K")
_crop_l = os.environ.get("CROP_L")
if _crop_h or _crop_k or _crop_l:
    _def_h = str(np.abs(vol.h_axis).max()) if not _crop_h else _crop_h
    _def_k = str(np.abs(vol.k_axis).max()) if not _crop_k else _crop_k
    _def_l = str(np.abs(vol.l_axis).max()) if not _crop_l else _crop_l
    crop_hkl = (float(_def_h), float(_def_k), float(_def_l))
else:
    crop_hkl = None

_sbg = os.environ.get("SUBTRACT_BG", "0")
if "," in _sbg:
    subtract_bg = tuple(float(x) for x in _sbg.split(","))   # (σ_H, σ_K, σ_L)
else:
    subtract_bg = float(_sbg) or None

rmax_h = float(os.environ.get("RMAX_H", "20.0"))
rmax_k = float(os.environ.get("RMAX_K", "20.0"))
rmax_l = float(os.environ.get("RMAX_L", "20.0"))

crop_str = f"  crop_hkl={crop_hkl}" if crop_hkl else ""
bg_str = f"  subtract_bg={subtract_bg} rlu" if subtract_bg else ""
print(
    f"computing 3D-ΔPDF  apodize={apodize}  zero_pad={zero_pad}"
    f"  subtract_mean={subtract_mean}{crop_str}{bg_str}",
    flush=True,
)
dpdf = compute_delta_pdf(
    vol,
    apodization=apodize,
    gaussian_sigma=gaussian_sigma,
    zero_pad=zero_pad,
    subtract_mean=subtract_mean,
    real_space_angstrom=True,
    crop_hkl=crop_hkl,
    subtract_smooth_bg=subtract_bg,
)
print(f"  output shape: {dpdf.data.shape}", flush=True)
print(f"  |Q|_max = {dpdf.q_max:.2f} Å⁻¹", flush=True)
print(f"  real-space range: x ±{dpdf.x_axis.max():.1f} Å,"
      f"  y ±{dpdf.y_axis.max():.1f} Å,"
      f"  z ±{dpdf.z_axis.max():.1f} Å", flush=True)


def _param_string(value):
    if value is None:
        return ""
    if isinstance(value, tuple):
        return ",".join(f"{float(v):.12g}" for v in value)
    return f"{float(value):.12g}"


transform_config = ";".join(
    (
        f"apodize={apodize}",
        f"gaussian_sigma={gaussian_sigma:.12g}",
        f"zero_pad={int(zero_pad)}",
        f"subtract_mean={int(subtract_mean)}",
        f"crop_hkl={_param_string(crop_hkl)}",
        f"subtract_bg={_param_string(subtract_bg)}",
    )
)

# Save DeltaPDF to HDF5 so it can be reloaded without recomputing
import h5py
_default_out = Path(__file__).parent / "_delta_pdf.h5"
out_h5 = Path(os.environ.get("OUT_FILE", str(_default_out)))
with h5py.File(out_h5, "w") as fh:
    fh.create_dataset("data", data=dpdf.data, compression="gzip", compression_opts=4)
    fh.create_dataset("x_axis", data=dpdf.x_axis)
    fh.create_dataset("y_axis", data=dpdf.y_axis)
    fh.create_dataset("z_axis", data=dpdf.z_axis)
    fh.attrs["q_max"]       = dpdf.q_max
    fh.attrs["apodization"] = dpdf.apodization
    fh.attrs["source_file"] = proc_path.name
    fh.attrs["crop_hkl"] = _param_string(crop_hkl)
    fh.attrs["subtract_smooth_bg"] = _param_string(subtract_bg)
    fh.attrs["gaussian_sigma"] = gaussian_sigma
    fh.attrs["zero_pad"] = int(zero_pad)
    fh.attrs["subtract_mean"] = int(subtract_mean)
    fh.attrs["transform_config"] = transform_config
    # store direct-lattice constants (Å) so viewers can draw unit-cell gridlines
    try:
        _direct = 2 * np.pi * np.linalg.inv(vol.ub_matrix).T
        fh.attrs["lat_a"] = float(np.linalg.norm(_direct[:, 0]))
        fh.attrs["lat_b"] = float(np.linalg.norm(_direct[:, 1]))
        fh.attrs["lat_c"] = float(np.linalg.norm(_direct[:, 2]))
    except np.linalg.LinAlgError:
        pass
print(f"saved {out_h5.name}  ({out_h5.stat().st_size/1e6:.0f} MB)", flush=True)

# Colour scale: set by the p99 of |DeltaPDF| at r>3 Å to avoid the near-origin
# spike (from backfill discontinuities) dominating the scale.  The r<1 Å region
# can be 10^3–10^4× larger than the physical signal at r=3–20 Å.
xg, yg, zg = np.meshgrid(dpdf.x_axis, dpdf.y_axis, dpdf.z_axis, indexing='ij')
r_grid = np.sqrt(xg**2 + yg**2 + zg**2)
del xg, yg, zg  # free memory; r_grid is saved for the linecut

far_mask = r_grid > 3.0
vmax_auto = float(np.percentile(np.abs(dpdf.data[far_mask]), 99))
vmax = float(os.environ.get("VMAX", str(vmax_auto)))
print(f"  colour scale: ±{vmax:.4g}  (auto p99 at r>3Å = {vmax_auto:.4g})", flush=True)

# lattice constants from UB (for annotation)
try:
    import numpy.linalg as la
    direct = 2 * np.pi * la.inv(vol.ub_matrix).T
    lat_a = la.norm(direct[:, 0])
    lat_b = la.norm(direct[:, 1])
    lat_c = la.norm(direct[:, 2])
    print(f"  lattice: a={lat_a:.3f} b={lat_b:.3f} c={lat_c:.3f} Å", flush=True)
except Exception:
    lat_a = lat_b = lat_c = None

# ------------------------------------------------------------------
# helper: trim axes and slice to ±rmax
# ------------------------------------------------------------------
def _trim(axis: np.ndarray, rmax: float):
    idx = np.where(np.abs(axis) <= rmax)[0]
    if len(idx) == 0:
        idx = np.arange(len(axis))
    return idx[[0, -1]]          # first, last index


def _slice2d(arr2d, ax1, ax2, r1, r2, title, xlabel, ylabel, cscale=None):
    i0, i1 = _trim(ax1, r1)
    j0, j1 = _trim(ax2, r2)
    img = arr2d[i0:i1+1, j0:j1+1]
    x = ax1[i0:i1+1]
    y = ax2[j0:j1+1]
    cv = cscale if cscale is not None else vmax
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        img.T,
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        cmap="RdBu_r",
        vmin=-cv, vmax=cv,
        aspect="equal",
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, label="ΔPDF (arb. units)")
    return fig


# tight scale for the r>3 Å view (excluding near-origin spike)
vmax_far = vmax  # already set by p99 at r>3 Å

# ------------------------------------------------------------------
# hk0 slice  (z=0 → L=0)
# ------------------------------------------------------------------
slc_hk0 = dpdf.slice_hk0()   # shape (nh, nk)
fig_hk0 = _slice2d(
    slc_hk0, dpdf.x_axis, dpdf.y_axis,
    rmax_h, rmax_k,
    f"3D-ΔPDF  H-K plane (L=0)  [scale ±{vmax_far:.0f}]",
    "x_H (Å)", "y_K (Å)",
)
out_hk0 = Path(__file__).parent / "_delta_pdf_hk0.png"
fig_hk0.savefig(out_hk0, dpi=150, bbox_inches="tight")
plt.close(fig_hk0)
print(f"saved {out_hk0.name}", flush=True)

# ------------------------------------------------------------------
# h0l slice  (y=0 → K=0)
# ------------------------------------------------------------------
slc_h0l = dpdf.slice_h0l()   # shape (nh, nl)
fig_h0l = _slice2d(
    slc_h0l, dpdf.x_axis, dpdf.z_axis,
    rmax_h, rmax_l,
    f"3D-ΔPDF  H-L plane (K=0)  [scale ±{vmax_far:.0f}]",
    "x_H (Å)", "z_L (Å)",
)
out_h0l = Path(__file__).parent / "_delta_pdf_h0l.png"
fig_h0l.savefig(out_h0l, dpi=150, bbox_inches="tight")
plt.close(fig_h0l)
print(f"saved {out_h0l.name}", flush=True)

# ------------------------------------------------------------------
# 0kl slice  (x=0 → H=0)
# ------------------------------------------------------------------
slc_0kl = dpdf.slice_0kl()   # shape (nk, nl)
fig_0kl = _slice2d(
    slc_0kl, dpdf.y_axis, dpdf.z_axis,
    rmax_k, rmax_l,
    f"3D-ΔPDF  K-L plane (H=0)  [scale ±{vmax_far:.0f}]",
    "y_K (Å)", "z_L (Å)",
)
out_0kl = Path(__file__).parent / "_delta_pdf_0kl.png"
fig_0kl.savefig(out_0kl, dpi=150, bbox_inches="tight")
plt.close(fig_0kl)
print(f"saved {out_0kl.name}", flush=True)

# ------------------------------------------------------------------
# radial profile: rms vs r (log-log) — shows artifact decay
# ------------------------------------------------------------------
r_edges = np.arange(0, 50.1, 0.5)
r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
rms_r = []
for r0, r1 in zip(r_edges[:-1], r_edges[1:]):
    mask = (r_grid >= r0) & (r_grid < r1)
    vals = dpdf.data[mask]
    rms_r.append(vals.std() if len(vals) > 0 else 0)

fig_rp, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(r_mid, rms_r, "b-", lw=1.2)
ax.set_xlabel("r (Å)")
ax.set_ylabel("RMS ΔPDF")
ax.set_title("Radial RMS of ΔPDF  (indicates near-origin artifact scale)")
ax.axhline(rms_r[-1] * 2, ls="--", color="gray", lw=0.8, label=f"noise floor ~{rms_r[-1]:.0f}")
if lat_a:
    for dist, lbl in [(lat_a, "a"), (lat_b, "b"), (lat_c, "c")]:
        if dist < 50:
            ax.axvline(dist, color="r", lw=0.7, ls=":")
            ax.text(dist + 0.3, ax.get_ylim()[1] * 0.5, lbl, color="r", fontsize=8)
ax.legend()
ax.grid(True, which="both", alpha=0.3)
out_rp = Path(__file__).parent / "_delta_pdf_radial.png"
fig_rp.tight_layout()
fig_rp.savefig(out_rp, dpi=150, bbox_inches="tight")
plt.close(fig_rp)
print(f"saved {out_rp.name}", flush=True)

# ------------------------------------------------------------------
# 1-D line cuts through the origin (r>2 Å only, log-scale y)
# ------------------------------------------------------------------
imid_h = dpdf.data.shape[0] // 2
imid_k = dpdf.data.shape[1] // 2
imid_l = dpdf.data.shape[2] // 2

cut_h = dpdf.data[:, imid_k, imid_l]
cut_k = dpdf.data[imid_h, :, imid_l]
cut_l = dpdf.data[imid_h, imid_k, :]

fig_lc, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, cut, axis, rmax, label, lat_dist in [
    (axes[0], cut_h, dpdf.x_axis, rmax_h, "x_H (Å)", lat_a),
    (axes[1], cut_k, dpdf.y_axis, rmax_k, "y_K (Å)", lat_b),
    (axes[2], cut_l, dpdf.z_axis, rmax_l, "z_L (Å)", lat_c),
]:
    mask = (np.abs(axis) >= 2) & (np.abs(axis) <= rmax)
    ax.plot(axis[mask], cut[mask], lw=0.8)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    if lat_dist:
        for sign in [-1, 1]:
            ax.axvline(sign * lat_dist, color="r", lw=0.7, ls=":", alpha=0.6)
    ax.set_xlabel(label)
    ax.set_ylabel("ΔPDF")
    ax.set_title(f"Cut along {label.split()[0]}  (|r|>2 Å)")

fig_lc.suptitle(
    f"3D-ΔPDF line cuts  (apodize={apodize}, |r|>2 Å to skip near-origin spike)", y=1.02
)
fig_lc.tight_layout()
out_lc = Path(__file__).parent / "_delta_pdf_linecuts.png"
fig_lc.savefig(out_lc, dpi=150, bbox_inches="tight")
plt.close(fig_lc)
print(f"saved {out_lc.name}", flush=True)

print("done.", flush=True)
