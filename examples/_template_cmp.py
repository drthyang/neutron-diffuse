"""Compare ring removal WITHOUT vs WITH linecut radial templates."""
import matplotlib
matplotlib.use("Agg")
import dataclasses
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import ndiff
from ndiff.preprocessing import (PatchedRadialRingModel, azimuthal_sampling_mask,
                                 line_profile, fit_ring_profiles)
from ndiff.visualization import extract_slice

data = ndiff.load([p for p in sorted(Path("data/raw").glob("*.nxs"))
                   if not p.stem.endswith(("_bkg", "_sub_bkg"))][0])
ih0 = int(np.argmin(np.abs(data.h_axis)))
d = dataclasses.replace(data, data=data.data[ih0:ih0+1], sigma=data.sigma[ih0:ih0+1],
                        mask=data.mask[ih0:ih0+1], h_axis=data.h_axis[ih0:ih0+1])
keep = azimuthal_sampling_mask(d, plane="0kl", min_count_frac=0.25, q_range=(1.5, 10.5))
src = dataclasses.replace(d, mask=keep)

# Ring templates from the Bragg-free linecuts: (0, ±1, l), l=0 → ±30.
L = float(max(abs(data.l_axis.min()), abs(data.l_axis.max())))
q_ref = None
cuts = []
for k0 in (-1.0, 1.0):
    for l1 in (-L, L):
        q, I, _ = line_profile(data, (0.0, k0, 0.0), (0.0, k0, l1), 900)
        if q_ref is None:
            q_ref = q
        elif not np.allclose(q, q_ref, rtol=0.0, atol=1e-8):
            I = np.interp(q_ref, q, I, left=np.nan, right=np.nan)
        cuts.append(I)
templates = fit_ring_profiles(q_ref, np.nanmean(np.vstack(cuts), 0),
                              cluster_gap=0.35, half_window=0.24)
print("linecut templates:")
for r in templates:
    print(f"  q={r.q_center:6.3f} sigma={r.sigma:6.4f} FWHM={r.fwhm:6.4f}")

common = dict(n_patches=36, plane="0kl", q_step=0.02, ring_width=0.24,
              baseline_smooth=0.06, texture_model="fourier")
models = [
    ("n3 no-template", PatchedRadialRingModel(n_fourier=3, **common)),
    ("n6 no-template", PatchedRadialRingModel(n_fourier=6, **common)),
    ("n6 + linecut templates",
     PatchedRadialRingModel(n_fourier=6, ring_templates=templates, **common)),
]
ring_fits = []
for label, model in models:
    p = model.fit(src, q_range=(1.5, 10.5))
    _, I = model.subtract(src, p)
    ring_fits.append((label, I))

q = src.q_magnitude(); v = keep & np.isfinite(d.data)
# Leftover ring metric: residual median minus a morphological baseline, at peaks.
from scipy.ndimage import grey_opening
edges = np.arange(1.5, 8.0, 0.02); qc = 0.5*(edges[:-1]+edges[1:])
binv = np.digitize(q[v], edges) - 1
def medprof(arr):
    a = arr[v]; out = np.full(len(qc), np.nan)
    for b in range(len(qc)):
        s = a[binv == b]
        if s.size: out[b] = np.median(s)
    return out
print(" ring    data  " + "  ".join([f"{label:>20}" for label, _ in ring_fits]))
dprof = medprof(d.data)
for r in templates:
    b = int(np.argmin(np.abs(qc - r.q_center)))
    leftovers = []
    for _, I in ring_fits:
        rp = medprof(d.data - I)
        base = np.nanpercentile(rp[max(0, b - 12):b + 12], 20)
        leftovers.append(rp[b] - base)
    print(f"  {r.q_center:5.2f}  {dprof[b]:6.3f}  "
          + "  ".join([f"{x:20.4f}" for x in leftovers]))

# Visual: residual at tight scale, side by side.
fig, axes = plt.subplots(1, len(ring_fits), figsize=(18, 6))
for ax, (label, I) in zip(axes, ring_fits):
    vol = dataclasses.replace(src, data=np.where(keep, d.data - I, np.nan))
    sl = extract_slice(vol, plane="0kl", value=0.0)
    ext=[sl.x_axis[0],sl.x_axis[-1],sl.y_axis[0],sl.y_axis[-1]]
    im=ax.imshow(np.ma.masked_invalid(sl.data),origin="lower",extent=ext,aspect="auto",
                 cmap="inferno",vmin=0,vmax=0.05)
    ax.set_title(f"residual: {label}"); ax.set_xlabel(sl.x_label); ax.set_ylabel(sl.y_label)
    fig.colorbar(im,ax=ax,fraction=0.046)
fig.tight_layout(); fig.savefig("examples/_template_cmp.png", dpi=110)
print("wrote examples/_template_cmp.png")
