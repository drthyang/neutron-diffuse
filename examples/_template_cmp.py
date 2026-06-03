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

# Ring templates from the Bragg-free linecut.
L = float(data.l_axis.max())
q1, I1, _ = line_profile(data, (0, 1, 0), (0, 1, L), 800)
q2, I2, _ = line_profile(data, (0, -1, 0), (0, -1, L), 800)
templates = fit_ring_profiles(q1, np.nanmean(np.vstack([I1, I2]), 0))

common = dict(n_patches=36, plane="0kl", q_step=0.02, ring_width=0.24,
              baseline_smooth=0.06, texture_model="fourier", n_fourier=3)
m0 = PatchedRadialRingModel(**common)
m1 = PatchedRadialRingModel(ring_templates=templates, **common)
p0 = m0.fit(src, q_range=(1.5, 10.5)); _, I0 = m0.subtract(src, p0)
p1 = m1.fit(src, q_range=(1.5, 10.5)); _, I1r = m1.subtract(src, p1)

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
print(" ring    data   resid(no-tmpl)  resid(tmpl)   leftover_no  leftover_tmpl")
dprof = medprof(d.data)
for r in templates:
    b = int(np.argmin(np.abs(qc - r.q_center)))
    r0 = medprof(d.data - I0); r1 = medprof(d.data - I1r)
    base0 = np.nanpercentile(r0[max(0,b-12):b+12], 20)
    lo0 = r0[b] - base0; lo1 = r1[b] - np.nanpercentile(r1[max(0,b-12):b+12],20)
    print(f"  {r.q_center:5.2f}  {dprof[b]:6.3f}    {r0[b]:8.3f}      {r1[b]:8.3f}    "
          f"{lo0:7.4f}      {lo1:7.4f}")

# Visual: residual at tight scale, side by side.
res0 = dataclasses.replace(src, data=np.where(keep, d.data - I0, np.nan))
res1 = dataclasses.replace(src, data=np.where(keep, d.data - I1r, np.nan))
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, vol, t in zip(axes, [res0, res1], ["residual: non-parametric", "residual: + templates"]):
    sl = extract_slice(vol, plane="0kl", value=0.0)
    ext=[sl.x_axis[0],sl.x_axis[-1],sl.y_axis[0],sl.y_axis[-1]]
    im=ax.imshow(np.ma.masked_invalid(sl.data),origin="lower",extent=ext,aspect="auto",
                 cmap="inferno",vmin=0,vmax=0.05)
    ax.set_title(t); ax.set_xlabel(sl.x_label); ax.set_ylabel(sl.y_label)
    fig.colorbar(im,ax=ax,fraction=0.046)
fig.tight_layout(); fig.savefig("examples/_template_cmp.png", dpi=110)
print("wrote examples/_template_cmp.png")
