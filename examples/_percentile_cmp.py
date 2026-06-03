"""Compare 0kl ring removal for robust-profile percentile bands."""
import matplotlib
matplotlib.use("Agg")

import dataclasses
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ndiff
from ndiff.preprocessing import (
    PatchedRadialRingModel,
    azimuthal_sampling_mask,
    fit_ring_profiles,
    line_profile,
)
from ndiff.visualization import extract_slice


raw = Path("data/raw")
data_file = os.environ.get("DATA_FILE")
data = ndiff.load(Path(data_file) if data_file else [p for p in sorted(raw.glob("*.nxs"))
                   if not p.stem.endswith(("_bkg", "_sub_bkg"))][0])
ih0 = int(np.argmin(np.abs(data.h_axis)))
d = dataclasses.replace(
    data,
    data=data.data[ih0:ih0 + 1],
    sigma=data.sigma[ih0:ih0 + 1],
    mask=data.mask[ih0:ih0 + 1],
    h_axis=data.h_axis[ih0:ih0 + 1],
)
keep = azimuthal_sampling_mask(d, plane="0kl", min_count_frac=0.25,
                               q_range=(1.5, 10.5))
src = dataclasses.replace(d, mask=keep)

# Clean linecuts only identify diagnostic ring centers.
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
ring_centers = [r.q_center for r in fit_ring_profiles(q_ref, np.nanmean(np.vstack(cuts), 0))]

variants = [
    ("q=.0025 p10-80", 0.0025, (10.0, 80.0)),
    ("q=.0025 p0-80", 0.0025, (0.0, 80.0)),
    ("q=.005 p10-80", 0.005, (10.0, 80.0)),
    ("q=.005 p0-80", 0.005, (0.0, 80.0)),
    ("q=.02 p10-80", 0.02, (10.0, 80.0)),
    ("q=.02 p0-80", 0.02, (0.0, 80.0)),
]
ring_fits = []
for label, q_step, pct in variants:
    model = PatchedRadialRingModel(
        n_patches=36,
        plane="0kl",
        q_step=q_step,
        ring_width=0.24,
        baseline_smooth=0.06,
        profile_percentiles=pct,
        texture_model="fourier",
        n_fourier=3,
        texture_symmetric=False,
    )
    profiles = model.fit(src, q_range=(1.5, 10.5))
    _, I_ring = model.subtract(src, profiles)
    ring_fits.append((label, I_ring))

q = src.q_magnitude()
valid = keep & np.isfinite(d.data)
edges = np.arange(1.5, 8.0, 0.02)
qc = 0.5 * (edges[:-1] + edges[1:])
binv = np.digitize(q[valid], edges) - 1


def medprof(arr):
    a = arr[valid]
    out = np.full(len(qc), np.nan)
    for b in range(len(qc)):
        s = a[binv == b]
        if s.size:
            out[b] = np.median(s)
    return out


print(" ring    data  " + "  ".join([f"{label:>15}" for label, _ in ring_fits]))
dprof = medprof(d.data)
totals = np.zeros(len(ring_fits))
for q0 in ring_centers:
    b = int(np.argmin(np.abs(qc - q0)))
    leftovers = []
    for i, (_, I_ring) in enumerate(ring_fits):
        rp = medprof(d.data - I_ring)
        base = np.nanpercentile(rp[max(0, b - 12):b + 12], 20)
        leftover = rp[b] - base
        leftovers.append(leftover)
        totals[i] += max(0.0, leftover)
    print(f"  {q0:5.2f}  {dprof[b]:6.3f}  "
          + "  ".join([f"{x:15.4f}" for x in leftovers]))
print(" total positive leftover  "
      + "  ".join([f"{x:15.4f}" for x in totals]))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, (label, I_ring) in zip(axes.ravel(), ring_fits):
    vol = dataclasses.replace(src, data=np.where(keep, d.data - I_ring, np.nan))
    sl = extract_slice(vol, plane="0kl", value=0.0)
    ext = [sl.x_axis[0], sl.x_axis[-1], sl.y_axis[0], sl.y_axis[-1]]
    im = ax.imshow(
        np.ma.masked_invalid(sl.data),
        origin="lower",
        extent=ext,
        aspect="auto",
        cmap="inferno",
        vmin=0,
        vmax=0.05,
    )
    ax.set_title(label)
    ax.set_xlabel(sl.x_label)
    ax.set_ylabel(sl.y_label)
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig("examples/_percentile_cmp.png", dpi=110)
print("wrote examples/_percentile_cmp.png")
