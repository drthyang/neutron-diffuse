"""Diagnose how well each texture model captures the ring's azimuthal profile.

For each detected ring |Q|, plot the *measured* per-patch ring amplitude vs φ
(point area ∝ voxel count, so well-sampled arcs are visually dominant) and
overlay the texture T(φ) reconstructed by each model.  This reveals whether the
low-order Fourier series is under-fitting genuine azimuthal inhomogeneity.
"""
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


raw = Path("data/raw")
data_file = os.environ.get("DATA_FILE")
if data_file:
    data = ndiff.load(Path(data_file))
else:
    cands = [p for p in sorted(raw.glob("*.nxs"))
             if not p.stem.endswith(("_bkg", "_sub_bkg"))]
    data = ndiff.load(next((p for p in cands if "22K_mmm" in p.stem), cands[0]))
H_VALUE = float(os.environ.get("H_VALUE", "0.3333"))
ih0 = int(np.argmin(np.abs(data.h_axis - H_VALUE)))
d = dataclasses.replace(
    data, data=data.data[ih0:ih0 + 1], sigma=data.sigma[ih0:ih0 + 1],
    mask=data.mask[ih0:ih0 + 1], h_axis=data.h_axis[ih0:ih0 + 1],
)
print(f"slice H={float(d.h_axis[0]):.4f}")
keep = azimuthal_sampling_mask(d, plane="0kl", min_count_frac=0.25, q_range=(1.5, 10.5))
src = dataclasses.replace(d, mask=keep)

# Detected ring centers from clean Bragg-free linecuts.
L = float(max(abs(data.l_axis.min()), abs(data.l_axis.max())))
q_ref, cuts = None, []
for k0 in (-1.0, 1.0):
    for l1 in (-L, L):
        q, I, _ = line_profile(data, (0.0, k0, 0.0), (0.0, k0, l1), 900)
        if q_ref is None:
            q_ref = q
        elif not np.allclose(q, q_ref, atol=1e-8):
            I = np.interp(q_ref, q, I, left=np.nan, right=np.nan)
        cuts.append(I)
ring_centers = [r.q_center for r in fit_ring_profiles(q_ref, np.nanmean(np.vstack(cuts), 0))]

common = dict(n_patches=36, plane="0kl", q_step=0.02, ring_width=0.24,
              baseline_method="snip", baseline_smooth=0.06,
              profile_percentiles=(10.0, 80.0), profile_method="trimmed_mean",
              texture_symmetric=False)
models = {
    # old: per-bin fits (no |Q| pooling)
    "f3 (old ref)":   PatchedRadialRingModel(texture_model="fourier", n_fourier=3,  texture_ridge=0.3,  texture_q_smooth=0.0, **common),
    "f8 no-pool":     PatchedRadialRingModel(texture_model="fourier", n_fourier=8,  texture_ridge=0.05, texture_q_smooth=0.0, **common),
    # new: same high order but pooled along |Q|
    "f8 qpool":       PatchedRadialRingModel(texture_model="fourier", n_fourier=8,  texture_ridge=0.05, texture_q_smooth=0.06, **common),
    "f12 qpool":      PatchedRadialRingModel(texture_model="fourier", n_fourier=12, texture_ridge=0.02, texture_q_smooth=0.08, **common),
}
profs = {name: m.fit(src, q_range=(1.5, 10.5)) for name, m in models.items()}

# Reference profile object (any) for the measured per-patch amplitudes + counts.
ref = profs["f3 (old ref)"]
pc = ref.patch_centers
phi_dense = np.linspace(0, 2 * np.pi, 361)

# Quantitative fit quality.  The single-|Q|-bin measured amplitude is noisy, so
# we compare each model to the |Q|-POOLED measured texture: per patch, the
# amplitude-weighted mean of the measured ring amplitude over a ±0.10 Å⁻¹ window
# around each ring centre.  That averages out per-bin noise and is the coherent
# azimuthal texture the model should reproduce.  RMS is taken on well-sampled
# patches only.  extrap_swing = peak-to-peak of T on the under-sampled azimuths
# (proxy for ringing into unmeasured regions; lower is better).
half = int(round(0.10 / 0.02))
print(f"{'model':14s}  RMS_to_pooled_measured  extrap_swing")
for name, prof in profs.items():
    num = den = swing = 0.0
    for q0 in ring_centers:
        b = int(np.argmin(np.abs(ref.q_grid - q0)))
        lo, hi = max(0, b - half), min(ref.q_grid.size, b + half + 1)
        block = ref.ring_profile[:, lo:hi]            # (P, w)
        wq = block.mean(axis=0, keepdims=True)        # |Q|-bin amplitude weight
        pooled = (block * wq).sum(1) / np.maximum(wq.sum(), 1e-12)  # (P,)
        cnt = ref.counts[:, b]
        if cnt.max() <= 0:
            continue
        well = cnt >= 0.4 * cnt.max()
        T_patch = prof.evaluate(np.full(pc.shape, q0), pc)
        num += float(np.sum(cnt[well] * (T_patch[well] - pooled[well]) ** 2))
        den += float(np.sum(cnt[well]))
        poor = ~well
        if np.any(poor):
            swing += float(np.ptp(T_patch[poor]))
    rms = np.sqrt(num / den) if den > 0 else np.nan
    print(f"{name:14s}  {rms:22.5f}  {swing:12.4f}")

rings_to_plot = ring_centers[:6]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, q0 in zip(axes.ravel(), rings_to_plot):
    b = int(np.argmin(np.abs(ref.q_grid - q0)))
    meas = ref.ring_profile[:, b]          # measured per-patch amplitude
    cnt = ref.counts[:, b]
    sizes = 10 + 200 * (cnt / cnt.max()) if cnt.max() > 0 else np.full_like(cnt, 20)
    ax.scatter(np.degrees(pc), meas, s=sizes, c="0.5", alpha=0.6,
               label="measured (size∝count)", zorder=2)
    for name, prof in profs.items():
        T = prof.evaluate(np.full_like(phi_dense, q0), phi_dense)
        ax.plot(np.degrees(phi_dense), T, lw=1.6, label=name, zorder=3)
    ax.set_title(f"ring |Q|={q0:.3f}")
    ax.set_xlabel("azimuth φ (deg)")
    ax.set_ylabel("ring amplitude")
    ax.set_xlim(0, 360)
axes[0, 0].legend(fontsize=8)
fig.suptitle(f"Azimuthal ring texture T(φ) — H={float(d.h_axis[0]):.4f}", fontsize=13)
fig.tight_layout()
out = "examples/_azimuthal_texture_cmp.png"
fig.savefig(out, dpi=110)
print(f"wrote {out}")
