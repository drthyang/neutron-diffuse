"""Check radial continuity of residual backgrounds through powder rings."""
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
data = ndiff.load(Path(data_file) if data_file else [p for p in sorted(raw.glob("*.nxs"))
                   if not p.stem.endswith(("_bkg", "_sub_bkg"))][0])
H_VALUE = float(os.environ.get("H_VALUE", "0.3333"))
ih0 = int(np.argmin(np.abs(data.h_axis - H_VALUE)))
d = dataclasses.replace(
    data,
    data=data.data[ih0:ih0 + 1],
    sigma=data.sigma[ih0:ih0 + 1],
    mask=data.mask[ih0:ih0 + 1],
    h_axis=data.h_axis[ih0:ih0 + 1],
)
print(f"slice H target={H_VALUE:.4f}, using H={float(d.h_axis[0]):.4f}")

keep = azimuthal_sampling_mask(d, plane="0kl", min_count_frac=0.25,
                               q_range=(1.5, 10.5))
src = dataclasses.replace(d, mask=keep)

# Clean linecuts identify diagnostic ring centers.
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
    # SNIP (new default) vs opening (old) side-by-side for the reference config
    ("q.02 f3 snip", dict(q_step=0.02, texture_model="fourier", n_fourier=3,
                          texture_ridge=0.3, baseline_method="snip", ring_smooth=0.0)),
    ("q.02 f3 open", dict(q_step=0.02, texture_model="fourier", n_fourier=3,
                          texture_ridge=0.3, baseline_method="opening", ring_smooth=0.0)),
    # SNIP variants
    ("q.01 f6 snip", dict(q_step=0.01, texture_model="fourier", n_fourier=6,
                          texture_ridge=0.1, baseline_method="snip", ring_smooth=0.0)),
    ("q.02 smooth10 snip", dict(q_step=0.02, texture_model="smooth",
                                texture_smoothness=10.0, baseline_method="snip",
                                ring_smooth=0.0)),
    ("q.02 smooth30 snip", dict(q_step=0.02, texture_model="smooth",
                                texture_smoothness=30.0, baseline_method="snip",
                                ring_smooth=0.0)),
    ("q.02 f10 r.1 snip", dict(q_step=0.02, texture_model="fourier", n_fourier=10,
                                texture_ridge=0.1, baseline_method="snip",
                                ring_smooth=0.0)),
    ("q.02 f10 r.02 snip", dict(q_step=0.02, texture_model="fourier", n_fourier=10,
                                 texture_ridge=0.02, baseline_method="snip",
                                 ring_smooth=0.0)),
    ("q.0025 f3 snip", dict(q_step=0.0025, texture_model="fourier", n_fourier=3,
                             texture_ridge=0.3, baseline_method="snip",
                             ring_smooth=0.0)),
    ("q.0025 f6 snip", dict(q_step=0.0025, texture_model="fourier", n_fourier=6,
                             texture_ridge=0.1, baseline_method="snip",
                             ring_smooth=0.0)),
]

q = src.q_magnitude()
valid = keep & np.isfinite(d.data)
edges = np.arange(1.5, 8.0, 0.01)
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


def local_line_baseline(profile, b, half=24, gap=7):
    lo = max(0, b - half)
    hi = min(profile.size, b + half + 1)
    idx = np.arange(lo, hi)
    side = np.abs(idx - b) >= gap
    side &= np.isfinite(profile[idx])
    if side.sum() < 4:
        return float(np.nanpercentile(profile[lo:hi], 20))
    x = qc[idx[side]]
    y = profile[idx[side]]
    coef = np.polyfit(x, y, deg=1)
    return float(np.polyval(coef, qc[b]))


fits = []
for label, params in variants:
    params = params.copy()
    baseline_method = params.pop("baseline_method", "snip")
    model = PatchedRadialRingModel(
        n_patches=36,
        plane="0kl",
        ring_width=0.24,
        baseline_method=baseline_method,
        baseline_smooth=0.06,
        profile_percentiles=(10.0, 80.0),
        profile_method="trimmed_mean",
        texture_symmetric=False,
        **params,
    )
    profiles = model.fit(src, q_range=(1.5, 10.5))
    _, I_ring = model.subtract(src, profiles)
    resid_prof = medprof(d.data - I_ring)
    fits.append((label, I_ring, resid_prof))

print("variant              abs_resid  neg_trough  radial_rough  offring_p95")
for label, I_ring, rp in fits:
    abs_sum = 0.0
    neg_sum = 0.0
    rough_sum = 0.0
    for q0 in ring_centers:
        b = int(np.argmin(np.abs(qc - q0)))
        base = local_line_baseline(rp, b)
        resid = rp[b] - base
        abs_sum += abs(resid)
        neg_sum += max(0.0, -resid)
        lo = max(1, b - 10)
        hi = min(rp.size - 1, b + 11)
        second = rp[lo + 1:hi + 1] - 2 * rp[lo:hi] + rp[lo - 1:hi - 1]
        rough_sum += float(np.nanmedian(np.abs(second)))
    off_ring = valid & (q >= 1.5) & (q <= 8.0)
    for q0 in ring_centers:
        off_ring &= np.abs(q - q0) > 0.18
    print(f"{label:20s} {abs_sum:9.4f}  {neg_sum:10.4f}  "
          f"{rough_sum:12.5f}  {np.percentile(I_ring[off_ring], 95):10.5f}")

fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=False, sharey=False)
plot_centers = [3.113, 4.405, 6.798, 6.962]
for ax, q0 in zip(axes.ravel(), plot_centers):
    b = int(np.argmin(np.abs(qc - q0)))
    lo = max(0, b - 35)
    hi = min(qc.size, b + 36)
    for label, _, rp in fits:
        ax.plot(qc[lo:hi], rp[lo:hi], lw=1.0, label=label)
    ax.axvline(q0, color="k", lw=0.8, alpha=0.35)
    ax.set_title(f"Residual radial profile near |Q|={q0:.3f}")
    ax.set_xlabel("|Q| (A^-1)")
    ax.set_ylabel("median residual")
axes[0, 0].legend(fontsize=8)
fig.tight_layout()
tag = f"h{float(d.h_axis[0]):+.4f}".replace("+", "p").replace("-", "m").replace(".", "p")
out = f"examples/_radial_continuity_cmp_{tag}.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
