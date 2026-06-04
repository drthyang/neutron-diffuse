"""Large, clear 0kl slice views: data, removed rings, residual, signed residual.

The signed-residual panel uses a diverging colormap centered at 0, so
over-subtraction (negative troughs at ring positions) shows up directly as
blue rings.  Set VARIANT to pick which removal config to inspect.
"""
import matplotlib
matplotlib.use("Agg")

import dataclasses
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
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

# Removal config to inspect (env VARIANT:
# 'cc_off' | 'cc_on' | 'cc_on_soft' | 'cc_on_template' | 'default' |
# 'f3old' | 'smooth10').
#   default : new reference — n_fourier=8, texture_q_smooth=0.06 (|Q|-pooled
#             azimuthal texture, the inhomogeneous-texture fix)
#   f3old   : the previous reference (low-order, unpooled) for comparison
VARIANT = os.environ.get("VARIANT", os.environ.get("RING_PRESET", "cc_off"))
common = dict(n_patches=36, plane="0kl", ring_width=0.24,
              baseline_method="snip", baseline_smooth=0.06,
              profile_percentiles=(10.0, 80.0),
              texture_symmetric=False, q_step=0.02)
templates = None
if VARIANT == "cc_on_template":
    lmax = float(max(abs(data.l_axis.min()), abs(data.l_axis.max())))
    q_ref = None
    cuts = []
    for k0 in (-1.0, 1.0):
        for l1 in (-lmax, lmax):
            q, I, _ = line_profile(data, (0.0, k0, 0.0), (0.0, k0, l1), n_points=900)
            if q_ref is None:
                q_ref = q
            elif not np.allclose(q, q_ref, rtol=0.0, atol=1e-8):
                I = np.interp(q_ref, q, I, left=np.nan, right=np.nan)
            cuts.append(I)
    templates = fit_ring_profiles(
        q_ref, np.nanmean(np.vstack(cuts), axis=0), prominence=0.04,
        min_distance=8, cluster_gap=0.35, half_window=0.24, sigma0=0.04,
    )
    print("linecut templates:")
    for tmpl in templates:
        print(f"  |Q|={tmpl.q_center:6.3f} sigma={tmpl.sigma:6.4f}")

if VARIANT == "cc_on":
    model = PatchedRadialRingModel(
        texture_model="fourier", n_fourier=6, texture_ridge=0.08,
        texture_q_smooth=0.02, profile_method="median", **common)
elif VARIANT == "cc_on_template":
    model = PatchedRadialRingModel(
        texture_model="fourier", n_fourier=6, texture_ridge=0.08,
        texture_q_smooth=0.02, profile_method="median",
        ring_templates=templates, **common)
elif VARIANT == "cc_on_soft":
    model = PatchedRadialRingModel(
        texture_model="fourier", n_fourier=3, texture_ridge=0.15,
        texture_q_smooth=0.06, profile_method="median", **common)
elif VARIANT == "cc_off":
    model = PatchedRadialRingModel(
        texture_model="fourier", n_fourier=8, texture_ridge=0.05,
        texture_q_smooth=0.0, profile_method="median", **common)
elif VARIANT == "smooth10":
    model = PatchedRadialRingModel(texture_model="smooth", texture_smoothness=10.0,
                                   profile_method="trimmed_mean", **common)
elif VARIANT == "f3old":
    model = PatchedRadialRingModel(texture_model="fourier", n_fourier=3,
                                   texture_ridge=0.3, texture_q_smooth=0.0,
                                   adaptive_ring_width=False,
                                   profile_method="trimmed_mean", **common)
elif VARIANT == "globalw":  # new texture but fixed global ring_width (no adaptive)
    model = PatchedRadialRingModel(texture_model="fourier",
                                   adaptive_ring_width=False,
                                   profile_method="trimmed_mean", **common)
else:  # 'default' — |Q|-pooled texture + adaptive per-ring thickness (class defaults)
    model = PatchedRadialRingModel(texture_model="fourier",
                                   profile_method="trimmed_mean", **common)
profiles = model.fit(src, q_range=(1.5, 10.5))
_, I_ring = model.subtract(src, profiles)
resid = d.data - I_ring


def slc(arr):
    vol = dataclasses.replace(src, data=np.where(keep, arr, np.nan))
    return extract_slice(vol, plane="0kl", value=0.0)


# Linear-scale display limits (env-overridable).  Rings saturate; the point is
# to see the diffuse star and any negative troughs at the diffuse level.
VMIN = float(os.environ.get("VMIN", "0.0"))
VMAX = float(os.environ.get("VMAX", "0.3"))
CMAP = os.environ.get("CMAP", "viridis")

panels = [
    ("data",              d.data, "lin"),
    (f"removed rings ({VARIANT})", I_ring, "lin_pos"),
    ("residual = data - rings", resid, "lin"),
    ("signed residual (diverging)", resid, "diverge"),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 18))
for ax, (title, arr, mode) in zip(axes.ravel(), panels):
    sl = slc(arr)
    ext = [sl.x_axis[0], sl.x_axis[-1], sl.y_axis[0], sl.y_axis[-1]]
    pdata = np.ma.masked_invalid(sl.data)
    if mode == "lin":
        im = ax.imshow(pdata, origin="lower", extent=ext, aspect="auto",
                       cmap=CMAP, vmin=VMIN, vmax=VMAX)
    elif mode == "lin_pos":
        im = ax.imshow(pdata, origin="lower", extent=ext, aspect="auto",
                       cmap=CMAP, vmin=0.0, vmax=VMAX)
    else:  # diverging, centered at 0 — blue = over-subtraction
        im = ax.imshow(pdata, origin="lower", extent=ext, aspect="auto",
                       cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-0.1, vcenter=0.0, vmax=VMAX))
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(sl.x_label)
    ax.set_ylabel(sl.y_label)
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
tag = f"h{float(d.h_axis[0]):+.4f}".replace("+", "p").replace("-", "m").replace(".", "p")
out = f"examples/_slice_view_{VARIANT}_{tag}.png"
fig.savefig(out, dpi=110)
print(f"wrote {out}")
