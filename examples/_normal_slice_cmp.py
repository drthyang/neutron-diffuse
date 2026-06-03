"""Compare conservative ring-removal variants on the diffuse H slice."""
import matplotlib
matplotlib.use("Agg")

import dataclasses
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

import ndiff
from ndiff.preprocessing import (
    PatchedRadialRingModel,
    azimuthal_sampling_mask,
)
from ndiff.visualization import extract_slice


raw = Path("data/raw")
data_file = os.environ.get("DATA_FILE")
data = ndiff.load(Path(data_file) if data_file else [p for p in sorted(raw.glob("*.nxs"))
                   if not p.stem.endswith(("_bkg", "_sub_bkg"))][0])
H_VALUE = float(os.environ.get("H_VALUE", "0.3333"))
VMAX = float(os.environ.get("VMAX", "0.10"))
VMIN = float(os.environ.get("VMIN", "0.0"))
LOG_SCALE = os.environ.get("LOG_SCALE", "0") == "1"
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

variants = [
    # label, extra_params, scale, center_offset
    ("data", None, 0.0, (0.0, 0.0)),
    # SNIP baseline (new default) — should eliminate opening's downward bias on slopes
    ("q.02 f3 snip", dict(q_step=0.02, n_fourier=3, texture_ridge=0.3,
                          baseline_method="snip"), 1.0, (0.0, 0.0)),
    # Opening baseline (old default) — comparison target to verify improvement
    ("q.02 f3 open", dict(q_step=0.02, n_fourier=3, texture_ridge=0.3,
                          baseline_method="opening"), 1.0, (0.0, 0.0)),
    ("q.02 f6 r.1 snip", dict(q_step=0.02, n_fourier=6, texture_ridge=0.1,
                               baseline_method="snip"), 1.0, (0.0, 0.0)),
    ("q.02 smooth10 snip", dict(q_step=0.02, texture_model="smooth",
                                texture_smoothness=10.0,
                                baseline_method="snip"), 1.0, (0.0, 0.0)),
    ("q.02 smooth30 snip", dict(q_step=0.02, texture_model="smooth",
                                texture_smoothness=30.0,
                                baseline_method="snip"), 1.0, (0.0, 0.0)),
]

residuals = []
for label, params, scale, offset in variants:
    if params is None:
        residuals.append((label, src.data))
        continue
    params = params.copy()
    texture_model = params.pop("texture_model", "fourier")
    baseline_method = params.pop("baseline_method", "snip")
    model = PatchedRadialRingModel(
        n_patches=36,
        plane="0kl",
        ring_width=0.24,
        baseline_method=baseline_method,
        baseline_smooth=0.06,
        profile_percentiles=(10.0, 80.0),
        profile_method="trimmed_mean",
        texture_model=texture_model,
        texture_symmetric=False,
        center_offset=offset,
        **params,
    )
    profiles = model.fit(src, q_range=(1.5, 10.5))
    _, I_ring = model.subtract(src, profiles)
    residuals.append((label, src.data - scale * I_ring))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, (label, arr) in zip(axes.ravel(), residuals):
    vol = dataclasses.replace(src, data=np.where(keep, arr, np.nan))
    sl = extract_slice(vol, plane="0kl", value=0.0)
    ext = [sl.x_axis[0], sl.x_axis[-1], sl.y_axis[0], sl.y_axis[-1]]
    plot_data = np.ma.masked_invalid(sl.data)
    if LOG_SCALE:
        plot_data = np.ma.masked_less_equal(plot_data, 0.0)
        norm = LogNorm(vmin=max(VMIN, 1e-12), vmax=VMAX)
        vmin = vmax = None
    else:
        norm = None
        vmin = VMIN
        vmax = VMAX
    im = ax.imshow(
        plot_data,
        origin="lower",
        extent=ext,
        aspect="auto",
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        norm=norm,
    )
    ax.set_title(label)
    ax.set_xlabel(sl.x_label)
    ax.set_ylabel(sl.y_label)
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
tag = f"h{float(d.h_axis[0]):+.4f}".replace("+", "p").replace("-", "m").replace(".", "p")
out = f"examples/_normal_slice_cmp_{tag}.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
