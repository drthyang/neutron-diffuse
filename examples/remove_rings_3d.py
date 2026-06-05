"""Full 3D powder-ring removal — promote the slice-validated model to the volume.

Applies the slice-validated ``PatchedRadialRingModel`` (current class defaults:
median profile, per-|Q| azimuthal texture, SNIP baseline, adaptive ring width)
to every H-plane of the volume **independently** and stacks the residuals into a
clean 3D diffuse volume.

Why per-slice rather than one global fit:  the model fits the azimuthal texture
T(φ) in the in-plane (b*,c*) frame and bins by the full 3D |Q|.  The ring's
radial WIDTH and texture vary with H (this is exactly what the
``texture_q_smooth=0`` default captures); a single volume-global fit would pool
all H into one texture per |Q| and re-homogenise that H-dependence.  Looping the
validated 0kl-slice pipeline over H keeps each plane's own texture/width — it is
literally "carry the slice-validated config to the whole volume".

Run (headless — writes the residual volume + spot-check PNGs)::

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
      /Users/tt9/miniforge3/envs/rmc-discord/bin/python3 examples/remove_rings_3d.py

Env overrides:
    DATA_FILE   input .nxs (default: the 22K mmm validation file)
    OUT_FILE    output .h5 (default: data/processed/<stem>_ringremoved.h5)
    Q_MIN,Q_MAX radial fit range (default 1.5, 10.5 — matches the slice harness)
    RING_PRESET cc_off|cc_on. cc_off keeps the previous aggressive defaults;
                cc_on is less flexible in texture so diffuse scattering at
                H≈1/3 is less likely to enter I_ring.
    Q_STEP,N_FOURIER,N_PATCHES,PROFILE_METHOD,TEXTURE_Q_SMOOTH,TEXTURE_RIDGE
                override the selected preset/model defaults.
"""
import matplotlib
matplotlib.use("Agg")              # headless: write PNGs, no window

import dataclasses
import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import ndiff
from ndiff.preprocessing import (
    PatchedRadialRingModel,
    azimuthal_sampling_mask,
    confirm_ring_shells_across_h,
)
from ndiff.visualization import extract_slice

raw = Path("data/raw")
data_file = os.environ.get("DATA_FILE")
if data_file:
    in_path = Path(data_file)
else:
    def is_empty_background(path: Path) -> bool:
        return (
            path.stem.endswith("_bkg")
            and not path.stem.endswith(("_sub_bkg", "_cc_sub_bkg"))
        )

    cands = [p for p in sorted(raw.glob("*.nxs")) if not is_empty_background(p)]
    if not cands:
        raise FileNotFoundError(
            "No input .nxs files found in data/raw. Set DATA_FILE=/path/to/input.nxs."
        )
    in_path = next(
        (p for p in cands if "22K_mmm" in p.stem and "cc_sub_bkg" in p.stem),
        next((p for p in cands if "22K_mmm" in p.stem), cands[0]),
    )

q_range = (float(os.environ.get("Q_MIN", "1.5")),
           float(os.environ.get("Q_MAX", "10.5")))

PRESETS = {
    # Historical reference used for the cc-off volumes.  It follows strong
    # azimuthal ring texture closely, but on cc-on data it can also follow the
    # H≈1/3 diffuse structure visible in the ring panel.
    "cc_off": {
        "PROFILE_METHOD": "median",
        "N_FOURIER": "8",
        "N_PATCHES": "36",
        "Q_STEP": "0.02",
        "TEXTURE_Q_SMOOTH": "0.0",
        "TEXTURE_RIDGE": "0.05",
        "RING_AMP_CAP": "4.0",
    },
    # Cleaner cc-on / cc-sub-bkg data needs less freedom in the ring texture.
    # A lower harmonic order plus slight |Q|-pooling makes I_ring smoother and
    # reduces subtraction of structured diffuse signal while keeping true rings.
    "cc_on": {
        "PROFILE_METHOD": "median",
        "N_FOURIER": "6",
        "N_PATCHES": "36",
        "Q_STEP": "0.02",
        "TEXTURE_Q_SMOOTH": "0.02",
        "TEXTURE_RIDGE": "0.08",
        "RING_AMP_CAP": "3.0",
    },
}

preset_name = os.environ.get("RING_PRESET", "cc_off")
if preset_name not in PRESETS:
    raise ValueError(
        f"Unknown RING_PRESET={preset_name!r}; choose one of {sorted(PRESETS)}"
    )
preset = PRESETS[preset_name]


def env_default(name: str, default: str) -> str:
    return os.environ.get(name, preset.get(name, default))

out_file = os.environ.get("OUT_FILE")
if out_file:
    out_path = Path(out_file)
else:
    out_path = Path("data/processed") / f"{in_path.stem}_ringremoved.h5"
out_path.parent.mkdir(parents=True, exist_ok=True)

print(f"loading {in_path.name}", flush=True)
vol = ndiff.load(in_path)
nh, nk, nl = vol.data.shape
print(f"volume (H,K,L)=({nh},{nk},{nl})  |Q| fit range {q_range}", flush=True)

# Pre-pass: confirm the real powder-ring |Q| shells ACROSS H.  A real ring sits
# at the same 3D |Q| on every plane that samples it; a Bragg-fed phantom (which
# corrupts only integer-H planes) washes out of the across-plane median.  Passing
# the confirmed shells to the model restricts subtraction to them, killing the
# phantom-ring over-subtraction troughs AND making the subtracted shells the same
# on every plane (continuous in H — no discontinuity for the ΔPDF FFT).  Set
# CONFIRM_RINGS=0 to disable and reproduce the old per-slice behaviour.
#
# A second, complementary guard caps each shell's per-plane amplitude: where a
# Bragg peak lands ON a real ring (e.g. |Q|≈4.32 inside the 4.39 Å^-1 ring at
# H=±4) it inflates that ring's amplitude on the one plane and over-subtracts
# along the ring — which the |Q|-envelope cannot catch (the shell is real).
# The ceiling = RING_AMP_CAP × the across-H typical amplitude caps the spike back
# to the cross-plane norm; normal planes (amplitude below the ceiling) are
# untouched.  RING_AMP_CAP=0 disables the cap.
confirm = os.environ.get("CONFIRM_RINGS", "1") != "0"
amp_cap = float(env_default("RING_AMP_CAP", "4.0"))
ring_centers = ring_halfwidths = ring_ceilings = None
if confirm:
    t_pre = time.time()
    ring_centers, ring_halfwidths, ring_amps = confirm_ring_shells_across_h(
        vol, plane="0kl", q_range=q_range)
    if amp_cap > 0 and ring_amps.size:
        ring_ceilings = amp_cap * ring_amps
    print(f"cross-H confirmed {ring_centers.size} ring shells in "
          f"{time.time() - t_pre:.1f}s "
          f"(amplitude cap = {amp_cap}× across-H):", flush=True)
    for i, (c, w) in enumerate(zip(ring_centers, ring_halfwidths)):
        ceil = "" if ring_ceilings is None else f"  ceiling={ring_ceilings[i]:7.3f}"
        print(f"    |Q|={c:6.3f} Å^-1  FWHM={w:6.3f}  amp={ring_amps[i]:6.3f}{ceil}",
              flush=True)

# One model instance, reused per slice (fit() is called fresh on each plane, so
# the per-slice profiles never leak between H).  All knobs at class defaults,
# plus the cross-H confirmed shells and per-shell amplitude ceilings.
model = PatchedRadialRingModel(
    plane="0kl",
    q_step=float(env_default("Q_STEP", "0.02")),
    n_patches=int(env_default("N_PATCHES", "36")),
    n_fourier=int(env_default("N_FOURIER", "8")),
    profile_method=env_default("PROFILE_METHOD", "median"),
    texture_q_smooth=float(env_default("TEXTURE_Q_SMOOTH", "0.0")),
    texture_ridge=float(env_default("TEXTURE_RIDGE", "0.05")),
    allowed_ring_centers=ring_centers,
    allowed_ring_halfwidths=ring_halfwidths,
    allowed_ring_ceilings=ring_ceilings,
)
print(f"model: preset={preset_name} profile={model.profile_method} "
      f"n_fourier={model.n_fourier} "
      f"q_step={model.q_step} q_smooth={model.texture_q_smooth} "
      f"ridge={model.texture_ridge} baseline={model.baseline_method} "
      f"adaptive_width={model.adaptive_ring_width} "
      f"confirmed_shells={'none' if ring_centers is None else ring_centers.size} "
      f"amp_cap={amp_cap}",
      flush=True)

res_data = np.empty_like(vol.data)         # data - rings, per voxel
out_mask = vol.mask.copy()                 # propagate the sparse-azimuth drops
n_skipped = 0
ring_sum = 0.0                             # total positive ring intensity removed
neg_voxels = 0                             # over-subtraction tally (res < -0.05)


def _hslice(v, ih):
    s = slice(ih, ih + 1)
    return dataclasses.replace(
        v, data=v.data[s], sigma=v.sigma[s], mask=v.mask[s], h_axis=v.h_axis[s],
    )


t0 = time.time()
for ih in range(nh):
    sl = _hslice(vol, ih)
    valid = sl.mask & np.isfinite(sl.data)
    if int(valid.sum()) < model.min_voxels_per_patch:
        # Plane too empty to fit (extreme |H|) — leave it unchanged.
        res_data[ih] = sl.data[0]
        n_skipped += 1
        continue

    # Drop anomalously sparse (|Q|,φ) cells from the FIT so they don't bias the
    # ring estimate (same as the slice harness); the residual is still written
    # for every voxel (subtractive), and the drops are recorded in out_mask.
    keep = azimuthal_sampling_mask(sl, plane="0kl", min_count_frac=0.25,
                                   q_range=(1.5, 10.5))
    src = dataclasses.replace(sl, mask=keep)
    out_mask[ih] = keep[0]

    try:
        prof = model.fit(src, q_range=q_range)
        _, I_ring = model.subtract(src, prof)
    except Exception as exc:                       # numerical edge case on a plane
        print(f"  H[{ih}]={vol.h_axis[ih]:+.3f}: fit failed ({exc}); left as-is",
              flush=True)
        res_data[ih] = sl.data[0]
        n_skipped += 1
        continue

    I_ring2d = I_ring[0]
    res_data[ih] = sl.data[0] - I_ring2d
    ring_sum += float(np.nansum(I_ring2d[valid[0]]))
    neg_voxels += int(np.count_nonzero(res_data[ih][valid[0]] < -0.05))

    if ih % 30 == 0 or ih == nh - 1:
        dt = time.time() - t0
        eta = dt / (ih + 1) * (nh - ih - 1)
        print(f"  H[{ih:3d}]={vol.h_axis[ih]:+.3f}  "
              f"{dt:5.1f}s elapsed, ~{eta:4.0f}s left", flush=True)

dt = time.time() - t0
n_valid = int((vol.mask & np.isfinite(vol.data)).sum())
print(f"\ndone in {dt:.1f}s  ({n_skipped} planes left unchanged)", flush=True)
print(f"total ring intensity removed: {ring_sum:.4g}", flush=True)
print(f"over-subtracted voxels (residual < -0.05): {neg_voxels} "
      f"({100.0 * neg_voxels / max(n_valid, 1):.3f}% of valid)", flush=True)

out_vol = dataclasses.replace(vol, data=res_data, mask=out_mask)
print(f"\nsaving residual volume -> {out_path}", flush=True)
ndiff.save(out_vol, out_path)

# ---- spot-check PNGs: data vs residual on the validated H slices ----------
for H in (0.0, 0.3333, 0.666):
    ih = int(np.argmin(np.abs(vol.h_axis - H)))
    before = extract_slice(vol, plane="0kl", value=float(vol.h_axis[ih]))
    after = extract_slice(out_vol, plane="0kl", value=float(vol.h_axis[ih]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, sd, title in ((axes[0], before, "data"),
                          (axes[1], after, "residual = data - rings")):
        im = ax.imshow(sd.data, origin="lower", cmap="viridis",
                       vmin=0.0, vmax=0.3, aspect="equal",
                       extent=[sd.x_axis[0], sd.x_axis[-1],
                               sd.y_axis[0], sd.y_axis[-1]])
        ax.set_title(f"{title}  (H={vol.h_axis[ih]:+.3f})")
        ax.set_xlabel(sd.x_label)
        ax.set_ylabel(sd.y_label)
        fig.colorbar(im, ax=ax, shrink=0.8)
    png = Path("examples") / f"_remove_rings_3d_H{vol.h_axis[ih]:+.3f}.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"  wrote {png}", flush=True)

print("\n3D ring removal complete.", flush=True)
