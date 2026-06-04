"""Interactive cleanup viewer for 0kl slices across H.

Fast 2D development harness: extracts the H=0 plane and runs Step 2
(``PatchedRadialRingModel``, plane='0kl') — non-parametric per-patch radial
background subtraction — then punches Bragg/satellite peaks and backfills the
holes.  The viewer shows the four processing states:

    (1) data, (2) ring removed, (3) punched, (4) backfilled.

Run with an interactive backend, e.g.::

    PYTHONPATH=src python examples/explore_slice.py
    # or, to keep an IPython prompt afterwards:
    PYTHONPATH=src ipython --matplotlib=macosx -i examples/explore_slice.py
"""
import matplotlib
matplotlib.use("macosx")          # interactive backend (use "qt" on Linux/Win)

import dataclasses
import os
from pathlib import Path

import numpy as np

import ndiff
from ndiff.analysis import BraggRemover, backfill_bragg
from ndiff.preprocessing import (
    EmptySubtractor,
    PatchedRadialRingModel,
    azimuthal_sampling_mask,
    fit_ring_profiles,
    line_profile,
)
from ndiff.visualization import interactive_slices

raw = Path("data/raw")
# Default to the preferred 22K mmm validation file; the alphabetically-first
# .nxs is the older 28K file, so select the 22K one explicitly unless DATA_FILE
# overrides.  H_VALUE picks the initial H plane in the viewer.
data_file = os.environ.get("DATA_FILE")
if data_file:
    data = ndiff.load(Path(data_file))
else:
    cands = [p for p in sorted(raw.glob("*.nxs"))
             if not p.stem.endswith(("_bkg", "_sub_bkg"))]
    data = ndiff.load(next((p for p in cands if "22K_mmm" in p.stem), cands[0]))
H_VALUE = float(os.environ.get("H_VALUE", "0.3333"))
bkg = ndiff.load_mantid_nxs(
    [p for p in sorted(raw.glob("*.nxs")) if p.stem.endswith("_bkg")][0],
    ub_matrix=data.ub_matrix,
)

# Validate the ring model in isolation: skip the empty/background subtraction
# (it over-subtracts and imprints the background detector gap).  Fit and remove
# rings straight from the data:  residual = data - rings.
USE_BACKGROUND = False

q_range = (float(os.environ.get("Q_MIN", "1.5")),
           float(os.environ.get("Q_MAX", "10.5")))


def _hslice(v, ih):
    s = slice(ih, ih + 1)
    return dataclasses.replace(
        v, data=v.data[s], sigma=v.sigma[s], mask=v.mask[s], h_axis=v.h_axis[s],
    )

# Non-parametric per-patch radial background subtraction (the current reference
# config — all knobs at their class defaults: SNIP baseline, n_fourier=8,
# profile_method='median', texture_q_smooth=0.0 = per-|Q| azimuthal texture).
#   ring_width        : max ring full-width in |Q| removed as a peak (Å^-1)
#   q_step            : radial bin width (finer than the ring to resolve peaks)
#   baseline_method   : 'snip' = slope-aware peak-clipping baseline
#   profile_percentiles: trim band per |Q| bin (low=gaps, high=Bragg)
#   texture_model     : 'fourier' = T(φ) (anisotropy, Bragg-immune)
#   n_fourier         : azimuthal harmonics (8 — resolves multi-lobed texture)
#   texture_q_smooth  : pool the texture shape across the ring's |Q| width so
#                       high harmonics don't ring on sparse azimuths
#   ring_templates    : fixed Gaussian radial shapes from clean linecuts.  The
#                       clean cuts here run along (0, ±1, l) from l=0 to ±30,
#                       which avoids Bragg peaks and resolves close rings.
USE_LINECUT_TEMPLATES = False


def _linecut_templates():
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
    prof = np.nanmean(np.vstack(cuts), axis=0)
    return fit_ring_profiles(
        q_ref, prof, prominence=0.04, min_distance=8,
        cluster_gap=0.35, half_window=0.24, sigma0=0.04,
    )


templates = _linecut_templates() if USE_LINECUT_TEMPLATES else None
if templates:
    print("linecut ring templates from (0, ±1, ±30):")
    for r in templates:
        print(f"  q={r.q_center:6.3f} Å^-1  sigma={r.sigma:6.4f}  "
              f"FWHM={r.fwhm:6.4f}  amp={r.amplitude:6.3f}")

center_offset = (
    float(os.environ.get("CENTER_OFFSET_X", "0.0")),
    float(os.environ.get("CENTER_OFFSET_Y", "0.0")),
)
center_offset_h_slope = (
    float(os.environ.get("CENTER_H_SLOPE_X", "0.0")),
    float(os.environ.get("CENTER_H_SLOPE_Y", "0.0")),
)

# q_step (radial bin width): the dominant lever on the residual ring leftover.
# The ring estimate is a *robust* (Bragg-trimmed) profile, which sits slightly
# below the true peak — so a positive ring residual is left by construction, and
# it shrinks as the bins resolve the peak more finely.  q_step=0.015 cuts the
# 28K leftover ~15% with NO over-subtraction and the close-pair valley preserved;
# but on a slice with rich diffuse, too-fine bins can eat broad diffuse, so A/B
# this on YOUR validation slice (the 22K H=0.3333) before adopting it.  Finer
# than ~0.01 starts cutting troughs.  Override with Q_STEP.
q_step = float(os.environ.get("Q_STEP", "0.02"))

# profile_method: the per-(patch,|Q|) central estimator.  The default
# 'trimmed_mean' (10–80) is ASYMMETRIC — it trims 20% off the top (to reject
# Bragg) but only 10% off the bottom, so on a right-skewed cell it sits BELOW the
# true ring level and under-subtracts the bright ring arcs (~12% of the arc
# residual on 28K).  'median' is the symmetric unbiased robust centre — Bragg is
# a small fraction of each cell so it can't move the median — and cuts the arc
# under-fill ~10–12% with no extra over-subtraction.  A/B on YOUR 22K H=0.3333
# slice before adopting.  Override with PROFILE_METHOD.
profile_method = os.environ.get("PROFILE_METHOD", "median")

# texture_q_smooth: pools the azimuthal texture SHAPE across the ring's radial
# width.  It assumes the ring's azimuthal pattern is the same at the peak and the
# wings — but that only holds if the ring's WIDTH is azimuthally uniform.  When
# the width varies with φ (strong at H≠0: the powder ring is broad at some
# azimuths, narrow at others), pooling across |Q| forces one shared pattern and
# HOMOGENISES the width → under-subtracts the broad arcs and over-subtracts the
# narrow ones.  Setting it to 0 lets each |Q| bin keep its own azimuthal pattern
# (the low-order Fourier basis still smooths in φ), capturing the inhomogeneous
# width: on 28K H=0.32 this cuts both under-fill (−26%) and over-fill (−33%) with
# no diffuse cost.  The trade-off it was added for is ringing into UNMEASURED
# azimuths (one-sided coverage) — if your slice has sparse arcs, a small value
# (~0.02) compromises.  Override with TEXTURE_Q_SMOOTH.
texture_q_smooth = float(os.environ.get("TEXTURE_Q_SMOOTH", "0.0"))

# n_patches: number of overlapping azimuthal wedges the ring profile is fit in.
# More patches = finer azimuthal resolution of the texture, but each wedge holds
# fewer voxels so the per-patch robust profile gets noisier (and sparse arcs
# empty out sooner).  Override with N_PATCHES.
n_patches = int(os.environ.get("N_PATCHES", "36"))

prm = PatchedRadialRingModel(
    n_patches=n_patches, plane="0kl", q_step=q_step, ring_width=0.24,
    baseline_method="snip", baseline_smooth=0.06,
    profile_percentiles=(10.0, 80.0), profile_method=profile_method,
    texture_model="fourier", texture_symmetric=False, texture_q_smooth=texture_q_smooth,
    ring_templates=templates,
    center_offset=center_offset, center_offset_h_slope=center_offset_h_slope,
)
print(f"ring model: texture={prm.texture_model} n_fourier={prm.n_fourier} "
      f"symmetric={prm.texture_symmetric} q_step={q_step} profile={profile_method} "
      f"q_smooth={texture_q_smooth} n_patches={n_patches}")
print(f"center offset={center_offset}  H slope={center_offset_h_slope} Å^-1/H")

# Bragg/satellite punch on the ring-removed volume.  The defaults mirror the
# cleaner cc-on 3D punch preset; override with the same env vars used by
# punch_bragg_3d.py when A/B testing thresholds.
PUNCH_PRESETS = {
    "cc_off": {
        "MODE": "auto",
        "R_HKL": "0.09,0.12,0.45",
        "SEARCH_NMAD": "4.0",
        "SEARCH_MIN_I": "1.0",
        "SEARCH_PROM": "1.0",
        "MARGIN": "0.02",
        "MAX_SCALE": "2.0",
        "PHI_TAIL_HKL": "0.12",
        "INCIDENT_R_HKL": "0.24,0.24,0.90",
        "INCIDENT_MARGIN": "0.12",
    },
    "cc_on": {
        "MODE": "auto",
        "R_HKL": "0.09,0.12,0.45",
        "SEARCH_NMAD": "4.0",
        "SEARCH_MIN_I": "1.5",
        "SEARCH_PROM": "1.0",
        "MARGIN": "0.02",
        "MAX_SCALE": "2.0",
        "PHI_TAIL_HKL": "0.12",
        "INCIDENT_R_HKL": "0.24,0.24,0.90",
        "INCIDENT_MARGIN": "0.12",
    },
}

punch_preset_name = os.environ.get("PUNCH_PRESET", "cc_on")
if punch_preset_name not in PUNCH_PRESETS:
    raise ValueError(
        f"Unknown PUNCH_PRESET={punch_preset_name!r}; choose one of "
        f"{sorted(PUNCH_PRESETS)}"
    )
punch_preset = PUNCH_PRESETS[punch_preset_name]


def punch_default(name: str, default: str) -> str:
    return os.environ.get(name, punch_preset.get(name, default))


mode = punch_default("MODE", "auto")
r_hkl = tuple(float(x) for x in punch_default("R_HKL", "0.09,0.12,0.45").split(","))
min_i_env = punch_default("MIN_I", "")
min_i = None if min_i_env == "" else float(min_i_env)
min_prom = float(punch_default("MIN_PROM", "1.0"))
search_nmad = float(punch_default("SEARCH_NMAD", "4.0"))
search_min_i = float(punch_default("SEARCH_MIN_I", "1.5"))
search_prom = float(punch_default("SEARCH_PROM", "1.0"))
margin = float(punch_default("MARGIN", "0.02"))
max_scale = float(punch_default("MAX_SCALE", "2.0"))
phi_tail_hkl = float(punch_default("PHI_TAIL_HKL", "0.12"))
incident_r_hkl = tuple(
    float(x) for x in punch_default("INCIDENT_R_HKL", "0.24,0.24,0.90").split(",")
)
incident_margin = float(punch_default("INCIDENT_MARGIN", "0.12"))
incident_phi_tail = float(punch_default("INCIDENT_PHI_TAIL_HKL", "0.0"))

remover = BraggRemover(
    mode=mode,
    punch_radii=r_hkl,
    min_intensity=min_i,
    min_prominence=min_prom,
    intensity_scale=True,
    max_radius_scale=max_scale,
    margin=margin,
    punch_incident_beam=True,
    incident_beam_radii=incident_r_hkl,
    incident_beam_margin=incident_margin,
    incident_beam_phi_tail_hkl=incident_phi_tail,
    phi_tail_hkl=phi_tail_hkl,
    search_n_mad=search_nmad,
    search_min_intensity=search_min_i,
    search_min_prominence=search_prom,
)

# Compute ring removal for every H plane so the viewer can scrub H.  This keeps
# the slice-validated fit independent on each plane, matching remove_rings_3d.py.
MASK_SPARSE = True
residual_data = np.empty_like(data.data)
residual_mask = data.mask.copy()
sparse_dropped = 0
n_skipped = 0

print(f"processing {data.shape[0]} H planes for the slider...", flush=True)
for ih in range(data.shape[0]):
    d = _hslice(data, ih)
    if USE_BACKGROUND:
        sub = EmptySubtractor(_hslice(bkg, ih), scale_q_range=(2.4, 3.3))
        src = sub.subtract(d)
    else:
        src = d

    if MASK_SPARSE:
        keep_sparse = azimuthal_sampling_mask(
            src, plane="0kl", min_count_frac=0.25, q_range=q_range
        )
        sparse_dropped += int((src.mask & ~keep_sparse).sum())
        src = dataclasses.replace(src, mask=keep_sparse)

    valid = src.mask & np.isfinite(src.data)
    if int(valid.sum()) < prm.min_voxels_per_patch:
        residual_data[ih] = src.data[0]
        residual_mask[ih] = src.mask[0]
        n_skipped += 1
        continue

    try:
        prof = prm.fit(src, q_range=q_range)
        _, I_ring = prm.subtract(src, prof)
    except Exception as exc:
        print(f"  H[{ih}]={data.h_axis[ih]:+.3f}: fit failed ({exc}); left as-is",
              flush=True)
        residual_data[ih] = src.data[0]
        residual_mask[ih] = src.mask[0]
        n_skipped += 1
        continue

    residual_data[ih] = src.data[0] - I_ring[0]
    residual_mask[ih] = src.mask[0]
    if ih % 30 == 0 or ih == data.shape[0] - 1:
        print(f"  H[{ih:3d}]={data.h_axis[ih]:+.3f}", flush=True)

residual = dataclasses.replace(data, data=residual_data, mask=residual_mask)
print(f"sparse-azimuth mask: dropped {sparse_dropped} voxels across H")
if n_skipped:
    print(f"ring removal: {n_skipped} sparse/failed H planes left unchanged")

peaks = remover.detect_peaks(residual)
keep = remover.build_mask(residual)
punched_mask = residual.mask & keep
punched_voxels = residual.mask & ~keep
punched = dataclasses.replace(residual, mask=punched_mask)
punch_only = dataclasses.replace(residual, mask=~punched_voxels)

backfill_method = os.environ.get("BACKFILL_METHOD", "local")
backfilled = backfill_bragg(
    punch_only,
    method=backfill_method,
    tv_lam=float(os.environ.get("TV_LAM", "0.2")),
    tv_iter=int(os.environ.get("TV_ITER", "300")),
    local_radius=int(os.environ.get("LOCAL_RADIUS", "2")),
    local_min_count=int(os.environ.get("LOCAL_MIN_COUNT", "8")),
)
backfilled = dataclasses.replace(backfilled, mask=residual.mask)

print(f"0kl volume viewer initial H={H_VALUE:.4g}  "
      f"background={'on' if USE_BACKGROUND else 'OFF'}  "
      f"non-parametric radial-background subtraction")
valid = residual.mask & np.isfinite(residual.data)
print(f"Bragg punch: preset={punch_preset_name} mode={mode} radii={r_hkl} "
      f"phi_tail={phi_tail_hkl} peaks={len(peaks)}")
print(f"Incident beam punch: radii={incident_r_hkl} margin={incident_margin} "
      f"phi_tail={incident_phi_tail}")
print(f"Total punched: {int(punched_voxels.sum())} voxels "
      f"({100 * punched_voxels.sum() / max(int(valid.sum()), 1):.2f}% of valid)")
print(f"Backfill: method={backfill_method}")
print("Drag the H slider to scrub planes; drag the vmin/vmax sliders; "
      "toggle linear/log₁₀ (bottom-left). "
      "Close the window to exit.")

# Tight slider travel so the pullbar gives fine control near the diffuse level
# (~0–0.3) instead of spanning the full bright-ring data range.  Override with
# SLIDER_MIN / SLIDER_MAX env vars.
slider_min = float(os.environ.get("SLIDER_MIN", "0.0"))
slider_max = float(os.environ.get("SLIDER_MAX", "1.0"))
interactive_slices(
    [("data", data),
     ("Removed ring", residual),
     ("Punched", punched),
     ("Backfilled", backfilled)],
    plane="0kl", value=H_VALUE, cmap="viridis", vmin=0.0, vmax=0.3,
    slider_min=slider_min, slider_max=slider_max, value_slider=True,
)
