"""Interactive cleanup viewer for principal HKL slices.

Fast 2D development harness.  When the cleaned volumes are not supplied via
RING_FILE / PUNCH_FILE / BACKFILL_FILE, it computes them here: every H plane is
ring-removed with Step 2 (``PatchedRadialRingModel``, plane='0kl', a
non-parametric per-patch radial background subtraction), then Bragg/satellite
peaks are punched and the holes backfilled.  The viewer shows the four
processing states

    (1) data, (2) ring removed, (3) punched, (4) backfilled

and carries an H/K/L selector for browsing 0kl, h0l, or hk0 slices of the same
volumes.

Note on axes — the H/K/L selector only changes how the volumes are *displayed*.
The in-script ring-removal compute path always works along H (0kl planes); to
ring-remove along K or L, process the volume with ``remove_rings_3d.py
SLICE_AXIS=K`` (or ``L``) and load the result via RING_FILE.

Run with an interactive backend, e.g.::

    PYTHONPATH=src python examples/explore_slice.py
    # or, to keep an IPython prompt afterwards:
    PYTHONPATH=src ipython --matplotlib=macosx -i examples/explore_slice.py

Env: VIEW_AXIS=H|K|L sets the initial slice orientation (default H); the cut
value comes from {H,K,L}_VALUE (default 0.3333 for H, else 0.0).
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
# overrides.  VIEW_AXIS and H_VALUE/K_VALUE/L_VALUE pick the initial viewer cut.
USE_BACKGROUND = os.environ.get("USE_BACKGROUND", "0") not in {"0", "false", "False", ""}
data_file = os.environ.get("DATA_FILE")
if data_file:
    data = ndiff.load(Path(data_file))
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
    data = ndiff.load(next(
        (p for p in cands if "22K_mmm" in p.stem and "cc_sub_bkg" in p.stem),
        next((p for p in cands if "22K_mmm" in p.stem), cands[0]),
    ))

VIEW_PLANES = {"H": "0kl", "K": "h0l", "L": "hk0"}
VIEW_AXIS = os.environ.get("VIEW_AXIS", "H").strip().upper()
if VIEW_AXIS not in VIEW_PLANES:
    raise ValueError("VIEW_AXIS must be one of H, K, or L")
VIEW_VALUE = float(os.environ.get(f"{VIEW_AXIS}_VALUE",
                                  "0.3333" if VIEW_AXIS == "H" else "0.0"))

# Validate the ring model in isolation: skip the empty/background subtraction
# (it over-subtracts and imprints the background detector gap).  Fit and remove
# rings straight from the data:  residual = data - rings.
bkg = None
if USE_BACKGROUND:
    bkg_cands = [
        p for p in sorted(raw.glob("*.nxs"))
        if p.stem.endswith("_bkg") and not p.stem.endswith(("_sub_bkg", "_cc_sub_bkg"))
    ]
    if not bkg_cands:
        raise FileNotFoundError(
            "USE_BACKGROUND=1 but no empty/background *_bkg.nxs file was found in data/raw."
        )
    bkg = ndiff.load_mantid_nxs(bkg_cands[0], ub_matrix=data.ub_matrix)

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

# Radial bin width.  Finer bins can reduce residual ring arcs but may absorb
# broad diffuse signal, so keep 0.02 as the stable default unless A/B testing.
q_step = float(os.environ.get("Q_STEP", "0.02"))

# Per-(patch, |Q|) central estimator.  Median is the current default because it
# is robust to small Bragg contamination in each azimuthal/radial cell.
profile_method = os.environ.get("PROFILE_METHOD", "median")

# Optional pooling of azimuthal texture across the ring's radial width.  Zero
# keeps each |Q| bin independent; small nonzero values can reduce ringing in
# sparse-azimuth regions.
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
        # Lower floor + prominence to catch the small Bragg the old 1.0/1.0 missed
        # (validated in 3D: captures ~90% of sharp interior Bragg while the
        # H=0.333/0.667 magnetic diffuse is preserved — see SEARCH_MIN_I note).
        "SEARCH_MIN_I": "0.6",
        "SEARCH_PROM": "0.8",
        "MARGIN": "0.02",
        "MAX_SCALE": "2.0",
        "PHI_TAIL_HKL": "0.12",
        "INCIDENT_R_HKL": "0.24,0.24,0.90",
        "INCIDENT_MARGIN": "0.12",
        "INCIDENT_ELLIPSOID_R_HKL": "0.15,0.50,1.00",
        "INCIDENT_SPHERE_R_HKL": "",
    },
    "cc_on": {
        "MODE": "auto",
        "R_HKL": "0.09,0.12,0.45",
        "SEARCH_NMAD": "4.0",
        # Was 1.5/1.0 — too conservative, left ~25% of small Bragg unpunched.
        # 0.8/0.8 captures ~89% of sharp interior Bragg at H=0 with only ~0.7%
        # collateral on the H=0.333 diffuse (which has no real sharp peaks).
        "SEARCH_MIN_I": "0.8",
        "SEARCH_PROM": "0.8",
        "MARGIN": "0.02",
        "MAX_SCALE": "2.0",
        "PHI_TAIL_HKL": "0.12",
        "INCIDENT_R_HKL": "0.24,0.24,0.90",
        "INCIDENT_MARGIN": "0.12",
        "INCIDENT_ELLIPSOID_R_HKL": "0.15,0.50,1.00",
        "INCIDENT_SPHERE_R_HKL": "",
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


def punch_bool(name: str, default: str = "0") -> bool:
    return punch_default(name, default).strip().lower() in {"1", "true", "yes", "on"}


mode = punch_default("MODE", "auto")
r_hkl = tuple(float(x) for x in punch_default("R_HKL", "0.09,0.12,0.45").split(","))
min_i_env = punch_default("MIN_I", "")
min_i = None if min_i_env == "" else float(min_i_env)
min_prom = float(punch_default("MIN_PROM", "1.0"))
integer_nmad_env = punch_default("INTEGER_NMAD", "")
integer_nmad = None if integer_nmad_env == "" else float(integer_nmad_env)
integer_q_step_env = punch_default("INTEGER_Q_STEP", "")
integer_q_step = None if integer_q_step_env == "" else float(integer_q_step_env)
integer_fit_position = punch_bool("INTEGER_FIT_POSITION")
integer_fit_shape = punch_bool("INTEGER_FIT_SHAPE")
integer_fit_frac = float(punch_default("INTEGER_FIT_FRAC", "0.35"))
integer_fit_nsigma = float(punch_default("INTEGER_FIT_NSIGMA", "2.5"))
integer_fit_max_env = punch_default("INTEGER_FIT_MAX_R_HKL", "")
integer_fit_max = (
    tuple(float(x) for x in integer_fit_max_env.split(","))
    if integer_fit_max_env else None
)
integer_h_guard_env = punch_default("INTEGER_H_GUARD", "")
integer_h_guard = None if integer_h_guard_env == "" else float(integer_h_guard_env)
search_nmad = float(punch_default("SEARCH_NMAD", "4.0"))
search_min_i = float(punch_default("SEARCH_MIN_I", "1.5"))
search_prom = float(punch_default("SEARCH_PROM", "1.0"))
search_exclude_env = punch_default("SEARCH_EXCLUDE_H", "")
search_exclude_h = (
    tuple(float(x) for x in search_exclude_env.split(",") if x.strip())
    if search_exclude_env else None
)
search_exclude_h_width = float(punch_default("SEARCH_EXCLUDE_H_WIDTH", "0.0"))
margin = float(punch_default("MARGIN", "0.02"))
max_scale = float(punch_default("MAX_SCALE", "2.0"))
phi_tail_hkl = float(punch_default("PHI_TAIL_HKL", "0.12"))
incident_r_hkl = tuple(
    float(x) for x in punch_default("INCIDENT_R_HKL", "0.24,0.24,0.90").split(",")
)
incident_margin = float(punch_default("INCIDENT_MARGIN", "0.12"))
incident_phi_tail = float(punch_default("INCIDENT_PHI_TAIL_HKL", "0.0"))
incident_ellipsoid_env = punch_default("INCIDENT_ELLIPSOID_R_HKL", "")
incident_ellipsoid_radii = (
    tuple(float(x) for x in incident_ellipsoid_env.split(","))
    if incident_ellipsoid_env else None
)
incident_sphere_env = punch_default("INCIDENT_SPHERE_R_HKL", "")
incident_sphere_radius = (
    None if incident_sphere_env == "" else float(incident_sphere_env)
)

remover = BraggRemover(
    mode=mode,
    punch_radii=r_hkl,
    min_intensity=min_i,
    min_prominence=min_prom,
    integer_n_mad=integer_nmad,
    integer_q_step=integer_q_step,
    integer_optimize_position=integer_fit_position,
    integer_optimize_shape=integer_fit_shape,
    integer_fit_threshold_frac=integer_fit_frac,
    integer_fit_radius_n_sigma=integer_fit_nsigma,
    integer_fit_max_radius_hkl=integer_fit_max,
    integer_h_guard_hkl=integer_h_guard,
    intensity_scale=True,
    max_radius_scale=max_scale,
    margin=margin,
    punch_incident_beam=True,
    incident_beam_radii=incident_r_hkl,
    incident_beam_margin=incident_margin,
    incident_beam_phi_tail_hkl=incident_phi_tail,
    incident_beam_ellipsoid_radii_hkl=incident_ellipsoid_radii,
    incident_beam_sphere_radius_hkl=incident_sphere_radius,
    phi_tail_hkl=phi_tail_hkl,
    search_n_mad=search_nmad,
    search_min_intensity=search_min_i,
    search_min_prominence=search_prom,
    search_exclude_h_centers=search_exclude_h,
    search_exclude_h_half_width=search_exclude_h_width,
)

# ------------------------------------------------------------------
# Step 1: ring removal — compute or load from RING_FILE
# ------------------------------------------------------------------
ring_file = os.environ.get("RING_FILE")
if ring_file:
    print(f"loading ring-removed volume from {ring_file} ...", flush=True)
    residual = ndiff.load(Path(ring_file))
else:
    # Compute ring removal for every H plane so the viewer can scrub H.
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

# ------------------------------------------------------------------
# Step 2: Bragg punch — compute or load from PUNCH_FILE
# ------------------------------------------------------------------
punch_file = os.environ.get("PUNCH_FILE")
if punch_file:
    print(f"loading punched volume from {punch_file} ...", flush=True)
    punched = ndiff.load(Path(punch_file))
    punched_voxels = residual.mask & ~punched.mask
else:
    peaks = remover.detect_peaks(residual)
    keep = remover.build_mask(residual)
    punched_mask = residual.mask & keep
    punched_voxels = residual.mask & ~keep
    punched = dataclasses.replace(residual, mask=punched_mask)

# ------------------------------------------------------------------
# Step 3: backfill — compute or load from BACKFILL_FILE
# ------------------------------------------------------------------
backfill_file = os.environ.get("BACKFILL_FILE")
if backfill_file:
    print(f"loading backfilled volume from {backfill_file} ...", flush=True)
    backfilled = ndiff.load(Path(backfill_file))
else:
    punch_only = dataclasses.replace(residual, mask=~punched_voxels)
    backfill_method = os.environ.get("BACKFILL_METHOD", "local")
    backfilled = backfill_bragg(
        punch_only,
        method=backfill_method,
        tv_lam=float(os.environ.get("TV_LAM", "0.2")),
        tv_iter=int(os.environ.get("TV_ITER", "300")),
        local_radius=int(os.environ.get("LOCAL_RADIUS", "2")),
        local_min_count=int(os.environ.get("LOCAL_MIN_COUNT", "8")),
        q_shell_step=float(os.environ.get("Q_SHELL_STEP", "0.05")),
        q_shell_min_count=int(os.environ.get("Q_SHELL_MIN_COUNT", "20")),
    )

# Keep the original mask for display, but reveal the small unmeasured direct-beam
# shadow at the very origin (|Q|≈0): those voxels were deliberately filled with the
# just-outside diffuse background, so they should show rather than being re-hidden
# by residual.mask.  The punched-hole part of the beam was already valid pre-punch
# (residual.mask True), so it shows without a reveal.  Keep this radius small — a
# large |Q| ball would un-mask the origin column of *other* H planes (e.g. the
# H=0.333 diffuse), since |Q| is isotropic but H steps are coarse.
direct_beam_show_q = float(os.environ.get("DIRECT_BEAM_SHOW_Q", "0.15"))
beam_ball = residual.q_magnitude() <= direct_beam_show_q
backfilled = dataclasses.replace(backfilled, mask=backfilled.mask | beam_ball)

print(f"HKL volume viewer initial {VIEW_AXIS}={VIEW_VALUE:.4g}  "
      f"background={'on' if USE_BACKGROUND else 'OFF'}  "
      f"ring={'loaded' if ring_file else 'computed'}  "
      f"punch={'loaded' if punch_file else 'computed'}  "
      f"backfill={'loaded' if backfill_file else 'computed'}")
valid = residual.mask & np.isfinite(residual.data)
if not punch_file:
    print(f"Bragg punch: preset={punch_preset_name} mode={mode} radii={r_hkl} "
          f"phi_tail={phi_tail_hkl} peaks={len(peaks)}")
print(f"Total punched: {int(punched_voxels.sum())} voxels "
      f"({100 * punched_voxels.sum() / max(int(valid.sum()), 1):.2f}% of valid)")
if not backfill_file:
    print(f"Backfill: method={os.environ.get('BACKFILL_METHOD', 'local')}")
print("Choose H/K/L to switch slice orientation; drag the HKL plane slider, "
      "drag the vmin/vmax sliders, or toggle linear/log₁₀ (bottom-left). "
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
    plane=VIEW_PLANES[VIEW_AXIS], value=VIEW_VALUE, cmap="viridis",
    vmin=0.0, vmax=0.3, slider_min=slider_min, slider_max=slider_max,
    value_slider=True, plane_selector=True,
)
