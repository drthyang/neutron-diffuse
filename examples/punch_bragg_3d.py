"""Bragg punch on the ring-removed 3D volume — compute, save, and preview.

Loads the ring-removed volume (output of ``remove_rings_3d.py``), detects the
real Bragg peaks (data-driven — skips systematic absences), punches an
anisotropic, intensity-scaled ellipsoid at each, saves the punched volume, and
opens the H-slider viewer so you can scrub planes and see what was removed (grey
holes) vs what survives (bright spots — e.g. off-integer satellite reflections).

Run::

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
      python3 \
      examples/punch_bragg_3d.py

Env overrides:
    DATA_FILE    ring-removed input .h5 (default: data/processed/<22K>_ringremoved.h5)
    OUT_FILE     punched output .h5  (default: <stem>_braggpunched.h5)
    PUNCH_PRESET "cc_off" | "cc_on" | unset (cc = CORELLI correlation chopper).
                 "cc_off" saves the earlier aggressive weak-peak setup; "cc_on"
                 is less aggressive for the cleaner correlation-chopper
                 (*_cc_sub_bkg) data so diffuse scattering is not over-masked.
    MODE         "integer" | "auto" | "search" | "both" (default both).
                 "auto"/"search" detects sharp high-tail outliers above the
                 robust per-|Q| diffuse level, analogous to the ring-removal
                 robust profile logic; catches off-integer satellites /
                 small-domain Bragg.
    R_HKL        per-axis base punch radii "rh,rk,rl"  (default 0.12,0.12,0.45)
    MIN_I        integer-mode detection intensity threshold (default 2.0)
    INTEGER_NMAD optional integer-node per-|Q|-shell threshold in MADs
    INTEGER_Q_STEP
                 |Q| shell width for INTEGER_NMAD (default: SEARCH_Q_STEP)
    INTEGER_FIT_POSITION
                 1 to refine integer-node peaks to a weighted HKL centre
    INTEGER_FIT_SHAPE
                 1 to fit per-peak anisotropic radii from local moments
    INTEGER_FIT_FRAC
                 fraction of peak excess used by integer fit (default 0.35)
    INTEGER_FIT_NSIGMA
                 convert fitted widths to radii by this factor (default 2.5)
    INTEGER_FIT_MAX_R_HKL
                 optional max fitted radii "rh,rk,rl"
    INTEGER_H_GUARD
                 optional H half-width around source integer-H plane
    SEARCH_NMAD  search-mode outlier threshold in MADs (default 6.0)
    SEARCH_MIN_I search-mode absolute intensity floor (default 2.0)
    SEARCH_PROM  search-mode local 3x3x3 prominence floor (default 0.0)
    SEARCH_EXCLUDE_H
                 comma-separated H planes excluded from search mode
    SEARCH_EXCLUDE_H_WIDTH
                 H half-width around SEARCH_EXCLUDE_H planes
    MARGIN       guard band added to every radius (default 0.03)
    MAX_SCALE    max intensity radius multiplier (default 3.0)
    PHI_TAIL_HKL extra Bragg-punch width along the local powder-ring direction
    INCIDENT_R_HKL
                 incident-beam punch radii "rh,rk,rl" (default 0.24,0.24,0.90)
    INCIDENT_MARGIN
                 guard band for the incident-beam punch (default 0.12)
    INCIDENT_PHI_TAIL_HKL
                 optional incident-beam tail width along the local ring direction
    INCIDENT_SPHERE_R_HKL
                 if set, use this isotropic HKL sphere radius for the incident beam
    PREVIEW      1 (default) opens the viewer; 0 just saves
    H_VALUE      initial H plane in the viewer (default 0.0)
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("macosx" if os.environ.get("PREVIEW", "1") != "0" else "Agg")

import dataclasses
import time

import numpy as np

import ndiff
from ndiff.analysis import BraggRemover

proc = Path("data/processed")
PRESETS = {
    # Earlier setup tuned on the cc-off style volume.  Good weak-peak capture,
    # but too aggressive for cleaner cc-on data: it can mask diffuse features at
    # H≈0.333/0.666.
    "cc_off": {
        "MODE": "auto",
        "R_HKL": "0.09,0.12,0.45",
        "SEARCH_NMAD": "4.0",
        # Lowered floor/prominence to capture small Bragg (validated in 3D to
        # preserve the H=0.333/0.667 magnetic diffuse).
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
    # Cleaner cc-on data has better-shaped Bragg peaks.  Floor 0.8 / prominence
    # 0.8 captures ~89% of sharp interior Bragg at H=0 while leaving the diffuse
    # planes essentially untouched (~0.7% collateral, no real peaks there).
    "cc_on": {
        "MODE": "auto",
        "R_HKL": "0.09,0.12,0.45",
        "SEARCH_NMAD": "4.0",
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

preset_name = os.environ.get("PUNCH_PRESET", "")
if preset_name and preset_name not in PRESETS:
    raise ValueError(f"Unknown PUNCH_PRESET={preset_name!r}; choose one of {sorted(PRESETS)}")
preset = PRESETS.get(preset_name, {})


def env_default(name: str, default: str) -> str:
    return os.environ.get(name, preset.get(name, default))


def env_bool(name: str, default: str = "0") -> bool:
    return env_default(name, default).strip().lower() in {"1", "true", "yes", "on"}


data_file = os.environ.get("DATA_FILE")
if data_file:
    in_path = Path(data_file)
else:
    cands = sorted(proc.glob("*_ringremoved.h5"))
    if not cands:
        raise FileNotFoundError(
            "No ring-removed input found in data/processed. Run "
            "`PYTHONPATH=src python3 "
            "examples/remove_rings_3d.py` first, "
            "or set DATA_FILE=/path/to/*_ringremoved.h5."
        )
    in_path = next((p for p in cands if "22K_mmm" in p.stem), cands[0])

mode = env_default("MODE", "both")
r_hkl = tuple(float(x) for x in env_default("R_HKL", "0.12,0.12,0.45").split(","))
min_i = float(env_default("MIN_I", "2.0"))
integer_nmad_env = env_default("INTEGER_NMAD", "")
integer_nmad = None if integer_nmad_env == "" else float(integer_nmad_env)
integer_q_step_env = env_default("INTEGER_Q_STEP", "")
integer_q_step = None if integer_q_step_env == "" else float(integer_q_step_env)
integer_fit_position = env_bool("INTEGER_FIT_POSITION")
integer_fit_shape = env_bool("INTEGER_FIT_SHAPE")
integer_fit_frac = float(env_default("INTEGER_FIT_FRAC", "0.35"))
integer_fit_nsigma = float(env_default("INTEGER_FIT_NSIGMA", "2.5"))
integer_fit_max_env = env_default("INTEGER_FIT_MAX_R_HKL", "")
integer_fit_max = (
    tuple(float(x) for x in integer_fit_max_env.split(","))
    if integer_fit_max_env else None
)
integer_h_guard_env = env_default("INTEGER_H_GUARD", "")
integer_h_guard = None if integer_h_guard_env == "" else float(integer_h_guard_env)
search_nmad = float(env_default("SEARCH_NMAD", "6.0"))
search_min_i = float(env_default("SEARCH_MIN_I", "2.0"))
search_prom = float(env_default("SEARCH_PROM", "0.0"))
search_exclude_env = env_default("SEARCH_EXCLUDE_H", "")
search_exclude_h = (
    tuple(float(x) for x in search_exclude_env.split(",") if x.strip())
    if search_exclude_env else None
)
search_exclude_h_width = float(env_default("SEARCH_EXCLUDE_H_WIDTH", "0.0"))
margin = float(env_default("MARGIN", "0.03"))
max_scale = float(env_default("MAX_SCALE", "3.0"))
phi_tail_hkl = float(env_default("PHI_TAIL_HKL", "0.12"))
incident_r_hkl = tuple(
    float(x) for x in env_default("INCIDENT_R_HKL", "0.24,0.24,0.90").split(",")
)
incident_margin = float(env_default("INCIDENT_MARGIN", "0.12"))
incident_phi_tail = float(env_default("INCIDENT_PHI_TAIL_HKL", "0.0"))
incident_ellipsoid_env = env_default("INCIDENT_ELLIPSOID_R_HKL", "")
incident_ellipsoid_radii = (
    tuple(float(x) for x in incident_ellipsoid_env.split(","))
    if incident_ellipsoid_env else None
)
incident_sphere_env = env_default("INCIDENT_SPHERE_R_HKL", "")
incident_sphere_radius = (
    None if incident_sphere_env == "" else float(incident_sphere_env)
)

out_file = os.environ.get("OUT_FILE")
out_path = Path(out_file) if out_file else proc / f"{in_path.stem}_braggpunched.h5"

print(f"loading {in_path.name}", flush=True)
vol = ndiff.load(in_path)

remover = BraggRemover(
    mode=mode, punch_radii=r_hkl, min_intensity=min_i, min_prominence=1.0,
    integer_n_mad=integer_nmad, integer_q_step=integer_q_step,
    integer_optimize_position=integer_fit_position,
    integer_optimize_shape=integer_fit_shape,
    integer_fit_threshold_frac=integer_fit_frac,
    integer_fit_radius_n_sigma=integer_fit_nsigma,
    integer_fit_max_radius_hkl=integer_fit_max,
    integer_h_guard_hkl=integer_h_guard,
    intensity_scale=True, max_radius_scale=max_scale, margin=margin,
    punch_incident_beam=True, incident_beam_radii=incident_r_hkl,
    incident_beam_margin=incident_margin,
    incident_beam_phi_tail_hkl=incident_phi_tail,
    incident_beam_ellipsoid_radii_hkl=incident_ellipsoid_radii,
    incident_beam_sphere_radius_hkl=incident_sphere_radius,
    phi_tail_hkl=phi_tail_hkl,
    search_n_mad=search_nmad, search_min_intensity=search_min_i,
    search_min_prominence=search_prom,
    search_exclude_h_centers=search_exclude_h,
    search_exclude_h_half_width=search_exclude_h_width,
)
print(f"preset={preset_name or 'none'}  mode={mode}  radii={r_hkl}  min_I={min_i}  "
      f"integer_nmad={integer_nmad}  integer_q_step={integer_q_step}  "
      f"integer_fit_position={integer_fit_position}  "
      f"integer_fit_shape={integer_fit_shape}  "
      f"integer_h_guard={integer_h_guard}  "
      f"search_nmad={search_nmad}  search_min_I={search_min_i}  "
      f"search_prom={search_prom}  search_exclude_h={search_exclude_h}  "
      f"search_exclude_width={search_exclude_h_width}  phi_tail={phi_tail_hkl}",
      flush=True)
print(f"incident beam: radii={incident_r_hkl} margin={incident_margin} "
      f"phi_tail={incident_phi_tail} "
      f"ellipsoid={incident_ellipsoid_radii} sphere_r={incident_sphere_radius}", flush=True)
t = time.time()
peaks = remover.detect_peaks(vol)
keep = remover.build_mask(vol)
dt = time.time() - t

valid = vol.mask & np.isfinite(vol.data)
punched = valid & ~keep
d = np.where(valid, vol.data, np.nan)
kept_max = float(np.nanmax(np.where(keep, d, np.nan)))
print(f"detected {len(peaks)} Bragg peaks; punched "
      f"{int(punched.sum()):,} voxels ({100 * punched.sum() / valid.sum():.2f}% "
      f"of valid) in {dt:.1f}s", flush=True)
print(f"max intensity: {float(np.nanmax(d)):.1f} (before) -> {kept_max:.1f} (after "
      f"punch; the survivors are the off-integer satellites)", flush=True)

punched_vol = dataclasses.replace(vol, mask=vol.mask & keep)
print(f"saving punched volume -> {out_path}", flush=True)
ndiff.save(punched_vol, out_path)

if os.environ.get("PREVIEW", "1") != "0":
    from ndiff.visualization import interactive_slices
    H_VALUE = float(os.environ.get("H_VALUE", "0.0"))
    print("Drag the H plane slider; punched Bragg holes show grey, surviving "
          "(satellite) peaks stay bright.  Close the window to exit.", flush=True)
    interactive_slices(
        [("ring-removed", vol),
         ("Bragg-punched (holes grey)", punched_vol)],
        plane="0kl", value=H_VALUE, cmap="viridis", vmin=0.0, vmax=0.3,
        slider_min=0.0, slider_max=1.0, value_slider=True,
    )
