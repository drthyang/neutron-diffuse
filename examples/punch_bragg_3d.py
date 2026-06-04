"""Bragg punch on the ring-removed 3D volume — compute, save, and preview.

Loads the ring-removed volume (output of ``remove_rings_3d.py``), detects the
real Bragg peaks (data-driven — skips systematic absences), punches an
anisotropic, intensity-scaled ellipsoid at each, saves the punched volume, and
opens the H-slider viewer so you can scrub planes and see what was removed (grey
holes) vs what survives (bright spots — e.g. off-integer satellite reflections).

Run::

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
      /Users/tt9/miniforge3/envs/rmc-discord/bin/python3 examples/punch_bragg_3d.py

Env overrides:
    DATA_FILE    ring-removed input .h5 (default: data/processed/<22K>_ringremoved.h5)
    OUT_FILE     punched output .h5  (default: <stem>_braggpunched.h5)
    MODE         "integer" | "auto" | "search" | "both" (default both).
                 "auto"/"search" detects sharp high-tail outliers above the
                 robust per-|Q| diffuse level, analogous to the ring-removal
                 robust profile logic; catches off-integer satellites /
                 small-domain Bragg.
    R_HKL        per-axis base punch radii "rh,rk,rl"  (default 0.12,0.12,0.45)
    MIN_I        integer-mode detection intensity threshold (default 2.0)
    SEARCH_NMAD  search-mode outlier threshold in MADs (default 6.0)
    SEARCH_MIN_I search-mode absolute intensity floor (default 2.0)
    SEARCH_PROM  search-mode local 3x3x3 prominence floor (default 0.0)
    MARGIN       guard band added to every radius (default 0.03)
    MAX_SCALE    max intensity radius multiplier (default 3.0)
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
data_file = os.environ.get("DATA_FILE")
if data_file:
    in_path = Path(data_file)
else:
    cands = sorted(proc.glob("*_ringremoved.h5"))
    in_path = next((p for p in cands if "22K_mmm" in p.stem), cands[0])

mode = os.environ.get("MODE", "both")
r_hkl = tuple(float(x) for x in os.environ.get("R_HKL", "0.12,0.12,0.45").split(","))
min_i = float(os.environ.get("MIN_I", "2.0"))
search_nmad = float(os.environ.get("SEARCH_NMAD", "6.0"))
search_min_i = float(os.environ.get("SEARCH_MIN_I", "2.0"))
search_prom = float(os.environ.get("SEARCH_PROM", "0.0"))
margin = float(os.environ.get("MARGIN", "0.03"))
max_scale = float(os.environ.get("MAX_SCALE", "3.0"))

out_file = os.environ.get("OUT_FILE")
out_path = Path(out_file) if out_file else proc / f"{in_path.stem}_braggpunched.h5"

print(f"loading {in_path.name}", flush=True)
vol = ndiff.load(in_path)

remover = BraggRemover(
    mode=mode, punch_radii=r_hkl, min_intensity=min_i, min_prominence=1.0,
    intensity_scale=True, max_radius_scale=max_scale, margin=margin,
    search_n_mad=search_nmad, search_min_intensity=search_min_i,
    search_min_prominence=search_prom,
)
print(f"mode={mode}  radii={r_hkl}  min_I={min_i}  "
      f"search_nmad={search_nmad}  search_min_I={search_min_i}  "
      f"search_prom={search_prom}", flush=True)
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
