"""Interactive ring-removal viewer on a single 0kl slice (H=0).

Fast 2D development harness: extracts the H=0 plane and runs Step 2
(``PatchedRadialRingModel``, plane='0kl') — non-parametric per-patch radial
background subtraction — then opens an interactive window with live
colour-scale (vmin/vmax) sliders and a linear/log toggle so you can drag the
scale to reveal weak rings/diffuse.

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
from ndiff.preprocessing import (
    EmptySubtractor,
    PatchedRadialRingModel,
    azimuthal_sampling_mask,
    fit_ring_profiles,
    line_profile,
    replace_masked_ring_regions,
)
from ndiff.visualization import interactive_slices

raw = Path("data/raw")
# Default to the preferred 22K mmm validation file; the alphabetically-first
# .nxs is the older 28K file, so select the 22K one explicitly unless DATA_FILE
# overrides.  H_VALUE picks the slice (default 0.3333, where the diffuse signal
# is clearest).
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

ih0 = int(np.argmin(np.abs(data.h_axis - H_VALUE)))
def _hslice(v):
    return dataclasses.replace(
        v, data=v.data[ih0:ih0 + 1], sigma=v.sigma[ih0:ih0 + 1],
        mask=v.mask[ih0:ih0 + 1], h_axis=v.h_axis[ih0:ih0 + 1],
    )

d = _hslice(data)

if USE_BACKGROUND:
    sub = EmptySubtractor(_hslice(bkg), scale_q_range=(2.4, 3.3))
    src = sub.subtract(d)
    print(f"step1 empty subtraction: s={sub.scale:.3f}")
else:
    src = d   # ring model fits the total ring signal directly

# Mask azimuthally under-sampled (|Q|, φ) cells (anomalously sparse sectors whose
# few measurements are unreliable).  The threshold is relative to each |Q|-shell
# so uniformly low-density shells (e.g. small |Q|) are NOT carved out.
#   min_count_frac : drop cells below this fraction of the shell's median count
MASK_SPARSE = True
if MASK_SPARSE:
    keep = azimuthal_sampling_mask(src, plane="0kl", min_count_frac=0.25,
                                   q_range=(1.5, 10.5))
    src = dataclasses.replace(src, mask=keep)
    print(f"sparse-azimuth mask: dropped {int((d.mask & ~keep).sum())} voxels")

# Non-parametric per-patch radial background subtraction with a |Q|-pooled
# azimuthal-texture model (the current reference config — all knobs at their
# class defaults: SNIP baseline, n_fourier=8, texture_q_smooth=0.06).
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

prm = PatchedRadialRingModel(
    n_patches=36, plane="0kl", q_step=0.02, ring_width=0.24,
    baseline_method="snip", baseline_smooth=0.06,
    profile_percentiles=(10.0, 80.0), texture_model="fourier",
    texture_symmetric=False, ring_templates=templates,
    center_offset=center_offset, center_offset_h_slope=center_offset_h_slope,
)
prof = prm.fit(src, q_range=(1.5, 10.5))
print(f"ring model: texture={prm.texture_model} n_fourier={prm.n_fourier} "
      f"symmetric={prm.texture_symmetric}")
print(f"center offset={center_offset}  H slope={center_offset_h_slope} Å^-1/H")
res, I_ring = prm.subtract(src, prof)
removed = dataclasses.replace(src, data=I_ring)               # the fitted rings
residual = dataclasses.replace(src, data=src.data - I_ring)   # data - rings

cleanup_mode = os.environ.get("RING_CLEANUP_MODE", "subtract")
if cleanup_mode == "mask_replace":
    repl = replace_masked_ring_regions(
        src,
        prm,
        prof,
        model_threshold_frac=float(os.environ.get("RING_MASK_MODEL_FRAC", "0.18")),
        excess_sigma=float(os.environ.get("RING_MASK_EXCESS_SIGMA", "2.5")),
        dilation_iter=int(os.environ.get("RING_MASK_DILATE", "1")),
        closing_iter=int(os.environ.get("RING_MASK_CLOSE", "1")),
        fill_method=os.environ.get("RING_FILL_METHOD", "sideband"),
        n_phi_bins=int(os.environ.get("RING_FILL_PHI_BINS", "180")),
    )
    mask_panel = dataclasses.replace(src, data=repl.mask.astype(float))
    background = dataclasses.replace(src, data=repl.background)
    residual = repl.clean
    removed = mask_panel
    print(f"mask-replace cleanup: fill={os.environ.get('RING_FILL_METHOD', 'sideband')} "
          f"masked {int(repl.mask.sum())} voxels "
          f"({100.0 * repl.mask.mean():.3f}% of slice volume)")
elif cleanup_mode != "subtract":
    raise ValueError(f"Unknown RING_CLEANUP_MODE={cleanup_mode!r}")

print(f"0kl slice H={float(d.h_axis[0]):.3g}  background={'on' if USE_BACKGROUND else 'OFF'}  "
      f"non-parametric radial-background model  cleanup={cleanup_mode}")
print("Drag the vmin/vmax sliders; toggle linear/log₁₀ (bottom-left). "
      "Close the window to exit.")

# Tight slider travel so the pullbar gives fine control near the diffuse level
# (~0–0.3) instead of spanning the full bright-ring data range.  Override with
# SLIDER_MIN / SLIDER_MAX env vars.
slider_min = float(os.environ.get("SLIDER_MIN", "0.0"))
slider_max = float(os.environ.get("SLIDER_MAX", "1.0"))
interactive_slices(
    [("0kl data", src), ("ring mask" if cleanup_mode == "mask_replace" else "removed rings (I_ring)", removed),
     ("cleaned" if cleanup_mode == "mask_replace" else "residual = data - rings", residual)],
    plane="0kl", value=0.0, cmap="viridis", vmin=0.0, vmax=0.3,
    slider_min=slider_min, slider_max=slider_max,
)
