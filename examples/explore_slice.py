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
profile_method = os.environ.get("PROFILE_METHOD", "trimmed_mean")

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
texture_q_smooth = float(os.environ.get("TEXTURE_Q_SMOOTH", "0.06"))

prm = PatchedRadialRingModel(
    n_patches=36, plane="0kl", q_step=q_step, ring_width=0.24,
    baseline_method="snip", baseline_smooth=0.06,
    profile_percentiles=(10.0, 80.0), profile_method=profile_method,
    texture_model="fourier", texture_symmetric=False, texture_q_smooth=texture_q_smooth,
    ring_templates=templates,
    center_offset=center_offset, center_offset_h_slope=center_offset_h_slope,
)
prof = prm.fit(src, q_range=(1.5, 10.5))
print(f"ring model: texture={prm.texture_model} n_fourier={prm.n_fourier} "
      f"symmetric={prm.texture_symmetric} q_step={q_step} profile={profile_method} "
      f"q_smooth={texture_q_smooth}")
print(f"center offset={center_offset}  H slope={center_offset_h_slope} Å^-1/H")
res, I_ring = prm.subtract(src, prof)
removed = dataclasses.replace(src, data=I_ring)               # the fitted rings
residual = dataclasses.replace(src, data=src.data - I_ring)   # data - rings

print(f"0kl slice H={float(d.h_axis[0]):.3g}  background={'on' if USE_BACKGROUND else 'OFF'}  "
      f"non-parametric radial-background subtraction")
print("Drag the vmin/vmax sliders; toggle linear/log₁₀ (bottom-left). "
      "Close the window to exit.")

# Tight slider travel so the pullbar gives fine control near the diffuse level
# (~0–0.3) instead of spanning the full bright-ring data range.  Override with
# SLIDER_MIN / SLIDER_MAX env vars.
slider_min = float(os.environ.get("SLIDER_MIN", "0.0"))
slider_max = float(os.environ.get("SLIDER_MAX", "1.0"))
interactive_slices(
    [("0kl data", src), ("removed rings (I_ring)", removed),
     ("residual = data - rings", residual)],
    plane="0kl", value=0.0, cmap="viridis", vmin=0.0, vmax=0.3,
    slider_min=slider_min, slider_max=slider_max,
)
