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
data = ndiff.load([p for p in sorted(raw.glob("*.nxs"))
                   if not p.stem.endswith(("_bkg", "_sub_bkg"))][0])
bkg = ndiff.load_mantid_nxs(
    [p for p in sorted(raw.glob("*.nxs")) if p.stem.endswith("_bkg")][0],
    ub_matrix=data.ub_matrix,
)

# Validate the ring model in isolation: skip the empty/background subtraction
# (it over-subtracts and imprints the background detector gap).  Fit and remove
# rings straight from the data:  residual = data - rings.
USE_BACKGROUND = False

ih0 = int(np.argmin(np.abs(data.h_axis)))
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

# Non-parametric per-patch radial background subtraction with a low-order,
# count-weighted azimuthal-texture model.
#   ring_width        : max ring full-width in |Q| removed as a peak (Å^-1)
#   q_step            : radial bin width (finer than the ring to resolve peaks)
#   baseline_smooth   : σ of the post-opening baseline smoothing (Å^-1)
#   profile_percentiles: trim band per |Q| bin (low=gaps, high=Bragg)
#   texture_model     : 'fourier' = low-order T(φ) (anisotropy, Bragg-immune);
#                       'patch' = discrete Hann blend
#   n_fourier         : azimuthal harmonics (general Fourier; long-wavelength)
#   texture_symmetric : False = general T(φ) (no mmm assumption)
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

prm = PatchedRadialRingModel(
    n_patches=36, plane="0kl", q_step=0.02, ring_width=0.24,
    baseline_smooth=0.06, profile_percentiles=(10.0, 80.0),
    texture_model="fourier", n_fourier=3, texture_symmetric=False,
    ring_templates=templates,
)
prof = prm.fit(src, q_range=(1.5, 10.5))
print(f"ring model: texture={prm.texture_model} n_fourier={prm.n_fourier} "
      f"symmetric={prm.texture_symmetric}")
res, I_ring = prm.subtract(src, prof)
removed = dataclasses.replace(src, data=I_ring)               # the fitted rings
residual = dataclasses.replace(src, data=src.data - I_ring)   # data - rings

print(f"0kl slice H={float(d.h_axis[0]):.3g}  background={'on' if USE_BACKGROUND else 'OFF'}  "
      f"non-parametric radial-background model")
print("Drag the vmin/vmax sliders; toggle linear/log₁₀ (bottom-left). "
      "Close the window to exit.")

interactive_slices(
    [("0kl data", src), ("removed rings (I_ring)", removed), ("residual = data - rings", residual)],
    plane="0kl", value=0.0, cmap="inferno", vmax=0.3,
)
