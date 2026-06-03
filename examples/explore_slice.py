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
from ndiff.preprocessing import EmptySubtractor, PatchedRadialRingModel
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

# Non-parametric per-patch radial background subtraction.
#   ring_width        : max ring full-width in |Q| removed as a peak (Å^-1)
#   q_step            : radial bin width (finer than the ring to resolve peaks)
#   baseline_smooth   : σ of the post-opening baseline smoothing (Å^-1)
#   profile_percentiles: trim band per |Q| bin (low=gaps, high=Bragg)
prm = PatchedRadialRingModel(
    n_patches=36, plane="0kl", q_step=0.02, ring_width=0.18,
    baseline_smooth=0.06, profile_percentiles=(10.0, 80.0),
)
prof = prm.fit(src, q_range=(1.5, 10.5))
res, I_ring = prm.subtract(src, prof)
removed = dataclasses.replace(d, data=I_ring)                 # the fitted rings
residual = dataclasses.replace(d, data=src.data - I_ring)     # data - rings

print(f"0kl slice H={float(d.h_axis[0]):.3g}  background={'on' if USE_BACKGROUND else 'OFF'}  "
      f"non-parametric radial-background model")
print("Drag the vmin/vmax sliders; toggle linear/log₁₀ (bottom-left). "
      "Close the window to exit.")

interactive_slices(
    [("0kl data", src), ("removed rings (I_ring)", removed), ("residual = data - rings", residual)],
    plane="0kl", value=0.0, cmap="inferno", vmax=0.3,
)
