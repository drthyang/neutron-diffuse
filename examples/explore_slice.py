"""Interactive ring-removal viewer on a single 0kl slice (H=0).

Fast 2D development harness: extracts the H=0 plane, runs Step 1 (empty
subtraction) + Step 2 (PatchedRingModel, plane='0kl'), and opens an
interactive window with live colour-scale (vmin/vmax) sliders and a
linear/log toggle so you can drag the scale to reveal weak rings/diffuse.

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
from ndiff.preprocessing import EmptySubtractor, PatchedRingModel
from ndiff.preprocessing.powder_rings import al_ring_q_positions
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

FLATNESS_CV = 0.5     # None = baseline subtraction only; 0.5 = gate rough/Bragg shells
prm = PatchedRingModel(n_patches=36, n_fourier=6, plane="0kl", flatness_cv=FLATNESS_CV)
model = prm.fit(src, ring_hints=list(al_ring_q_positions(q_max=10.5)),
                q_range=(1.5, 10.5))
res, I_ring = prm.subtract(src, model)
removed = dataclasses.replace(d, data=I_ring)                 # the fitted rings
residual = dataclasses.replace(d, data=src.data - I_ring)     # data - rings

print(f"0kl slice H={float(d.h_axis[0]):.3g}  background={'on' if USE_BACKGROUND else 'OFF'}  "
      f"ring-model rank1_var={model.rank1_variance:.3f}")
print("Drag the vmin/vmax sliders; toggle linear/log₁₀ (bottom-left). "
      "Close the window to exit.")

interactive_slices(
    [("0kl data", src), ("removed rings (I_ring)", removed), ("residual = data - rings", residual)],
    plane="0kl", value=0.0, cmap="inferno", vmax=0.3,
)
