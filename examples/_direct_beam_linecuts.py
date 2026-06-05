"""Three linecuts through (0,0,0) along H, K, L to size the direct-beam remnant.

Plots intensity vs signed HKL coordinate for each cut so that both lobes of the
direct-beam spike are visible and its half-width in each direction can be read off.

Run::

    PYTHONPATH=src python examples/_direct_beam_linecuts.py

Env overrides:
    DATA_FILE   path to the ring-removed / cc_sub_bkg .nxs or processed .h5
                (default: the 22K mmm cc_sub_bkg file)
    N_POINTS    samples per linecut (default: 600)
    OUT         output PNG path (default: examples/_direct_beam_linecuts.png)
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "src")
import ndiff
from ndiff.preprocessing import line_profile

raw = Path("data/raw")

data_file = os.environ.get("DATA_FILE")
if data_file:
    vol = ndiff.load(Path(data_file))
else:
    cands = sorted(p for p in raw.glob("*.nxs") if "cc_sub_bkg" in p.stem)
    if not cands:
        cands = [p for p in sorted(raw.glob("*.nxs"))
                 if not p.stem.endswith(("_bkg", "_sub_bkg"))]
    vol = ndiff.load(cands[0])
    print(f"loaded: {cands[0].name}")

N = int(os.environ.get("N_POINTS", "600"))
OUT = os.environ.get("OUT", "examples/_direct_beam_linecuts.png")

h_max = float(min(abs(vol.h_axis.min()), abs(vol.h_axis.max())))
k_max = float(min(abs(vol.k_axis.min()), abs(vol.k_axis.max())))
l_max = float(min(abs(vol.l_axis.min()), abs(vol.l_axis.max())))

cuts = [
    ("H", (-h_max, 0.0, 0.0), (h_max, 0.0, 0.0)),
    ("K", (0.0, -k_max, 0.0), (0.0,  k_max, 0.0)),
    ("L", (0.0, 0.0, -l_max), (0.0,  0.0,   l_max)),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, (label, start, end) in zip(axes, cuts):
    _, intensity, hkl = line_profile(vol, start, end, n_points=N)
    coord_idx = {"H": 0, "K": 1, "L": 2}[label]
    x = hkl[:, coord_idx]

    finite = np.isfinite(intensity)
    ax.plot(x[finite], intensity[finite], lw=1.0, color="C0")
    ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.4)

    # Annotate peak at origin
    origin_idx = int(np.argmin(np.abs(x)))
    peak_val = intensity[origin_idx]
    if np.isfinite(peak_val):
        ax.annotate(f"{peak_val:.1f}", xy=(0, peak_val),
                    xytext=(0.05, 0.92), textcoords="axes fraction",
                    fontsize=8, color="C3")

    ax.set_xlabel(f"{label} (r.l.u.)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Cut along {label} through (0,0,0)")

fig.suptitle("Direct-beam linecuts — size the sphere punch radius", fontsize=10)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"saved: {OUT}")
