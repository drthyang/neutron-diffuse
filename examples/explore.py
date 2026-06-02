"""Interactive plotting session for the real 28K dataset.

Launch with an interactive Matplotlib backend so figures appear in live
windows you can pan/zoom (and stay at an IPython prompt afterwards):

    cd <repo root>
    PYTHONPATH=src ipython --matplotlib=macosx -i examples/explore.py

(use --matplotlib=qt on Linux/Windows, or run inside Jupyter).

After it loads you have these in scope:
    data, bkg, sub        HKLVolume for data / background / data-bkg
    plot_slice, plot_radial_profile, plot_azimuthal_map, plot_overview
    extract_slice, plt

Try:
    plot_overview(data, log_scale=True)            # 2x2 diagnostic
    plot_slice(bkg, "kl", value=0.0, log_scale=True)
    plot_slice(data, "hk", value=0.3333, interp=True)   # exact L=1/3 plane
    plot_slice(data, "hk", value=0.3333, interp=True,
               vmin=0.0, vmax=0.4)                 # manual colour limits
    plot_radial_profile(data, mark_q=[2.69])       # Al(111)
    plot_azimuthal_map(data, q_center=2.69)        # ring texture T(phi)
    plt.show()                                     # if not auto-shown
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import ndiff
from ndiff.visualization import (
    extract_slice,
    plot_azimuthal_map,
    plot_overview,
    plot_radial_profile,
    plot_slice,
)

# Locate the raw files (long Mantid-style names) without hard-coding them.
_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def _find(suffix: str) -> Path:
    """Return the single .nxs whose stem ends with *suffix* ('' = the data file)."""
    hits = [
        p
        for p in sorted(_RAW.glob("*.nxs"))
        if (p.stem.endswith(suffix) if suffix else not p.stem.endswith(("_bkg", "_sub_bkg")))
    ]
    if not hits:
        raise FileNotFoundError(f"No .nxs in {_RAW} matching suffix {suffix!r}")
    return hits[0]


data = ndiff.load(_find(""))
# Background scan has no UB of its own -> inherit the data's for a consistent |Q|.
bkg = ndiff.load_mantid_nxs(_find("_bkg"), ub_matrix=data.ub_matrix)
sub = ndiff.load(_find("_sub_bkg"))

plt.ion()  # interactive: figures show without blocking

print(__doc__)
for _name, _v in (("data", data), ("bkg", bkg), ("sub", sub)):
    print(f"  {_name:4s}  shape={_v.data.shape}  valid={100 * _v.mask.mean():.1f}%")
