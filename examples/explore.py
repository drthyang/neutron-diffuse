"""Interactive plotting session for a raw HKL volume.

Launch with an interactive Matplotlib backend so figures appear in live
windows you can pan/zoom (and stay at an IPython prompt afterwards):

    cd <repo root>
    PYTHONPATH=src ipython --matplotlib=macosx -i examples/explore.py

(use --matplotlib=qt on Linux/Windows, or run inside Jupyter).

After it loads you have these in scope:
    data, bkg, sub        HKLVolume for data / optional background / data-bkg
    plot_slice, plot_radial_profile, plot_azimuthal_map, plot_overview
    extract_slice, plt

Try:
    plot_overview(data, log_scale=True)            # 2x2 diagnostic
    plot_slice(bkg, "kl", value=0.0, log_scale=True)  # if bkg is not None
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


def _is_empty_background(path: Path) -> bool:
    return (
        path.stem.endswith("_bkg")
        and not path.stem.endswith(("_sub_bkg", "_cc_sub_bkg"))
    )


def _find_data() -> Path:
    """Return the preferred data .nxs file."""
    hits = [p for p in sorted(_RAW.glob("*.nxs")) if not _is_empty_background(p)]
    if not hits:
        raise FileNotFoundError(f"No data .nxs files found in {_RAW}")
    return next(
        (p for p in hits if "22K_mmm" in p.stem and "cc_sub_bkg" in p.stem),
        next((p for p in hits if "22K_mmm" in p.stem), hits[0]),
    )


def _find_background() -> Path | None:
    hits = [p for p in sorted(_RAW.glob("*.nxs")) if _is_empty_background(p)]
    return hits[0] if hits else None


data = ndiff.load(_find_data())
_bkg_path = _find_background()
if _bkg_path is None:
    bkg = None
    sub = data
else:
    # Background scan has no UB of its own -> inherit the data's for consistent |Q|.
    bkg = ndiff.load_mantid_nxs(_bkg_path, ub_matrix=data.ub_matrix)
    sub = data

plt.ion()  # interactive: figures show without blocking

print(__doc__)
for _name, _v in (("data", data), ("bkg", bkg), ("sub", sub)):
    print(f"  {_name:4s}  shape={_v.data.shape}  valid={100 * _v.mask.mean():.1f}%")
