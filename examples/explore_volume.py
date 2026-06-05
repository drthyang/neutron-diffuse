"""Interactive 3D-volume ring-removal preview — same 3 panels as
``explore_slice.py`` (data | removed rings | residual) but with an extra **H
slider** so you can scrub through every plane of the *already-processed* volume.

Unlike ``explore_slice.py`` (which fits ONE slice live), this loads the full 3D
residual written by ``examples/remove_rings_3d.py`` and the raw data, then scrubs
in place — the heavy compute already happened in the driver, so moving the H
slider just re-indexes the in-memory volumes (responsive even at 301 planes).
Loading both 689 MB volumes is the only heavy part (~1.4 GB RAM).

Run::

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
      python3 \
      examples/explore_volume.py

Env overrides:
    DATA_FILE   raw input .nxs (default: the 22K mmm validation file)
    PROC_FILE   processed residual .h5 (default: data/processed/<stem>_ringremoved.h5)
    H_VALUE     initial H plane (default 0.0; drag the H slider to move)
    SLIDER_MIN/SLIDER_MAX   vmin/vmax slider travel (default 0..1, tight near diffuse)
"""
import matplotlib
matplotlib.use("macosx")          # interactive backend (use "qt" on Linux/Win)

import dataclasses
import os
import sys
from pathlib import Path

import ndiff
from ndiff.visualization import interactive_slices

raw_dir = Path("data/raw")
data_file = os.environ.get("DATA_FILE")
if data_file:
    in_path = Path(data_file)
else:
    def is_empty_background(path: Path) -> bool:
        return (
            path.stem.endswith("_bkg")
            and not path.stem.endswith(("_sub_bkg", "_cc_sub_bkg"))
        )

    cands = [p for p in sorted(raw_dir.glob("*.nxs")) if not is_empty_background(p)]
    if not cands:
        raise FileNotFoundError(
            "No input .nxs files found in data/raw. Set DATA_FILE=/path/to/input.nxs."
        )
    in_path = next(
        (p for p in cands if "22K_mmm" in p.stem and "cc_sub_bkg" in p.stem),
        next((p for p in cands if "22K_mmm" in p.stem), cands[0]),
    )

proc_file = os.environ.get("PROC_FILE")
proc_path = (Path(proc_file) if proc_file
             else Path("data/processed") / f"{in_path.stem}_ringremoved.h5")
if not proc_path.exists():
    sys.exit(
        f"processed volume not found: {proc_path}\n"
        "Run the 3D ring removal first:\n"
        "  PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \\\n"
        "    python3 "
        "examples/remove_rings_3d.py"
    )

print(f"loading raw      {in_path.name}", flush=True)
data = ndiff.load(in_path)
print(f"loading residual {proc_path.name}", flush=True)
residual = ndiff.load(proc_path)

# The removed-ring intensity is just data − residual (the subtraction is
# voxel-wise).  Carry the residual's mask (it records the sparse-azimuth drops).
removed = dataclasses.replace(residual, data=data.data - residual.data)

H_VALUE = float(os.environ.get("H_VALUE", "0.0"))
slider_min = float(os.environ.get("SLIDER_MIN", "0.0"))
slider_max = float(os.environ.get("SLIDER_MAX", "1.0"))

print(f"volume (H,K,L)={data.data.shape}  H range "
      f"[{data.h_axis.min():.3g}, {data.h_axis.max():.3g}]  "
      f"start H={H_VALUE}", flush=True)
print("Drag the H slider to scrub planes; vmin/vmax sliders + linear/log toggle "
      "as in explore_slice.  Close the window to exit.", flush=True)

interactive_slices(
    [("0kl data", data), ("removed rings (I_ring)", removed),
     ("residual = data - rings", residual)],
    plane="0kl", value=H_VALUE, cmap="viridis", vmin=0.0, vmax=0.3,
    slider_min=slider_min, slider_max=slider_max, value_slider=True,
)
