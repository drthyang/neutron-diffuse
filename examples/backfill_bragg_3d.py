"""Backfill Bragg-punched holes in a 3D volume and save the filled result.

Run after ``examples/punch_bragg_3d.py``.  The input volume's mask marks punched
Bragg/satellite holes as invalid; this script fills those voxels and writes a
new all-valid volume for DeltaPDF.

Run::

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
      /opt/homebrew/Caskroom/miniforge/base/envs/sci-general/bin/python3 \
      examples/backfill_bragg_3d.py

Env overrides:
    DATA_FILE   punched input .h5
    OUT_FILE    output .h5 (default: <stem>_backfilled.h5)
    METHOD      "local" | "q_shell" | "tv" | "symmetry+tv" | "symmetry" | ...
                (default local).  "q_shell" fills ordinary Bragg holes from the
                robust background level at the same |Q|.
    LAUE        Laue class for symmetry fill (default mmm)
    LOCAL_RADIUS     dilation radius for local background shell (default 2)
    LOCAL_MIN_COUNT  minimum shell samples before global-median fallback (default 8)
    Q_SHELL_STEP      |Q| bin width for METHOD=q_shell (default 0.05 Å^-1)
    Q_SHELL_MIN_COUNT minimum radial-bin samples for METHOD=q_shell (default 20)
    TV_LAM      TV regularisation weight (default 0.2)
    TV_ITER     TV iterations, only used by TV modes (default 80)
"""
import os
import time
from pathlib import Path

import numpy as np

import ndiff
from ndiff.analysis import backfill_bragg

proc = Path("data/processed")
data_file = os.environ.get("DATA_FILE")
if data_file:
    in_path = Path(data_file)
else:
    cands = sorted(proc.glob("*_braggpunched*.h5"))
    if not cands:
        raise FileNotFoundError(
            "No Bragg-punched input found in data/processed. Run "
            "`PYTHONPATH=src /opt/homebrew/Caskroom/miniforge/base/envs/"
            "sci-general/bin/python3 examples/punch_bragg_3d.py` first, "
            "or set DATA_FILE=/path/to/*_braggpunched.h5."
        )
    in_path = next((p for p in cands if "hmid_min1_prom1" in p.stem), cands[-1])

out_file = os.environ.get("OUT_FILE")
out_path = Path(out_file) if out_file else proc / f"{in_path.stem}_backfilled.h5"

method = os.environ.get("METHOD", "local")
laue = os.environ.get("LAUE", "mmm")
local_radius = int(os.environ.get("LOCAL_RADIUS", "2"))
local_min_count = int(os.environ.get("LOCAL_MIN_COUNT", "8"))
q_shell_step = float(os.environ.get("Q_SHELL_STEP", "0.05"))
q_shell_min_count = int(os.environ.get("Q_SHELL_MIN_COUNT", "20"))
tv_lam = float(os.environ.get("TV_LAM", "0.2"))
tv_iter = int(os.environ.get("TV_ITER", "80"))

print(f"loading {in_path}", flush=True)
vol = ndiff.load(in_path)
valid = vol.mask & np.isfinite(vol.data)
holes = (~vol.mask) & np.isfinite(vol.data)
print(f"volume {vol.shape}; holes={int(holes.sum()):,} "
      f"({100.0 * holes.sum() / max(valid.sum() + holes.sum(), 1):.2f}% observed grid)",
      flush=True)
print(f"backfill method={method} laue={laue} local_radius={local_radius} "
      f"local_min_count={local_min_count} q_shell_step={q_shell_step} "
      f"q_shell_min_count={q_shell_min_count} tv_lam={tv_lam} tv_iter={tv_iter}",
      flush=True)

t0 = time.time()
filled = backfill_bragg(
    vol, method=method, laue_class=laue, local_radius=local_radius,
    local_min_count=local_min_count, q_shell_step=q_shell_step,
    q_shell_min_count=q_shell_min_count, tv_lam=tv_lam, tv_iter=tv_iter,
)
dt = time.time() - t0

still_masked = int((~filled.mask).sum())
vals = filled.data[holes]
print(f"filled in {dt:.1f}s; still masked={still_masked}", flush=True)
if vals.size:
    print(f"filled values: median={float(np.nanmedian(vals)):.4g} "
          f"p01={float(np.nanpercentile(vals, 1)):.4g} "
          f"p99={float(np.nanpercentile(vals, 99)):.4g}", flush=True)

print(f"saving -> {out_path}", flush=True)
ndiff.save(filled, out_path)
print("Bragg backfill complete.", flush=True)
