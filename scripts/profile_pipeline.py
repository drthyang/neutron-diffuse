#!/usr/bin/env python
"""Per-stage wall-clock profile of the full reduction on a real volume.

Loads a raw HKL volume and runs each stage in sequence with default params,
timing each, so we can see where the pipeline actually spends time (and whether
the 3D FFT is the bottleneck that would justify a WebGPU kernel).

    python scripts/profile_pipeline.py [path-to-raw.nxs]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import ndiff
from ndiff.analysis.delta_pdf import compute_delta_pdf, invert_delta_pdf
from ndiff.pipeline import backfill, flatten, punch_bragg, remove_rings

DEFAULT = "data/raw/TbTi3Bi4_22K_mmm_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg.nxs"


def timed(label: str, fn):
    t = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t
    print(f"  {label:<22} {dt:8.2f} s")
    return out, dt


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
    print(f"Loading {path.name} …")
    vol, t_load = timed("load (.nxs → HKLVolume)", lambda: ndiff.load(path))
    print(f"  grid {vol.data.shape}  ({vol.data.size/1e6:.1f} M voxels)\n")

    times: dict[str, float] = {"load": t_load}
    v1, times["remove_rings"] = timed("remove_rings", lambda: remove_rings(vol))
    v2, times["punch_bragg"] = timed("punch_bragg", lambda: punch_bragg(v1))
    v3, times["backfill"] = timed("backfill", lambda: backfill(v2))
    v4, times["flatten"] = timed("flatten", lambda: flatten(v3))
    dpdf, times["compute_delta_pdf (FFT)"] = timed(
        "compute_delta_pdf (FFT)", lambda: compute_delta_pdf(v4))
    _, times["invert_delta_pdf (FFT)"] = timed(
        "invert_delta_pdf (FFT)", lambda: invert_delta_pdf(dpdf, deapodize=True))

    compute = {k: v for k, v in times.items() if k != "load"}
    total = sum(compute.values())
    fft = times["compute_delta_pdf (FFT)"] + times["invert_delta_pdf (FFT)"]
    print(f"\n  {'TOTAL compute':<22} {total:8.2f} s")
    print(f"  {'of which FFT':<22} {fft:8.2f} s  ({100*fft/total:.0f}%)")
    print("\n  share of compute time:")
    for k, v in sorted(compute.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<24} {100*v/total:5.1f}%")


if __name__ == "__main__":
    main()
