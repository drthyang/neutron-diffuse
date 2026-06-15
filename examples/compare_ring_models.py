"""A/B-compare the two powder-ring removers on real data, per 0kl plane.

Runs the **patched** (`PatchedRadialRingModel`, the shipped non-parametric
per-patch estimator) and the **parametric** (`ParametricRingModel`, separable
pseudo-Voigt × per-ring Fourier texture) on the SAME input planes with the same
confirmed shells, then reports, for a few representative H planes:

* **ring suppression** at each confirmed |Q| shell (on-shell minus off-shell
  radial median, before vs after — higher % removed is better);
* **diffuse preservation** off-ring (|I_ring| should be ≈0 between shells);
* **over-subtraction** (fraction of plane driven below −0.05 — lower is better);
* **Bragg preservation** (the brightest voxels survive both — sanity check).

It also writes a side-by-side PNG: rows = H planes, columns = data | patched
residual | parametric residual (shared robust colour scale per row).

Usage::

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
    python3 examples/compare_ring_models.py            # H = 0, 1/3, 1 on 22K

Env: ``DATA_FILE`` (input .nxs), ``H_VALUES`` (comma list, default
``0,0.3333,1.0``), ``N_FOURIER`` (default 6, matched across both),
``Q_MIN``/``Q_MAX`` (fit range), ``OUT_PNG`` (figure path).
"""

from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ndiff
from ndiff.preprocessing import (
    ParametricRingModel,
    PatchedRadialRingModel,
    azimuthal_sampling_mask,
    confirm_ring_shells_across_h,
)
from ndiff.preprocessing.radial_background import _offset_q_magnitude

PLANE = "0kl"               # slice_axis = H → fit the b*-c* planes
AXIS_DIM = 0
Q_MIN = float(os.environ.get("Q_MIN", "1.5"))
Q_MAX = float(os.environ.get("Q_MAX", "10.5"))
N_FOURIER = int(os.environ.get("N_FOURIER", "6"))   # matched across both models


def _find_input() -> Path:
    df = os.environ.get("DATA_FILE")
    if df:
        return Path(df)
    raw = Path("data/raw")
    cands = sorted(raw.glob("*22K*cc_sub_bkg.nxs"))
    if not cands:
        cands = sorted(raw.glob("*.nxs"))
    if not cands:
        raise FileNotFoundError("No input .nxs found; set DATA_FILE=…")
    return cands[0]


def _plane_at(vol, h_value: float):
    """One-plane (0kl) HKLVolume view nearest to *h_value* (+ its H index)."""
    ih = int(np.argmin(np.abs(vol.h_axis - h_value)))
    sl = (slice(ih, ih + 1), slice(None), slice(None))
    return dataclasses.replace(
        vol, data=vol.data[sl], sigma=vol.sigma[sl], mask=vol.mask[sl],
        h_axis=vol.h_axis[ih:ih + 1]), ih


def _suppression(q2d, before2d, after2d, shells, halfwidths):
    """Per-shell on/off radial-median suppression, robust to Bragg."""
    out = []
    for c, w in zip(shells, halfwidths):
        on = (q2d >= c - w) & (q2d <= c + w)
        off = ((q2d >= c - 3 * w) & (q2d < c - w)) | (
            (q2d > c + w) & (q2d <= c + 3 * w))
        if on.sum() < 10 or off.sum() < 10:
            continue
        base = float(np.nanmedian(after2d[off]))
        b = float(np.nanmedian(before2d[on])) - base
        a = float(np.nanmedian(after2d[on])) - base
        out.append((c, b, a))
    return out


def main() -> None:
    in_path = _find_input()
    print(f"loading {in_path.name}", flush=True)
    vol = ndiff.load(in_path)
    print(f"volume {vol.data.shape}  |Q| fit {(Q_MIN, Q_MAX)}", flush=True)

    t0 = time.time()
    centers, halfwidths, amps = confirm_ring_shells_across_h(
        vol, plane=PLANE, q_range=(Q_MIN, Q_MAX))
    ceilings = 3.0 * amps if amps.size else None
    print(f"confirmed {centers.size} ring shells across H in "
          f"{time.time() - t0:.1f}s: "
          f"{', '.join(f'{c:.2f}' for c in centers)} Å⁻¹", flush=True)

    common = dict(plane=PLANE, q_step=0.02, n_fourier=N_FOURIER,
                  profile_method="median", allowed_ring_centers=centers,
                  allowed_ring_halfwidths=halfwidths,
                  allowed_ring_ceilings=ceilings)
    patched = PatchedRadialRingModel(texture_q_smooth=0.02, texture_ridge=0.08,
                                     **common)
    parametric = ParametricRingModel(texture_ridge=0.05, **common)

    h_values = [float(x) for x in
                os.environ.get("H_VALUES", "0,0.3333,1.0").split(",")]
    rows = []
    for hv in h_values:
        pl, ih = _plane_at(vol, hv)
        keep = azimuthal_sampling_mask(pl, plane=PLANE, min_count_frac=0.25,
                                       q_range=(Q_MIN, Q_MAX))
        src = dataclasses.replace(pl, mask=keep)
        q2d = _offset_q_magnitude(pl, PLANE)[0]
        data2d = pl.data[0]

        res = {}
        for name, model in (("patched", patched), ("parametric", parametric)):
            t = time.time()
            model.fit(src, q_range=(Q_MIN, Q_MAX))
            _, I_ring = model.subtract(src)
            res[name] = I_ring[0]
            dt = time.time() - t

            after = data2d - I_ring[0]
            supp = _suppression(q2d, data2d, after, centers, halfwidths)
            mean_rem = (np.mean([(b - a) / b for _, b, a in supp if b > 1e-6])
                        if supp else float("nan"))
            offring = np.ones_like(q2d, bool)
            for c, w in zip(centers, halfwidths):
                offring &= np.abs(q2d - c) > 3 * w
            offring &= (q2d > Q_MIN) & (q2d < Q_MAX)
            diffuse_touch = float(np.nanmedian(np.abs(I_ring[0][offring])))
            neg_frac = float(np.mean(after[(q2d > Q_MIN) & (q2d < Q_MAX)] < -0.05))
            print(f"H={hv:+.3f} [{name:>10}] {dt:5.1f}s  "
                  f"ring removed={mean_rem * 100:5.1f}%  "
                  f"off-ring |I_ring| median={diffuse_touch:.4f}  "
                  f"over-subtract frac={neg_frac * 100:.2f}%", flush=True)

        rows.append((hv, data2d, res["patched"], res["parametric"], q2d))

    # ---- figure: rows = H, cols = data | patched resid | parametric resid ----
    out_png = Path(os.environ.get(
        "OUT_PNG", "data/processed/compare_ring_models_22K.png"))
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)
    col_titles = ["data", "patched (data − ring)", "parametric (data − ring)"]
    for r, (hv, data2d, ip, iq, q2d) in enumerate(rows):
        panels = [data2d, data2d - ip, data2d - iq]
        finite = data2d[np.isfinite(data2d) & (q2d > Q_MIN)]
        vmax = float(np.nanpercentile(finite, 99)) if finite.size else 1.0
        for c, (ax, panel) in enumerate(zip(axes[r], panels)):
            ax.imshow(panel.T, origin="lower", vmin=0, vmax=vmax,
                      cmap="magma", aspect="auto")
            ax.set_title(f"H={hv:+.3f}  {col_titles[c]}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    print(f"\nwrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
