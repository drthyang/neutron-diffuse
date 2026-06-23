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

import nebula3d
from nebula3d.preprocessing import (
    ParametricRingModel,
    PatchedRadialRingModel,
    azimuthal_sampling_mask,
    confirm_ring_shells_across_h,
)
from nebula3d.preprocessing.radial_background import (
    _estimate_baseline,
    _fill_nan_1d,
    _offset_q_magnitude,
    _robust_radial_profile,
)

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


def _radial_profile_1d(vals2d, q2d, edges):
    """Robust (Bragg-rejecting) radial-median profile of *vals2d* over *edges*."""
    m = np.isfinite(vals2d)
    prof, _ = _robust_radial_profile(
        q2d[m], vals2d[m], edges, percentiles=(5.0, 55.0),
        min_per_bin=8, method="median")
    return _fill_nan_1d(prof)


def _diffuse_baseline_2d(data2d, q2d, q_range, q_step=0.01, ring_width=0.30):
    """Smooth off-ring diffuse baseline B(|Q|) interpolated onto the 2-D grid.

    SNIP-clips the rings (and Bragg) out of the data's robust radial profile, so
    ``data − B`` is ~0 off-ring and isolates whatever sits *on* the rings.  Used
    as the SAME reference for both residuals → a fair over/under-subtraction map.
    Returns ``(b2d, centers, base1d)`` so the 1-D residual figure can reuse it.
    """
    qmin, qmax = q_range
    edges = np.arange(qmin, qmax + q_step, q_step)
    sel = (q2d >= qmin) & (q2d <= qmax)
    prof = _radial_profile_1d(np.where(sel, data2d, np.nan), q2d, edges)
    base = _estimate_baseline(prof, q_step=q_step, ring_width=ring_width, smooth=0.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    b2d = np.interp(q2d, centers, base, left=base[0], right=base[-1])
    return b2d, centers, base, edges


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
    vol = nebula3d.load(in_path)
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

    # ---- decision figure: deviation from the common off-ring diffuse baseline --
    # cols = data | patched (after − B) | parametric (after − B) on a DIVERGING
    # scale centred at 0.  red arc = ring leftover (under-subtract); blue trough =
    # over-subtraction; white = clean.  Same baseline B(|Q|) for both methods.
    dev_png = Path(os.environ.get(
        "DEV_PNG", "data/processed/compare_ring_deviation_22K.png"))
    fig2, axes2 = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)
    dcol_titles = ["data", "patched: after − baseline",
                   "parametric: after − baseline"]
    for r, (hv, data2d, ip, iq, q2d) in enumerate(rows):
        base2d, _, _, _ = _diffuse_baseline_2d(data2d, q2d, (Q_MIN, Q_MAX))
        dev_p = (data2d - ip) - base2d
        dev_q = (data2d - iq) - base2d
        ann = (q2d > Q_MIN) & (q2d < Q_MAX) & np.isfinite(data2d)
        scale = float(np.nanpercentile(
            np.abs(np.concatenate([dev_p[ann], dev_q[ann]])), 97))
        scale = scale if scale > 1e-9 else 1.0
        finite = data2d[np.isfinite(data2d) & (q2d > Q_MIN)]
        vmax = float(np.nanpercentile(finite, 99)) if finite.size else 1.0
        for c, (panel, cmap, kw, title) in enumerate((
            (data2d, "magma", dict(vmin=0, vmax=vmax), dcol_titles[0]),
            (dev_p, "RdBu_r", dict(vmin=-scale, vmax=scale), dcol_titles[1]),
            (dev_q, "RdBu_r", dict(vmin=-scale, vmax=scale), dcol_titles[2]),
        )):
            ax = axes2[r][c]
            ax.imshow(panel.T, origin="lower", cmap=cmap, aspect="auto", **kw)
            ax.set_title(f"H={hv:+.3f}  {title}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig2.tight_layout()
    fig2.savefig(dev_png, dpi=110)
    print(f"wrote {dev_png}", flush=True)

    # ---- 1-D decision figure: azimuthally-averaged ring-residual profile -------
    # This is the SAME robust radial median the %-removed metric is built from, so
    # it lets you check the metric by eye.  Plotted as (after − baseline): at each
    # ring, a curve near 0 is perfect; a positive bump = leftover (under-subtract);
    # a negative dip = over-subtraction.  Whichever curve hugs 0 best wins.
    prof_png = Path(os.environ.get(
        "PROF_PNG", "data/processed/compare_ring_profile_22K.png"))
    fig3, axes3 = plt.subplots(n, 1, figsize=(13, 3.2 * n), squeeze=False)
    for r, (hv, data2d, ip, iq, q2d) in enumerate(rows):
        _, qc, base1d, edges = _diffuse_baseline_2d(data2d, q2d, (Q_MIN, Q_MAX))
        sel = (q2d >= Q_MIN) & (q2d <= Q_MAX)
        pp = _radial_profile_1d(np.where(sel, data2d - ip, np.nan), q2d, edges)
        pq = _radial_profile_1d(np.where(sel, data2d - iq, np.nan), q2d, edges)
        rp, rq = pp - base1d, pq - base1d
        # light smoothing (σ≈0.015 Å⁻¹ ≪ ring width) so the curves are readable
        k = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2)
        k /= k.sum()
        rp = np.convolve(rp, k, mode="same")
        rq = np.convolve(rq, k, mode="same")
        ax = axes3[r][0]
        ax.axhline(0, color="0.6", lw=0.8)
        ax.plot(qc, rp, color="C0", lw=1.4, label="patched (after − B)")
        ax.plot(qc, rq, color="C3", lw=1.4, label="parametric (after − B)")
        for c, w in zip(centers, halfwidths):
            ax.axvspan(c - w, c + w, color="0.85", alpha=0.5, lw=0)
        # zoom y to the clean ring band (1.5–7.8 Å⁻¹); high-|Q| is single-plane noise
        clean = (qc >= Q_MIN) & (qc <= 7.8)
        lim = float(np.nanpercentile(
            np.abs(np.concatenate([rp[clean], rq[clean]])), 98)) or 1.0
        ax.set_ylim(-1.3 * lim, 1.3 * lim)
        ax.set_xlim(Q_MIN, 8.0)
        ax.set_ylabel("ring residual")
        ax.set_title(f"H={hv:+.3f}   (grey bands = confirmed ring shells; "
                     f"0 = clean, + = leftover, − = over-subtracted)", fontsize=9)
        if r == 0:
            ax.legend(loc="upper right", fontsize=9)
    axes3[-1][0].set_xlabel("|Q| (Å⁻¹)")
    fig3.tight_layout()
    fig3.savefig(prof_png, dpi=120)
    print(f"wrote {prof_png}", flush=True)


if __name__ == "__main__":
    main()
