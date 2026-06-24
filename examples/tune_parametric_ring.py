"""Tune + diagnose the parametric ring remover's azimuthal-texture capture.

Motivation
----------
On the 0kl planes the **patched** (``PatchedRadialRingModel``) and **parametric**
(``ParametricRingModel``) removers produce visually similar ``data − ring``
residuals, yet :mod:`compare_ring_models` shows they fail in *opposite*
directions: patched over-subtracts (>100 % at the shell, digs below baseline)
while parametric under-subtracts (~60–75 %, leaves ring behind).  Under-
subtraction is the signature of a **damped azimuthal texture** ``T(φ)`` — the
bright arcs are not given enough amplitude.  This script makes that visible and
finds parameters that fix it.

Two outputs
-----------
1. **Sweep** (stdout): for a grid of ``n_fourier`` × ``texture_ridge`` (and the
   ceiling on/off), the mean per-shell ring-removed % and over-subtract % across
   the H planes.  The goal is removal ≈100 % with over-subtract no worse than
   patched.
2. **Texture overlay PNG**: per (shell, H plane) the *azimuthal* ring excess —
   the data truth ``median_on(φ) − median_off(φ)`` (Bragg-robust) overlaid with
   each model's ``I_ring(φ)``.  This shows directly whether parametric tracks the
   real arc-to-arc texture better than patched, and how the best swept params
   compare to the defaults.

Usage::

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
    python3 examples/tune_parametric_ring.py

Env: ``DATA_FILE``, ``H_VALUES`` (default ``0,0.3333,1.0``),
``Q_MIN``/``Q_MAX`` (default 1.5/10.5), ``SHELLS`` (comma |Q| list for the
overlay; default = the two strongest confirmed shells), ``OUT_PNG``.
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
from nebula3d.analysis import BraggRemover
from nebula3d.preprocessing import (
    ParametricRingModel,
    PatchedRadialRingModel,
    azimuthal_sampling_mask,
    confirm_ring_shells_across_h,
)
from nebula3d.preprocessing.radial_background import (
    _azimuthal_angle,
    _offset_q_magnitude,
)

PLANE = "0kl"
Q_MIN = float(os.environ.get("Q_MIN", "1.5"))
Q_MAX = float(os.environ.get("Q_MAX", "10.5"))

# texture knobs to sweep (parametric only — the fit is ~0.2 s/plane)
N_FOURIER_GRID = [int(x) for x in os.environ.get("NF_GRID", "6,8,12").split(",")]
RIDGE_GRID = [float(x) for x in os.environ.get("RIDGE_GRID", "0.01,0.02,0.05").split(",")]


def _find_input() -> Path:
    df = os.environ.get("DATA_FILE")
    if df:
        return Path(df)
    raw = Path("data/raw")
    cands = sorted(raw.glob("*cc_sub_bkg.nxs")) or sorted(raw.glob("*.nxs"))
    if not cands:
        raise FileNotFoundError("No input .nxs found; set DATA_FILE=…")
    return cands[0]


def _plane_at(vol, h_value: float):
    ih = int(np.argmin(np.abs(vol.h_axis - h_value)))
    sl = (slice(ih, ih + 1), slice(None), slice(None))
    return dataclasses.replace(
        vol, data=vol.data[sl], sigma=vol.sigma[sl], mask=vol.mask[sl],
        h_axis=vol.h_axis[ih:ih + 1]), ih


def _metrics(q2d, data2d, after2d, shells, halfwidths):
    """(mean ring-removed fraction, over-subtract fraction) for one plane."""
    rems = []
    for c, w in zip(shells, halfwidths):
        on = (q2d >= c - w) & (q2d <= c + w)
        off = ((q2d >= c - 3 * w) & (q2d < c - w)) | (
            (q2d > c + w) & (q2d <= c + 3 * w))
        if on.sum() < 10 or off.sum() < 10:
            continue
        base = float(np.nanmedian(after2d[off]))
        b = float(np.nanmedian(data2d[on])) - base
        a = float(np.nanmedian(after2d[on])) - base
        if b > 1e-6:
            rems.append((b - a) / b)
    rng = (q2d > Q_MIN) & (q2d < Q_MAX)
    neg = float(np.mean(after2d[rng] < -0.05))
    return (float(np.mean(rems)) if rems else float("nan")), neg


def _azimuthal_excess(phi2d, val2d, q2d, c, w, nbin, *, model: bool):
    """Per-φ-bin ring excess at shell *c*.

    Data (``model=False``): ``median_on(φ) − median_off(φ)`` — Bragg-robust truth.
    Model I_ring (``model=True``): ``mean_on(φ)`` — already an excess above the
    diffuse baseline by construction, so no off-shell term.
    """
    on = (q2d >= c - w) & (q2d <= c + w) & np.isfinite(val2d)
    ph = phi2d[on]
    v = val2d[on]
    if ph.size < nbin:
        return None, None
    edges = np.linspace(-np.pi, np.pi, nbin + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(ph, edges) - 1, 0, nbin - 1)
    on_stat = np.array([
        np.median(v[idx == b]) if np.any(idx == b) else np.nan
        for b in range(nbin)])
    if model:
        return ctr, on_stat
    off = (((q2d >= c - 3 * w) & (q2d < c - w)) |
           ((q2d > c + w) & (q2d <= c + 3 * w))) & np.isfinite(val2d)
    pho = phi2d[off]
    vo = val2d[off]
    ido = np.clip(np.digitize(pho, edges) - 1, 0, nbin - 1)
    off_stat = np.array([
        np.median(vo[ido == b]) if np.any(ido == b) else np.nan
        for b in range(nbin)])
    return ctr, on_stat - off_stat


def main() -> None:
    in_path = _find_input()
    print(f"loading {in_path.name}", flush=True)
    vol = nebula3d.load(in_path)
    print(f"volume {vol.data.shape}  |Q| fit {(Q_MIN, Q_MAX)}", flush=True)

    t0 = time.time()
    centers, halfwidths, amps = confirm_ring_shells_across_h(
        vol, plane=PLANE, q_range=(Q_MIN, Q_MAX))
    ceilings = 3.0 * amps if amps.size else None
    print(f"confirmed {centers.size} shells in {time.time() - t0:.1f}s\n", flush=True)

    h_values = [float(x) for x in
                os.environ.get("H_VALUES", "0,0.3333,1.0").split(",")]

    # Bragg keep-mask: detect+punch the sharp single-crystal reflections in the
    # FULL 3D volume (the detector needs 3D context — a depth-1 plane finds
    # nothing), then slice per plane.  This mirrors how the 3D driver would apply
    # a punch mask to each per-slice ring fit.
    bragg = BraggRemover(mode="integer", intensity_scale=True,
                         punch_incident_beam=True)
    t = time.time()
    bkeep_full = bragg.build_mask(vol)
    print(f"3D Bragg mask: punched {int((~bkeep_full).sum()):,} voxels in "
          f"{time.time() - t:.1f}s\n", flush=True)

    # pre-extract the masked planes once (+ per-plane Bragg keep-mask slice)
    planes = []
    for hv in h_values:
        pl, ih = _plane_at(vol, hv)
        keep = azimuthal_sampling_mask(pl, plane=PLANE, min_count_frac=0.25,
                                       q_range=(Q_MIN, Q_MAX))
        src = dataclasses.replace(pl, mask=keep)
        q2d = _offset_q_magnitude(pl, PLANE)[0]
        phi2d = _azimuthal_angle(pl, PLANE, (0.0, 0.0), (0.0, 0.0))[0]
        bkeep = bkeep_full[ih:ih + 1]
        inq = (q2d > Q_MIN) & (q2d < Q_MAX)
        punched = int((~bkeep[0] & inq & keep[0]).sum())
        print(f"  H={hv:+.3f}: Bragg-punched {punched:,} of "
              f"{int((inq & keep[0]).sum()):,} in-range voxels", flush=True)
        planes.append((hv, src, pl.data[0], q2d, phi2d, bkeep))

    common = dict(plane=PLANE, q_step=0.02, profile_method="median",
                  allowed_ring_centers=centers,
                  allowed_ring_halfwidths=halfwidths)

    # ---- baseline references: patched + parametric default -------------------
    patched = PatchedRadialRingModel(
        n_fourier=6, texture_q_smooth=0.02, texture_ridge=0.08,
        allowed_ring_ceilings=ceilings, **common)
    param_default = ParametricRingModel(
        n_fourier=8, texture_ridge=0.05, allowed_ring_ceilings=ceilings, **common)

    def _run(model, label, use_bragg_mask=False):
        rem, neg = [], []
        rings = {}
        for hv, src, data2d, q2d, _phi, bkeep in planes:
            kw = {"bragg_keep_mask": bkeep} if use_bragg_mask else {}
            model.fit(src, q_range=(Q_MIN, Q_MAX), **kw)
            _, I_ring = model.subtract(src)
            rings[hv] = I_ring[0]
            r, n = _metrics(q2d, data2d, data2d - I_ring[0], centers, halfwidths)
            rem.append(r)
            neg.append(n)
        per = "  ".join(f"H{hv:+.2f}={r * 100:.0f}%"
                         for (hv, *_), r in zip(planes, rem))
        print(f"  {label:<38} ring removed={np.nanmean(rem) * 100:5.1f}%  "
              f"over-subtract={np.mean(neg) * 100:.2f}%   [{per}]", flush=True)
        return rings, np.nanmean(rem), np.mean(neg)

    print("reference models:")
    rings_patched, _, _ = _run(patched, "patched (nf=6, ridge=0.08)")
    rings_def, _, _ = _run(param_default, "parametric default (nf=8, ridge=0.05)")

    # ---- THE FIX: Bragg-mask before the ring fit, then gentle IRLS -----------
    # The texture knobs (n_fourier, ridge, ceiling) barely move removal, and the
    # high-side IRLS that recovers the amplitude (irls=1) only does so by eating
    # Bragg.  Masking the sharp reflections OUT of the fit lets a gentle IRLS
    # capture the true bright-arc amplitude without the Bragg contamination.
    # Probes pair the mask with a few IRLS settings; baselines (no mask) included
    # so the effect of the mask is isolated.
    probes = [
        ("legacy high-side (spike_reject=off)",
         dict(texture_spike_reject=False), False),
        ("bragg-mask + legacy irls=2",
         dict(texture_spike_reject=False, texture_irls_iter=2), True),
        ("phi-shape reject (NEW default)",
         dict(texture_spike_reject=True), False),
        ("phi-shape + bragg-mask",
         dict(texture_spike_reject=True), True),
    ]
    print("\nparametric Bragg-mask probe  [nf=8, ceiling off]:")
    best = None  # (score, label, rings)
    for label, kw, use_mask in probes:
        m = ParametricRingModel(
            n_fourier=8, allowed_ring_ceilings=None, **{**common, **kw})
        rings, rem, neg = _run(m, label, use_bragg_mask=use_mask)
        score = abs(rem - 1.0) + 2.0 * neg
        if best is None or score < best[0]:
            best = (score, label, rings)

    _, blabel, rings_best = best
    print(f"\nbest parametric probe: {blabel}", flush=True)

    # ---- texture overlay figure ----------------------------------------------
    if os.environ.get("SHELLS"):
        shells = [float(x) for x in os.environ["SHELLS"].split(",")]
        hw = [float(np.interp(s, centers, halfwidths)) for s in shells]
    else:
        # two strongest confirmed shells
        order = np.argsort(amps)[::-1] if amps.size else np.arange(centers.size)
        pick = order[:2]
        shells = [float(centers[i]) for i in pick]
        hw = [float(halfwidths[i]) for i in pick]

    nbin = 48
    nrow = len(planes)
    ncol = len(shells)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 3.2 * nrow),
                             squeeze=False)
    for r, (hv, _src, data2d, q2d, phi2d, _bk) in enumerate(planes):
        for c, (sc, sw) in enumerate(zip(shells, hw)):
            ax = axes[r][c]
            x, y_data = _azimuthal_excess(phi2d, data2d, q2d, sc, sw, nbin, model=False)
            if x is None:
                ax.set_axis_off()
                continue
            ax.plot(x, y_data, color="k", lw=1.6, label="data excess (truth)")
            for rings, lab, style in (
                (rings_patched, "patched", dict(color="tab:blue", ls="--")),
                (rings_def, "param default", dict(color="tab:orange", ls="-")),
                (rings_best, f"param best [{blabel}]",
                 dict(color="tab:green", ls="-")),
            ):
                _, y = _azimuthal_excess(phi2d, rings[hv], q2d, sc, sw, nbin, model=True)
                ax.plot(x, y, lw=1.3, label=lab, **style)
            ax.set_title(f"H={hv:+.3f}  |Q|={sc:.2f} Å⁻¹", fontsize=9)
            ax.set_xlabel("φ (rad)")
            if c == 0:
                ax.set_ylabel("ring excess")
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out_png = Path(os.environ.get(
        "OUT_PNG", "data/processed/tune_parametric_ring_texture.png"))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    print(f"\nwrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
