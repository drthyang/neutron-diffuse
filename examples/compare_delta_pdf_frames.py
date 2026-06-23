"""ΔPDF-level A/B: HKL vs Q-anisotropic-adaptive punch through the full pipeline.

The final gate for ROADMAP Phase 6 / Phase 4: does the ~8% Bragg-punch difference
between the HKL default and the proposed Q-frame default actually change the
real-space 3D-ΔPDF?  From one ring-removed volume this runs
punch → backfill → flatten → ΔPDF *both ways* and compares the resulting maps:

* global metrics — relative RMS difference and Pearson correlation,
* an orthoslice figure (3 planes × HKL | Q | difference), same diverging scale,
  so a real change in a correlation feature would stand out against the data.

Run (no venv; see the run-environment note):

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl DATASET=22K \
    /path/to/sci-general/python examples/compare_delta_pdf_frames.py
"""

from __future__ import annotations

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import nebula3d  # noqa: E402
from nebula3d.pipeline import (  # noqa: E402
    BackfillParams,
    DeltaPdfParams,
    FlattenParams,
    PunchParams,
    backfill,
    delta_pdf,
    flatten,
    punch_bragg,
)

Q_RADII = (0.097, 0.072, 0.115)
WINDOW_A = 25.0  # half-window (Å) shown per orthoslice


def run_arm(vol: nebula3d.HKLVolume, punch_params: PunchParams):
    """punch → backfill → flatten → ΔPDF, returning the DeltaPDF."""
    v = punch_bragg(vol, punch_params)
    v = backfill(v, BackfillParams())
    v = flatten(v, FlattenParams())
    return delta_pdf(v, DeltaPdfParams())


def _crop(s: np.ndarray, ar: np.ndarray, ac: np.ndarray, w: float):
    ri, ci = np.abs(ar) <= w, np.abs(ac) <= w
    return s[np.ix_(ri, ci)], ar[ri], ac[ci]


def main() -> None:
    tag = os.environ.get("DATASET", "22K")
    paths = sorted(p for p in glob.glob(f"data/processed/*{tag}*ringremoved.h5")
                   if "braggpunched" not in p)
    if not paths:
        raise SystemExit(f"no *_ringremoved.h5 for DATASET={tag}")
    vol = nebula3d.load(paths[0])

    print("running HKL arm …")
    a = run_arm(vol, PunchParams())
    print("running Q-anisotropic-adaptive arm …")
    b = run_arm(vol, PunchParams(punch_frame="q", punch_q_radii=Q_RADII))

    da, db = a.data, b.data
    diff = db - da
    rms = float(np.linalg.norm(diff) / (np.linalg.norm(da) + 1e-30))
    corr = float(np.corrcoef(da.ravel(), db.ravel())[0, 1])
    rel_max = float(np.max(np.abs(diff)) / (np.max(np.abs(da)) + 1e-30))
    print(f"\nΔPDF HKL vs Q:  relative RMS diff = {rms:.4f}   "
          f"Pearson r = {corr:.5f}   max|Δ|/max|HKL| = {rel_max:.4f}")

    # ---- orthoslice figure ------------------------------------------------
    planes = [
        ("x_H – y_K", a.slice_hk0(), b.slice_hk0(), a.x_axis, a.y_axis),
        ("x_H – z_L", a.slice_h0l(), b.slice_h0l(), a.x_axis, a.z_axis),
        ("y_K – z_L", a.slice_0kl(), b.slice_0kl(), a.y_axis, a.z_axis),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(13, 12), squeeze=False)
    cols = ["HKL punch", "Q punch", "difference (Q − HKL)"]
    for r, (title, sa, sb, ar, ac) in enumerate(planes):
        ca, ar2, ac2 = _crop(sa, ar, ac, WINDOW_A)
        cb, _, _ = _crop(sb, ar, ac, WINDOW_A)
        vmax = float(np.percentile(np.abs(ca), 99.5)) or 1.0
        ext = [ac2[0], ac2[-1], ar2[0], ar2[-1]]
        for c, img in enumerate([ca, cb, cb - ca]):
            ax = axes[r][c]
            ax.imshow(img, origin="lower", extent=ext, aspect="equal",
                      cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            if r == 0:
                ax.set_title(cols[c], fontsize=11)
            ax.set_xlabel("Å"); ax.set_ylabel("Å")
        axes[r][0].text(0.02, 0.98, title, transform=axes[r][0].transAxes,
                        va="top", color="k", fontsize=11,
                        bbox=dict(facecolor="white", alpha=0.7, pad=2))

    fig.suptitle(
        f"3D-ΔPDF A/B — {os.path.basename(paths[0])[:42]}\n"
        f"HKL vs Q-anisotropic-adaptive punch · relRMS={rms:.3f} · r={corr:.4f} "
        f"· diff at same colour scale", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.environ.get("OUT", f"/tmp/dpdf_ab_{tag}.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
