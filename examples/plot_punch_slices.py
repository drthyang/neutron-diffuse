"""Visual side-by-side of the punched slices: HKL vs Q-anisotropic-adaptive.

Phase-4 validation companion to ``compare_punch_frames.py`` (which reports the
numbers).  Runs the *production* default punch (HKL) and the proposed Q-frame
default (anisotropic Å⁻¹, adaptive) on a real ``*_ringremoved.h5`` volume and
draws, for a few K-L planes:

    data | HKL holes (red) | Q holes (blue) | difference (HKL-only red / Q-only blue)

so the punch *footprint* and the diffuse-plane preservation can be eyeballed.

Run (no venv; see the run-environment note):

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl DATASET=22K \
    /path/to/sci-general/python examples/plot_punch_slices.py
"""

from __future__ import annotations

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

import nebula3d  # noqa: E402
from nebula3d.pipeline import PunchParams, punch_bragg  # noqa: E402

# Proposed Q default = the current HKL footprint expressed in Å⁻¹ (≈ HKL × b*).
Q_RADII = (0.097, 0.072, 0.115)

# (label, H plane, (k0,k1,l0,l1) zoom window in r.l.u.)  — square windows so the
# equal-aspect footprints show their true r.l.u. shape.
PLANES = [
    ("H = 0  (Bragg, tight)", 0.0, (-2.5, 2.5, -2.5, 2.5)),
    ("H = 0  (Bragg, wide)", 0.0, (-9.0, 9.0, -9.0, 9.0)),
    ("H = 1/3  (diffuse plane)", 0.3333, (-2.5, 2.5, -2.5, 2.5)),
]

_RED = ListedColormap([(0.90, 0.10, 0.10, 1.0)])
_BLUE = ListedColormap([(0.20, 0.45, 0.95, 1.0)])


def _overlay(mask2d: np.ndarray) -> np.ndarray:
    """1.0 where punched, NaN elsewhere (so imshow draws only the holes)."""
    return np.where(mask2d, 1.0, np.nan)


def main() -> None:
    tag = os.environ.get("DATASET", "22K")
    paths = sorted(p for p in glob.glob(f"data/processed/*{tag}*ringremoved.h5")
                   if "braggpunched" not in p)
    if not paths:
        raise SystemExit(f"no *_ringremoved.h5 for DATASET={tag}")
    vol = nebula3d.load(paths[0])
    valid = vol.mask & np.isfinite(vol.data)

    punched_hkl = valid & ~punch_bragg(vol, PunchParams()).mask
    punched_q = valid & ~punch_bragg(
        vol, PunchParams(punch_frame="q", punch_q_radii=Q_RADII)).mask
    print(f"punched: HKL={int(punched_hkl.sum()):,}  Q={int(punched_q.sum()):,}")

    k, l = vol.k_axis, vol.l_axis
    cols = ["data", "HKL punch (red)", "Q punch (blue)", "diff: HKL-only / Q-only"]
    fig, axes = plt.subplots(len(PLANES), 4,
                             figsize=(16, 4.2 * len(PLANES)), squeeze=False)

    for r, (title, hval, zoom) in enumerate(PLANES):
        ih = int(np.argmin(np.abs(vol.h_axis - hval)))
        data = vol.data[ih]
        ph, pq = punched_hkl[ih], punched_q[ih]
        finite = data[np.isfinite(data)]
        vmax = float(np.percentile(finite[finite > 0], 98)) if finite.size else 1.0
        extent = [l[0], l[-1], k[0], k[-1]]

        def _bg(ax, dim=1.0):
            ax.imshow(data, origin="lower", extent=extent, aspect="equal",
                      cmap="gray", vmin=0, vmax=vmax * dim)
            ax.set_xlim(zoom[2], zoom[3])
            ax.set_ylim(zoom[0], zoom[1])
            ax.set_xlabel("L"); ax.set_ylabel("K")

        for c in range(4):
            ax = axes[r][c]
            # dim the data under the diff panel so the colours read clearly
            _bg(ax, dim=2.5 if c == 3 else 1.0)
            if c == 1:
                ax.imshow(_overlay(ph), origin="lower", extent=extent,
                          aspect="equal", cmap=_RED, alpha=0.6)
            elif c == 2:
                ax.imshow(_overlay(pq), origin="lower", extent=extent,
                          aspect="equal", cmap=_BLUE, alpha=0.6)
            elif c == 3:
                ax.imshow(_overlay(ph & ~pq), origin="lower", extent=extent,
                          aspect="equal", cmap=_RED, alpha=0.95)
                ax.imshow(_overlay(pq & ~ph), origin="lower", extent=extent,
                          aspect="equal", cmap=_BLUE, alpha=0.95)
            if r == 0:
                ax.set_title(cols[c], fontsize=11)
        axes[r][0].text(0.01, 0.99, title, transform=axes[r][0].transAxes,
                        va="top", ha="left", color="w", fontsize=11,
                        bbox=dict(facecolor="black", alpha=0.5, pad=2))

    fig.suptitle(
        f"Punched slices — {os.path.basename(paths[0])[:42]}\n"
        f"HKL default vs Q-anisotropic-adaptive  (Q radii {Q_RADII} Å⁻¹)",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.environ.get("OUT", f"/tmp/punch_slices_{tag}.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
