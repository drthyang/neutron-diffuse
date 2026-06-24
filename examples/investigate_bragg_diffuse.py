"""Investigate Bragg vs co-located diffuse scattering.

Diagnostic for the case where diffuse scattering sits at or near sharp Bragg or
satellite features. Punch + backfill would destroy that diffuse, so before
separating we measure the peak shape on line cuts:

    1. Calibrate the instrument resolution σ(|Q|) on resolution-limited *nuclear*
       (integer-node) Bragg peaks — these set the sharp-core width.
    2. Locate satellite candidates from configurable fractional-H planes and any
       auto-detected node that carries a broad component.
    3. Decompose each along H, K, L into a sharp (resolution) Gaussian core + a
       broad (diffuse) Lorentzian / squared-Lorentzian, keeping the better AIC.
    4. Report per-axis points-across-FWHM (which axes even resolve the core),
       correlation length ξ = 1/κ, and the diffuse fraction.

Outputs (in OUTDIR, default examples/):
    _investigate_bragg_diffuse_<label>.png   per-satellite 3-axis cuts + fits
    _investigate_bragg_summary_<label>.png   resolution σ(|Q|) + points/FWHM + ξ/frac
    _investigate_bragg_diffuse_<label>.csv   the full per-axis table
    _investigate_bragg_series.png            (SERIES=1) broad component overlay

Run::

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \\
      python3 examples/investigate_bragg_diffuse.py

Env:
    DATA_FILE      ring-removed .h5 (default: auto-detect *_ringremoved.h5)
    DATASET        substring used to pick one ringremoved file
    SERIES         1 → process all detected files + the series overlay
    FRACTIONS      comma list of H fractional parts (default 0.3333,0.6667)
    H_HALF_WIDTH   half-width (rlu) around each fraction plane (default 0.08)
    MAX_SATELLITES detailed cut panels to draw (default 6)
    HALF_WINDOW    cut half-length in rlu (default 0.6)
    RES_MAX_PEAKS  nuclear peaks used for resolution calibration (default 40)
    BROAD          auto | lorentzian | squared_lorentzian (default auto)
    OUTDIR         output directory (default: this script's dir)
"""
import matplotlib

matplotlib.use("Agg")

import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import nebula3d

# Import the new diagnostic straight from its module (not via nebula3d.analysis) so
# this feature stays fully self-contained — it adds zero surface to the existing
# nebula3d.analysis package and no existing import path loads it.
from nebula3d.analysis.peak_profile import (
    AXIS_NAMES,
    build_interpolator,
    calibrate_resolution,
    decompose_peak,
    evaluate_components,
    extract_orthogonal_cuts,
    magnetic_satellite_centers,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PROC = REPO / "data" / "processed"
OUTDIR = Path(os.environ.get("OUTDIR", str(HERE)))

FRACTIONS = tuple(float(x) for x in os.environ.get("FRACTIONS", "0.3333,0.6667").split(","))
H_HALF_WIDTH = float(os.environ.get("H_HALF_WIDTH", "0.08"))
MAX_SATELLITES = int(os.environ.get("MAX_SATELLITES", "6"))
HALF_WINDOW = float(os.environ.get("HALF_WINDOW", "0.6"))
RES_MAX_PEAKS = int(os.environ.get("RES_MAX_PEAKS", "40"))
_BROAD = os.environ.get("BROAD", "auto")
BROAD = None if _BROAD == "auto" else _BROAD


def _label_of(path: Path) -> str:
    stem = path.stem
    return stem[:-len("_ringremoved")] if stem.endswith("_ringremoved") else stem


def _detect_ringremoved() -> list[Path]:
    env = os.environ.get("DATA_FILE")
    if env:
        return [Path(env)]
    cands = sorted(PROC.glob("*_ringremoved.h5"))
    cands = [c for c in cands if "braggpunched" not in c.name]  # the raw ring-removed stage
    dataset = os.environ.get("DATASET")
    if dataset:
        cands = [c for c in cands if dataset in c.name]
    if os.environ.get("SERIES", "0") != "1" and cands:
        cands = [cands[0]]
    return cands


# ---------------------------------------------------------------------------
# Per-volume analysis
# ---------------------------------------------------------------------------
def analyse(path: Path):
    label = _label_of(path)
    print(f"\n{'='*70}\n▶ {label}: {path.name}\n{'='*70}", flush=True)
    vol = nebula3d.load(path)
    print(f"  shape (H,K,L) = {vol.data.shape}", flush=True)
    interp = build_interpolator(vol)   # one masked-volume copy, shared by all cuts

    print("  calibrating resolution on nuclear Bragg ...", flush=True)
    res = calibrate_resolution(vol, max_peaks=RES_MAX_PEAKS, interp=interp)
    for a in range(3):
        print(f"    σ_{AXIS_NAMES[a]}(|Q|) = {res.intercept[a]:.4f} + "
              f"{res.slope[a]:.4f}·|Q|  ({res.n_ref[a]} refs)", flush=True)

    print("  locating satellite candidates ...", flush=True)
    centers = magnetic_satellite_centers(
        vol, fractions=FRACTIONS, h_half_width=H_HALF_WIDTH, max_peaks=40,
    )
    print(f"    found {len(centers)} satellite candidates", flush=True)

    rows = []
    decomps = []
    for c in centers:
        dec = decompose_peak(vol, c, res, half_window=HALF_WINDOW, broad=BROAD, interp=interp)
        decomps.append((c, dec))
        for a in range(3):
            d = dec[a]
            rows.append({
                "label": label, "h": round(c[0], 4), "k": round(c[1], 4), "l": round(c[2], 4),
                "axis": AXIS_NAMES[a], "q": round(d.q_center, 4),
                "sharp_fwhm_rlu": round(d.sharp_fwhm, 4),
                "broad_fwhm_rlu": round(d.broad_fwhm, 4),
                "pts_across_fwhm": round(d.points_across_fwhm, 2),
                "xi_angstrom": round(d.xi_angstrom, 2),
                "diffuse_fraction": round(d.diffuse_fraction, 3),
                "broad_shape": d.broad_shape, "is_diffuse": d.is_diffuse,
                "success": d.success,
            })
    return label, vol, res, decomps, rows, interp


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_satellite_cuts(label, vol, res, decomps, interp):
    sats = decomps[:MAX_SATELLITES]
    if not sats:
        return
    n = len(sats)
    fig, axes = plt.subplots(n, 3, figsize=(13, 3.2 * n), squeeze=False)
    for r, (center, dec) in enumerate(sats):
        cuts = extract_orthogonal_cuts(vol, center, half_window=HALF_WINDOW,
                                       n_points=241, interp=interp)
        for a in range(3):
            ax = axes[r][a]
            coord, inten = cuts[a]
            d = dec[a]
            ax.plot(coord, inten, ".", ms=3, color="0.4", label="data")
            if d.success:
                xf = np.linspace(coord.min(), coord.max(), 400)
                sharp, broad, base, total = evaluate_components(d, xf)
                ax.plot(xf, total, "-", color="C3", lw=1.6, label="total")
                ax.plot(xf, base + sharp, "-", color="C0", lw=1.1, label="sharp (Bragg)")
                ax.plot(xf, base + broad, "-", color="C2", lw=1.1, label="broad (diffuse)")
            ax.set_yscale("log")
            ax.set_xlabel(f"{AXIS_NAMES[a]} (rlu)")
            if a == 0:
                ax.set_ylabel("I")
            title = (f"({center[0]:.3f},{center[1]:.2f},{center[2]:.2f})  "
                     f"|Q|={d.q_center:.2f}\n"
                     f"frac={d.diffuse_fraction:.2f}  ξ={d.xi_angstrom:.1f}Å  "
                     f"pts/FWHM={d.points_across_fwhm:.1f}")
            ax.set_title(title, fontsize=8)
            if r == 0 and a == 2:
                ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{label}: satellite line cuts — sharp (Bragg) vs broad (diffuse)",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    out = OUTDIR / f"_investigate_bragg_diffuse_{label}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}", flush=True)


def plot_summary(label, vol, res, rows):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # (1) resolution σ(|Q|) per axis with the reference points
    ax = axes[0]
    for a in range(3):
        qa, sa = res.q_ref[a], res.sigma_ref[a]
        if qa.size:
            ax.plot(qa, sa, ".", color=f"C{a}", ms=4, alpha=0.6)
            qline = np.linspace(0, max(float(qa.max()), 1.0), 50)
            ax.plot(qline, res.intercept[a] + res.slope[a] * qline, "-",
                    color=f"C{a}", lw=1.4, label=f"σ_{AXIS_NAMES[a]}")
    ax.set_xlabel("|Q| (Å⁻¹)")
    ax.set_ylabel("resolution σ (rlu)")
    ax.set_title("Resolution from nuclear Bragg")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ok = [r for r in rows if r["success"]]
    # (2) points-across-FWHM per axis — which axes resolve the core
    ax = axes[1]
    for a in range(3):
        vals = [r["pts_across_fwhm"] for r in ok if r["axis"] == AXIS_NAMES[a]]
        if vals:
            ax.scatter([a] * len(vals), vals, color=f"C{a}", alpha=0.5, s=18)
    ax.axhline(3.0, ls="--", color="gray", lw=0.8, label="≈resolved (3 pts)")
    ax.set_xticks(range(3))
    ax.set_xticklabels(AXIS_NAMES)
    ax.set_ylabel("sharp FWHM / grid step")
    ax.set_title("Points across the Bragg core")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (3) diffuse fraction vs ξ, coloured by axis
    ax = axes[2]
    for a in range(3):
        fr = [r["diffuse_fraction"] for r in ok if r["axis"] == AXIS_NAMES[a]]
        xi = [r["xi_angstrom"] for r in ok if r["axis"] == AXIS_NAMES[a]]
        xi = [min(v, 60) for v in xi]   # clip ∞ for display
        if fr:
            ax.scatter(xi, fr, color=f"C{a}", alpha=0.6, s=18, label=AXIS_NAMES[a])
    ax.set_xlabel("ξ = 1/κ (Å)")
    ax.set_ylabel("diffuse fraction")
    ax.set_title("Diffuse share vs correlation length")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"{label}: Bragg/diffuse separation diagnostics", fontsize=12)
    fig.tight_layout()
    out = OUTDIR / f"_investigate_bragg_summary_{label}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}", flush=True)


def write_csv(label, rows):
    out = OUTDIR / f"_investigate_bragg_diffuse_{label}.csv"
    if not rows:
        return
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {out.name}  ({len(rows)} rows)", flush=True)


def print_table(label, rows):
    diffuse = [r for r in rows if r["is_diffuse"]]
    print(f"\n  {label}: {len(diffuse)}/{len(rows)} (node,axis) entries flagged diffuse "
          f"(broad, >2× resolution, >20% area)", flush=True)
    hdr = f"    {'node':>20} {'ax':>2} {'|Q|':>5} {'sharpFWHM':>9} {'broadFWHM':>9} "\
          f"{'pts/FW':>6} {'xi(Å)':>6} {'frac':>5} {'shape':>17}"
    print(hdr, flush=True)
    for r in rows[:30]:
        node = f"({r['h']:.3f},{r['k']:.2f},{r['l']:.2f})"
        flag = "*" if r["is_diffuse"] else " "
        print(f"  {flag} {node:>20} {r['axis']:>2} {r['q']:>5.2f} "
              f"{r['sharp_fwhm_rlu']:>9.3f} {r['broad_fwhm_rlu']:>9.3f} "
              f"{r['pts_across_fwhm']:>6.1f} {r['xi_angstrom']:>6.1f} "
              f"{r['diffuse_fraction']:>5.2f} {r['broad_shape']:>17}", flush=True)
    if len(rows) > 30:
        print(f"    … {len(rows)-30} more rows in the CSV", flush=True)


def _best_axis(dec):
    """Axis index with the strongest *flagged* diffuse, else the max-fraction axis."""
    flagged = [(a, dec[a].diffuse_fraction) for a in range(3) if dec[a].is_diffuse]
    if flagged:
        return max(flagged, key=lambda t: t[1])[0]
    return max(range(3), key=lambda a: dec[a].diffuse_fraction)


def plot_series(per_items):
    """Overlay the broad (diffuse) component for a fixed satellite candidate.

    Picks the satellite with the strongest separable diffuse in the first
    dataset, then re-decomposes that **same (h,k,l) position** (no recentering)
    in every loaded file so the comparison holds Q fixed.
    """
    series = list(per_items)
    _ref_label, _ref_vol, _ref_res, ref_decomps, _ref_interp = series[0]

    best = None
    for center, dec in ref_decomps:
        a = _best_axis(dec)
        if dec[a].is_diffuse and (best is None or dec[a].diffuse_fraction > best[0]):
            best = (dec[a].diffuse_fraction, center, a)
    if best is None:
        print("  [series] no flagged-diffuse satellite in the reference dataset — skipped",
              flush=True)
        return
    _, center0, axis = best

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, vol, res, _decomps, interp) in enumerate(series):
        # hold the position fixed (refine=False) so every file is the same Q
        dec = decompose_peak(vol, center0, res, half_window=HALF_WINDOW,
                             broad=BROAD, interp=interp, refine=False)[axis]
        coord, inten = extract_orthogonal_cuts(
            vol, center0, half_window=HALF_WINDOW, n_points=241, interp=interp)[axis]
        color = f"C{i}"
        ax.plot(coord, inten, ".", ms=3, color=color, alpha=0.45)
        if dec.success:
            xf = np.linspace(coord.min(), coord.max(), 400)
            _, broad, base, total = evaluate_components(dec, xf)
            tag = (f"{label}: frac={dec.diffuse_fraction:.2f} ξ={dec.xi_angstrom:.1f}Å"
                   if dec.is_diffuse else f"{label}: no diffuse")
            ax.plot(xf, total, "-", color=color, lw=1.5, label=tag)
            ax.plot(xf, base + broad, "--", color=color, lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel(f"{AXIS_NAMES[axis]} (rlu)")
    ax.set_ylabel("I")
    ax.set_title(
        f"Diffuse component comparison at "
        f"({center0[0]:.3f},{center0[1]:.2f},{center0[2]:.2f}) — {AXIS_NAMES[axis]} cut\n"
        "(— total fit, -- broad/diffuse component)"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = OUTDIR / "_investigate_bragg_series.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out.name}", flush=True)


# ---------------------------------------------------------------------------
def main():
    files = _detect_ringremoved()
    if not files:
        sys.exit("No *_ringremoved.h5 in data/processed/ — run the pipeline first, "
                 "or set DATA_FILE=/path/to/file.h5.")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    per_items = []
    for path in files:
        if not path.exists():
            sys.exit(f"input not found: {path}")
        label, vol, res, decomps, rows, interp = analyse(path)
        plot_satellite_cuts(label, vol, res, decomps, interp)
        plot_summary(label, vol, res, rows)
        print_table(label, rows)
        write_csv(label, rows)
        per_items.append((label, vol, res, decomps, interp))

    if os.environ.get("SERIES", "0") == "1" and len(per_items) > 1:
        plot_series(per_items)
    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
