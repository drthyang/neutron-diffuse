"""Robustness validation for the isotropic radial-background flatten.

The QA in ``flatten_background_3d.py`` reports the per-shell *median* spread
before vs after.  That check is nearly **circular**: the flatten subtracts a
curve derived from each shell, so the per-shell statistic it is built from is
guaranteed to collapse.  It says little about whether the background is *really*
removed or whether real features are altered.

This script answers the two questions the user actually cares about, with
checks that are **not** tautologically satisfied by the subtraction itself:

1.  **Is what we subtract genuinely isotropic background?**  The flatten removes
    a single level ``bg(|Q|)`` from every voxel in a shell.  That is only the
    right thing to do if the background is azimuthally flat within the shell.
    We test it directly: split each shell into 8 Cartesian octants, estimate the
    floor in each, and report the spread across octants vs |Q|.  Small spread →
    the isotropy assumption holds and a radial-only subtraction is valid; large
    spread → the "background" has direction structure that this stage cannot
    flatten (and would mislabel as signal/over-subtract).

2.  **Does it alter real features / over- or under-subtract?**  Removal is purely
    radial (``bg(|Q|)`` depends only on |Q|), so it provably removes the *same*
    amount from a feature voxel and a background voxel at equal |Q|.  The only
    ways it can hurt are (a) ``bg(|Q|)`` set too high → the background population
    is driven negative and real intensity is removed, or (b) set too low → a
    residual radial pedestal survives.  We measure the **background-population
    residual** after subtraction per shell (should sit at ≈0), the **negative
    fraction** and its depth vs the noise σ, and the **feature contrast** of the
    strongest anisotropic voxels (should stay strongly positive).

It also characterises each shell's distribution shape (floor↔median gap, high-
tail asymmetry) so the estimator choice (floor / mode / median) can be made from
the data rather than assumed, and prints a structured PASS/FLAG report.

Run::

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python3 examples/validate_flatten.py

Env:
    DATA_FILE   backfilled input .h5 (default: auto-detect 22K *_backfilled.h5)
    ESTIMATOR FLOOR_PCT Q_STEP SMOOTH MIN_COUNT   flatten knobs (production defaults)
    Q_MIN Q_MAX   restrict the validated |Q| range
    NO_PLOT     1 -> skip the QA PNG
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import nebula3d
from nebula3d.preprocessing import flatten_radial_background

HERE = Path(__file__).resolve().parent
PROC = Path("data/processed")


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------
def _percentiles_per_bin(
    values: np.ndarray, bin_idx: np.ndarray, n_bins: int, pcts: np.ndarray,
    min_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin percentiles + counts via one stable sort (segmented scan).

    Returns ``(out[n_bins, len(pcts)], counts[n_bins])``; bins below
    ``min_count`` are NaN.
    """
    order = np.argsort(bin_idx, kind="stable")
    sb = bin_idx[order]
    sv = values[order]
    bounds = np.searchsorted(sb, np.arange(n_bins + 1))
    out = np.full((n_bins, pcts.size), np.nan)
    counts = np.diff(bounds)
    for b in range(n_bins):
        seg = sv[bounds[b]:bounds[b + 1]]
        if seg.size >= min_count:
            out[b] = np.percentile(seg, pcts)
    return out, counts


def _shell_means_after(
    after: np.ndarray, bin_idx: np.ndarray, n_bins: int, lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:
    """Mean of the *background population* (values within [lo, hi] per shell) of
    the flattened data — an estimate of the residual background level per shell.

    ``lo``/``hi`` are per-shell bounds (the before p10/p60 here), shifted by the
    same subtraction, so this isolates the bulk that was background and asks
    where it now sits (≈0 = correctly removed; <0 = over-subtracted).
    """
    out = np.full(n_bins, np.nan)
    order = np.argsort(bin_idx, kind="stable")
    sb = bin_idx[order]
    sv = after[order]
    bounds = np.searchsorted(sb, np.arange(n_bins + 1))
    for b in range(n_bins):
        seg = sv[bounds[b]:bounds[b + 1]]
        if seg.size and np.isfinite(lo[b]) and np.isfinite(hi[b]):
            bulk = seg[(seg >= lo[b]) & (seg <= hi[b])]
            if bulk.size:
                out[b] = float(np.mean(bulk))
    return out


# ----------------------------------------------------------------------
# load
# ----------------------------------------------------------------------
data_file = os.environ.get("DATA_FILE")
if data_file:
    in_path = Path(data_file)
else:
    cands = [p for p in sorted(PROC.glob("*_backfilled.h5")) if "flattened" not in p.name]
    pref = [p for p in cands if "22K" in p.name]
    if not cands:
        raise FileNotFoundError("No *_backfilled.h5 in data/processed; set DATA_FILE.")
    in_path = (pref or cands)[0]

estimator = os.environ.get("ESTIMATOR", "floor")
floor_pct = float(os.environ.get("FLOOR_PCT", "25"))
q_step = float(os.environ.get("Q_STEP", "0.05"))
smooth = float(os.environ.get("SMOOTH", "0.10"))
min_count = int(os.environ.get("MIN_COUNT", "20"))
q_min = os.environ.get("Q_MIN")
q_max = os.environ.get("Q_MAX")
q_range = (float(q_min), float(q_max)) if q_min and q_max else None

print(f"loading {in_path.name}", flush=True)
vol = nebula3d.load(in_path)
q = vol.q_magnitude()
valid = vol.mask & np.isfinite(vol.data)
print(f"  shape={vol.shape}  valid={valid.mean()*100:.1f}%  "
      f"|Q| {q[valid].min():.3f}..{q[valid].max():.3f}", flush=True)

# ----------------------------------------------------------------------
# run the flatten (production estimator) — gives the curve actually subtracted
# ----------------------------------------------------------------------
res = flatten_radial_background(
    vol, q_step=q_step, estimator=estimator, floor_percentile=floor_pct,
    smooth=smooth, min_count=min_count, q_range=q_range,
)
qg = res.q_grid
nb = qg.size
edges = np.concatenate([qg - 0.5 * q_step, qg[-1:] + 0.5 * q_step])
bin_idx_full = np.clip(np.digitize(q, edges) - 1, 0, nb - 1)

# flat views over valid voxels
fb = bin_idx_full[valid]
before = vol.data[valid].astype(np.float64)
after = res.volume.data[valid].astype(np.float64)

# ----------------------------------------------------------------------
# (A) per-shell distribution shape  (before)
# ----------------------------------------------------------------------
pcts = np.array([1, 5, 10, 25, 50, 60, 75, 90, 95, 99], dtype=float)
P, counts = _percentiles_per_bin(before, fb, nb, pcts, min_count)
col = {int(p): i for i, p in enumerate(pcts)}
p5, p10, p25 = P[:, col[5]], P[:, col[10]], P[:, col[25]]
p50, p60, p75 = P[:, col[50]], P[:, col[60]], P[:, col[75]]
p95 = P[:, col[95]]

floor_med_gap = p50 - p25                       # pedestal the floor leaves behind
# high-tail asymmetry: upper half vs lower half of the bulk (robust, noise-safe)
denom = np.where((p50 - p5) > 1e-9, p50 - p5, np.nan)
tail_ratio = (p95 - p50) / denom                # >~1.5 => real anisotropic high tail

# ----------------------------------------------------------------------
# (B) isotropy test — per (shell x octant) floor spread
# ----------------------------------------------------------------------
H, K, L = vol.hkl_grid()
qc = np.stack([H, K, L], axis=-1) @ vol.ub_matrix.T
oct_idx_full = (
    (qc[..., 0] > 0).astype(np.int64) * 4
    + (qc[..., 1] > 0).astype(np.int64) * 2
    + (qc[..., 2] > 0).astype(np.int64)
)
del H, K, L, qc
so = oct_idx_full[valid]
combo = fb * 8 + so                              # shell x octant
Poct, coct = _percentiles_per_bin(
    before, combo, nb * 8, np.array([floor_pct]), max(min_count // 4, 5)
)
oct_floor = Poct[:, 0].reshape(nb, 8)            # [shell, octant]
with np.errstate(invalid="ignore"):
    oct_spread = np.nanmax(oct_floor, axis=1) - np.nanmin(oct_floor, axis=1)
n_oct = np.sum(np.isfinite(oct_floor), axis=1)

# Coverage / reliability: a shell sampled by only a few voxels or a few octants
# gives an unreliable floor (the box corners at high |Q| are a biased azimuthal
# wedge).  Flag the |Q| above which coverage degrades so the curve there is
# read as smoothing-dominated extrapolation, not a measured level.
full_cov = (n_oct >= 8) & (counts >= max(min_count, 200))
q_reliable = float(qg[full_cov].max()) if full_cov.any() else float("nan")
frac_beyond = float(np.mean(q[valid] > q_reliable)) if np.isfinite(q_reliable) else 0.0

# ----------------------------------------------------------------------
# (C) background-population residual after subtraction (≈0 if correct)
# ----------------------------------------------------------------------
shift = res.bg_curve                             # subtracted per shell (≈ smoothed floor)
lo_after = p10 - shift                           # background bulk window, shifted
hi_after = p60 - shift
bg_resid_after = _shell_means_after(after, fb, nb, lo_after, hi_after)

# ----------------------------------------------------------------------
# (D) negatives & feature contrast
# ----------------------------------------------------------------------
neg_frac = float(np.mean(after < 0.0))
sig = vol.sigma[valid].astype(np.float64)
sig_med = float(np.median(sig[np.isfinite(sig) & (sig > 0)])) if np.isfinite(sig).any() else np.nan
deep_neg = float(np.mean(after < -3.0 * sig_med)) if np.isfinite(sig_med) else float("nan")

# strongest anisotropic features: top voxels by contrast-above-floor (before).
floor_at = np.interp(q, qg, res.raw_levels if np.isfinite(res.raw_levels).all()
                     else res.bg_curve, left=res.bg_curve[0], right=res.bg_curve[-1])
contrast_before = before - floor_at[valid]
k_top = max(1000, before.size // 100000)
top = np.argpartition(contrast_before, -k_top)[-k_top:]
# retention of contrast above the *subtracted* background, feature vs its shell
feat_after = after[top]
feat_keep = float(np.median(feat_after / np.maximum(contrast_before[top], 1e-9)))

# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------
m = counts >= min_count
bg_span = float(np.nanmax(res.bg_curve) - np.nanmin(res.bg_curve)) or 1.0
median_gap_level = float(np.nanmedian(floor_med_gap[m]))
resid_level = float(np.nanmedian(np.abs(bg_resid_after[m])))
resid_trend = float(np.nanstd(bg_resid_after[m]))
iso_rel = float(np.nanmedian((oct_spread / np.maximum(np.abs(shift), 1e-9))[m]))
frac_tailed = float(np.mean(tail_ratio[m] > 1.5))

print("\n================  FLATTEN ROBUSTNESS REPORT  ================", flush=True)
print(f"estimator={estimator}  floor_pct={floor_pct}  q_step={q_step}  smooth={smooth}")
print(f"shells used: {int(m.sum())}/{nb}   bg(|Q|) span: {bg_span:.4g}")
print("\n[1] Isotropy of the subtracted background (the key assumption)")
print(f"    octant-floor spread / |bg|, median over shells : {iso_rel:.3f}")
print("    (≲0.3 good: bg is ~azimuthally flat; ≫0.5 means anisotropic bg)")
print("\n[2] Background removal completeness (non-circular)")
print(f"    bulk-centre offset after |mean|, median over shells  : {resid_level:.4g}")
print(f"    bulk-centre offset after std over shells (|Q|-trend) : {resid_trend:.4g}")
print(f"    floor↔median gap left by estimator (median)          : {median_gap_level:.4g}")
print(f"    bg span being removed                                : {bg_span:.4g}")
print("\n[2b] High-|Q| coverage (estimate reliability)")
print(f"    |Q| reliably sampled (≥8 octants, ≥200 vox/shell)    : {q_reliable:.2f} Å⁻¹")
print(f"    valid voxels beyond that (smoothing-extrapolated bg) : {frac_beyond*100:.2f}%")
print("\n[3] Distribution shape")
print(f"    shells with a real anisotropic high tail (ratio>1.5) : {frac_tailed*100:.0f}%")
print("\n[4] Over-subtraction / feature preservation")
print(f"    negative fraction after                : {neg_frac*100:.1f}%")
print(f"    deep negatives (< -3 sigma_med={sig_med:.3g}) : {deep_neg*100:.2f}%")
print(f"    strong-feature contrast retained (med) : {feat_keep*100:.0f}%")

flags = []
if iso_rel > 0.5:
    flags.append("anisotropic background — radial-only subtraction insufficient")
if resid_level > 0.25 * bg_span:
    flags.append("residual pedestal: estimator under-removes background")
if neg_frac > 0.45:
    flags.append("excessive negatives: estimator may over-subtract")
if deep_neg > 0.02:
    flags.append("deep negatives vs noise: likely over-subtraction")
if feat_keep < 0.9:
    flags.append("feature contrast lost: real signal altered")
print("\n[SUMMARY]", "PASS — no robustness flags" if not flags else "FLAGS:")
for f in flags:
    print(f"   ⚠ {f}")
print("=============================================================\n", flush=True)

# ----------------------------------------------------------------------
# QA figure
# ----------------------------------------------------------------------
if os.environ.get("NO_PLOT", "0") != "1":
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        a = ax[0, 0]
        a.plot(qg[m], p50[m], color="0.4", lw=1.3, label="shell median (before)")
        a.fill_between(qg[m], p10[m], p75[m], color="0.8", label="p10–p75 (before)")
        a.plot(qg[m], res.bg_curve[m], color="#d9892a", lw=1.8, label="subtracted bg(|Q|)")
        a.plot(qg[m], bg_resid_after[m], color="#2e9e6b", lw=1.3, label="bg residual (after)")
        a.axhline(0, color="0.8", lw=0.8, zorder=0)
        a.set_title("background curve & residual"); a.set_xlabel("|Q| (Å⁻¹)")
        a.legend(fontsize=8, frameon=False)

        a = ax[0, 1]
        a.plot(qg[m], (oct_spread / np.maximum(np.abs(shift), 1e-9))[m],
               color="#3b6fb0", lw=1.3)
        a.axhline(0.3, color="0.6", ls="--", lw=0.8)
        a.axhline(0.5, color="#b03b3b", ls="--", lw=0.8, label="anisotropy concern")
        a.set_title("isotropy: octant-floor spread / |bg|"); a.set_xlabel("|Q| (Å⁻¹)")
        a.set_ylim(0, max(1.0, float(np.nanpercentile(
            (oct_spread / np.maximum(np.abs(shift), 1e-9))[m], 98))))
        a.legend(fontsize=8, frameon=False)

        a = ax[1, 0]
        a.plot(qg[m], floor_med_gap[m], color="#9467bd", lw=1.3, label="floor↔median gap")
        a.plot(qg[m], tail_ratio[m], color="#c46210", lw=1.0, label="high-tail ratio")
        a.axhline(1.5, color="0.6", ls="--", lw=0.8)
        a.set_title("shell shape: pedestal left & tail asymmetry")
        a.set_xlabel("|Q| (Å⁻¹)"); a.legend(fontsize=8, frameon=False)

        a = ax[1, 1]
        lo, hi = np.percentile(after, [0.5, 99.5])
        a.hist(after[(after > lo) & (after < hi)], bins=200, color="0.6")
        a.axvline(0, color="#b03b3b", lw=1.0)
        a.set_title(f"flattened values (neg frac {neg_frac*100:.0f}%)")
        a.set_xlabel("intensity after")
        a.set_yscale("log")

        fig.suptitle(f"flatten robustness — {in_path.stem[:48]} — estimator={estimator}")
        fig.tight_layout()
        png = HERE / "_validate_flatten.png"
        fig.savefig(png, dpi=130)
        print(f"QA figure -> {png}", flush=True)
    except Exception as exc:  # noqa: BLE001 - QA plot best-effort
        print(f"(skipped QA plot: {exc})", flush=True)
