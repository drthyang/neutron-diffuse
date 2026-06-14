"""Compare HKL-frame vs Q-frame (isotropic) Bragg punch on the real volumes.

Phase-4 validation for ROADMAP Phase 6: before making the Q frame the default,
quantify how much switching the Bragg punch from the anisotropic HKL radii
``(0.09, 0.12, 0.45)`` r.l.u. to an isotropic Q-sphere ``|δQ| ≤ ρ`` (Å⁻¹) changes
the punched mask, and where the two disagree.

The two frames are run with *identical* detection/guard settings and a *fixed*
shape (per-peak shape-fit off), so the only difference is the punch footprint —
an HKL-axis ellipsoid vs a Q-ball.  We report, per temperature:

* punched-voxel counts and the symmetric-difference / Jaccard agreement,
* the |Q|-shell distribution of the disagreement (does it grow with |Q|?),
* the HKL radii re-expressed in Å⁻¹ and the volume-matched ρ.

Run (no venv; see the run-environment note):

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
    /path/to/sci-general/python examples/compare_punch_frames.py
"""

from __future__ import annotations

import glob
import os

import numpy as np

import ndiff
from ndiff.analysis.bragg import BraggRemover

HKL_RADII = (0.09, 0.12, 0.45)

# Identical detection + guards in both arms; fixed shape (fit off) and no
# intensity scaling so the *only* variable is the punch frame/footprint.
COMMON = dict(
    mode="both", min_intensity=0.8, min_prominence=1.0,
    integer_optimize_position=True, integer_optimize_shape=False,
    integer_h_guard_hkl=0.12, integer_local_prominence_n_mad=8.0,
    search_n_mad=4.0, search_min_intensity=0.8, search_min_prominence=0.8,
    search_exclude_h_fractions=(0.3333, 0.6667), search_exclude_h_half_width=0.08,
    intensity_scale=False, margin=0.02, phi_tail_hkl=0.0,
    punch_incident_beam=True, incident_beam_ellipsoid_radii_hkl=(0.15, 0.50, 1.00),
)


def bstar(vol: ndiff.HKLVolume) -> np.ndarray:
    """Reciprocal-axis lengths |a*|,|b*|,|c*| (Å⁻¹)."""
    return np.sqrt(np.diag(vol.ub_matrix.T @ vol.ub_matrix))


def _punched(keep: np.ndarray) -> np.ndarray:
    return ~keep


def compare_one(path: str) -> None:
    vol = ndiff.load(path)
    name = os.path.basename(path)
    bs = bstar(vol)
    hkl_in_A = np.array(HKL_RADII) * bs
    rho_match = float(np.prod(hkl_in_A) ** (1.0 / 3.0))  # volume-matched Q sphere

    keep_hkl = BraggRemover(punch_radii=HKL_RADII, **COMMON).build_mask(vol)
    keep_q = BraggRemover(
        punch_frame="q", punch_q_radius=rho_match, **COMMON).build_mask(vol)
    # anisotropic Q = the HKL footprint in Å⁻¹ → should reproduce HKL (units only)
    keep_qa = BraggRemover(
        punch_frame="q", punch_q_radii=tuple(hkl_in_A), **COMMON).build_mask(vol)

    p_hkl, p_q = _punched(keep_hkl), _punched(keep_q)
    p_qa = _punched(keep_qa)
    qa_dis = int((p_hkl ^ p_qa).sum())
    qa_union = int((p_hkl | p_qa).sum())
    both = int((p_hkl & p_q).sum())
    only_hkl = int((p_hkl & ~p_q).sum())   # HKL punches, Q keeps
    only_q = int((~p_hkl & p_q).sum())     # Q punches, HKL keeps
    union = both + only_hkl + only_q
    jaccard = both / union if union else 1.0

    print(f"\n=== {name[:60]} ===")
    print(f"  |a*,b*,c*| (Å⁻¹)      : {np.array2string(bs, precision=3)}")
    print(f"  HKL radii in Å⁻¹       : {np.array2string(hkl_in_A, precision=3)} "
          f"(max/min {hkl_in_A.max()/hkl_in_A.min():.2f})")
    print(f"  volume-matched ρ (Å⁻¹) : {rho_match:.4f}")
    print(f"  punched  HKL={int(p_hkl.sum()):>9,}   Q-iso={int(p_q.sum()):>9,}   "
          f"Q-aniso={int(p_qa.sum()):>9,}")
    print("  -- isotropic Q (single ρ) vs HKL --")
    print(f"  agreement (Jaccard)    : {jaccard:.4f}")
    print(f"  HKL-only (Q keeps)     : {only_hkl:>9,}  "
          f"({100*only_hkl/int(p_hkl.sum()):.1f}% of HKL)")
    print(f"  Q-only   (HKL keeps)   : {only_q:>9,}  "
          f"({100*only_q/int(p_q.sum()):.1f}% of Q)")
    print("  -- anisotropic Q (= HKL in Å⁻¹) vs HKL --")
    print(f"  disagree / union       : {qa_dis:,} / {qa_union:,}  "
          f"(Jaccard {1 - qa_dis/qa_union:.4f}) — units-only check")

    # where do they disagree along |Q|?  (disagreement is the Q-ball vs L-ellipsoid
    # mismatch — expect it concentrated, and to scale with peak density)
    disagree = p_hkl ^ p_q
    if disagree.any():
        q = vol.q_magnitude()
        qd = q[disagree]
        lo, mid, hi = np.percentile(q[p_hkl | p_q], [33, 66, 100])
        frac_lo = float((qd <= lo).mean())
        frac_hi = float((qd > mid).mean())
        print(f"  disagree vs |Q|        : {100*frac_lo:.0f}% in low third "
              f"(≤{lo:.1f}), {100*frac_hi:.0f}% in high third (>{mid:.1f} Å⁻¹)")


def main() -> None:
    pat = os.environ.get("PATTERN", "*ringremoved.h5")
    paths = sorted(
        p for p in glob.glob(f"data/processed/{pat}")
        if "braggpunched" not in p
    )
    if not paths:
        raise SystemExit("no *_ringremoved.h5 volumes under data/processed/")
    print("HKL-frame vs Q-frame (isotropic) Bragg punch — identical detection, "
          "fixed shape.")
    for p in paths:
        compare_one(p)


if __name__ == "__main__":
    main()
