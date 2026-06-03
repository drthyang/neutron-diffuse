"""Fit apparent powder-ring centers on a chosen HKL slice.

This diagnoses whether residual over/under subtraction could come from rings
being slightly off-centered relative to the model's |Q|-centered shells.
"""
import dataclasses
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import ndiff
from ndiff.preprocessing import fit_ring_profiles, line_profile
from ndiff.preprocessing.radial_background import _azimuthal_angle


def _plane_coords(vol, plane):
    axes = {"hk0": (0, 1), "h0l": (0, 2), "0kl": (1, 2)}
    i, j = axes[plane]
    H, K, L = vol.hkl_grid()
    Q = np.stack([H, K, L], axis=-1) @ vol.ub_matrix.T

    a1 = vol.ub_matrix[:, i].astype(np.float64)
    a2 = vol.ub_matrix[:, j].astype(np.float64)
    e1 = a1 / (np.linalg.norm(a1) + 1e-12)
    a2_perp = a2 - (a2 @ e1) * e1
    e2 = a2_perp / (np.linalg.norm(a2_perp) + 1e-12)
    return Q @ e1, Q @ e2


def _fit_center(x, y, intensity, r0, width=0.18, top_frac=0.08):
    r = np.sqrt(x * x + y * y)
    ann = np.isfinite(intensity) & (np.abs(r - r0) < width)
    if int(ann.sum()) < 30:
        return None
    vals = intensity[ann]
    thresh = np.nanpercentile(vals, 100.0 * (1.0 - top_frac))
    sel = ann & (intensity >= thresh)
    xx = x[sel].ravel()
    yy = y[sel].ravel()
    if xx.size < 12:
        return None

    # Fit x^2+y^2+Ax+By+C=0, then refine robustly.
    A = np.column_stack([xx, yy, np.ones_like(xx)])
    b = -(xx * xx + yy * yy)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx0 = -0.5 * coef[0]
    cy0 = -0.5 * coef[1]
    rr0 = np.sqrt(max(1e-12, cx0 * cx0 + cy0 * cy0 - coef[2]))

    def residual(p):
        cx, cy, rr = p
        return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - rr

    fit = least_squares(
        residual, np.array([cx0, cy0, rr0]), loss="soft_l1", f_scale=0.02,
        max_nfev=2000,
    )
    cx, cy, rr = fit.x
    resid = residual(fit.x)
    return {
        "cx": float(cx),
        "cy": float(cy),
        "r": float(rr),
        "n": int(xx.size),
        "mad": float(np.median(np.abs(resid - np.median(resid)))),
    }


raw = Path("data/raw")
data_file = os.environ.get("DATA_FILE")
data = ndiff.load(Path(data_file) if data_file else [p for p in sorted(raw.glob("*.nxs"))
                   if not p.stem.endswith(("_bkg", "_sub_bkg"))][0])

H_VALUE = float(os.environ.get("H_VALUE", "0.3333"))
ih = int(np.argmin(np.abs(data.h_axis - H_VALUE)))
sl = dataclasses.replace(
    data,
    data=data.data[ih:ih + 1],
    sigma=data.sigma[ih:ih + 1],
    mask=data.mask[ih:ih + 1],
    h_axis=data.h_axis[ih:ih + 1],
)
print(f"slice H target={H_VALUE:.4f}, using H={float(sl.h_axis[0]):.4f}")

L = float(max(abs(data.l_axis.min()), abs(data.l_axis.max())))
q_ref = None
cuts = []
for k0 in (-1.0, 1.0):
    for l1 in (-L, L):
        q, I, _ = line_profile(data, (0.0, k0, 0.0), (0.0, k0, l1), 900)
        if q_ref is None:
            q_ref = q
        elif not np.allclose(q, q_ref, rtol=0.0, atol=1e-8):
            I = np.interp(q_ref, q, I, left=np.nan, right=np.nan)
        cuts.append(I)
rings = fit_ring_profiles(q_ref, np.nanmean(np.vstack(cuts), axis=0))

x, y = _plane_coords(sl, "0kl")
phi = _azimuthal_angle(sl, "0kl")
valid = sl.mask & np.isfinite(sl.data)
print(" ring_q  rho0    cx      cy      |c|     r_fit   npts   mad")
centers = []
for ring in rings:
    q0 = ring.q_center
    # In this H slice, the ring radius in the plane is the median in-plane
    # radius of voxels whose full |Q| lies close to q0.
    qmag = sl.q_magnitude()
    near = valid & (np.abs(qmag - q0) < 0.04)
    if int(near.sum()) < 20:
        continue
    rho0 = float(np.nanmedian(np.sqrt(x[near] ** 2 + y[near] ** 2)))
    fit = _fit_center(x, y, sl.data, rho0)
    if fit is None:
        continue
    cmag = float(np.hypot(fit["cx"], fit["cy"]))
    centers.append((fit["cx"], fit["cy"], fit["n"]))
    print(f" {q0:6.3f} {rho0:6.3f} {fit['cx']:7.4f} {fit['cy']:7.4f} "
          f"{cmag:7.4f} {fit['r']:7.3f} {fit['n']:6d} {fit['mad']:6.4f}")

if centers:
    w = np.array([n for _, _, n in centers], dtype=float)
    cx = np.average([c[0] for c in centers], weights=w)
    cy = np.average([c[1] for c in centers], weights=w)
    print(f"weighted mean center: cx={cx:.5f}, cy={cy:.5f}, |c|={np.hypot(cx, cy):.5f} A^-1")

_ = phi  # keep import path exercised for parity with model coordinates
