"""Fit each powder ring (center, width, amplitude, baseline) on the clean linecut."""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import ndiff
from ndiff.preprocessing import line_profile, al_ring_q_positions

data = ndiff.load([p for p in sorted(Path("data/raw").glob("*.nxs"))
                   if not p.stem.endswith(("_bkg", "_sub_bkg"))][0])
L = float(data.l_axis.max())
q1, I1, _ = line_profile(data, (0, 1, 0), (0, 1, L), 800)
q2, I2, _ = line_profile(data, (0, -1, 0), (0, -1, L), 800)
q = q1
I = np.nanmean(np.vstack([I1, I2]), axis=0)
fin = np.isfinite(I)

peaks, _ = find_peaks(np.where(fin, I, 0.0), prominence=0.04, distance=8)
centers = q[peaks]

# Cluster nearby peaks (fit overlapping rings jointly).
clusters = []
cur = [centers[0]]
for c in centers[1:]:
    if c - cur[-1] < 0.3:
        cur.append(c)
    else:
        clusters.append(cur); cur = [c]
clusters.append(cur)

def model(x, *p):
    n = (len(p) - 2) // 3
    b0, b1 = p[-2], p[-1]
    y = b0 + b1 * (x - x.mean())
    for i in range(n):
        amp, x0, sig = p[3*i:3*i+3]
        y = y + amp * np.exp(-0.5 * ((x - x0) / sig) ** 2)
    return y

print(" ring  q_center   FWHM     amplitude  baseline")
rows = []
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(q[fin], I[fin], lw=1.0, color="0.4", label="(0,±1,l) linecut")
for cl in clusters:
    lo, hi = min(cl) - 0.22, max(cl) + 0.22
    sel = fin & (q >= lo) & (q <= hi)
    qq, II = q[sel], I[sel]
    base0 = np.percentile(II, 20)
    p0, lb, ub = [], [], []
    for c in cl:
        p0 += [max(1e-3, II.max() - base0), c, 0.04]
        lb += [0.0, c - 0.1, 0.005]; ub += [np.inf, c + 0.1, 0.2]
    p0 += [base0, 0.0]; lb += [0.0, -np.inf]; ub += [np.inf, np.inf]
    try:
        popt, _ = curve_fit(model, qq, II, p0=p0, bounds=(lb, ub), maxfev=20000)
    except Exception as e:
        print("  fit failed for", cl, e); continue
    xfit = np.linspace(lo, hi, 300)
    ax.plot(xfit, model(xfit, *popt), color="C3", lw=1.2)
    b0 = popt[-2]
    for i in range(len(cl)):
        amp, x0, sig = popt[3*i:3*i+3]
        fwhm = 2.3548 * sig
        rows.append((x0, fwhm, amp, b0))
        print(f"        {x0:6.3f}   {fwhm:6.3f}    {amp:7.3f}   {b0:7.3f}")
ax.legend(); ax.set_xlabel("|Q| (1/A)"); ax.set_ylabel("I"); ax.set_ylim(0, None)
ax.set_title("Per-ring Gaussian + baseline fits on the Bragg-free linecut")
fig.tight_layout(); fig.savefig("examples/_ringfit.png", dpi=110)
print("\nmean FWHM = %.3f A^-1, mean sigma = %.3f" %
      (np.mean([r[1] for r in rows]), np.mean([r[1]/2.3548 for r in rows])))
print("wrote examples/_ringfit.png")
