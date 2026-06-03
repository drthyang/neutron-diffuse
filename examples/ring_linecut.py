"""Identify powder-ring |Q| positions from a Bragg-free linecut.

The crystal's ``0kl`` reflections with odd k are systematically absent, so a line
along ``(0, ±1, l)`` threads between all Bragg peaks.  The resulting radial
linecut is the powder-ring signal alone (no Bragg contamination), giving clean,
directly-readable ring |Q| positions.

Run::

    PYTHONPATH=src python examples/ring_linecut.py
"""
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

import ndiff
from ndiff.preprocessing import line_profile, al_ring_q_positions

raw = Path("data/raw")
data = ndiff.load([p for p in sorted(raw.glob("*.nxs"))
                   if not p.stem.endswith(("_bkg", "_sub_bkg"))][0])

# Bragg-free radial cuts along (0, +1, l) and (0, -1, l); they share the same
# |Q| sampling, so average them to beat down noise.
L_MAX = float(data.l_axis.max())
q1, I1, _ = line_profile(data, (0, 1, 0), (0, 1, L_MAX), n_points=800)
q2, I2, _ = line_profile(data, (0, -1, 0), (0, -1, L_MAX), n_points=800)
qmag = q1
prof = np.nanmean(np.vstack([I1, I2]), axis=0)

# Detect ring peaks (baseline ~0.05; rings rise well above it).
finite = np.isfinite(prof)
peaks, props = find_peaks(np.where(finite, prof, 0.0),
                          prominence=0.04, distance=8)
ring_q = qmag[peaks]

al = np.array(al_ring_q_positions(q_max=qmag.max() + 0.2))
print(f"Bragg-free linecut (0, ±1, l), |Q| = {qmag.min():.2f}–{qmag.max():.2f} Å⁻¹")
print(f"max intensity {np.nanmax(prof):.2f} (no Bragg — clean)\n")
print(" detected |Q|   peak I    nearest Al |Q|   Δ")
for idx in peaks:
    qp = qmag[idx]
    j = int(np.argmin(np.abs(al - qp)))
    print(f"   {qp:6.3f}     {prof[idx]:6.3f}     {al[j]:6.3f}        {qp - al[j]:+.3f}")

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(qmag[finite], prof[finite], lw=1.0, color="C0", label="(0, ±1, l) linecut")
    ax.plot(ring_q, prof[peaks], "v", color="C3", label="detected rings")
    for qa in al:
        ax.axvline(qa, color="C1", lw=0.5, alpha=0.4)
    ax.set_xlabel("|Q| (Å⁻¹)  (orange = Al reference)")
    ax.set_ylabel("Intensity (Bragg-free)")
    ax.set_title("Powder-ring |Q| from the Bragg-free (0, ±1, l) linecut")
    ax.legend()
    plt.show()
except Exception as exc:  # headless / no display
    print(f"\n(plot skipped: {exc})")
