"""Compare centered vs H-dependent powder-ring coordinates."""
import dataclasses
import os
from pathlib import Path

import numpy as np

import ndiff
from ndiff.preprocessing import PatchedRadialRingModel, azimuthal_sampling_mask


raw = Path("data/raw")
data_file = os.environ.get("DATA_FILE")
data = ndiff.load(Path(data_file) if data_file else [p for p in sorted(raw.glob("*.nxs"))
                   if not p.stem.endswith(("_bkg", "_sub_bkg"))][0])
H_VALUE = float(os.environ.get("H_VALUE", "0.3333"))
ih0 = int(np.argmin(np.abs(data.h_axis - H_VALUE)))
d = dataclasses.replace(
    data,
    data=data.data[ih0:ih0 + 1],
    sigma=data.sigma[ih0:ih0 + 1],
    mask=data.mask[ih0:ih0 + 1],
    h_axis=data.h_axis[ih0:ih0 + 1],
)
print(f"slice H target={H_VALUE:.4f}, using H={float(d.h_axis[0]):.4f}")

keep = azimuthal_sampling_mask(d, plane="0kl", min_count_frac=0.25,
                               q_range=(1.5, 10.5))
src = dataclasses.replace(d, mask=keep)

variants = [
    ("centered", (0.0, 0.0), (0.0, 0.0)),
    ("h-slope weighted", (0.0, 0.0), (0.004, 0.001)),
    ("h-slope lowmid", (0.0, 0.0), (0.010, 0.006)),
    ("h-slope xonly", (0.0, 0.0), (0.010, 0.0)),
    ("fixed lowmid@H", (0.0034, 0.0020), (0.0, 0.0)),
]

ring_q = [1.914, 2.687, 3.115, 4.401, 5.164, 5.847,
          6.195, 6.790, 6.963, 7.218, 7.631]
q = src.q_magnitude()
valid = keep & np.isfinite(d.data)
edges = np.arange(1.5, 8.0, 0.02)
qc = 0.5 * (edges[:-1] + edges[1:])
binv = np.digitize(q[valid], edges) - 1


def medprof(arr):
    a = arr[valid]
    out = np.full(len(qc), np.nan)
    for b in range(len(qc)):
        s = a[binv == b]
        if s.size:
            out[b] = np.median(s)
    return out


print("variant              abs_resid  neg_trough  offring_p95")
for label, offset, h_slope in variants:
    model = PatchedRadialRingModel(
        n_patches=36,
        plane="0kl",
        ring_width=0.24,
        baseline_smooth=0.06,
        profile_percentiles=(10.0, 80.0),
        texture_model="fourier",
        texture_symmetric=False,
        center_offset=offset,
        center_offset_h_slope=h_slope,
    )
    profiles = model.fit(src, q_range=(1.5, 10.5))
    _, I_ring = model.subtract(src, profiles)
    rp = medprof(d.data - I_ring)
    abs_sum = 0.0
    neg_sum = 0.0
    for q0 in ring_q:
        b = int(np.argmin(np.abs(qc - q0)))
        base = np.nanpercentile(rp[max(0, b - 12):b + 12], 20)
        resid = rp[b] - base
        abs_sum += abs(resid)
        neg_sum += max(0.0, -resid)
    off_ring = valid & (q >= 1.5) & (q <= 8.0)
    for q0 in ring_q:
        off_ring &= np.abs(q - q0) > 0.18
    print(f"{label:20s} {abs_sum:9.4f}  {neg_sum:10.4f}  "
          f"{np.percentile(I_ring[off_ring], 95):10.5f}")
