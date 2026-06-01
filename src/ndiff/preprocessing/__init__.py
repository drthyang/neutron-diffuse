"""Data processing pipeline.

Input: symmetrised 3D HKL volume from Mantid.

Step 1 — Empty-scan subtraction  (``EmptySubtractor``)
    Removes the environment ring (cryostat, furnace, etc.).
    Residual rings from the sample holder remain.

Step 2 — Ring detection in 1D |Q|  (``detect_ring_shells``)
    Rolling-median baseline on the radial profile — no assumption on the
    diffuse signal.  Returns |Q| ranges of detected ring shells.

Step 3 — Masking  (``mask_ring_shells``)
    Marks ring-shell voxels invalid.  Sigmoid-tapered boundary for C¹
    continuity at the shell edge.

Step 4 — Backfill  (``backfill_ring_shells``)
    Fills masked voxels by radial interpolation from nearest uncontaminated
    neighbours.  C¹ continuity is guaranteed by the interpolation itself.
"""

from ndiff.preprocessing.empty_subtraction import EmptySubtractor
from ndiff.preprocessing.powder_rings import (
    RingShell,
    detect_ring_shells,
    mask_ring_shells,
    radial_profile,
    al_ring_q_positions,
)
from ndiff.preprocessing.backfill import backfill_ring_shells

__all__ = [
    "EmptySubtractor",
    "RingShell",
    "detect_ring_shells",
    "mask_ring_shells",
    "radial_profile",
    "al_ring_q_positions",
    "backfill_ring_shells",
]
