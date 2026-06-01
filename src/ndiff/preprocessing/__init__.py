"""Data processing pipeline.

Input: symmetrised 3D HKL volume from Mantid.

Step 1 — Empty scan subtraction:
    Removes the bulk of the powder ring from the sample environment.
    Residual rings from the sample holder remain.

Step 2 — Residual ring detection & fill:
    Uses Laue symmetry to detect voxels that are anomalously high compared
    to their crystal equivalents, then fills them from the clean equivalents.
    Works because the residual ring has uneven amplitude around the |Q| shell,
    so not all Laue equivalents are equally contaminated.

Step 3 — Inpainting fallback:
    Fills any voxels that could not be filled by symmetry (too few clean
    equivalents) using smooth 3D interpolation.
"""

from ndiff.preprocessing.empty_subtraction import EmptySubtractor
from ndiff.preprocessing.residual_rings import detect_and_fill_residual
from ndiff.preprocessing.backfill import backfill
from ndiff.preprocessing.powder_rings import (
    PowderRing,
    PowderRingRemover,
    detect_rings,
    subtract_rings,
    al_ring_q_positions,
)

__all__ = [
    # Primary pipeline
    "EmptySubtractor",
    "detect_and_fill_residual",
    "backfill",
    # Exploratory / diagnostic tools
    "PowderRing",
    "PowderRingRemover",
    "detect_rings",
    "subtract_rings",
    "al_ring_q_positions",
]
