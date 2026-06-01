"""Data processing pipeline.

Steps (applied to symmetrized input from Mantid or equivalent):
    (1) Remove powder rings  → powder_rings.PowderRingRemover
    (2) Backfill ring holes  → backfill.backfill
"""

from ndiff.preprocessing.powder_rings import (
    PowderRing,
    PowderRingRemover,
    detect_rings,
    subtract_rings,
    al_ring_q_positions,
)
from ndiff.preprocessing.backfill import backfill

__all__ = [
    "PowderRing",
    "PowderRingRemover",
    "detect_rings",
    "subtract_rings",
    "al_ring_q_positions",
    "backfill",
]
