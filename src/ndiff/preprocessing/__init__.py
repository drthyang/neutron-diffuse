"""Data processing pipeline.

Input: symmetrised 3D HKL volume from Mantid.

Step 1 — Empty-scan subtraction  (``EmptySubtractor``)
    Removes the environment ring (cryostat, furnace, etc.).
    Residual rings from the sample holder remain.

Step 2 — Factored ring model  (``PatchedRingModel``)
    Fits  I_ring(Q, φ) = T(φ) × Σᵢ Aᵢ G(|Q| − qᵢ, σᵢ)
    by dividing φ into overlapping patches, fitting Gaussians per patch,
    then extracting T(φ) via SVD + Fourier smoothing.
    Subtracts the model; masks voxels where ring dominates.

Step 3 — Backfill  (``backfill_ring_shells``)
    Fills masked voxels by radial interpolation from nearest uncontaminated
    neighbours.  C¹ continuity comes from the interpolation itself.
"""

from ndiff.preprocessing.empty_subtraction import EmptySubtractor
from ndiff.preprocessing.ring_model import PatchedRingModel, RingParams, FittedRingModel
from ndiff.preprocessing.radial_background import (
    PatchedRadialRingModel,
    RadialRingProfiles,
)
from ndiff.preprocessing.sampling import azimuthal_sampling_mask
from ndiff.preprocessing.backfill import backfill_ring_shells
from ndiff.preprocessing.powder_rings import (
    RingShell,
    RingProfile,
    detect_ring_shells,
    mask_ring_shells,
    radial_profile,
    line_profile,
    fit_ring_profiles,
    al_ring_q_positions,
)

__all__ = [
    # Primary pipeline
    "EmptySubtractor",
    "PatchedRingModel",
    "RingParams",
    "FittedRingModel",
    "PatchedRadialRingModel",
    "RadialRingProfiles",
    "azimuthal_sampling_mask",
    "backfill_ring_shells",
    # Utilities / diagnostics
    "RingShell",
    "RingProfile",
    "detect_ring_shells",
    "mask_ring_shells",
    "radial_profile",
    "line_profile",
    "fit_ring_profiles",
    "al_ring_q_positions",
]
