# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tsung-han Yang

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

from nebula3d.preprocessing.backfill import backfill_ring_shells
from nebula3d.preprocessing.empty_subtraction import EmptySubtractor
from nebula3d.preprocessing.parametric_ring import (
    FittedParametricRingModel,
    ParametricRing,
    ParametricRingModel,
)
from nebula3d.preprocessing.powder_rings import (
    RingProfile,
    RingShell,
    al_ring_q_positions,
    detect_ring_shells,
    fit_ring_profiles,
    line_profile,
    mask_ring_shells,
    radial_profile,
)
from nebula3d.preprocessing.radial_background import (
    PatchedRadialRingModel,
    RadialRingProfiles,
    confirm_ring_shells_across_h,
)
from nebula3d.preprocessing.radial_flatten import (
    RadialFlattenResult,
    flatten_radial_background,
)
from nebula3d.preprocessing.ring_model import FittedRingModel, PatchedRingModel, RingParams
from nebula3d.preprocessing.sampling import azimuthal_sampling_mask

__all__ = [
    # Primary pipeline
    "EmptySubtractor",
    "PatchedRingModel",
    "RingParams",
    "FittedRingModel",
    "PatchedRadialRingModel",
    "RadialRingProfiles",
    "ParametricRingModel",
    "ParametricRing",
    "FittedParametricRingModel",
    "confirm_ring_shells_across_h",
    "flatten_radial_background",
    "RadialFlattenResult",
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
