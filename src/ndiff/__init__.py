"""ndiff — neutron diffuse scattering processing toolkit.

Input: symmetrized 3D HKL volume (e.g. from Mantid).

Pipeline
--------
Data processing:
    preprocessing.EmptySubtractor      → (1) subtract empty-environment scan
    preprocessing.PatchedRingModel     → (2) fit & subtract residual ring
    preprocessing.backfill_ring_shells → (3) fill ring holes in diffuse signal

Further analysis:
    analysis.BraggRemover              → (4) punch Bragg peaks
    analysis.backfill_bragg            → (5) fill Bragg holes
    analysis.compute_delta_pdf         → (6) 3D-ΔPDF via FFT
"""

from ndiff import analysis, inpainting, preprocessing, utils, visualization
from ndiff._version import __version__
from ndiff.core import HKLVolume
from ndiff.io.hkl_reader import load, save
from ndiff.io.mantid_nxs import is_mantid_nxs, load_mantid_nxs

__all__ = [
    "__version__",
    "HKLVolume",
    "load",
    "save",
    "load_mantid_nxs",
    "is_mantid_nxs",
    "preprocessing",
    "analysis",
    "inpainting",
    "utils",
    "visualization",
]
