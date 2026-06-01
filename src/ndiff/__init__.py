"""ndiff — neutron diffuse scattering processing toolkit.

Pipeline
--------
Data processing:
    preprocessing.symmetrize     → (1) symmetrize data
    preprocessing.aluminum_mask  → (2) remove Al signals
    preprocessing.backfill_al    → (3) backfill Al holes

Further analysis:
    analysis.bragg_mask          → (4) remove Bragg peaks (punch)
    analysis.backfill_bragg      → (5) backfill Bragg holes
    analysis.compute_delta_pdf   → (6) 3D-ΔPDF via FFT
"""

from ndiff._version import __version__
from ndiff.core import HKLVolume
from ndiff.io.hkl_reader import load, save
from ndiff import preprocessing, analysis, inpainting, utils

__all__ = [
    "__version__",
    "HKLVolume",
    "load",
    "save",
    "preprocessing",
    "analysis",
    "inpainting",
    "utils",
]
