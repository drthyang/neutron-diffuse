"""ndiff — neutron diffuse scattering processing toolkit.

Input: symmetrized 3D HKL volume (e.g. from Mantid).

Pipeline
--------
Data processing:
    preprocessing.PowderRingRemover  → (1) detect & subtract powder rings
    preprocessing.backfill           → (2) fill ring holes in diffuse signal

Further analysis:
    analysis.BraggRemover            → (3) punch Bragg peaks
    analysis.backfill_bragg          → (4) fill Bragg holes
    analysis.compute_delta_pdf       → (5) 3D-ΔPDF via FFT
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
