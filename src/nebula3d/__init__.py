"""nebula3d — neutron diffuse scattering processing toolkit.

Input: symmetrized 3D HKL volume (e.g. from Mantid).

Pipeline
--------
Current diffuse workflow:
    (1) powder-ring subtraction
    (2) Bragg/satellite punch
    (3) Bragg-hole backfill
    (4) radial-background flatten
    (5) 3D-ΔPDF transform
    (6) back-FFT consistency check
"""

from nebula3d import analysis, inpainting, preprocessing, utils, visualization
from nebula3d._version import __version__
from nebula3d.core import HKLVolume
from nebula3d.io.hkl_reader import load, save
from nebula3d.io.mantid_nxs import is_mantid_nxs, load_mantid_nxs

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
