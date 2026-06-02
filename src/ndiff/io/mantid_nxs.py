"""Reader for Mantid MDHistoWorkspace NeXus files (.nxs).

File layout
-----------
MDHistoWorkspace/data/
    D0, D1, D2          bin-edge arrays  (N+1 values for N bins)
    signal              intensity  shape (n_D2, n_D1, n_D0)
    errors_squared      σ²         same shape
    mask                int8, 0 = valid, 1 = masked by Mantid
MDHistoWorkspace/experiment0/sample/oriented_lattice/
    orientation_matrix  UB matrix (3×3), Q = UB @ [h,k,l]ᵀ
    unit_cell_*         lattice parameters (provenance only)

Convention note
---------------
Mantid stores a *crystallographic* orientation matrix (|b*| = 1/d, no 2π).
ndiff uses the *physics* convention everywhere (Q = 2π/d, see
``ub_from_lattice`` and ``al_ring_q_positions``), so the stored matrix is
scaled by 2π on read to keep |Q| consistent across the package.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray

from ndiff.core import HKLVolume

if TYPE_CHECKING:
    import h5py

_PathLike = Union[str, Path]

_HKL_LETTERS = ("H", "K", "L")

# signal has shape (n_D2, n_D1, n_D0), so the array axis for each label is:
_DIM_TO_FILE_AXIS: dict[str, int] = {"D0": 2, "D1": 1, "D2": 0}

# Mantid's orientation_matrix is crystallographic (|b*| = 1/d); ndiff works in
# the physics convention (Q = 2π/d), so scale the stored matrix on read.
_TWO_PI: float = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_mantid_nxs(
    path: _PathLike,
    ub_matrix: NDArray[np.float64] | None = None,
) -> HKLVolume:
    """Load a Mantid MDHistoWorkspace NeXus file into an HKLVolume.

    Voxels with NaN signal or Mantid mask=1 are zeroed and set mask=False.

    Parameters
    ----------
    path
        Path to the ``.nxs`` file.
    ub_matrix
        Optional 3×3 UB matrix (physics convention, Q = 2π/d) to use instead
        of the one stored in the file.  Background/empty-can scans often lack
        an ``experiment0`` group; pass the paired data volume's
        ``ub_matrix`` so both share a consistent |Q| scale.
    """
    path = Path(path)
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to read NeXus files.") from exc

    with h5py.File(path, "r") as f:
        root = _require_md_histo(f)
        axes = _parse_dim_axes(root["data"])
        signal_raw = root["data/signal"][:].astype(np.float64)
        err2_raw = root["data/errors_squared"][:].astype(np.float64)
        mask_raw = root["data/mask"][:].astype(np.int8)
        ub = _resolve_ub(root, ub_matrix)

    data, sigma, mask, h_ax, k_ax, l_ax = _assemble(axes, signal_raw, err2_raw, mask_raw)
    return HKLVolume(
        data=data,
        sigma=sigma,
        mask=mask,
        h_axis=h_ax,
        k_axis=k_ax,
        l_axis=l_ax,
        ub_matrix=ub,
        instrument=path.stem,
    )


def is_mantid_nxs(path: _PathLike) -> bool:
    """Return True if *path* is a Mantid MDHistoWorkspace NeXus file."""
    try:
        import h5py
    except ImportError:
        return False
    try:
        with h5py.File(path, "r") as f:
            return "MDHistoWorkspace" in f
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Private helpers — each does exactly one thing
# ---------------------------------------------------------------------------


def _require_md_histo(f: "h5py.File") -> "h5py.Group":
    if "MDHistoWorkspace" not in f:
        raise ValueError(f"{f.filename!r} is not a Mantid MDHistoWorkspace file.")
    return f["MDHistoWorkspace"]  # type: ignore[return-value]


def _parse_dim_axes(
    data_grp: "h5py.Group",
) -> dict[str, tuple[int, NDArray[np.float64]]]:
    """Return {hkl_char: (file_array_axis, bin_centers)} for H, K, L.

    Mantid stores signal as (n_D2, n_D1, n_D0), so D2 → axis 0,
    D1 → axis 1, D0 → axis 2.  The long_name attribute on each dim
    (e.g. '[0,K,0]') identifies which HKL coordinate it represents.
    """
    result: dict[str, tuple[int, NDArray[np.float64]]] = {}
    for d_label, file_axis in _DIM_TO_FILE_AXIS.items():
        ds = data_grp[d_label]
        edges: NDArray[np.float64] = ds[:].astype(np.float64)
        long_name: str = ds.attrs["long_name"].decode()
        hkl_char = _identify_hkl_char(long_name)
        result[hkl_char] = (file_axis, _bin_centers(edges))
    return result


def _identify_hkl_char(long_name: str) -> str:
    """Return 'H', 'K', or 'L' from a Mantid dim long_name like '[0,K,0]'.

    Assumes an orthogonal cut where exactly one HKL component varies.
    Oblique cuts (e.g. '[H,H,0]') are not supported.
    """
    upper = long_name.upper()
    for ch in _HKL_LETTERS:
        if ch in upper:
            return ch
    raise ValueError(f"Cannot identify H/K/L component in dim long_name {long_name!r}")


def _bin_centers(edges: NDArray[np.float64]) -> NDArray[np.float64]:
    return (edges[:-1] + edges[1:]) * 0.5


def _resolve_ub(
    root: "h5py.Group",
    override: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Pick the UB matrix: explicit override > file value > identity fallback."""
    if override is not None:
        return np.asarray(override, dtype=np.float64)
    if "experiment0" in root:
        return _read_ub_matrix(root["experiment0/sample/oriented_lattice"])
    return np.eye(3, dtype=np.float64)


def _read_ub_matrix(lattice_grp: "h5py.Group") -> NDArray[np.float64]:
    """Read the orientation matrix and scale to the physics (2π) convention."""
    return lattice_grp["orientation_matrix"][:].astype(np.float64) * _TWO_PI


def _assemble(
    axes: dict[str, tuple[int, NDArray[np.float64]]],
    signal_raw: NDArray[np.float64],
    err2_raw: NDArray[np.float64],
    mask_raw: NDArray[np.int8],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Permute file arrays from (D2,D1,D0) to canonical (H,K,L) axis order."""
    perm = (axes["H"][0], axes["K"][0], axes["L"][0])
    sig = np.transpose(signal_raw, perm)
    err2 = np.transpose(err2_raw, perm)
    fmask = np.transpose(mask_raw, perm)

    valid: NDArray[np.bool_] = np.isfinite(sig) & (fmask == 0)
    sigma = np.sqrt(np.maximum(err2, 0.0))

    # zero out invalid voxels so downstream code never sees NaN
    sig = np.where(valid, sig, 0.0)
    sigma = np.where(valid, sigma, 0.0)

    return sig, sigma, valid, axes["H"][1], axes["K"][1], axes["L"][1]
