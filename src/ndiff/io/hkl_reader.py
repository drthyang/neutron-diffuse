"""Load and save HKLVolume from/to HDF5 or ASCII files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ndiff.core import HKLVolume

_PathLike = str | Path


def load(path: _PathLike, **kwargs: object) -> HKLVolume:
    """Load an HKLVolume from *path*.

    Supported formats (auto-detected by extension and file content):
    - ``.nxs``: Mantid MDHistoWorkspace (auto-detected) or ndiff HDF5
    - ``.h5`` / ``.hdf5``: ndiff HDF5
    - ``.txt`` / ``.dat`` / ``.hkl``: whitespace-delimited ASCII (h k l I [sigma])
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in {".h5", ".hdf5", ".nxs"}:
        from ndiff.io.mantid_nxs import is_mantid_nxs, load_mantid_nxs
        if is_mantid_nxs(path):
            return load_mantid_nxs(path)
        entry = _pop_only_kwarg(kwargs, "entry", "/entry")
        return _load_hdf5(path, entry=entry)
    if ext in {".txt", ".dat", ".hkl"}:
        _reject_kwargs(kwargs)
        return _load_ascii(path)
    raise ValueError(f"Unrecognised file extension: {ext!r}")


def save(vol: HKLVolume, path: _PathLike, **kwargs: object) -> None:
    """Save *vol* to *path* (format auto-detected by extension)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in {".h5", ".hdf5", ".nxs"}:
        entry = _pop_only_kwarg(kwargs, "entry", "/entry")
        _save_hdf5(vol, path, entry=entry)
    elif ext in {".txt", ".dat", ".hkl"}:
        _reject_kwargs(kwargs)
        _save_ascii(vol, path)
    else:
        raise ValueError(f"Unrecognised file extension: {ext!r}")


def _pop_only_kwarg(kwargs: dict[str, object], name: str, default: str) -> str:
    value = kwargs.pop(name, default)
    _reject_kwargs(kwargs)
    return str(value)


def _reject_kwargs(kwargs: dict[str, object]) -> None:
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {names}")


# ------------------------------------------------------------------
# HDF5
# ------------------------------------------------------------------


def _load_hdf5(path: Path, entry: str = "/entry") -> HKLVolume:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to read HDF5 files.") from exc

    with h5py.File(path, "r") as f:
        grp = f[entry]
        data = np.array(grp["data"], dtype=np.float64)
        sigma = (np.array(grp["sigma"], dtype=np.float64) if "sigma" in grp
                 else np.sqrt(np.abs(data)))
        mask = (np.array(grp["mask"], dtype=bool) if "mask" in grp
                else np.ones(data.shape, dtype=bool))
        h_axis = np.array(grp["h_axis"], dtype=np.float64)
        k_axis = np.array(grp["k_axis"], dtype=np.float64)
        l_axis = np.array(grp["l_axis"], dtype=np.float64)
        ub = np.array(grp["ub_matrix"], dtype=np.float64) if "ub_matrix" in grp else np.eye(3)
        instrument = str(grp.attrs.get("instrument", ""))

    return HKLVolume(
        data=data,
        sigma=sigma,
        mask=mask,
        h_axis=h_axis,
        k_axis=k_axis,
        l_axis=l_axis,
        ub_matrix=ub,
        instrument=instrument,
    )


def _save_hdf5(vol: HKLVolume, path: Path, entry: str = "/entry") -> None:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to write HDF5 files.") from exc

    with h5py.File(path, "w") as f:
        grp = f.require_group(entry)
        grp.create_dataset("data", data=vol.data, compression="gzip")
        grp.create_dataset("sigma", data=vol.sigma, compression="gzip")
        grp.create_dataset("mask", data=vol.mask, compression="gzip")
        grp.create_dataset("h_axis", data=vol.h_axis)
        grp.create_dataset("k_axis", data=vol.k_axis)
        grp.create_dataset("l_axis", data=vol.l_axis)
        grp.create_dataset("ub_matrix", data=vol.ub_matrix)
        grp.attrs["instrument"] = vol.instrument


# ------------------------------------------------------------------
# ASCII (h k l I sigma)
# ------------------------------------------------------------------


def _load_ascii(path: Path) -> HKLVolume:
    cols = np.loadtxt(path)
    if cols.ndim != 2 or cols.shape[1] < 4:
        raise ValueError("ASCII file must have at least 4 columns: h k l I [sigma]")

    h, k, l, intensity = cols[:, 0], cols[:, 1], cols[:, 2], cols[:, 3]
    sigma = cols[:, 4] if cols.shape[1] >= 5 else np.sqrt(np.abs(intensity))

    h_vals = np.unique(h)
    k_vals = np.unique(k)
    l_vals = np.unique(l)
    nh, nk, nl = len(h_vals), len(k_vals), len(l_vals)

    h_idx = np.searchsorted(h_vals, h)
    k_idx = np.searchsorted(k_vals, k)
    l_idx = np.searchsorted(l_vals, l)

    data = np.full((nh, nk, nl), np.nan)
    sig = np.full((nh, nk, nl), np.nan)
    data[h_idx, k_idx, l_idx] = intensity
    sig[h_idx, k_idx, l_idx] = sigma

    mask = np.isfinite(data)
    data = np.where(mask, data, 0.0)
    sig = np.where(mask, sig, 0.0)

    return HKLVolume(
        data=data,
        sigma=sig,
        mask=mask,
        h_axis=h_vals,
        k_axis=k_vals,
        l_axis=l_vals,
    )


def _save_ascii(vol: HKLVolume, path: Path) -> None:
    H, K, L = vol.hkl_grid()
    rows = np.column_stack([
        H.ravel(), K.ravel(), L.ravel(),
        vol.data.ravel(), vol.sigma.ravel(),
        vol.mask.ravel().astype(int),
    ])
    np.savetxt(path, rows, fmt="%10.4f %10.4f %10.4f %14.6e %14.6e %1d",
               header="h k l I sigma valid")
