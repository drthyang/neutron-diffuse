"""Real-space 3D-ΔPDF loading and orthoslice extraction for the API.

ΔPDF ``.h5`` files (written by :func:`ndiff.pipeline.write_delta_pdf_h5` /
``examples/delta_pdf.py``) have a different schema from an :class:`HKLVolume`:
a signed real-space ``data`` array indexed ``[ix, iy, iz]`` with separate
``x_axis`` (x_H), ``y_axis`` (y_K), ``z_axis`` (z_L) in Å, plus direct-lattice
constants in the attrs.

The three orthoslice planes match the matplotlib viewers:

    xy : x_H–y_K  (fix z_L)   — slice_hk0
    xz : x_H–z_L  (fix y_K)   — slice_h0l
    yz : y_K–z_L  (fix x_H)   — slice_0kl

Slices are packed into the same binary envelope as the reciprocal-space slices
(see :mod:`ndiff.server.volumes`).  Because the ΔPDF is signed, the colour scale
hint is a robust *far-field* level (p98 of ``|ΔPDF|`` at in-plane r > 3 Å) so the
huge near-origin spike does not dominate; the client renders it diverging about 0.
"""

from __future__ import annotations

import json
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

#: Orthoslice plane keys (and the axis each one fixes).
DPDF_PLANES: tuple[str, ...] = ("xy", "xz", "yz")

_CACHE_MAX = 3  # keep all three temperatures resident for the multi-temp viewer
_cache: OrderedDict[tuple[str, float], DeltaPdfData] = OrderedDict()
_lock = threading.Lock()


@dataclass
class DeltaPdfData:
    data: np.ndarray  # (nx, ny, nz)
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray
    lat_a: float | None
    lat_b: float | None
    lat_c: float | None
    q_max: float | None


def _attr(fh: h5py.File, key: str) -> float | None:
    v = fh.attrs.get(key)
    return float(v) if v is not None else None


def load_dpdf(path: Path) -> DeltaPdfData:
    """Load a ΔPDF ``.h5``, caching by ``(path, mtime)``."""
    key = (str(path), path.stat().st_mtime)
    with _lock:
        d = _cache.get(key)
        if d is not None:
            _cache.move_to_end(key)
            return d
    with h5py.File(path, "r") as fh:
        d = DeltaPdfData(
            data=np.asarray(fh["data"][()], dtype=np.float64),
            x_axis=np.asarray(fh["x_axis"][()], dtype=float),
            y_axis=np.asarray(fh["y_axis"][()], dtype=float),
            z_axis=np.asarray(fh["z_axis"][()], dtype=float),
            lat_a=_attr(fh, "lat_a"),
            lat_b=_attr(fh, "lat_b"),
            lat_c=_attr(fh, "lat_c"),
            q_max=_attr(fh, "q_max"),
        )
    with _lock:
        _cache[key] = d
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return d


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def dpdf_meta(path: Path) -> dict:
    d = load_dpdf(path)
    return {
        "shape": [int(n) for n in d.data.shape],
        "x_range": [float(d.x_axis[0]), float(d.x_axis[-1])],
        "y_range": [float(d.y_axis[0]), float(d.y_axis[-1])],
        "z_range": [float(d.z_axis[0]), float(d.z_axis[-1])],
        "lattice": {"a": d.lat_a, "b": d.lat_b, "c": d.lat_c},
        "q_max": d.q_max,
        "planes": list(DPDF_PLANES),
    }


def _nearest(axis: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(axis - value)))


def _robust_far(data2d: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                r_min: float = 3.0, pct: float = 98.0) -> float:
    """p<pct> of |ΔPDF| at in-plane radius > r_min Å (skip near-origin spike)."""
    xg, yg = np.meshgrid(xs, ys, indexing="xy")  # (ny, nx), matching data2d [y, x]
    r = np.sqrt(xg**2 + yg**2)
    vals = np.abs(data2d[r > r_min])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        vals = np.abs(data2d[np.isfinite(data2d)])
    if vals.size == 0:
        return 1.0
    return float(np.percentile(vals, pct)) or 1.0


def dpdf_slice_envelope(path: Path, plane: str, value: float) -> bytes:
    """Extract one orthoslice and pack it into the binary slice envelope."""
    d = load_dpdf(path)
    if plane == "xy":  # x_H–y_K, fix z_L
        iz = _nearest(d.z_axis, value)
        data2d = np.ascontiguousarray(d.data[:, :, iz].T, dtype="<f4")
        xs, ys, xl, yl = d.x_axis, d.y_axis, "x_H (Å)", "y_K (Å)"
        cut = f"z_L = {float(d.z_axis[iz]):.3g} Å"
    elif plane == "xz":  # x_H–z_L, fix y_K
        iy = _nearest(d.y_axis, value)
        data2d = np.ascontiguousarray(d.data[:, iy, :].T, dtype="<f4")
        xs, ys, xl, yl = d.x_axis, d.z_axis, "x_H (Å)", "z_L (Å)"
        cut = f"y_K = {float(d.y_axis[iy]):.3g} Å"
    elif plane == "yz":  # y_K–z_L, fix x_H
        ix = _nearest(d.x_axis, value)
        data2d = np.ascontiguousarray(d.data[ix, :, :].T, dtype="<f4")
        xs, ys, xl, yl = d.y_axis, d.z_axis, "y_K (Å)", "z_L (Å)"
        cut = f"x_H = {float(d.x_axis[ix]):.3g} Å"
    else:
        raise ValueError(f"unknown ΔPDF plane {plane!r}")

    header = {
        "ny": int(data2d.shape[0]),
        "nx": int(data2d.shape[1]),
        "x_axis": np.asarray(xs, dtype=float).tolist(),
        "y_axis": np.asarray(ys, dtype=float).tolist(),
        "x_label": xl,
        "y_label": yl,
        "cut_label": cut,
        "robust_max": _robust_far(data2d, xs, ys),
    }
    hb = json.dumps(header).encode("utf-8")
    return struct.pack("<I", len(hb)) + hb + data2d.tobytes()
