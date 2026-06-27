# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tsung-han Yang

"""Volume loading (LRU cached) and 2D-slice extraction for the API.

Slices are extracted with the *same* :func:`nebula3d.visualization.extract_slice`
primitive the matplotlib viewers use, so the web viewer is pixel-for-pixel
consistent with them.  A loaded :class:`~nebula3d.core.HKLVolume` is ~130 MB, so a
small LRU keyed by ``(path, mtime)`` keeps cut-slider scrubbing responsive while
bounding memory.

Wire format for a slice (one request, self-contained):

    [uint32 LE: header_len][JSON header][float32 LE data, ny*nx, C-order]

The JSON header carries the plane's x/y axes, labels, the cut label, and a robust
colour-scale hint; masked voxels are NaN in the float32 payload (drawn grey by the
client).
"""

from __future__ import annotations

import json
import struct
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

import nebula3d
from nebula3d.core import HKLVolume
from nebula3d.visualization import extract_slice
from nebula3d.visualization.slices import _ALIASES, _PLANE, SliceData

#: Plane keys accepted by the slice endpoint (principal pairs + Mantid aliases).
PLANES: tuple[str, ...] = tuple(_PLANE.keys()) + tuple(_ALIASES.keys())

# The reciprocal-space viewer displays every cleanup stage of one dataset at
# once (raw → ring-removed → Bragg-punched → backfilled → flattened = up to 5
# HKLVolumes), all sharing one cut slider.  The cache must hold all of them
# simultaneously or each slider move evicts and re-loads ~130 MB volumes from
# disk, stalling the panels.  6 keeps a full dataset warm with one slot of
# headroom (~0.8 GB worst case).
_CACHE_MAX = 6
_cache: OrderedDict[tuple[str, float], HKLVolume] = OrderedDict()
_lock = threading.Lock()


def load_volume(path: Path) -> HKLVolume:
    """Load an nebula3d/Mantid volume, caching by ``(path, mtime)``."""
    key = (str(path), path.stat().st_mtime)
    with _lock:
        vol = _cache.get(key)
        if vol is not None:
            _cache.move_to_end(key)
            return vol
    vol = nebula3d.load(path)  # heavy I/O outside the lock
    with _lock:
        _cache[key] = vol
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return vol


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def lattice_constants(vol: HKLVolume) -> tuple[float | None, float | None, float | None]:
    """Direct-lattice a/b/c (Å) from the UB matrix, or ``None`` if singular."""
    try:
        direct = 2 * np.pi * np.linalg.inv(vol.ub_matrix).T
        return (
            float(np.linalg.norm(direct[:, 0])),
            float(np.linalg.norm(direct[:, 1])),
            float(np.linalg.norm(direct[:, 2])),
        )
    except np.linalg.LinAlgError:
        return None, None, None


def volume_meta(path: Path) -> dict:
    """Compact metadata for a volume: shape, axis ranges, lattice."""
    vol = load_volume(path)
    a, b, c = lattice_constants(vol)
    return {
        "shape": [int(n) for n in vol.data.shape],
        "h_range": [float(vol.h_axis[0]), float(vol.h_axis[-1])],
        "k_range": [float(vol.k_axis[0]), float(vol.k_axis[-1])],
        "l_range": [float(vol.l_axis[0]), float(vol.l_axis[-1])],
        "lattice": {"a": a, "b": b, "c": c},
        "ub_matrix": np.asarray(vol.ub_matrix, dtype=float).tolist(),
        "planes": list(PLANES),
    }


def _robust_max(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    return float(np.percentile(np.abs(finite), 99)) or 1.0


def pack_slice(sd: SliceData) -> bytes:
    """Pack a :class:`~nebula3d.visualization.slices.SliceData` into the wire format."""
    data = np.ascontiguousarray(sd.data, dtype="<f4")  # masked voxels are NaN
    header = {
        "ny": int(data.shape[0]),
        "nx": int(data.shape[1]),
        "x_axis": np.asarray(sd.x_axis, dtype=float).tolist(),
        "y_axis": np.asarray(sd.y_axis, dtype=float).tolist(),
        "x_label": sd.x_label,
        "y_label": sd.y_label,
        "cut_label": sd.cut_label,
        "robust_max": _robust_max(data),
    }
    hb = json.dumps(header).encode("utf-8")
    return struct.pack("<I", len(hb)) + hb + data.tobytes()


def slice_envelope(path: Path, plane: str, value: float, interp: bool) -> bytes:
    """Extract a 2D slice and pack it into the binary wire format above."""
    vol = load_volume(path)
    sd = extract_slice(vol, plane=plane, value=value, interp=interp)
    return pack_slice(sd)
