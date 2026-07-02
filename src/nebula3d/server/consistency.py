# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tsung-han Yang

"""Back-FFT consistency check for the API.

Runs the |Q|-band-limited ΔPDF round trip (``nebula3d.pipeline.consistency_reconstruction``)
for a dataset and serves the resulting reciprocal-space volumes — ``data`` (the
band-limited input), ``recon`` (the inverse-FFT of that band's ΔPDF), and
``residual`` — as the same binary slice envelope the cleanup viewer uses, plus
the agreement metrics.

A reconstruction (one forward+inverse FFT) is the expensive part, so it is LRU
cached by ``(pdf-input path, mtime, |Q| band)``: changing the displayed plane/cut
re-slices the cached volumes (cheap), and only a new |Q| band recomputes.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

from nebula3d.pipeline import (
    DeltaPdfParams,
    consistency_reconstruction,
    write_delta_pdf_h5,
)
from nebula3d.server.config import ServerConfig
from nebula3d.server.datasets import find_dataset
from nebula3d.server.volumes import PLANES, lattice_constants, load_volume, pack_slice
from nebula3d.visualization import extract_slice
from nebula3d.visualization.slices import extract_slice_dpdf

#: The three comparison volumes the viewer can slice.
PANELS: tuple[str, ...] = ("data", "recon", "residual", "dpdf")

_CACHE_MAX = 4
_cache: OrderedDict[tuple, dict] = OrderedDict()
_lock = threading.Lock()


def set_cache_max(n: int) -> None:
    """Cap the reconstruction cache (evicting oldest); the browser build
    shrinks it — each entry holds four volume-sized arrays (data / recon /
    residual / ΔPDF)."""
    global _CACHE_MAX
    with _lock:
        _CACHE_MAX = max(1, int(n))
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)


def pdf_input_path(cfg: ServerConfig, dataset_id: str) -> Path | None:
    """The volume the ΔPDF was built from: flattened if present, else backfilled."""
    ds = find_dataset(cfg, dataset_id)
    if ds is None:
        return None
    by_name = {s.name: s for s in ds.stages}
    for name in ("flattened", "backfilled"):
        stage = by_name.get(name)
        if stage is not None and stage.exists:
            return stage.path
    return None


def _band_key(q_band: tuple[float, float] | None, r_band: tuple[float, float] | None) -> tuple:
    qk = (round(float(q_band[0]), 4), round(float(q_band[1]), 4)) if q_band else None
    rk = (round(float(r_band[0]), 4), round(float(r_band[1]), 4)) if r_band else None
    return (qk, rk)


def reconstruction(
    path: Path, q_band: tuple[float, float] | None, r_band: tuple[float, float] | None
) -> dict:
    """LRU-cached ``consistency_reconstruction`` for *path* + bands."""
    key = (str(path), path.stat().st_mtime, _band_key(q_band, r_band))
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit
    vol = load_volume(path)  # shared with the slice viewers' cache
    res = consistency_reconstruction(
        vol, DeltaPdfParams(crop_hkl=None), q_band=q_band, r_band=r_band
    )
    with _lock:
        _cache[key] = res
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return res


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def consistency_meta(
    path: Path, q_band: tuple[float, float] | None, r_band: tuple[float, float] | None
) -> dict:
    """Grid ranges, available span, and the agreement metrics."""
    res = reconstruction(path, q_band, r_band)
    recon = res["recon"]
    a, b, c = lattice_constants(recon)
    return {
        "shape": [int(n) for n in recon.data.shape],
        "h_range": [float(recon.h_axis[0]), float(recon.h_axis[-1])],
        "k_range": [float(recon.k_axis[0]), float(recon.k_axis[-1])],
        "l_range": [float(recon.l_axis[0]), float(recon.l_axis[-1])],
        "dpdf_shape": [int(n) for n in res["dpdf"].data.shape],
        "x_range": [float(res["dpdf"].x_axis[0]), float(res["dpdf"].x_axis[-1])],
        "y_range": [float(res["dpdf"].y_axis[0]), float(res["dpdf"].y_axis[-1])],
        "z_range": [float(res["dpdf"].z_axis[0]), float(res["dpdf"].z_axis[-1])],
        "lattice": {"a": a, "b": b, "c": c},
        "planes": list(PLANES),
        "q_data_max": res["metrics"]["q_data_max"],
        "r_data_max": res["metrics"]["r_data_max"],
        "metrics": res["metrics"],
    }


def _band_tag(q_band: tuple[float, float] | None,
              r_band: tuple[float, float] | None) -> str:
    """Filename suffix encoding the selected bands (so distinct selections
    don't overwrite each other); ``full`` when neither band is set."""
    parts = []
    if q_band is not None:
        parts.append(f"q{q_band[0]:g}-{q_band[1]:g}")
    if r_band is not None:
        parts.append(f"r{r_band[0]:g}-{r_band[1]:g}")
    return "_".join(parts) if parts else "full"


def save_reconstruction(
    path: Path, q_band: tuple[float, float] | None,
    r_band: tuple[float, float] | None, out_dir: Path,
) -> Path:
    """Write the band-limited 3D-ΔPDF (the workflow's final product) to HDF5.

    ``res["dpdf"]`` is the cleaned real-space ΔPDF after the selected |Q| band
    (applied before the forward FFT) and |R| band (masked after) — i.e. exactly
    what the viewer shows.  Reuses the cached reconstruction, so saving right
    after viewing recomputes nothing.
    """
    res = reconstruction(path, q_band, r_band)
    vol = load_volume(path)  # cached; supplies UB / lattice for provenance
    out_dir = Path(out_dir)
    out_path = out_dir / f"{path.stem}_delta_pdf_{_band_tag(q_band, r_band)}.h5"
    p = DeltaPdfParams(crop_hkl=None, q_band=q_band)
    write_delta_pdf_h5(
        res["dpdf"], vol, p, source_name=path.name, out_path=out_path,
        r_band=r_band)
    return out_path


def consistency_slice_envelope(
    path: Path, q_band: tuple[float, float] | None, r_band: tuple[float, float] | None,
    panel: str, plane: str, value: float,
) -> bytes:
    """Binary slice envelope of one comparison *panel* at *plane*/*value*."""
    res = reconstruction(path, q_band, r_band)
    if panel == "dpdf":
        sd = extract_slice_dpdf(res["dpdf"], plane=plane, value=value)
    else:
        sd = extract_slice(res[panel], plane=plane, value=value)
    return pack_slice(sd)
