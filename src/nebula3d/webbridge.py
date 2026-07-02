# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tsung-han Yang

"""In-browser (Pyodide) bridge: drive the real pipeline + viewers, no server.

This is the backend-less twin of :mod:`nebula3d.server` for the GitHub Pages build.
When the SPA runs under Pyodide there is no FastAPI process, so the React data
layer calls these functions directly instead of ``fetch("/api/...")``.  Each one
mirrors the corresponding API endpoint and returns the *same* payload — a JSON
string for metadata/dataset listings, or the binary slice envelope
(``[uint32 header_len][JSON header][float32 data]``) the viewers already decode —
so the front-end is unchanged apart from where the bytes come from.

The heavy lifting (slicing, ΔPDF, consistency, dataset discovery) is **not**
reimplemented here: it reuses the FastAPI-free helper modules under
:mod:`nebula3d.server` (``volumes``, ``deltapdf``, ``consistency``, ``datasets``,
``params``).  Those import only numpy/scipy/h5py/nebula3d — all available in Pyodide
— which is why :mod:`nebula3d.server` imports ``create_app`` lazily.

Workflow (one dataset per browser session):

    setup()                       → create the virtual workspace
    load_input(name, tmp_path)    → register the user's uploaded volume
    run(stages, params_json, …)   → run_pipeline (streams per-stage progress)
    datasets_json()               → dataset + per-stage status (for the viewers)
    volume_slice / dpdf_slice / consistency_slice → binary envelopes

All file I/O happens in Pyodide's in-memory filesystem; nothing is uploaded.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import nebula3d
from nebula3d.pipeline import STAGES, pipeline_paths, run_pipeline
from nebula3d.server import consistency as _cons
from nebula3d.server import datasets as _ds
from nebula3d.server import deltapdf as _dpdf
from nebula3d.server import volumes as _vol
from nebula3d.server.config import ServerConfig
from nebula3d.server.params import build_params

if TYPE_CHECKING:
    from nebula3d.server.schemas import PipelineRunRequest

__all__ = [
    "setup",
    "inspect_input",
    "load_input",
    "make_demo_input",
    "run",
    "datasets_json",
    "volume_meta_json",
    "volume_slice",
    "dpdf_meta_json",
    "dpdf_slice",
    "consistency_meta_json",
    "consistency_slice",
    "bragg_profile_json",
    "save_dpdf",
]

# In-browser memory budget.  Pyodide runs in a 32-bit-WASM heap whose hard
# ceiling is 4 GB (Pyodide ≥ 0.27; MAXIMUM_MEMORY=4GB), so a full float64
# reduction of a very large volume can simply not fit (see the memory-ceiling
# notes in docs/web.md, "In-browser run").  After the memory rework (broadcast
# |Q| accumulation instead of meshgrids, next_fast_len FFT padding, prompt
# frees, capped caches, consumed ΔPDF in the check stage) the worst stage —
# the back-FFT consistency check — measures ~7.3 live volume-sized float64
# arrays (58 B/voxel peak RSS on a 200³ run); every other stage is ≤6.
# 8×8 B/voxel keeps ~10% safety margin on top of that measurement.  Volumes
# whose estimated peak exceeds the budget are refused at load with a clear
# message rather than allowed to crash mid-pipeline with a numpy MemoryError.
# Native ``nebula3d-web`` has no such limit; this gate lives only here, in the
# browser bridge.
_PIPELINE_PEAK_BYTES_PER_VOXEL = 8 * 8
# 4 GB WASM ceiling minus the Pyodide runtime + packages (~0.5 GB) and headroom
# for growth fragmentation (wasm memory never shrinks) → 50 M voxels pass the
# gate (e.g. a 301×401×401 full-resolution volume = 48.4 M voxels ≈ 2.9 GB
# estimated peak).
_BROWSER_PEAK_BUDGET_BYTES = 3_200_000_000


# ---------------------------------------------------------------------------
# Session state (one workspace / one loaded dataset per browser tab)
# ---------------------------------------------------------------------------
class _State:
    cfg: ServerConfig | None = None
    input: Path | None = None
    dataset_id: str | None = None


_S = _State()


def _require_cfg() -> ServerConfig:
    if _S.cfg is None:
        setup()
    assert _S.cfg is not None
    return _S.cfg


def setup(workdir: str = "/work") -> str:
    """Create the virtual workspace (``raw/`` + ``processed/``); return its root.

    Idempotent: re-calling keeps any already-loaded input but ensures the dirs
    exist and resets the config.  Also shrinks the server-side slice caches:
    natively they keep whole float64 volumes warm for snappy sliders, but in
    the 4 GB WASM heap that residency would stack on top of the pipeline's own
    peak (a full dataset's cache alone can exceed the heap at full resolution).
    """
    root = Path(workdir)
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "processed").mkdir(parents=True, exist_ok=True)
    _S.cfg = ServerConfig(data_root=root)
    _vol.set_cache_max(2)
    _dpdf.set_cache_max(1)
    _cons.set_cache_max(1)
    return str(root)


def _clear_caches() -> None:
    _vol.clear_cache()
    _dpdf.clear_cache()
    _cons.clear_cache()


def _safe_stem(name: str) -> str:
    """Filename → a clean stem for the raw ``.nxs`` (drops the extension)."""
    stem = Path(name).name
    # Strip a known volume extension; keep the rest verbatim so condition labels
    # survive for dataset grouping / display.
    for ext in (".nxs", ".hdf5", ".h5", ".txt", ".dat", ".hkl"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    stem = stem.strip() or "volume"
    return stem


def _peek_voxel_count(path: Path) -> tuple[tuple[int, ...], int]:
    """Signal-grid shape + voxel count from HDF5 *metadata* only (no array read).

    Handles both supported layouts — Mantid ``MDHistoWorkspace/data/signal`` and
    nebula3d ``entry/data``.  Reading ``Dataset.shape`` never loads the data, so this
    is safe to call on a volume too large to fit in memory.  Returns ``((), 0)``
    for an unrecognised file (callers then skip the size gate).
    """
    import h5py

    with h5py.File(path, "r") as f:
        if "MDHistoWorkspace" in f:
            shape = tuple(f["MDHistoWorkspace/data/signal"].shape)
        elif "entry" in f and "data" in f["entry"]:
            shape = tuple(f["entry/data"].shape)
        else:
            return (), 0
    n = 1
    for s in shape:
        n *= int(s)
    return shape, n


def inspect_input(name: str, tmp_path: str) -> str:
    """Pre-flight memory estimate for an uploaded volume (metadata only).

    Reads just the signal-grid shape — never the arrays — so it cannot itself run
    out of memory, then estimates the full-pipeline peak and compares it to the
    browser budget.  Returns JSON ``{shape, n_voxels, est_peak_mb, ok, message}``;
    the engine refuses to load when ``ok`` is false, surfacing *message* instead
    of letting the reduction crash with an opaque numpy ``MemoryError``.
    """
    shape, n = _peek_voxel_count(Path(tmp_path))
    est_peak = n * _PIPELINE_PEAK_BYTES_PER_VOXEL
    ok = n == 0 or est_peak <= _BROWSER_PEAK_BUDGET_BYTES
    budget_voxels = _BROWSER_PEAK_BUDGET_BYTES // _PIPELINE_PEAK_BYTES_PER_VOXEL

    message = ""
    if not ok:
        dims = "×".join(str(s) for s in shape)
        message = (
            f"“{Path(name).name}” is {dims} ({n / 1e6:.1f} M voxels). Reducing it "
            f"would need roughly {est_peak / 1e9:.1f} GB of browser memory — more "
            f"than the in-browser engine can hold (it targets volumes up to about "
            f"{budget_voxels / 1e6:.0f} M voxels). For full-resolution data this "
            f"large, run the native build, which has no memory limit and opens the "
            f"same interface:\n"
            f'    pip install "nebula3d[web]"  &&  nebula3d-web'
        )
    return _json({
        "shape": list(shape),
        "n_voxels": n,
        "est_peak_mb": est_peak / 1e6,
        "ok": ok,
        "message": message,
    })


def load_input(name: str, tmp_path: str) -> str:
    """Register an uploaded volume (already written to *tmp_path* in the FS).

    Copies it to ``raw/<stem>.nxs`` (``nebula3d.load`` content-detects Mantid vs
    nebula3d-HDF5, so the ``.nxs`` extension is fine for either) and returns the
    dataset id the viewers will use.  Clears the slice caches so a re-load does
    not serve a previous volume.
    """
    cfg = _require_cfg()
    stem = _safe_stem(name)
    dest = cfg.raw_dir / f"{stem}.nxs"
    shutil.copyfile(tmp_path, dest)
    _S.input = dest
    _S.dataset_id = _ds._slug(stem)
    _clear_caches()
    return _S.dataset_id


def make_demo_input(n: int = 33) -> str:
    """Write a small synthetic **FCC** HKL volume to the workspace; return its id.

    A physically-flavoured demo for when the user has no data, so every pipeline
    stage has realistic structure to act on:

    - **FCC Bragg peaks** at integer (h,k,l) with *unmixed parity* (all even or
      all odd) — the FCC reflection condition; mixed-parity nodes (100, 210, …)
      are systematically absent.  Sharp, with a Debye-Waller-like |Q| falloff.
    - **Diffuse scattering along one axis**: continuous rods running along **L**
      through the in-plane nodes (transverse-narrow, L-extended, modulated so the
      diffuse is strongest between Bragg peaks) — the signature of 1-D
      correlations / planar disorder.
    - a smooth thermal-diffuse background, a faint powder ring, and a bright
      incident-beam spot at the origin, plus a little noise.

    Cubic lattice ``a = 4 Å`` (``ub = (2π/a)·I``) so |Q| is physical.  Kept tiny
    (``n³``) so the full chain runs in seconds under Pyodide.
    """
    import numpy as np

    cfg = _require_cfg()
    extent = 4.0
    a = 4.0  # cubic lattice parameter (Å) → isotropic reciprocal metric
    ub = (2.0 * np.pi / a) * np.eye(3, dtype=np.float64)

    h = np.linspace(-extent, extent, n)
    step = float(h[1] - h[0])
    H, K, L = np.meshgrid(h, h, h, indexing="ij")
    q2 = H**2 + K**2 + L**2  # |hkl|² in r.l.u.

    # Smooth thermal-diffuse background + a Debye-Waller-like high-|Q| falloff
    # shared by the Bragg peaks and the diffuse rods.
    data = 0.6 + 3.0 * np.exp(-q2 / 6.0)
    falloff = np.exp(-0.10 * q2)

    nodes = range(int(np.ceil(-extent)), int(np.floor(extent)) + 1)
    sig_b = max(0.10, 0.9 * step)  # Bragg half-width (r.l.u.)
    for ih in nodes:
        for ik in nodes:
            for il in nodes:
                all_even = (ih % 2 == 0) and (ik % 2 == 0) and (il % 2 == 0)
                all_odd = (ih % 2 != 0) and (ik % 2 != 0) and (il % 2 != 0)
                if not (all_even or all_odd):
                    continue  # FCC systematic absence (mixed parity)
                rn2 = ih * ih + ik * ik + il * il
                amp = 26.0 * np.exp(-0.10 * rn2)
                data += amp * np.exp(
                    -((H - ih) ** 2 + (K - ik) ** 2 + (L - il) ** 2)
                    / (2.0 * sig_b * sig_b)
                )

    # Diffuse rods along L through the in-plane same-parity columns (|h|,|k| ≤ 2),
    # so they thread the FCC nodes.  Narrow in H,K; extended in L; modulated to be
    # brightest midway between Bragg peaks (cos² along L) — 1-D-correlation diffuse.
    sig_t = max(0.16, 1.1 * step)
    l_mod = 0.5 - 0.5 * np.cos(2.0 * np.pi * L)  # 0 at integer L, 1 at half-integer
    for h0 in (-2, -1, 0, 1, 2):
        for k0 in (-2, -1, 0, 1, 2):
            if (h0 % 2) != (k0 % 2):
                continue  # keep columns that pass through FCC nodes
            amp = 3.2 * np.exp(-0.18 * (h0 * h0 + k0 * k0))
            data += amp * l_mod * np.exp(
                -((H - h0) ** 2 + (K - k0) ** 2) / (2.0 * sig_t * sig_t)
            )

    data *= falloff
    data += 1.2 * np.exp(-((np.sqrt(q2) - 3.0) ** 2) / 0.05)  # faint powder ring
    data += 60.0 * np.exp(-q2 / (2.0 * sig_b * sig_b))        # incident beam @ origin

    rng = np.random.default_rng(0)
    data = np.clip(data + 0.02 * rng.standard_normal(data.shape), 0.0, None)

    vol = nebula3d.core.HKLVolume.from_arrays(
        data.astype(np.float64),
        (-extent, extent), (-extent, extent), (-extent, extent),
        ub_matrix=ub,
    )
    dest = cfg.raw_dir / "demo_fcc.nxs"
    nebula3d.save(vol, dest)
    _S.input = dest
    _S.dataset_id = _ds._slug("demo_fcc")
    _clear_caches()
    return _S.dataset_id


# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------
class _ParamsNS:
    """Attribute view over the params dict; any unset field reads back as None.

    Matches the Pydantic ``StageParamsIn`` (all fields default to ``None``) that
    :func:`build_params` expects, without importing Pydantic.
    """

    def __init__(self, d: dict[str, object]) -> None:
        self.__dict__.update(d)

    def __getattr__(self, _name: str) -> None:  # only for fields not in __dict__
        return None


def _make_request(params_json: str, flatten_enabled: bool) -> PipelineRunRequest:
    """Duck-typed request for :func:`build_params` from a JSON params dict.

    ``build_params`` only does attribute access, so a namespace stands in for the
    Pydantic ``PipelineRunRequest`` (which we cannot import under Pyodide); the
    cast keeps the type checker happy without a runtime Pydantic dependency.
    """
    raw = json.loads(params_json) if params_json else {}
    req = SimpleNamespace(flatten_enabled=bool(flatten_enabled), params=_ParamsNS(raw))
    return cast("PipelineRunRequest", req)


def run(
    stages_csv: str,
    params_json: str,
    flatten_enabled: bool,
    force: bool = False,
    force_from: str | None = None,
    progress: Callable[[str, str, float | None, str], None] | None = None,
) -> str:
    """Run the selected pipeline *stages* on the loaded input; return datasets JSON.

    *stages_csv* is a comma-separated subset of :data:`nebula3d.pipeline.STAGES`
    (empty = all), so the JS caller can drive the pipeline one stage at a time
    and repaint between stages.  *params_json* is the curated ``StageParamsIn``
    override dict; *progress* is an optional JS callback
    ``progress(stage, status, fraction, message)`` streamed during the run.
    """
    cfg = _require_cfg()
    if _S.input is None:
        raise RuntimeError("no input loaded; call load_input() first")
    # Free any volumes the slice viewers have cached: the pipeline needs the
    # whole WASM heap for its own peak, and stale cache entries for outputs
    # this run will overwrite are useless anyway.
    _clear_caches()
    stages = tuple(s for s in (stages_csv.split(",") if stages_csv else [])
                   if s) or STAGES
    params = build_params(_make_request(params_json, flatten_enabled))

    cb = None
    if progress is not None:
        def cb(stage: str, status: str, fraction: float | None, message: str) -> None:
            progress(stage, status, fraction, message)  # type: ignore[misc]

    run_pipeline(
        _S.input, params, proc_dir=cfg.processed_dir, stages=stages,
        force=bool(force), force_from=force_from, progress=cb,
    )
    return datasets_json()


# ---------------------------------------------------------------------------
# JSON helpers (sanitise non-finite floats so JSON.parse in the browser is happy)
# ---------------------------------------------------------------------------
def _finite(obj: object) -> object:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def _json(obj: object) -> str:
    return json.dumps(_finite(obj))


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def datasets_json() -> str:
    """List discovered datasets + per-stage status (mirrors GET /api/datasets)."""
    cfg = _require_cfg()
    out = []
    for ds in _ds.discover_datasets(cfg):
        stages = [
            {"name": s.name, "exists": s.exists, "kind": s.kind,
             "volume_id": f"{ds.id}.{s.name}"}
            for s in ds.stages
        ]
        out.append({
            "id": ds.id, "temperature": ds.temperature, "raw_name": ds.raw_name,
            "stem": ds.stem, "stages": stages,
        })
    return _json(out)


def bragg_profile_json(dataset_id: str) -> str:
    """Per-peak Bragg punch metadata (mirrors GET /api/bragg/{id}/profile).

    The punch stage writes ``*_profile.json`` next to its output volume — in
    the browser that is the Pyodide virtual FS — so this only re-reads a small
    JSON file.  The ``has_profile: false`` shape matches the native
    ``BraggProfileOut`` defaults so the viewer's empty states are identical.
    """
    cfg = _require_cfg()
    ds = _ds.find_dataset(cfg, dataset_id)
    if ds is None:
        raise KeyError(f"unknown dataset id {dataset_id!r}")
    path = pipeline_paths(ds.raw_path, proc_dir=cfg.processed_dir).bragg_profile_json
    if not path.exists():
        return _json({
            "dataset_id": dataset_id, "profile_path": str(path),
            "has_profile": False, "schema_version": 1,
            "width_labels": [], "hkl_width_labels": [], "width_units": {},
            "n_peaks": 0, "fit_covariance": False, "punch_frame": None,
            "peaks": [],
        })
    data = json.loads(path.read_text(encoding="utf-8"))
    return _json({
        "dataset_id": dataset_id,
        "profile_path": str(path),
        "has_profile": True,
        "schema_version": int(data.get("schema_version", 1)),
        "width_labels": list(data.get("width_labels", [])),
        "hkl_width_labels": list(data.get("hkl_width_labels", [])),
        "width_units": dict(data.get("width_units", {})),
        "n_peaks": int(data.get("n_peaks", 0)),
        "fit_covariance": bool(data.get("fit_covariance", False)),
        "punch_frame": data.get("punch_frame"),
        "peaks": list(data.get("peaks", [])),
    })


def _resolve(volume_id: str, kind: str) -> _ds.StageStatus:
    cfg = _require_cfg()
    stage = _ds.resolve_volume(cfg, volume_id)
    if stage is None:
        raise KeyError(f"unknown volume id {volume_id!r}")
    if not stage.path.exists():
        raise FileNotFoundError(f"stage output not found for {volume_id!r}")
    if stage.kind != kind:
        raise ValueError(f"{volume_id!r} is a {stage.kind} volume, expected {kind}")
    return stage


# ---------------------------------------------------------------------------
# Reciprocal-space (HKL) volumes
# ---------------------------------------------------------------------------
def volume_meta_json(volume_id: str) -> str:
    """Metadata for an HKL stage (mirrors GET /api/volumes/{id}/meta)."""
    stage = _resolve(volume_id, "hkl")
    m = _vol.volume_meta(stage.path)
    m.update(id=volume_id, stage=stage.name, kind=stage.kind)
    return _json(m)


def volume_slice(volume_id: str, plane: str, value: float, interp: bool = False
                 ) -> bytes:
    """Binary slice envelope of an HKL stage (mirrors /api/volumes/{id}/slice)."""
    stage = _resolve(volume_id, "hkl")
    return _vol.slice_envelope(stage.path, plane=plane, value=float(value),
                               interp=bool(interp))


# ---------------------------------------------------------------------------
# Real-space ΔPDF
# ---------------------------------------------------------------------------
def dpdf_meta_json(volume_id: str) -> str:
    """Metadata for a ΔPDF stage (mirrors GET /api/deltapdf/{id}/meta)."""
    stage = _resolve(volume_id, "delta_pdf")
    m = _dpdf.dpdf_meta(stage.path)
    m["id"] = volume_id
    return _json(m)


def dpdf_slice(volume_id: str, plane: str, value: float) -> bytes:
    """Binary slice envelope of a ΔPDF stage (mirrors /api/deltapdf/{id}/slice)."""
    stage = _resolve(volume_id, "delta_pdf")
    return _dpdf.dpdf_slice_envelope(stage.path, plane=plane, value=float(value))


# ---------------------------------------------------------------------------
# Back-FFT consistency check
# ---------------------------------------------------------------------------
def _band(lo: float | None, hi: float | None) -> tuple[float, float] | None:
    return (float(lo), float(hi)) if lo is not None and hi is not None else None


def _pdf_input(dataset_id: str) -> Path:
    cfg = _require_cfg()
    path = _cons.pdf_input_path(cfg, dataset_id)
    if path is None:
        raise FileNotFoundError(
            f"no ΔPDF-input volume (flattened/backfilled) for {dataset_id!r}")
    return path


def consistency_meta_json(
    dataset_id: str,
    q_min: float | None = None,
    q_max: float | None = None,
    r_min: float | None = None,
    r_max: float | None = None,
) -> str:
    """Consistency grid + agreement metrics (mirrors /api/consistency/{id}/meta)."""
    path = _pdf_input(dataset_id)
    meta = _cons.consistency_meta(path, _band(q_min, q_max), _band(r_min, r_max))
    return _json(meta)


def consistency_slice(
    dataset_id: str,
    panel: str,
    plane: str,
    value: float,
    q_min: float | None = None,
    q_max: float | None = None,
    r_min: float | None = None,
    r_max: float | None = None,
) -> bytes:
    """Binary slice envelope of one comparison panel (data/recon/residual/dpdf)."""
    path = _pdf_input(dataset_id)
    return _cons.consistency_slice_envelope(
        path, _band(q_min, q_max), _band(r_min, r_max),
        panel=panel, plane=plane, value=float(value))


def save_dpdf(
    dataset_id: str,
    q_min: float | None = None,
    q_max: float | None = None,
    r_min: float | None = None,
    r_max: float | None = None,
) -> bytes:
    """Band-limited ΔPDF ``.h5`` as a download envelope
    (``[uint32 LE header_len][JSON {"filename"}][h5 bytes]``).

    Mirrors POST /api/consistency/{id}/save, but the browser build has no disk
    to save to: the file is written to the virtual FS, read back, unlinked (to
    reclaim MEMFS space and keep dataset discovery clean), and handed to JS,
    which turns it into a browser download.
    """
    cfg = _require_cfg()
    path = _pdf_input(dataset_id)
    out = _cons.save_reconstruction(
        path, _band(q_min, q_max), _band(r_min, r_max), cfg.processed_dir)
    payload = out.read_bytes()
    out.unlink()
    header = json.dumps({"filename": out.name}).encode("utf-8")
    return struct.pack("<I", len(header)) + header + payload
