#!/usr/bin/env python
"""Export downsampled ΔPDF volumes as static assets for the GitHub Pages build.

The hosted (static) build of the web console has no FastAPI backend, so it cannot
load the ~270 MB ΔPDF ``.h5`` volumes or slice them server-side.  This script
bakes a *downsampled* copy of each dataset's ΔPDF volume into ``web/public/data/``
as a compact binary, plus a ``manifest.json`` the SPA reads in place of the
``/api`` endpoints.  The browser then slices the volume client-side (see
``web/src/api/staticData.ts``), so the 3D-ΔPDF and Multi-temperature viewers stay
fully interactive offline.

Binary volume format (little-endian), one file per dataset::

    [uint32 header_len][JSON header][float16 data, C-order [ix, iy, iz]]

The JSON header carries ``nx/ny/nz``, the three real-space axes (Å), the direct
lattice, and ``q_max``.  Half-float halves the payload; the client converts to
float32 once on load.

Usage::

    python scripts/export_web_assets.py [--data-root ./data] [--stride 2]
                                        [--out web/public/data]

Run it whenever the ΔPDF outputs change, then commit ``web/public/data/`` and let
the Pages workflow deploy.  See docs/github-pages-webgpu-plan.md.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from ndiff.server.config import ServerConfig
from ndiff.server.datasets import discover_datasets
from ndiff.server.deltapdf import load_dpdf

REPO_ROOT = Path(__file__).resolve().parent.parent


def _downsample_axis(axis: np.ndarray, stride: int) -> np.ndarray:
    return np.asarray(axis[::stride], dtype=float)


def export_volume(path: Path, stride: int) -> tuple[bytes, dict]:
    """Load a ΔPDF h5, downsample by ``stride``, return (binary, meta dict)."""
    d = load_dpdf(path)
    data = np.asarray(d.data[::stride, ::stride, ::stride], dtype=np.float16)
    data = np.ascontiguousarray(data)  # C-order [ix, iy, iz]
    xs = _downsample_axis(d.x_axis, stride)
    ys = _downsample_axis(d.y_axis, stride)
    zs = _downsample_axis(d.z_axis, stride)
    nx, ny, nz = (int(n) for n in data.shape)

    header = {
        "nx": nx, "ny": ny, "nz": nz,
        "x_axis": xs.tolist(),
        "y_axis": ys.tolist(),
        "z_axis": zs.tolist(),
        "lattice": {"a": d.lat_a, "b": d.lat_b, "c": d.lat_c},
        "q_max": d.q_max,
    }
    hb = json.dumps(header).encode("utf-8")
    blob = struct.pack("<I", len(hb)) + hb + data.tobytes()

    meta = {
        "shape": [nx, ny, nz],
        "x_range": [float(xs[0]), float(xs[-1])],
        "y_range": [float(ys[0]), float(ys[-1])],
        "z_range": [float(zs[0]), float(zs[-1])],
        "lattice": {"a": d.lat_a, "b": d.lat_b, "c": d.lat_c},
        "q_max": d.q_max,
        "planes": ["xy", "xz", "yz"],
    }
    return blob, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=None,
                    help="Directory with raw/ and processed/ (default $NDIFF_DATA_ROOT or ./data)")
    ap.add_argument("--stride", type=int, default=2,
                    help="Downsample stride on each axis (default 2)")
    ap.add_argument("--out", default=str(REPO_ROOT / "web" / "public" / "data"),
                    help="Output directory for the static assets")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser().resolve() if args.data_root else (Path.cwd() / "data")
    cfg = ServerConfig(data_root=root)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(cfg)
    manifest_datasets = []
    total_bytes = 0

    for ds in datasets:
        dpdf_stage = next((s for s in ds.stages if s.kind == "delta_pdf" and s.exists), None)
        if dpdf_stage is None:
            print(f"  skip {ds.temperature or ds.stem}: no ΔPDF volume")
            continue

        blob, meta = export_volume(dpdf_stage.path, args.stride)
        asset_name = f"{ds.id}.dpdf.bin"
        (out_dir / asset_name).write_bytes(blob)
        total_bytes += len(blob)
        print(f"  {ds.temperature or ds.stem}: {meta['shape']}  {len(blob) / 1e6:.1f} MB  → {asset_name}")

        manifest_datasets.append({
            "id": ds.id,
            "temperature": ds.temperature,
            "raw_name": ds.raw_name,
            "stem": ds.stem,
            "stages": [{
                "name": "delta_pdf",
                "exists": True,
                "kind": "delta_pdf",
                "volume_id": f"{ds.id}.delta_pdf",
            }],
            "dpdf": {"asset": f"data/{asset_name}", "meta": meta},
        })

    manifest = {
        "version": 1,
        "mode": "static",
        "stride": args.stride,
        "datasets": manifest_datasets,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(manifest_datasets)} dataset(s), {total_bytes / 1e6:.1f} MB → {out_dir}")
    print(f"manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
