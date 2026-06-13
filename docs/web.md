# Web UI

`neutron-diffuse` ships a browser-based front-end (FastAPI backend + React SPA)
that unifies the cleanup and 3D-ΔPDF viewers and can drive the whole pipeline from
a raw `.nxs`. It is the recommended way to explore results; the standalone
matplotlib viewers in `examples/explore_*.py` remain as a fallback.

## Launch

```bash
pip install -e ".[web]"
ndiff-web                       # serves http://127.0.0.1:8000 and opens a browser
```

By default it reads `./data` (the `raw/` and `processed/` layout). Override:

```bash
ndiff-web --data-root /path/to/data --port 8000
ndiff-web --no-browser          # headless (e.g. remote)
```

The installed wheel bundles the built SPA, so `ndiff-web` is the only command
needed. If you are running from a source checkout that has not been built yet,
build the frontend first (see below) — otherwise only the API is served.

## What it does

A single-page console with a left sidebar; the four views (in sidebar order)
replace the standalone `examples/explore_*.py` viewers, which remain as a
fallback:

| View | Replaces | What |
| --- | --- | --- |
| **Run pipeline** | `run_pipeline.py` | Pick a dataset and tune the key parameters per stage — ring removal (azimuthal **patches** and texture **Fourier order**), punch, backfill, flatten, and ΔPDF — then run all stages with a live stage stepper and log. Existing outputs are skipped unless *force* is on. This is the default landing view. |
| **Reciprocal cleanup** | `explore_slice.py` | One panel per HKLVolume stage (raw / ring-removed / Bragg-punched / backfilled / flattened) sharing an H/K/L plane selector, cut, contrast, log, and colormap. All panels share **one fixed global colour scale** (pooled from the centre cut), so stages are directly comparable and the scale stays put while you scrub. The cut readout is an **editable box** — type a value (e.g. `0.3333`) and it snaps to the nearest plane. |
| **3D-ΔPDF** | `explore_delta_pdf_ortho.py` | Three linked real-space orthoslices (x_H–y_K, x_H–z_L, y_K–z_L) shown as square **windows** (adjustable size, default 80 Å), each with its own cut slider directly above it, plus contrast and a gray dashed unit-cell overlay. |
| **Multi-temperature** | `explore_delta_pdf_multi.py` | 22 / 45 / 100 K × the three planes as a square grid, sharing the cut, window, and contrast; a per-plane colour scale pooled across temperatures. |

## Architecture

```
Browser (React/TS SPA)  ──HTTP/SSE──►  FastAPI (uvicorn)  ──►  ndiff library
  client-side colormap                  /api/datasets          extract_slice
  cut / contrast / log                  /api/volumes/.../slice  (visualization)
  pipeline param forms                  /api/deltapdf/.../slice
  SSE progress + log                    /api/pipeline/run + SSE  ndiff.pipeline
```

- **Slices** are extracted server-side with the same
  `ndiff.visualization.extract_slice` the matplotlib viewers use, returned as a
  compact binary envelope (`[uint32 header_len][JSON header][float32 data]`), and
  colour-mapped in the browser — so contrast/log/colormap change instantly with no
  refetch.
- **Pipeline runs** execute `ndiff.pipeline.run_pipeline` in a separate process
  (`multiprocessing` spawn), streaming progress over Server-Sent Events. Cancel
  terminates the worker.
- Loaded volumes are kept in an LRU cache sized to hold every cleanup stage of a
  dataset at once, so the shared cut slider scrubs without re-reading the ~100 MB
  volumes from disk.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/datasets` | datasets grouped by temperature, with per-stage output status |
| GET | `/api/volumes/{id}/meta` | HKLVolume shape, axis ranges, lattice |
| GET | `/api/volumes/{id}/slice?plane=&value=&interp=` | binary 2D slice |
| GET | `/api/deltapdf/{id}/meta` | ΔPDF shape, ranges, lattice, |Q|max |
| GET | `/api/deltapdf/{id}/slice?plane=xy\|xz\|yz&value=` | binary ΔPDF orthoslice |
| POST | `/api/pipeline/run` | start a job; returns `{id, status, ...}` |
| GET | `/api/pipeline/jobs/{id}/events` | SSE progress stream |
| POST | `/api/pipeline/jobs/{id}/cancel` | terminate a running job |

Volume ids are `"<dataset_id>.<stage>"` (e.g. `…22K….backfilled`,
`…22K….delta_pdf`).

## Development

Run the backend and the Vite dev server (with hot reload) separately:

```bash
# terminal 1 — API on :8000
ndiff-web --no-browser --reload

# terminal 2 — Vite dev server on :5173 (proxies /api to :8000)
cd web && npm install && npm run dev
```

Open http://localhost:5173.

Build the production SPA (emitted into `src/ndiff/server/static/`, which the
FastAPI app then serves and the wheel bundles):

```bash
cd web && npm run build
```

Frontend checks (run in CI):

```bash
cd web && npm run lint && npm run typecheck && npm run build
```

## Packaging note

The built SPA in `src/ndiff/server/static/` is a build artifact (git-ignored).
A release build is: `cd web && npm ci && npm run build`, then build the wheel —
`package-data` bundles `static/**/*` so the published wheel serves the UI with no
Node toolchain on the user's side.
