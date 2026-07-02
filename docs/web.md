# Web UI

`nebula3d` ships **one** browser console — a React + TypeScript SPA (Vite)
that unifies the cleanup, 3D-ΔPDF, multi-volume, and consistency views and
drives the whole reduction pipeline. It has two interchangeable run modes that
share the same UI and the same `nebula3d` reduction code:

| Mode | `VITE_DATA_MODE` | Backend | Use it for |
| --- | --- | --- | --- |
| **Native** | unset (default) | FastAPI (`nebula3d-web`) over `/api` | Full-resolution local work, no size limit |
| **In-browser** | `pyodide` | none — `nebula3d` runs in the browser via Pyodide | No-install / sharing / demo; modest volumes |

The standalone matplotlib viewers in `examples/explore_*.py` remain as a
CLI fallback (see [commands.md](commands.md)).

## Native run (`nebula3d-web`)

```bash
pip install -e ".[web]"
nebula3d-web                       # serves http://127.0.0.1:8000 and opens a browser
```

By default it reads `./data` (the `raw/` + `processed/` layout). Override:

```bash
nebula3d-web --data-root /path/to/data --port 8000
nebula3d-web --no-browser          # headless (e.g. remote)
```

The installed wheel bundles the built SPA, so `nebula3d-web` is the only command
needed. From a source checkout that has not been built yet, build the frontend
first (`make ui`, see [Development](#development)) — otherwise only the API is
served.

## In-browser run (GitHub Pages / Pyodide)

The hosted build runs the **real** `nebula3d` pipeline in the user's browser via
[Pyodide](https://pyodide.org) (CPython + numpy/scipy/h5py compiled to
WebAssembly). Users load **their own** data file; nothing is uploaded, nothing is
hosted — the privacy-preserving path to a public, fully-functional app.

- Hosted at **https://drthyang.github.io/nebula3d/** (deployed by
  `.github/workflows/pages.yml` on push to `main`).
- The pipeline ships as a data-free `nebula3d` wheel that the page micropip-installs
  at runtime. Pyodide runs in a dedicated Web Worker
  (`web/src/workers/pyodideWorker.ts`) so the UI never blocks; a boot-progress
  panel covers the ~15 MB WASM download (cached after first load).
- **Memory ceiling.** Pyodide's 32-bit-WASM heap tops out at 4 GB (Pyodide
  ≥ 0.27). The pipeline is memory-lean in float64 — |Q| grids accumulate from
  broadcast 1-D axes instead of meshgrids, the ΔPDF pads to
  `scipy.fft.next_fast_len` instead of powers of two, intermediates are freed
  promptly, and the slice caches are capped in-browser — so the worst stage
  peaks at ~7.3 volume-sized arrays (measured; the gate assumes 8). That admits
  **up to ~50 M voxels at full float64 precision** (e.g. a 301×401×401
  full-resolution volume = 48.4 M voxels). Every upload is pre-flighted by
  `nebula3d.webbridge.inspect_input` (reads only the HDF5 shape, so it can't OOM)
  and rejected above the gate with a message pointing to the native build.

Local dev for this build: `cd web && npm run dev:pyodide` (loads `.env.pages`,
base `/`).

## What it does

A single-page console with a left sidebar; the five views (in sidebar order)
replace the standalone `examples/explore_*.py` viewers:

| View | Replaces | What |
| --- | --- | --- |
| **Configure / Run pipeline** | `run_pipeline.py` | Pick a dataset and tune the key parameters per stage — ring removal (azimuthal **patches**, texture **Fourier order**), punch (HKL ↔ Q-space frame), backfill, flatten, ΔPDF, consistency — then run all stages with a live stepper and log. Existing outputs are skipped unless *force* is on. Default landing view. |
| **Reciprocal cleanup** | `explore_slice.py` | One panel per HKLVolume stage (raw / ring-removed / punched / backfilled / flattened) sharing an H/K/L plane selector, cut, contrast, log, and colormap. All panels share **one fixed global colour scale** (pooled from the centre cut). The cut readout is an **editable box** — type `0.3333` and it snaps to the nearest plane. |
| **3D-ΔPDF** | `explore_delta_pdf_ortho.py` | Three linked real-space orthoslices (x_H–y_K, x_H–z_L, y_K–z_L) as square **windows** (adjustable, default 80 Å), each with its own cut slider, plus contrast and a gray dashed unit-cell overlay. |
| **Multi-volume** | `explore_delta_pdf_multi.py` | Related DeltaPDF files × the three planes as a square grid, sharing cut, window, and contrast; a per-plane colour scale pooled across files. |
| **Consistency check** | `delta_pdf_consistency.py` | Back-FFT check: inverse-transforms the ΔPDF to reciprocal space and shows **data \| back-FFT \| residual** at a shared plane/cut, with agreement metrics (Pearson r, normalised RMS, per-plane r). Adjustable **\|Q\|** and real-space **r** bands isolate which ranges support a signal. |

## Architecture

```
Native:    Browser (React/TS SPA)  ──HTTP/SSE──►  FastAPI (uvicorn)  ──►  nebula3d library
In-browser: Browser (React/TS SPA) ──RPC──►  Web Worker → Pyodide  ──►  nebula3d (same code)
```

- **Slices** are extracted with the same `nebula3d.visualization.extract_slice` the
  matplotlib viewers use, returned as a compact binary envelope
  (`[uint32 header_len][JSON header][float32 data]`), and colour-mapped in the
  browser — so contrast/log/colormap change instantly with no refetch.
- **Native** runs `nebula3d.pipeline.run_pipeline` in a separate process
  (`multiprocessing` spawn), streaming progress over Server-Sent Events; cancel
  terminates the worker. Loaded volumes are kept in an LRU cache sized to hold
  every cleanup stage of a dataset at once, so the shared cut slider scrubs
  without re-reading the ~100 MB volumes.
- **In-browser** drives the pipeline **stage-by-stage** from JS, yielding between
  stages so the stepper + log update per stage. The Python side is a thin
  in-process driver, **`nebula3d.webbridge`**, that reuses the FastAPI-free server
  helpers (`volumes`, `deltapdf`, `consistency`, `datasets`, `params`) against a
  virtual `/work` FS — slicing/discovery/consistency are *not* reimplemented in
  JS. `client.ts` branches on `PYODIDE_MODE` for every endpoint.

### Endpoints (native API)

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/datasets` | discovered datasets with per-stage output status |
| GET | `/api/volumes/{id}/meta` | HKLVolume shape, axis ranges, lattice |
| GET | `/api/volumes/{id}/slice?plane=&value=&interp=` | binary 2D slice |
| GET | `/api/deltapdf/{id}/meta` | ΔPDF shape, ranges, lattice, \|Q\|max |
| GET | `/api/deltapdf/{id}/slice?plane=xy\|xz\|yz&value=` | binary ΔPDF orthoslice |
| GET | `/api/consistency/{dataset_id}/meta?q_min=&q_max=&r_min=&r_max=` | back-FFT metadata and metrics |
| GET | `/api/consistency/{dataset_id}/slice?panel=data\|recon\|residual\|dpdf&...` | binary consistency slice |
| POST | `/api/pipeline/run` | start a job; returns `{id, status, ...}` |
| GET | `/api/pipeline/jobs/{id}/events` | SSE progress stream |
| POST | `/api/pipeline/jobs/{id}/cancel` | terminate a running job |

Volume ids are `"<dataset_id>.<stage>"` (e.g. `sample.backfilled`, `sample.delta_pdf`).

## Development

Run the backend and the Vite dev server (hot reload) separately:

```bash
# terminal 1 — API on :8000
nebula3d-web --no-browser --reload

# terminal 2 — Vite dev server on :5173 (proxies /api to :8000)
cd web && npm install && npm run dev      # http://localhost:5173
```

Frontend layout:

| Path | What |
| --- | --- |
| `web/src/App.tsx` | sidebar shell + view routing |
| `web/src/pages/` | one component per view (config, execution, reciprocal, ΔPDF, multi-volume, consistency) |
| `web/src/components/` | shared panels (`SliceCanvas`, `SlicePanel`, `DpdfPanel`, `UnitCellGrid`) + UI primitives (`ui.tsx`) |
| `web/src/api/` | typed fetch client (`client.ts`), the Pyodide engine (`pyodideEngine.ts`), React Query hooks, response types |
| `web/src/workers/` | the classic Web Worker hosting Pyodide |
| `web/src/state/` | zustand stores for cleanup + ΔPDF view state |
| `web/src/colormaps/` | client-side colormap LUTs |

The React app is built into **two** gitignored targets (rerun the matching one
after editing `web/src`):

```bash
make ui            # → src/nebula3d/server/static   (served by native nebula3d-web; bundled in the wheel)
make ui-pages      # → web/dist                  (GitHub Pages / Pyodide build)
```

> **Heads-up — the native bundle can go stale.** `src/nebula3d/server/static/` is a
> gitignored build artifact that only changes when you run `make ui`. If you edit
> `web/src` (or pull) and don't rebuild, `nebula3d-web` keeps serving the **old** UI
> (it prints a `[warn]` at startup when the bundle is older than `web/src`). After
> rebuilding, hard-refresh (Cmd/Ctrl-Shift-R). The two targets are independent.

Frontend checks (run in CI):

```bash
cd web && npm run lint && npm run typecheck && npm run build
```

## Packaging & the data-free wheel

The native SPA in `src/nebula3d/server/static/` is a build artifact (git-ignored);
`package-data` bundles `static/**/*` so the published wheel serves the UI with no
Node toolchain on the user's side. A release build is `cd web && npm ci &&
npm run build`, then build the wheel.

The **Pages** build instead micropip-installs an `nebula3d` wheel at runtime, so it
must be built **data-free** — a careless build can bundle experimental data via
the packaged `static/`. Always clean first:

```bash
rm -rf build src/*.egg-info src/nebula3d/server/static/data
python -m pip wheel . --no-deps --no-cache-dir -w web/public/wheels
unzip -l web/public/wheels/*.whl | grep -iqE '\.(bin|nxs|h5)' && echo "DATA LEAK — stop" || echo "clean"
```

A clean wheel is ~252 KB. The CI workflow performs this same data-leak check.

## In-browser design notes

- **Why Pyodide, not WebGPU or pre-baked data.** `nebula3d` is pure Python and its
  compute deps (numpy/scipy/h5py) are official Pyodide packages, so the existing,
  regression-gated pipeline runs in the browser essentially unchanged. WebGPU was
  shelved: profiling shows the 3D FFT is only ~4% of runtime while the dominant
  stage (ring removal, ~70%) is an irregular robust fit — a poor GPU fit — and a
  WGSL rewrite would discard the "real `nebula3d` runs unchanged" property. Pre-baked
  static volumes were rejected because they would require *hosting the data*.
- **Pyodide gotchas.** `import nebula3d` pulls in matplotlib; Pyodide ships
  matplotlib 3.5.2 (< the wheel's `>=3.7` pin), so install with `deps=False` to
  skip the version check. Pipeline entry points: `nebula3d.load`,
  `nebula3d.core.HKLVolume.from_arrays`, `nebula3d.pipeline.run_pipeline`,
  `nebula3d.analysis.compute_delta_pdf`.
- **Privacy.** The public app ships **no data**; users supply their own at
  runtime. `web/public/data/` and `web/public/wheels/` are gitignored, and the CI
  wheel build is data-free.
