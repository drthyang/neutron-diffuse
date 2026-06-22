# Browser-hosted app — plan & progress (pickup doc)

**Audience:** future agents and the maintainer picking this up later.
**Last updated:** 2026-06-22.
**Status:** **integration built & working end-to-end.** The full app (load →
6-stage pipeline → all six viewers) runs client-side via Pyodide. P1–P3 are
done; P4 (polish: Web Worker so the UI doesn't block, lazy load, large-volume
memory budget) remains. See §4.

---

## 1. The goal (one paragraph)

Ship a **fully-functioning version of the neutron-diffuse console as a static
site on GitHub Pages — with no backend server.** Users load **their own** data
file in the browser; the **real `ndiff` reduction pipeline runs locally in their
browser** via Pyodide (CPython + numpy/scipy/h5py compiled to WebAssembly).
Nothing is uploaded, nothing is hosted. "Full-functioning" = the whole workflow
(load → ring removal → Bragg punch → backfill → flatten → 3D-ΔPDF → consistency
check) plus all six viewer pages, all client-side.

Why this shape: GitHub Pages serves static files only, and the experimental data
must never be public (see §6). Running the existing Python pipeline in-browser on
user-supplied data satisfies both — it's static-hostable *and* private.

---

## 2. Architecture decision: Pyodide (not WebGPU, not pre-baked data)

`ndiff` is **pure Python**; its only compute deps (numpy, scipy, h5py, matplotlib)
are all official **Pyodide** packages. So the existing reduction code runs in the
browser essentially unchanged. This is the chosen path.

Two alternatives were considered and **rejected** (don't redo them):

- **Pre-baked static data** (ship downsampled volumes / slices as assets, slice
  in JS). Rejected for the interactive app because it requires *hosting the data*
  — which violates the privacy requirement — and only supports the ΔPDF viewers.
  The infrastructure for this still exists on `main` (see §4) but is a dead end
  for the real goal.
- **Hand-written WebGPU pipeline.** Rejected: profiling (§5) shows the 3D FFT is
  only ~4% of runtime, so a WebGPU FFT saves almost nothing, and the dominant
  stage (ring removal) is an irregular fit — a poor GPU fit. WebGPU is **shelved**
  unless a future profile shows a dominant, GPU-friendly stage.

---

## 3. Status — what's DONE

### Proven (the risky parts are de-risked)
- **Pyodide PoC works** — `web/public/poc-pyodide.html` boots Pyodide, installs
  the `ndiff` wheel, and runs the real `compute_delta_pdf` (scipy 3D FFT) on a
  synthetic volume, rendering a ΔPDF — all client-side. Validated live. Has a
  file picker (real `.nxs/.h5` → `ndiff.load` → `compute_delta_pdf`) and a
  Pyodide-vs-native benchmark button.
- **Performance characterized** (§5) and **ring removal optimized 25%** (§5).
- **Privacy handled** (§6): experimental data removed from repo + history.

### Shipped on `main` (the *pre-baked* static build — a partial dead end)
The pre-baked approach was built and deployed before the Pyodide pivot. It is
live but **shows an empty shell** (data removed). Reusable pieces:
- `web/vite.config.ts` — `vite build --mode pages` → base `/neutron-diffuse/`,
  outputs to `web/dist` (keeps the api-mode bundle in the package intact).
- `web/.env.pages` — sets `VITE_DATA_MODE=static`.
- `web/package.json` — `build:pages` script.
- `.github/workflows/pages.yml` — deploys `web/dist` to Pages on push to `main`.
  (Pages was enabled manually: Settings → Pages → Source: GitHub Actions.)
- `web/src/App.tsx` — in static mode the nav is restricted to the two viewers
  that *can* run without a backend (3D-ΔPDF, Multi-temperature) and defaults to
  3D-ΔPDF. **Revisit once Pyodide compute lands** — all six pages should return.
- `web/src/api/staticData.ts` + `client.ts` — `VITE_DATA_MODE==="static"`
  branch reads a pre-baked `manifest.json` + downsampled volumes. **This is the
  layer Pyodide should replace** (see §4, P1).
- `scripts/export_web_assets.py` — bakes downsampled f16 ΔPDF volumes. Output is
  gitignored; only needed if anyone revisits the pre-baked path.

### On branch `pyodide-browser-pipeline` (NOT merged to `main`)
- `web/public/poc-pyodide.html` — the PoC (see above).
- `scripts/profile_pipeline.py` — per-stage pipeline profiler.
- Ring-removal optimization in `ndiff/preprocessing/radial_background.py`
  (§5) — 79 s → 59 s, output byte-identical, 190/190 tests pass.
- This plan doc.

> **Recommended:** merge `pyodide-browser-pipeline` → `main` so the PoC, profiler,
> ring-removal speedup, and this doc are all canonical. The ring-removal fix is
> independent and safe to cherry-pick on its own if preferred.

---

## 4. Roadmap — status

**P1 — Pyodide data layer in the React app. ✅ DONE.**
`VITE_DATA_MODE=pyodide` (set in `web/.env.pages`) selects the in-browser engine
(`web/src/api/pyodideEngine.ts`): lazy Pyodide boot (CDN), `loadPackage`, micropip
the wheel, then typed methods that return the *same* `Slice`/`Dataset`/`Meta`
shapes the viewers consume. `web/src/api/client.ts` branches on `PYODIDE_MODE` for
every endpoint. The Python side is a thin in-process driver, **`ndiff.webbridge`**
(in the wheel), that reuses the FastAPI-free server helpers (`volumes`,
`deltapdf`, `consistency`, `datasets`, `params`) against a virtual `/work` FS — so
slicing/discovery/consistency are *not* reimplemented in JS. `ndiff.server.__init__`
is now lazy so those helpers import without FastAPI under Pyodide.

**P2 — File-load UX + full `run_pipeline`. ✅ DONE.**
Configure page (pyodide mode) shows a "Load volume… / Use demo" control →
`engine.loadFile()` writes into the Pyodide FS → registered as the dataset. Run
drives `run_pipeline` **stage-by-stage** from JS (`engine.runPipeline`), yielding
between stages so the Execution stepper + log update per stage. Results live in
the Pyodide FS and feed every viewer; queries are invalidated on completion.

**P3 — All six pages in the hosted build. ✅ DONE.**
The nav restriction was tied to the legacy `static` flag; pyodide mode uses a new
flag, so all six tabs are live. Verified end-to-end in a headless browser: boot →
demo volume → 6 stages "done" → 3D-ΔPDF / Consistency / Reciprocal viewers all
render, no console errors. CI (`.github/workflows/pages.yml`) builds the data-free
wheel before `vite build` (with a data-leak guard).

**P4 — Polish & robustness. ✅ DONE.**
- **Web Worker**: Pyodide now runs in a dedicated classic Web Worker
  (`web/src/workers/pyodideWorker.ts`, IIFE bundle via `worker.format: "iife"`).
  Message-passing RPC (`{ id, type, ...payload }` both ways); binary slice
  envelopes transferred as Transferable `ArrayBuffer` (zero-copy). Cancel
  (`cancelPipeline()`) terminates the Worker cleanly and resets boot state.
- **Boot progress panel**: a dedicated `BootProgressPanel` component in
  `PipelineConfig.tsx` subscribes to `BootStatus` and shows a phased progress
  bar while the ~15 MB WASM downloads — replaces the log-line-only approach.
- **Responsive UI**: the main thread is never blocked; React repaints freely
  while the Worker drives numpy/scipy stages over minutes.
- **Loader memory**: `ndiff/io/mantid_nxs.py` `_assemble` rewritten to read +
  transpose one array at a time (freeing each file-order temp), compute σ in
  place, and `astype(copy=False)` — peak load 1.88 GB → 1.04 GB at 48 M voxels,
  byte-identical output.

### Memory reality at full resolution (measured 2026-06-22)
Pyodide is a **32-bit-WASM heap**; large float64 volumes do not fit. Measured
native peak-RSS at 301×401×401 (= 48 M voxels, the size a user hit):
load ~1 GB, ring removal +0.5 GB, **Bragg punch +1.9 GB**, padded ΔPDF FFT ~1 GB
complex array → **full-pipeline peak ≈ 3 GB**. One real browser refused
allocations at ~1 GB, i.e. the volume cannot even be *held* loaded in float64.

**Decision (maintainer, 2026-06-22): native for full-res.** The in-browser
build targets modest volumes / sharing / the demo; full-resolution data goes
through native `ndiff-web` (same React UI, no memory limit). Rather than crash
mid-pipeline with a numpy `MemoryError`, the browser now **pre-flights** every
upload: `ndiff.webbridge.inspect_input` reads only the HDF5 signal *shape* (no
array load, so it can't OOM), estimates the full-pipeline peak
(`_PIPELINE_PEAK_BYTES_PER_VOXEL = 8×8`), and the Worker's `load_file` case
rejects volumes over `_BROWSER_PEAK_BUDGET_BYTES` (~1.5 GB → ~23 M voxels) with
a clear message pointing to the native build. Native loaders are untouched.

Deferred (only if browser full-res is ever required): **float32 in-browser
mode** — would ~halve memory but needs a precision flag threaded through *both*
readers (note `hkl_reader._load_hdf5` hard-casts to float64 on read) and every
transform, plus validation vs float64; even then the punch spike (~0.95 GB) is
borderline. Do **not** silently downsample reciprocal space (blurs Bragg peaks,
changing punch). Other low-pri: mobile caveat docs, optional brotli on the wheel.

Local dev for the in-browser build: `npm run dev:pyodide` (loads `.env.pages`,
base `/`); or the `ndiff-pyodide` entry in `.claude/launch.json`.

**P-opt — Further pipeline speedups (optional).**
Next target inside `remove_rings` is batching the per-patch loop in `fit`
(`radial_background.py`) — higher risk, needs per-step validation. Only if the
in-browser runtime needs to be shorter.

---

## 5. Technical reference

### Build the `ndiff` wheel (data-free!)
The wheel is gitignored (`web/public/wheels/`) because a careless build bundles
the experimental data via the packaged `ndiff/server/static/`. Always clean first:

```bash
rm -rf build src/*.egg-info src/ndiff/server/static/data
python -m pip wheel . --no-deps --no-cache-dir -w web/public/wheels
unzip -l web/public/wheels/*.whl | grep -i '\.bin' && echo "DATA LEAK — stop" || echo "clean"
```
A clean wheel is ~252 KB. Run the PoC with `npm run dev` in `web/`, open
`/poc-pyodide.html`.

### Pyodide gotchas (learned in the PoC)
- Pyodide v0.26.2 from jsDelivr CDN. `loadPackage(["numpy","scipy","h5py","matplotlib","micropip"])`.
- `import ndiff` pulls in matplotlib (via `ndiff.visualization`). Pyodide ships
  matplotlib **3.5.2**, but the wheel pins `>=3.7`. Install with **`deps=False`**
  to skip the version check: `await micropip.install(url, deps=False)` — and call
  it from `runPythonAsync` (a JS `{deps:false}` arg is passed positionally, not as
  a kwarg, and fails).
- Pipeline entry points: `ndiff.load(path)`, `ndiff.core.HKLVolume.from_arrays(...)`,
  `ndiff.pipeline.run_pipeline(...)`, `ndiff.analysis.delta_pdf.compute_delta_pdf(vol)`.

### Profiling (`scripts/profile_pipeline.py`, real 22 K volume, 48 M voxels)
| Stage | Native | Share |
|---|--:|--:|
| remove_rings | 59 s (was 78) | ~70% |
| punch_bragg | 16 s | 14% |
| backfill / flatten | 12 s | 11% |
| ΔPDF + back-FFT (FFT) | 4.5 s | **4%** |

Pyodide overhead ≈ **2× native** (FFT 1.9×, percentile 1.5×, Python loop 2.3× —
measured via the PoC benchmark button). ⇒ full pipeline ≈ **~3 min in browser**.
**The FFT is not the bottleneck; ring removal is.**

### Ring-removal optimization (done)
`_robust_radial_profile` in `radial_background.py` looped ~318 |Q| bins ×
~11 k (plane, patch) pairs, calling `np.median` per bin (3.5 M tiny calls). For
the default `median` estimator it now lexsorts once (values contiguous + sorted
per bin) and reads the middle order statistic per bin (bincount + cumsum). Other
estimators keep a sort-grouped loop. **79 s → 59 s, byte-identical output, all
tests pass.** Remaining cost is the inherent sort/gather + per-patch arithmetic.

---

## 6. Privacy constraint (hard requirement)

The experimental ΔPDF data must **never** be public. Actions taken:
- Removed `web/public/data/` from the repo **and from git history** (filter-branch
  + force-push); old commits purged locally.
- `web/public/data/` and `web/public/wheels/` are **gitignored**.
- The packaged wheel must be built data-free (§5).
- Note: GitHub may cache old commits by SHA for a while, and any pre-existing
  forks/clones still contain the data — outside our control.

Implication: the public app ships **no data**; users supply their own at runtime.
That is the whole point of the Pyodide architecture.

---

## 7. Key file map

| Path | Role |
|---|---|
| `ndiff/webbridge.py` | **In-browser driver** — run pipeline + serve slices/meta/consistency, reusing the server helpers, FastAPI-free (in the wheel). `inspect_input` = metadata-only memory pre-flight (size gate) |
| `ndiff/io/mantid_nxs.py` | Mantid reader; `_assemble` is memory-lean (load fix) |
| `tests/test_webbridge.py` | Native end-to-end test of the bridge (synthetic volume) |
| `ndiff/server/params.py` | `build_params` extracted here (FastAPI-free; shared by the router + bridge) |
| `ndiff/server/__init__.py` | `create_app` import made lazy so helper submodules import without FastAPI |
| `web/src/workers/pyodideWorker.ts` | **Classic Web Worker** — hosts Pyodide; classic IIFE bundle (`importScripts`); RPC dispatch |
| `web/src/api/pyodideEngine.ts` | **Engine main-thread side** — Worker lifecycle, RPC promise map, boot/progress observables, public `engine` API |
| `web/src/api/client.ts` | `PYODIDE_MODE` branch point for every endpoint |
| `web/src/api/queryClient.ts` | Shared QueryClient (store invalidates viewers after an in-browser run) |
| `web/src/state/pipelineStore.ts` | `run()` drives the engine stage-by-stage in pyodide mode |
| `web/src/pages/PipelineConfig.tsx` | "Load volume / Use demo" data source in pyodide mode |
| `web/.env.pages`, `web/vite.config.ts` | Pages build (`--mode pages`, `VITE_DATA_MODE=pyodide`) |
| `.github/workflows/pages.yml` | Pages deploy — builds the data-free wheel, then the SPA |
| `web/public/poc-pyodide.html` | Original Pyodide PoC (boot, self-test, file picker, benchmark) |
| `scripts/profile_pipeline.py` | Per-stage pipeline profiler |
| `ndiff/preprocessing/radial_background.py` | Ring removal (optimized) |
