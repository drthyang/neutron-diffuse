# Browser-hosted app — plan & progress (pickup doc)

**Audience:** future agents and the maintainer picking this up later.
**Last updated:** 2026-06-21.
**Status:** architecture validated (PoC works); the app integration is not built yet.

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

## 4. Roadmap — what's LEFT (the integration build)

This is the bulk of the remaining work: turn the PoC into the real app.

**P1 — Pyodide data layer in the React app.**
Replace the `static`-mode data path (`web/src/api/staticData.ts`) with one that
runs the pipeline in Pyodide and returns the same `Slice`/`Dataset`/`Meta` shapes
the viewers already consume. Boot Pyodide once (lazy, with a progress UI), install
the wheel, keep the loaded volumes/results in JS. Decide the mode flag: either a
new `VITE_DATA_MODE=pyodide` or repurpose `static`.

**P2 — File-load UX + full `run_pipeline`.**
A "Load volume" control (`<input type=file>`) → write into Pyodide FS →
`ndiff.run_pipeline(...)` (not just `compute_delta_pdf`). Surface per-stage
progress (the pipeline takes ~1–3 min in-browser). Hold results in memory and
feed every viewer.

**P3 — Re-enable all six pages in the hosted build.**
Once compute is client-side, undo the `App.tsx` nav restriction so Configure /
Execution / Reciprocal cleanup / Consistency check work again — now driven by
Pyodide instead of `/api`.

**P4 — Polish & robustness.**
Lazy Pyodide load + cached across pages; memory budget (use float32; offer a
downsample option for very large volumes — browsers cap WASM at ~2–4 GB);
one-time ~15–25 MB WASM download UX; mobile caveat. Optional brotli on the wheel.

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

| Path | Role | Where |
|---|---|---|
| `web/public/poc-pyodide.html` | Pyodide PoC (boot, self-test, file picker, benchmark) | branch |
| `scripts/profile_pipeline.py` | Per-stage pipeline profiler | branch |
| `scripts/export_web_assets.py` | Pre-baked f16 volume exporter (dead-end path) | main |
| `web/src/api/staticData.ts` | Pre-baked static data layer — replace with Pyodide (P1) | main |
| `web/src/api/client.ts` | `VITE_DATA_MODE` branch point | main |
| `web/vite.config.ts`, `web/.env.pages` | Pages build (`--mode pages`) | main |
| `.github/workflows/pages.yml` | Pages deploy | main |
| `web/src/App.tsx` | Static-mode nav restriction (undo in P3) | main |
| `ndiff/preprocessing/radial_background.py` | Ring removal (optimized) | branch |
