# GitHub Pages hosting + client-side compute (WebGPU) — migration plan

## The problem in one sentence

GitHub Pages serves **static files only**, but the console today depends on a
live FastAPI backend that loads ~130 MB `HKLVolume`s, extracts slices, and runs
FFTs — none of which can run on Pages. To host the whole app on Pages the
compute has to move into the browser (or be pre-baked).

This document is the architecture decision and the staged path to get there.

## Where the time actually goes (measured)

Backend work is already well-cached:

- `load_volume` — LRU(6) keyed by `(path, mtime)`; one full dataset (5 HKL
  stages) stays warm, so cut-slider scrubbing is fast. Cold first-load per
  stage is disk-bound (~130 MB read).
- `reconstruction` (back-FFT) — LRU cached per `(dataset, |Q| band, |R| band)`;
  the expensive forward+inverse FFT runs once per band.
- Client `SliceCanvas` — CPU per-pixel LUT map over ~120 K px/slice; sub-ms to
  a few ms. **Not a bottleneck.** Contrast/colormap re-render is already instant
  because it is applied client-side from the fetched float32 slice.

**Conclusion:** the dominant cost is the *server round-trip per slice* (network
+ cold volume load), not pixel mapping. So "add WebGPU to the canvas" alone buys
little. The win comes from **slicing the volume client-side** so cut changes
never hit the network — and that is exactly what also unblocks Pages.

## Target architecture for Pages

```
 static SPA (Pages)  ──fetch──►  pre-processed volume assets (Pages, same origin)
        │
        ├─ WebGPU: 3D texture slice  → 2D slice (replaces /api/volumes/.../slice)
        ├─ WebGPU/WASM FFT           → ΔPDF + back-FFT (replaces /api/consistency)
        └─ existing LUT/contrast      (already client-side)
```

The FastAPI backend stays the *development / full-resolution* path (and a
possible self-host option). Pages gets a **static-data build** of the same SPA.

## Data budget — the real constraint

Full volumes are 401×401×301 float32 ≈ 184 MB **each**, ×3 datasets ×5 stages.
That cannot ship to a browser. Options, in order of preference:

1. **Downsample + half-float (f16) + compression.** e.g. 200×200×150 f16 ≈
   11.5 MB/volume; gzip/brotli on the `.bin` cuts it further. Ship only the
   stages a page needs (ΔPDF viewer needs `delta_pdf`; reciprocal viewer needs
   the 5 HKL stages). Budget ~30–60 MB per dataset — acceptable for a demo.
2. **Pre-baked slices only (no client compute).** Render the binary slice
   envelopes for a fixed grid of cuts at build time; the SPA fetches them like
   the API does today. Zero GPU work, but interactivity limited to the baked
   cuts. Good fallback / first milestone.
3. **On-demand chunks.** Tile the volume and fetch only the slab a cut needs.
   More engineering; revisit only if (1) is too heavy.

## Why WebGPU (vs WASM / WebGL)

- **Slicing**: upload each volume once as a `r16float` 3D texture; a cut is a
  single textured quad draw / compute pass. Effectively free per cut → kills the
  round-trip. WebGL2 can do this too (3D textures), so WebGL is the **fallback**.
- **FFT** (ΔPDF + back-FFT): a Stockham radix-2 compute pipeline. This is the
  part that genuinely needs WebGPU compute; a 256³ complex FFT is feasible on
  the GPU and impractical in JS. WASM (e.g. pocketfft compiled) is the
  CPU fallback if `navigator.gpu` is absent.
- Feature-detect `navigator.gpu`; degrade WebGPU → WebGL slicing + WASM FFT →
  pre-baked slices. Never hard-fail.

## Staged rollout

- **M0 — Pages-ready shell.** Vite `base` set for project Pages; a Pages
  Actions workflow; a `VITE_DATA_MODE` flag (`api` | `static`). In `static`
  mode the API client reads pre-baked assets instead of `/api`. Ship the
  reciprocal + ΔPDF viewers on pre-baked slices (option 2) so *something* is
  live and correct on Pages. **This is the first shippable milestone.**
- **M1 — Client-side slicing.** Add a downsampled-volume asset pipeline (a
  `scripts/export_web_assets.py` that writes f16 `.bin` + a manifest) and a
  WebGPU 3D-texture slicer behind the existing `SliceCanvas` data path. Cut
  sliders become instant and offline. WebGL2 fallback.
- **M2 — Client-side ΔPDF / consistency.** Port `invert_delta_pdf` + the
  forward transform to a WebGPU FFT compute pipeline; the Consistency page's
  band sliders recompute in-browser. WASM-FFT fallback.
- **M3 — Polish.** Brotli precompression, lazy per-page asset loading, a
  loading/█ progress UI for the one-time volume download.

## Concrete first steps (M0)

1. `vite.config.ts`: `base: mode === "pages" ? "/neutron-diffuse/" : "/"` (done).
   The Pages build runs `vite build --mode pages`; `mode` is used rather than
   `process.env` so the config stays type-checkable without `@types/node`.
2. `.github/workflows/pages.yml`: build with `--mode pages`, upload the build
   output, deploy via `actions/deploy-pages`.
3. `web/src/api/client.ts`: branch on `import.meta.env.VITE_DATA_MODE`; in
   `static` mode resolve `import.meta.env.BASE_URL + "data/<id>/<stage>/<plane>/<cut>.bin"`.
4. `scripts/export_web_assets.py`: dump the pre-baked slice grid + a
   `manifest.json` (datasets, stages, available cuts, lattice/meta) into
   `web/public/data/`.

Do **not** deploy the `api`-mode SPA to Pages — it would render "API offline"
and nothing would work. M0's `static` mode is the gate.
