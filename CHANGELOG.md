# Changelog

## Unreleased

- **Milestone: fully static, GitHub Pages-hosted app with feature parity.** The
  browser console now runs the **complete** `nebula3d` reduction — every pipeline
  stage, cleanup, 3D-ΔPDF, multi-volume, and consistency view — entirely
  client-side via Pyodide, at **full-resolution float64** (up to ~50 M voxels;
  a 301×401×401 volume fits). No server, no upload, no install: the app is a
  static bundle served from **https://drthyang.github.io/nebula3d/**, deployed by
  `.github/workflows/pages.yml` on push to `main`. The in-browser build is now a
  first-class path alongside the native `nebula3d-web` backend, not a reduced
  demo. Under Pyodide (no OS threads) ring removal falls back to serial slice
  processing; native CPython still parallelises.
- **Spherical-frame Bragg punch.** The default punch ellipsoid axes now follow
  the local spherical frame at each peak — `(rρ, rθ, rφ)` in Å⁻¹ with rρ radial
  (along Q̂), rφ azimuthal (a*–b* ring tangent, c* pole), rθ polar — so every
  reflection is oriented correctly with no tilt angle. Added
  `punch_frame="spherical"` (now the `PunchParams` / web default) alongside the
  existing `"q"` (a*/b*/c*) and `"hkl"` frames; the legacy frames are unchanged.
  Configure and Bragg-profile pages gain a frame selector and rρ/rθ/rφ controls,
  and the punch preview renders the per-peak oriented ellipse.

## 0.2.0 - 2026-06-18

- Promoted the consistency check to the endpoint of the recommended 3D-DeltaPDF
  workflow.
- Added the FastAPI/React consistency viewer and `/api/consistency` endpoints
  for reciprocal-space back-FFT comparison with optional `|Q|` and real-space
  bands.
- Updated `examples/run_pipeline.py` to run the back-FFT consistency check by
  default after the DeltaPDF stage.
- Updated documentation around the full workflow, web UI, reproducibility
  commands, and output artifacts.
- Aligned package, API, and web app version metadata at `0.2.0`.

## 0.1.0 - Initial alpha

- Initial alpha toolkit for reciprocal-space diffuse-scattering cleanup and
  3D-DeltaPDF exploration.
