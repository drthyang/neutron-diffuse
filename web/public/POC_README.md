# In-browser pipeline PoC (Pyodide)

`poc-pyodide.html` runs the **real pure-Python `ndiff` pipeline entirely in the
browser** via [Pyodide](https://pyodide.org) (CPython + numpy/scipy/h5py compiled
to WebAssembly). The user's data file and all computation stay on their machine —
nothing is uploaded, nothing is hosted. This is the privacy-preserving path to a
public, fully-functional app: users bring their own data.

## Validated

Boot → load numpy/scipy/h5py/matplotlib → micropip-install the `ndiff` wheel →
`import ndiff` → `compute_delta_pdf` on a synthetic volume → render. Confirmed
working: scipy's 3D FFT and the real `ndiff` code produce a ΔPDF in-browser.

## Build the wheel (required — it is gitignored)

The `ndiff` wheel is a build artifact (and a careless build can bundle data), so
`web/public/wheels/` is gitignored. Build a clean, **data-free** wheel before
serving the page:

```bash
# from the repo root — clean stale build dirs first so no data is bundled
rm -rf build src/*.egg-info src/ndiff/server/static/data
python -m pip wheel . --no-deps --no-cache-dir -w web/public/wheels
# verify it is clean:
unzip -l web/public/wheels/*.whl | grep -i '\.bin' && echo "DATA LEAK" || echo "clean"
```

Then serve `web/` (e.g. `npm run dev`) and open `/poc-pyodide.html`.

## Notes / next steps

- First load pulls ~15–25 MB of WASM (Pyodide + the scientific stack); cached after.
- `deps=False` on `micropip.install` skips the version check — Pyodide ships
  matplotlib 3.5.2 (< our `>=3.7` pin) but it imports fine.
- Performance lever: profile `compute_delta_pdf`; if the 3D FFT dominates, a
  WebGPU FFT kernel (data dims are powers of two → clean radix-2) is the targeted
  acceleration. See `docs/github-pages-webgpu-plan.md`.
- Next: wire this path into the real React viewers (replace the API data layer
  with Pyodide results) and extend from `compute_delta_pdf` to the full
  `run_pipeline`.
