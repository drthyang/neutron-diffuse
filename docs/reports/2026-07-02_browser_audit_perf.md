# Browser (Pyodide) audit & performance pass — 2026-07-02

Scope: go through the in-browser (GitHub Pages / Pyodide) version end to end,
confirm there are no bugs, and improve pipeline performance by ≥10 % with **all
features and numerical results preserved exactly** (bit-for-bit).

## Outcome

- **No functional bugs found** in the browser path.
- **~22–31 % faster** pipeline wall time, with **bit-identical outputs**
  (SHA-256-verified on every stage artifact).
- Full test suite (219 passed), `ruff`, and `mypy` clean.
- End-to-end in-browser run verified with the optimized wheel: 6/6 stages
  complete, consistency check **r = 0.99963**, zero console errors, all six
  views (Configure, Execution, Reciprocal cleanup, Bragg profile, 3D-ΔPDF,
  Q–R Band Transform, Multi-volume) render without failures.

## Audit findings

Reviewed: `web/src/api/pyodideEngine.ts`, `web/src/workers/pyodideWorker.ts`,
`src/nebula3d/webbridge.py`, the FastAPI-free server helpers
(`server/volumes.py`, `server/deltapdf.py`, `server/consistency.py`), the
pipeline orchestration, and the ring-removal / save-load hot paths; exercised
the app in a Pyodide dev preview.

No bugs. Three observations worth recording:

1. **Per-slice `except Exception` in ring removal** (`pipeline.py`,
   `process_slice`) converts *any* per-slice failure into "plane left as-is"
   plus an event-log line. That is the intended degrade-gracefully behaviour
   for data-driven fit failures, but it also swallows unexpected programming
   errors — during this work it silently masked an induced `ValueError`,
   turning a broken fit into "rings ran fine, subtracted nothing". Left as
   designed; worth remembering when touching the rings stage.
2. **Spurious `RuntimeWarning: … in matmul` on macOS.** The
   divide-by-zero/overflow/invalid warnings seen at
   `radial_background.py` (`AtA = B.T @ (B * wn[:, None])`) reproduce with
   plain finite random matrices on this macOS/NumPy (Accelerate raises stray
   FP-status flags inside `dgemm`). Benign, upstream, and absent under WASM.
3. **Local dev wheel staleness.** `web/public/wheels/*.whl` is only rebuilt
   by CI on deploy; a local `dev:pyodide` session can silently test an old
   wheel. Rebuilt fresh here (data-free, verified no `.nxs/.h5/...` leak).
   The hosted site is unaffected (CI always builds from source).

## Performance changes (all bit-exact)

| Change | File | Why it's exact |
| --- | --- | --- |
| HDF5 writes: gzip level 4 → **gzip level 1 + byte-shuffle** | `io/hkl_reader.py` | Lossless filters; stored arrays identical; still standard HDF5 readable anywhere. Files are also ~8 % *smaller* (shuffle helps float data), reducing browser MEMFS pressure. |
| **In-memory pass-through between stages.** Each stage still writes its `.h5` (artifacts, resume, viewers unchanged), but the next stage consumes the just-computed volume instead of re-reading gzip'd HDF5. | `pipeline.py` | A reload of the just-written file is bit-identical to the in-memory object (lossless round-trip); residency is one input volume during a stage, exactly as before, so the browser memory gate is unaffected. |
| **Batched ridge solves** in the azimuthal-texture fit: per-|Q|-bin normal equations are built exactly as before, then solved in one stacked LAPACK call instead of ~n_q Python-level `np.linalg.solve` calls per slice. | `preprocessing/radial_background.py` | Same `dgesv` per matrix (explicit single-column RHS, `b[..., None]`, unambiguous across NumPy 1.x/2.x); verified 0/300 bit-differences on random systems, and by the output checksums below. |

## Verification

- **Bit-exactness:** SHA-256 over every HDF5 array (data/sigma/mask/axes/UB) of
  all five stage outputs, before vs. after, at 97³ and 129³, serial
  (browser-like) and parallel — **all identical**. Bragg-profile and
  consistency-metrics JSONs byte-identical.
- **Tests/lint/types:** `pytest` 219 passed; `ruff` clean; `mypy` clean.
- **Browser E2E (optimized wheel):** demo volume → full 6-stage run in
  Pyodide → all viewers checked; consistency r = 0.99963; no console errors.

## Measurements (native, serial = the browser's execution mode)

| Volume | Before | After | Δ |
| --- | --- | --- | --- |
| 97³ demo, serial | 2.58 s | 2.02 s | **−22 %** |
| 129³ demo, serial | 6.60 s | 4.52 s | **−31 %** |

(Native parallel at 129³ lands at 4.53 s with identical checksums.) The wins
concentrate where profiling showed the time actually goes: HDF5 gzip writes
were 41 % of wall time at 129³ (now ~2.6× cheaper and 4 of 5 inter-stage
reloads are gone), and the texture fit's ~38 k tiny solves per volume collapse
into ~130 stacked calls — Python-call overhead the WASM interpreter pays even
more dearly for.

## Remaining headroom (not taken — would break exactness or parity)

- Vectorising the texture-fit normal equations *across* |Q| bins (einsum)
  changes floating-point summation order → not bit-exact.
- `lzf` compression writes ~3× faster still, but is an h5py-only filter —
  downloaded artifacts would lose portability.
- Skipping the consistency QA PNG under Emscripten would drop an artifact the
  native build produces → parity break.
- Replacing `np.interp`-per-coefficient in `_evaluate_fourier` with a shared
  searchsorted + manual lerp risks last-ulp differences.

Within the "bit-exact, feature-identical" constraint, the low-hanging fruit is
now taken; further gains would need relaxing one of those constraints or
optimizing inside scipy/LAPACK kernels.
