# Roadmap

## Pipeline

```text
[ Mantid / symmetrised HKL volume ]
        |
        v
  1. Powder-ring subtraction        implemented, real-data QA active
  2. Bragg/satellite punch          implemented, real-data QA active
  3. Bragg-hole backfill            implemented, real-data QA active
  4. Radial-background flatten      implemented (background removal, default ON)
  5. 3D-DeltaPDF Fourier transform  implemented, centring bug fixed
  6. Back-FFT consistency check     implemented (inverse-FFT round trip vs data)
```

Stage 6 inverse-transforms the ΔPDF back to reciprocal space
(`nebula3d.analysis.invert_delta_pdf`) and compares it to the diffuse data the ΔPDF
was built from, writing a metric (Pearson r + normalised RMS residual) and a
`data | back-FFT | residual` figure. A gross mismatch would flag a transform bug or an over-aggressive
`crop_hkl` / apodization.  Default ON (`pdf_check_enabled`).

Background removal is an explicit step (4 — the radial flatten), not a hidden
blur inside the FFT; the transform's own Gaussian `SUBTRACT_BG` defaults off and
is only the alternative remover. The current development goal is to enter the
final 3D-DeltaPDF stage with a well-organized cleanup stack and clear defaults.

## Phase 1 — Foundation  Complete

| Task | Status |
|------|--------|
| `HKLVolume` data model | done |
| Mantid NeXus / HDF5 / legacy HKL I/O | done |
| Visualization primitives | done |
| Test suite | done |

## Phase 2 — Powder-Ring Subtraction  Implemented

Current production path:

- `examples/remove_rings_3d.py`
- Two selectable models (`RingParams.ring_model`): `PatchedRadialRingModel`
  (`"patched"`, **default**) and `ParametricRingModel` (`"parametric"`, separable
  pseudo-Voigt × per-ring Fourier texture; rolling / peaks radial modes).
- Per-H-plane fit over the full 3D volume.
- Cross-H shell confirmation and amplitude ceilings in the 3D driver.

Current decisions:

- Ring removal is subtractive only.
- **Patched is the default model.** A/B with `examples/compare_ring_models.py`
  showed that patched and parametric rolling are close but fail oppositely
  (patched over-subtracts at the ring centres, parametric under-subtracts on
  H=1/3); patched hugs the diffuse baseline better overall.
- Use the Mantid-background-subtracted `*_cc_sub_bkg.nxs` input when available
  (but see Open validation — the un-subtracted `*_cc.nxs` is under evaluation).
- `RING_PRESET=cc_on` is preferred for the cleaner background-subtracted data.
- Keep `q_step=0.02` as the stable default. Finer `0.015` can reduce ring
  leftovers but has not been adopted because it can remove broad diffuse signal.
- `profile_method="median"` and low/zero texture-Q pooling remain the preferred
  levers for ring residuals.

Resolved issues:

- Ring off-centering was rechecked and is not a blocker on testable data.
- Mask-and-replace ring cleanup was removed because radial excess is not a valid
  separator between ring and diffuse signal.
- Integer-H phantom ring troughs are suppressed by cross-H ring confirmation and
  amplitude ceilings.

Open validation:

- Inspect residual weak ring arcs after Bragg cleanup.
- Decide whether the `cc_on` preset should become the default script preset for
  all available real data.
- **Texture-contrast compression (both models).** The fitted azimuthal texture
  `T(φ)` is flattened to ≈half the data-truth contrast at bright shells
  (|Q|≈2.69, H=0) → bright arcs under-subtracted, dim arcs over-subtracted.
  Lever is contrast (lower `texture_ridge`, higher `n_fourier`), not a background
  term. The mean removal-% metric is blind to it; judge on the texture overlay
  (`examples/tune_parametric_ring.py`) and the diverging / per-φ residual figures
  (`examples/compare_ring_models.py`).
- **Input under evaluation:** raw `*_cc.nxs` (no Mantid empty-background
  subtraction) vs `*_cc_sub_bkg.nxs`, since the subtracted file may add artifacts.

## Phase 3 — Bragg Punch And Backfill  Implemented

Current production path:

- `examples/punch_bragg_3d.py`
- `examples/backfill_bragg_3d.py`
- `examples/explore_slice.py` for interactive all-H QA.

Current recommended QA settings:

```bash
MODE=both
MIN_I=0.8
MIN_PROM=0.8
INTEGER_FIT_POSITION=1
INTEGER_FIT_SHAPE=1
INTEGER_H_GUARD=0.12
SEARCH_EXCLUDE_H=-0.6667,-0.3333,0.3333,0.6667
SEARCH_EXCLUDE_H_WIDTH=0.08
BACKFILL_METHOD=q_shell
```

Current decisions:

- The direct beam is punched separately as an origin ellipsoid
  `INCIDENT_ELLIPSOID_R_HKL=0.15,0.50,1.00`.
- `MODE=both` is visually preferred over integer-only, but unrestricted search
  can damage fractional-H diffuse planes. Search exclusions protect those planes.
- Integer-node Bragg holes are guarded in H so they do not bleed from integer-H
  planes into `H=±1/3` or `H=±2/3` diffuse planes.
- Ordinary Bragg holes can be filled with `method="q_shell"` from the robust
  background level at the same `|Q|`.
- Direct-beam backfill remains a special just-outside-`|Q|` fill.

Open validation:

- Confirm guarded `MODE=both` leaves no important small Bragg peaks.
- Check whether `q_shell` fill creates radial banding around large holes.
- Decide whether search exclusions should be derived from known magnetic diffuse
  planes rather than passed manually.

## Phase 3b — Radial-Background Flatten  Implemented (default ON, step 4)

Isotropic complement to the per-plane ring removal: sweeps spherical `|Q|`
shells and subtracts a smooth, continuous per-shell background **floor** so the
radial pedestal flattens to ≈0 while anisotropic diffuse and Bragg residuals are
preserved.

- `nebula3d.preprocessing.flatten_radial_background` (`src/nebula3d/preprocessing/radial_flatten.py`)
- `examples/flatten_background_3d.py`; the explicit background-removal step 4 in
  `run_pipeline.py`, default ON (disable with `FLATTEN=0`). The ΔPDF's own
  Gaussian `SUBTRACT_BG` blur defaults off — it is the alternative remover, never
  combined with the flatten.
- Default estimator `floor` (p25); `mode` / `median` / `snip` also available.

Use the flatten instead of a K-L `SUBTRACT_BG` blur, not with it. The blur can
attenuate on-axis real-space signal at the same length scale as the smooth
background, while the isotropic flatten preserves anisotropic contrast by
subtracting only a function of `|Q|`. Judge the result on representative planes
and by the round-trip consistency check.

Robustness should be checked with `examples/validate_flatten.py`, a
**non-circular** QA (the per-shell-median check is nearly tautological). The
expected diagnostics are:

- **Background is isotropic** — octant-floor spread / |bg| ≈ 0.10–0.14 (≪ 0.3),
  so subtracting one level per |Q| shell is valid; we are not mislabeling
  anisotropic structure as background.
- **Features preserved** — strong anisotropic (Bragg/satellite) contrast 100%
  retained; the subtraction is a function of |Q| alone, so it cannot distort
  anisotropy (only shifts the radial mean).
- **No over-subtraction** — negative fraction a stable ~24.8% (the p25
  expectation), deep negatives (< −3σ) only ~1.2–1.7%.
- **`floor` (p25) is the validated default** — `median`/`mode` centre the shell
  (~50% negative) and flag as over-subtraction; `floor` keeps the bulk positive
  and preserves possibly-real isotropic diffuse. No default change warranted.
Open validation:

- Caveat (bounded): only ~2–3% of voxels sit beyond the reliably-sampled
  |Q| ≈ 10 Å⁻¹, where `bg(|Q|)` is tiny and smoothing-extrapolated.
- For a 3D-PDF (Bragg-kept) input the flatten also passes, but the un-punched
  direct beam dominates the innermost shells (bg span ~4.3 vs 0.12 backfilled);
  treat the direct beam before flattening if the flatten is used in the PDF path.
- Optional future work: an H-aware residual-cross reduction that lowers the
  leftover L=0 cross without harming the sharp H-axis peaks.

## Phase 4 — 3D-DeltaPDF  Implemented

Implemented API and drivers:

- `nebula3d.analysis.compute_delta_pdf` (algorithm: `docs/algorithms/delta_pdf.md`)
- apodization: `hann`, `gaussian`, `none`; optional mean subtraction; symmetric
  zero padding; real-space axes from the UB matrix
- `examples/delta_pdf.py` — full 3D transform; slice/line-cut/radial PNGs and a
  `_delta_pdf.h5` cache
- `examples/delta_pdf_plane.py` — single reciprocal H-plane 2D-ΔPDF
- `examples/explore_delta_pdf_ortho.py` — interactive viewer, all three
  orthoslice planes at once (recommended); `examples/explore_delta_pdf.py` —
  single y_K–z_L plane with an x_H slider
- `examples/run_pipeline.py` — one-command end-to-end ΔPDF runner (resume-aware)

Correct transform recipe `fftshift(fftn(ifftshift(·)))` with symmetric padding;
the real part is valid for the centrosymmetric (`mmm`) data.

Resolved issues:

- **Fourier-centring bug fixed (2026-06-05).** The transform lacked `ifftshift`
  on the centre-origin input and used one-sided padding, flipping real-space
  peak signs by pixel parity (mixed +/- lobes per feature; scrambled `x_H=0`).
  Guarded by `test_delta_pdf_centring_positive_peak`.
- DC handling: subtract the mean *after* windowing so `Σ I = 0` and the `r=0`
  self-correlation spike is suppressed.

Next work / open validation:

- Reduce the near-origin artifact (residual Bragg leakage, backfill
  discontinuities, direct-beam punch); consider tapered punch boundaries.
- Decide whether the principal-axis cross artifact needs masking.
- Compare apodization choices for peak sharpness vs termination ripple.
- Interpret the correlation lattice against the structure / H=±1/3 modulation.

## Phase 4b — 3D-PDF And Satellite Diagnostic  Implemented

Two siblings to the ΔPDF path, added 2026-06-08:

- **3D-PDF (total scattering, Bragg kept).** `examples/run_pipeline_pdf.py` +
  `examples/pdf_3d.py` skip the punch/backfill and transform the total
  scattering, giving a Patterson-like 3D-PDF (average structure plus diffuse).
  Reuses `compute_delta_pdf` and the shared `*_ringremoved.h5`; `SUBTRACT_BG` is
  off (it is a ΔPDF-only axis-cross fix). Output `*_3dpdf.h5` carries a `kind`
  attr so the ortho viewer labels 3D-PDF vs 3D-ΔPDF.
- **Bragg/diffuse separation diagnostic.** `examples/investigate_bragg_diffuse.py`
  + `nebula3d.analysis.peak_profile`, for diffuse scattering co-located with
  sharp Bragg or satellite features. Calibrates resolution σ(|Q|) on sharp
  Bragg peaks, fits a sharp core + broad diffuse per axis, and reports ξ and the
  diffuse fraction.

Open: decide "Phase B" — subtract the sharp core but keep the broad diffuse at
the satellites, rather than punching them.

## Phase 6 — Q-Space Bragg Punch  Complete (Q is the default since Phase 4)

Migrate the Bragg punch ([`src/nebula3d/analysis/bragg.py`](src/nebula3d/analysis/bragg.py))
from HKL-axis radii to a **Q-space resolution-ellipsoid** described by one
quadratic form `δhklᵀ A δhkl ≤ 1`. Motivation: the peak profile is a function of
**Q** (instrument resolution + size/strain/mosaic), not of the lattice constants;
HKL radii bake in the `b*` scaling and rotate the wrong way off-axis.

Key implementation point: the reciprocal metric `g = UBᵀUB` maps fractional HKL
radii into physical Q-space units. Large apparent anisotropy in HKL can become
near-isotropic in Q when lattice constants and grid sampling are accounted for.
The practical gain from Q-space punching is correct off-axis peak orientation,
support for oblique crystals, and lattice-/condition-portable parameters in
Å⁻¹.

Design — one kernel, multiple shape specs:

| Shape spec | `A` |
|------------|-----|
| legacy HKL radii `(rh,rk,rl)` | `diag(1/rh², 1/rk², 1/rl²)` |
| Q isotropic radius `ρ` (Å⁻¹) | `g / ρ²` |
| Q resolution ellipsoid `M` (radial/tangential) | `UBᵀ M UB`; φ-tail = rank-1 mod of `M` |

| Phase | Scope | Status |
|-------|-------|--------|
| 0 | Characterization + spec tests (golden masters, equivalence invariants) | **done** |
| 1 | Internal quadratic-form kernel routed through `A=diag(1/r²)`; prove equivalence | **done** |
| 2 | Opt-in Q-space spec (`punch_frame`, `punch_q_radius`, `punch_q_radii`); default = legacy | **done** |
| 3 | Unify φ-tail + shape-fit into `M` (fitter returns a 3×3 covariance) | **done** |
| 4 | Validate HKL vs Q on real data, make Q-mode adaptive, then flip the default | **done** |

Phase 0 (done): [`tests/test_bragg_qspace_phase0.py`](tests/test_bragg_qspace_phase0.py)
— 9 tests. Golden masters freeze the current default `punch_bragg` keep-mask
(sha256 + per-mechanism punch counts) on a synthetic volume built on a
representative UB; specification tests pin the HKL ↔ Q-axis equivalence the
future kernel must satisfy. No production code changed.

Phase 1 (done): the single shape kernel `_ellipsoid_inside(δ, radii | shape_matrix)`
in [`src/nebula3d/analysis/bragg.py`](src/nebula3d/analysis/bragg.py). Both `_punch_one`
and `_punch_origin_ellipsoid` now go through it. The `radii=` fast path keeps the
exact `(d/r)²` arithmetic, so the Phase 0 golden masters pass **bit-identical** —
production is unchanged. The `shape_matrix=` path is the general `δᵀ A δ ≤ 1` the
Q-space work will drive (φ-tail + shape-fit stay as-is until Phase 3).
[`tests/test_bragg_qspace_phase1.py`](tests/test_bragg_qspace_phase1.py) — 5 tests:
the diagonal-matrix path matches the radii path (continuous values; masks differ
only at boundary ties), `A = g/ρ²` reproduces the metric sphere, and an
off-diagonal `A` tilts the ellipsoid.

Phase 2 (done): opt-in Q-space punch. `BraggRemover` gains `punch_frame`
(`"hkl"` default / `"q"`), `punch_q_radius` (isotropic Å⁻¹ → `A = g/ρ²`), and
`punch_q_radii` (per a*,b*,c* Å⁻¹ → `A = Pᵀ diag(1/r²) P`); `_punch_one` punches
via `shape_matrix` in Q-mode (φ-tail + per-peak HKL fit stay legacy-only until
Phase 3). Threaded through `PunchParams` → `punch_bragg`, the server
`StageParamsIn`/`build_params`, and the web Run-pipeline **Frame** selector
(HKL ↔ Q-space, Å⁻¹ radius). Default behaviour unchanged.
[`tests/test_bragg_qspace_phase2.py`](tests/test_bragg_qspace_phase2.py) — 7 tests
(metric-sphere punch, diagonal-metric equivalence to the converted HKL punch,
per-axis anisotropy) plus a server `build_params` Q-override test.

Phase 3 (done): the per-peak integer-node fit can return a full 3×3 HKL
covariance (a *tilted* resolution ellipsoid following the peak's real
orientation) instead of three axis-aligned radii, and the φ-tail is folded in as
a rank-1 tangential inflation of that matrix — replacing the union-of-two-
ellipsoids. Opt-in via `integer_fit_covariance` (default off → diagonal-radii fit
+ union φ-tail bit-identical). The covariance fit reduces *exactly* to the
diagonal radii for an axis-aligned peak. Threaded through `PunchParams`, the
server `punch_fit_covariance`, and a web punch-card toggle.
[`tests/test_bragg_qspace_phase3.py`](tests/test_bragg_qspace_phase3.py) — 8 tests
(diagonal reduction, tilted orientation, φ-tail tangent-only inflation, fit
integration, Q-mode adaptivity) plus a server `build_params` test.

Phase 4 (validating): HKL-vs-Q comparison via
[`examples/compare_punch_frames.py`](examples/compare_punch_frames.py). The
validation path compares matched Q radii against the legacy HKL footprint and
checks whether an isotropic Q radius is justified for a given volume. Two fixes landed:
(a) Q-mode was silently ignoring `margin` — now applied; (b) Q-mode was
*fixed-shape* (bypassing the per-peak fit + φ-tail), which would under-punch ~41%
vs the intended punch — Q-mode is now **adaptive** (`_fit_base_radii` floors
the per-peak fit to the Q resolution, in Å⁻¹; the diagonal fit then punches via
the same radii path as HKL). The final gate is a **ΔPDF-level A/B** through
the full pipeline ([`examples/compare_delta_pdf_frames.py`](examples/compare_delta_pdf_frames.py)),
followed by slice-level checks with
[`examples/plot_punch_slices.py`](examples/plot_punch_slices.py).

**The default is now Q** (`PunchParams.punch_frame="q"`,
`punch_q_radii=(0.097, 0.072, 0.115)` Å⁻¹): the punch
footprint is a lattice-/temperature-portable reciprocal-Å⁻¹ resolution floor,
still modulated by the per-peak fit + φ-tail. The Phase 0 golden master was
regenerated to freeze the new default; set `punch_frame="hkl"` to restore the
legacy rlu footprint.

Two findings now encoded as tests / constraints:

- **"Bit-identical" has a caveat.** The real UB is orthogonal only to ~0.5%, so
  HKL- and Q-axis punches differ at up to ~10 boundary voxels on real data;
  bit-identical masks hold only for an *exactly* diagonal metric.
- **Phase-1 design constraint:** kernel-equivalence checks must compare
  **continuous quadratic values with a tolerance**, not thresholded boolean
  masks — boundary ties (`quad == 1`) flip a couple of voxels purely from
  floating-point path differences, even under an exact rotation.

Backward compatibility (a hard requirement): `punch_radii` keeps working forever
(converts to a diagonal `A`), preserving the existing bragg tests, saved
pipelines, and the web Run-pipeline ring/punch controls. Defaults move only in
Phase 4.

## Phase 7 — Performance  Planned (next focus)

Running the pipeline at **full ΔPDF resolution** (no downsampling of the HKL
volume) has made the end-to-end computation heavy. The goal of this phase is to
**find and remove wasted work without changing any numerical output** — i.e. a
quality-preserving optimization pass, not an accuracy/speed trade-off.

Approach (audit first, optimize second):

- **Map the hot path.** Profile `examples/run_pipeline.py` at full resolution
  (e.g. `cProfile` / `py-spy`) to rank the real cost centres before touching
  code; record wall-time + peak memory per stage (ring removal, punch, backfill,
  flatten, ΔPDF FFT, back-FFT consistency check).
- **Hunt duplicated computation.** Suspected redundancy to confirm:
  - the back-FFT consistency check (stage 6) re-deriving quantities the forward
    ΔPDF (stage 5) already computed (window, padded grid, UB-derived axes);
  - per-H-plane ring fits recomputing shared `|Q|`/shell geometry every slice;
  - Bragg masks / `|Q|` grids / metric `g = UBᵀUB` rebuilt in multiple stages
    instead of computed once and threaded through;
  - the resume cache (`*_ringremoved.h5`, `*_delta_pdf.h5`) reloading or
    recomputing intermediates that are already on disk.
- **Cheap structural wins to evaluate:** real-input FFTs (`rfftn`/`irfftn`)
  where the data is real and centrosymmetric; in-place / dtype-aware array ops to
  cut peak memory; vectorizing per-slice Python loops; caching the apodization
  window and real-space axes.

Hard constraint: **byte-for-byte (or within-tolerance) identical results.** Each
optimization must be guarded by an equivalence test against the current output
(reuse the Phase 0 golden-master pattern and the `pdf_check` round-trip metric)
before it is accepted. No change to defaults, masks, or transform recipe.

## Phase 5 — Release Hygiene  In Progress

Before treating the pipeline as a stable release candidate:

- Keep README, QUICKSTART, ROADMAP, and algorithm docs aligned with current
  defaults.
- Keep generated data and preview images out of commits unless they are explicit
  reference artifacts.
- Add a lightweight CLI or config-file runner if env-var scripts become hard to
  reproduce.
- Done: `scripts/check.sh` mirrors GitHub CI (`.github/workflows/ci.yml`) —
  pytest + `ruff check src/ tests/` + `mypy src/nebula3d` — and can be installed as
  a `pre-push` hook; the suite is at 162 passing tests.
- Done for `v0.2.0`: the recommended workflow now has a documented endpoint
  (`pdf_check` / consistency viewer), and package/web/API version metadata is
  aligned.
- Still open: add CI coverage that specifically exercises the Bragg
  guard/exclusion behavior, not just import/type checks.

## In-Browser Engine — Large-Volume Ceiling  Blocked (needs new architecture)

The GitHub Pages / Pyodide build runs the real pipeline in the browser, but the
**32-bit WASM heap (~2 GB practical ceiling)** caps how large a volume it can
reduce. Two prototypes on branch **`feat/in-browser-parallel-float32`** improve it
without changing native behaviour:

- **Parallel ring removal** — shards the ~70% ring stage across a pool of Web
  Workers (each its own Pyodide), bit-identical to serial.
- **float32 build** — computes in float32 in-browser (native stays float64),
  ~halving peak memory and roughly doubling the voxel cap (~23 M → ~56 M).

**Still blocked:** a real full-resolution file (301×401×401 = **48.4 M voxels**,
~1.55 GB float32 peak) OOMs near the WASM ceiling once Pyodide runtime overhead is
added. float32 is necessary but not sufficient at this size.

Options to revisit (a **different architecture** is likely needed):

- **memory64 (wasm64) Pyodide build** — a 64-bit heap removes the ~2 GB ceiling;
  experimental, larger download, Chrome/Firefox only (Safari lags).
- **Out-of-core / chunked reduction** — stream the volume in slabs so peak memory
  is bounded by the slab, not the whole grid (the parallel slab driver in
  `nebula3d.webparallel` is a starting point).
- **Server/serverless compute** — offload the reduction to a backend with the
  Pages site as a thin client (trades the privacy / no-upload property).
- **Optional downsampling** — a coarser in-browser preview mode for oversized
  inputs, full resolution via the native install.

Until then the in-browser engine targets modest volumes; full-resolution data
goes through the native build (`pip install "nebula3d[web]" && nebula3d-web`).
