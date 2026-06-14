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
```

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
- `PatchedRadialRingModel`
- Per-H-plane fit over the full 3D volume.
- Cross-H shell confirmation and amplitude ceilings in the 3D driver.

Current decisions:

- Ring removal is subtractive only.
- Use the Mantid-background-subtracted `*_cc_sub_bkg.nxs` input when available.
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

- `ndiff.preprocessing.flatten_radial_background` (`src/ndiff/preprocessing/radial_flatten.py`)
- `examples/flatten_background_3d.py`; the explicit background-removal step 4 in
  `run_pipeline.py`, default ON (disable with `FLATTEN=0`). The ΔPDF's own
  Gaussian `SUBTRACT_BG` blur defaults off — it is the alternative remover, never
  combined with the flatten.
- Default estimator `floor` (p25); `mode` / `median` / `snip` also available.

Validated on 22K (flatten preserves diffuse: ~94% of bright satellites retained;
flattens the radial pedestal). **Use the flatten instead of a K-L `SUBTRACT_BG`
blur, not with it** — the blur (σ_H=0) destroys the on-axis H-direction ΔPDF
signal (real lattice-`a` peaks → ~1-3%, any σ), which the isotropic flatten
preserves. Judge on the L=0 (H-K) plane.

Robustness validated across 22/45/100K (2026-06-10) via `examples/validate_flatten.py`
— a **non-circular** QA (the per-shell-median check is nearly tautological).
Results, all three temperatures **PASS**:

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
- Anisotropic-tail shell fraction tracks temperature (40% at 22K vs 16–17% at
  45/100K), consistent with stronger magnetic diffuse at low T — all retained.

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

- `ndiff.analysis.compute_delta_pdf` (algorithm: `docs/algorithms/delta_pdf.md`)
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
  + `ndiff.analysis.peak_profile`, for magnetic diffuse co-located at the
  q=integer±1/3 satellites. Calibrates resolution σ(|Q|) on nuclear Bragg, fits
  a sharp core + broad diffuse per axis, and reports ξ and the diffuse fraction.
  The T-series shows broad diffuse at 22/45 K and none at 100 K.

Open: decide "Phase B" — subtract the sharp core but keep the broad diffuse at
the satellites, rather than punching them.

## Phase 6 — Q-Space Bragg Punch  Complete (Q is the default since Phase 4)

Migrate the Bragg punch ([`src/ndiff/analysis/bragg.py`](src/ndiff/analysis/bragg.py))
from HKL-axis radii to a **Q-space resolution-ellipsoid** described by one
quadratic form `δhklᵀ A δhkl ≤ 1`. Motivation: the peak profile is a function of
**Q** (instrument resolution + size/strain/mosaic), not of the lattice constants;
HKL radii bake in the `b*` scaling and rotate the wrong way off-axis.

Key finding (TbTi3Bi4 22/45/100 K): the reciprocal metric `g = UBᵀUB` is diagonal
to ~0.5% (orthorhombic). The default `punch_radii=(0.09,0.12,0.45)` rlu — a 5×
HKL anisotropy — is `≈(0.097,0.072,0.115)` Å⁻¹, i.e. **near-isotropic in Q**
(max/min < 1.6). So for this data a Q-axis punch is mathematically the current
punch in different units; the genuine gain is off-axis peaks, oblique crystals,
and lattice-/temperature-portable parameters in Å⁻¹.

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
(sha256 + per-mechanism punch counts) on a synthetic volume built on the real
22 K UB; specification tests pin the HKL ↔ Q-axis equivalence the future kernel
must satisfy. No production code changed.

Phase 1 (done): the single shape kernel `_ellipsoid_inside(δ, radii | shape_matrix)`
in [`src/ndiff/analysis/bragg.py`](src/ndiff/analysis/bragg.py). Both `_punch_one`
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

Phase 4 (validating): real-data HKL-vs-Q comparison via
[`examples/compare_punch_frames.py`](examples/compare_punch_frames.py). Findings on
22/45/100 K: the volume-matched Q radius **ρ ≈ 0.093 Å⁻¹ is T-invariant**
(portability confirmed); isotropic Q differs ~22–27% from HKL because the peaks
are ~1.6× anisotropic in Q (so a single ρ is not justified). Two fixes landed:
(a) Q-mode was silently ignoring `margin` — now applied; (b) Q-mode was
*fixed-shape* (bypassing the per-peak fit + φ-tail), which would under-punch ~41%
vs the validated ~6% punch — Q-mode is now **adaptive** (`_fit_base_radii` floors
the per-peak fit to the Q resolution, in Å⁻¹; the diagonal fit then punches via
the same radii path as HKL). Production HKL vs Q-anisotropic-adaptive default now
agrees at **Jaccard 0.89–0.93**; the residual ~8% is search peaks (Q
metric-ellipsoid + folded φ-tail vs HKL axis-aligned + union φ-tail), a frame
difference rather than a regression. The final gate — a **ΔPDF-level A/B** through
the full pipeline ([`examples/compare_delta_pdf_frames.py`](examples/compare_delta_pdf_frames.py)) —
passed cleanly on 22 K: HKL vs Q-adaptive ΔPDF maps agree at **Pearson r =
0.9998**, relative RMS 1.9%, max difference 0.06% of the peak (the difference is
invisible at the data colour scale). The slice-level check
([`examples/plot_punch_slices.py`](examples/plot_punch_slices.py)) confirmed the
two punches differ only at Bragg-peak edges and leave the H=1/3 diffuse plane
untouched.

**The default is now Q** (`PunchParams.punch_frame="q"`,
`punch_q_radii=(0.097, 0.072, 0.115)` Å⁻¹ ≈ the old HKL footprint × b*): the punch
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

## Phase 5 — Release Hygiene  In Progress

Before treating the pipeline as a stable release candidate:

- Keep README, ROADMAP, HANDOFF, and algorithm docs aligned with current
  defaults.
- Keep generated data and preview images out of commits unless they are explicit
  reference artifacts.
- Add a lightweight CLI or config-file runner if env-var scripts become hard to
  reproduce.
- Done: `scripts/check.sh` mirrors GitHub CI (`.github/workflows/ci.yml`) —
  pytest + `ruff check src/ tests/` + `mypy src/ndiff` — and can be installed as
  a `pre-push` hook; the suite is at 162 passing tests.
- Still open: add CI coverage that specifically exercises the Bragg
  guard/exclusion behavior, not just import/type checks.
