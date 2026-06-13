# Hand-off Notes — neutron-diffuse

**Date:** 2026-06-13
**Repo:** `neutron-diffuse`
**Current branch:** `main`

## Current State

The full pipeline now runs end to end through the 3D-ΔPDF:

1. Full-3D powder-ring subtraction via `examples/remove_rings_3d.py`.
2. Guarded Bragg/satellite punch via `examples/punch_bragg_3d.py`.
3. Bragg-hole backfill via `examples/backfill_bragg_3d.py`.
4. Isotropic radial-background flatten via `examples/flatten_background_3d.py` —
   the **explicit background-removal step, ON by default** (disable with
   `FLATTEN=0`). Sweeps
   spherical `|Q|` shells from 0 to Qmax; in each shell it estimates a robust
   background **floor** (low percentile / mode, default p25) that sits below the
   diffuse and Bragg-residual high tail, smooths the per-shell levels into one
   continuous `bg(|Q|)`, and subtracts it. The radial pedestal flattens to ≈0
   while anisotropic diffuse and Bragg residuals are preserved. Feeds the ΔPDF
   in place of the backfilled volume. The ΔPDF's own Gaussian `SUBTRACT_BG` blur
   defaults **off** — it is the *alternative* remover. **Use the flatten
   *instead of* a K-L `SUBTRACT_BG` blur, never both**: validated on 22K, the
   blur (σ_H=0, per-H-plane) destroys the
   on-axis H-direction ΔPDF signal — it removes each H-plane's integrated
   intensity, which is the x_H Fourier component (real lattice-`a` peaks dropped
   to ~1-3%, for any σ). The isotropic flatten removes the background without
   touching per-plane DC, so it preserves that signal. **Judge the effect on the
   L=0 (H-K) plane** — the H=0 (K-L) plane barely differs between methods.
   (Core: `flatten_radial_background` in
   `src/ndiff/preprocessing/radial_flatten.py`.)

   **Robustness validated (2026-06-10).** `examples/validate_flatten.py` is a
   **non-circular** QA for this stage — the per-shell-median check in
   `flatten_background_3d.py` is nearly tautological (the subtraction is built
   from that statistic). It tests (1) **isotropy** of the subtracted level
   (octant-floor spread vs |Q|) and (2) **feature preservation** (background
   residual, negative fraction vs noise, strong-feature contrast retention),
   and prints a PASS/FLAG report. Run across 22/45/100K, all **PASS**:
   background is isotropic (octant spread ≈0.10–0.14 ≪ 0.3), strong-feature
   contrast **100% retained**, negatives a stable ~24.8% (the p25 expectation,
   no over-subtraction). The default `floor` (p25) is the validated operating
   point: `median`/`mode` centre the shell (~50% negative) and flag as
   over-subtraction. No default change was warranted. The anisotropic-tail
   shell fraction (40% at 22K vs 16–17% at 45/100K) tracks the stronger
   magnetic diffuse at low T, all retained. Caveat: ~2–3% of voxels sit beyond
   the reliably-sampled |Q|≈10 Å⁻¹ (tiny, smoothing-extrapolated bg). For a
   Bragg-kept (3D-PDF) input the flatten also passes but the un-punched direct
   beam dominates the innermost shells — handle the direct beam first if the
   flatten is used in the PDF path.
5. 3D-ΔPDF transform via `examples/delta_pdf.py` (full 3D),
   `examples/delta_pdf_plane.py` (single reciprocal H-plane 2D), and the
   interactive viewers `examples/explore_delta_pdf_ortho.py` (recommended — all
   three orthoslice planes at once) and `examples/explore_delta_pdf.py` (single
   y_K–z_L plane with an x_H slider).

A **Fourier-centring bug** in `compute_delta_pdf` was found and fixed
(2026-06-05): the transform was missing `ifftshift` on the centred input and
used one-sided zero-padding, which flipped real-space peak signs by pixel
parity (each correlation split into +/- lobes; `x_H=0` looked scrambled). The
correct recipe is `fftshift(fftn(ifftshift(·)))` with symmetric padding. See
`docs/algorithms/delta_pdf.md` and regression test
`test_delta_pdf_centring_positive_peak`. After the fix the ΔPDF shows coherent
single-sign correlation peaks on the expected b/c lattice.

Preferred input is the Mantid-background-subtracted file:

```text
data/raw/*_cc_sub_bkg.nxs
```

Run with a Python ≥3.10 environment that has the dependencies installed
(`pip install -e ".[dev]"`); the commands below use `python3`.

## Additional Workflows (2026-06-08)

Two siblings to the ΔPDF pipeline were added:

- **3D-PDF (total scattering, Bragg KEPT).** `examples/run_pipeline_pdf.py`
  chains ring removal → `examples/pdf_3d.py` → the ortho viewer, deliberately
  **skipping** the punch/backfill so the transform keeps the Bragg peaks (a
  Patterson-like 3D-PDF of the average structure plus the diffuse). It reuses
  the same `compute_delta_pdf` engine and the SAME `*_ringremoved.h5` files as
  the ΔPDF workflow; the only deliberate setting difference is `SUBTRACT_BG`
  off (smooth-bg subtraction is a ΔPDF-only axis-cross fix). Output
  `*_3dpdf.h5` carries a `kind` attr so the ortho viewer titles it 3D-PDF vs
  3D-ΔPDF.
- **Bragg/diffuse separation diagnostic.** `examples/investigate_bragg_diffuse.py`
  (via `ndiff.analysis.peak_profile`) handles the case where magnetic diffuse
  sits AT the q=integer±1/3 satellites, where punch+backfill would destroy it.
  It calibrates instrument resolution σ(|Q|) on nuclear (integer-node) Bragg,
  then decomposes each satellite along H/K/L into a sharp Gaussian core + a
  broad Lorentzian, reporting correlation length ξ and the diffuse fraction.
  The `T_SERIES=1` overlay shows the broad (diffuse) component present at 22 K
  and 45 K and absent at 100 K — the magnetic signature. It is a
  measurement/decision tool; it does not modify the pipeline volumes.

## Recommended QA Command

Interactive all-H cleanup viewer:

```bash
env PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl USE_BACKGROUND=0 \
PUNCH_PRESET=cc_on MODE=both MIN_I=0.8 MIN_PROM=0.8 \
INTEGER_FIT_POSITION=1 INTEGER_FIT_SHAPE=1 INTEGER_H_GUARD=0.12 \
SEARCH_EXCLUDE_H=-0.6667,-0.3333,0.3333,0.6667 SEARCH_EXCLUDE_H_WIDTH=0.08 \
BACKFILL_METHOD=q_shell H_VALUE=0.3333 \
python3 examples/explore_slice.py
```

This opens four panels: `data`, `Removed ring`, `Punched`, `Backfilled`.

Latest visual QA preference:

- `MODE=both` looked better than integer-only because search catches residual
  off-integer/satellite peaks.
- Protect fractional-H diffuse planes from search with `SEARCH_EXCLUDE_H`.
- Keep integer punches confined near integer-H planes with `INTEGER_H_GUARD=0.12`.
- The last guarded `MODE=both` viewer punched about `6.02%` of valid voxels
  versus `7.02%` for unrestricted `MODE=both` and `2.27%` for integer-only.

## One-Command Workflow

`examples/run_pipeline.py` chains every stage end-to-end — raw `.nxs` → (1) ring
removal → (2) Bragg punch → (3) backfill → (4) radial-background flatten →
(5) 3D-ΔPDF — then opens two interactive viewers: (6) the 4-panel KL cleanup QA
viewer (`explore_slice.py`: data | ring removed | punched | backfilled, with H +
vmin/vmax sliders) and (7) the ΔPDF orthoslice viewer
(`explore_delta_pdf_ortho.py`).  Close each window to advance.  It uses the
validated `cc_on` / clean-ΔPDF presets below.  Each compute stage is **skipped
if its output already exists** (resume); pass `FORCE=1` or
`FORCE_FROM=rings|punch|backfill|flatten|pdf` to recompute, `NO_VIEWER=1` to
stop after the ΔPDF.

**Background removal is step 4 (the radial flatten), ON by default** — disable
with `FLATTEN=0`.  The ΔPDF's own Gaussian `SUBTRACT_BG` blur defaults **off**:
it is the *alternative* remover, never combined with the flatten (running both
subtracts the background twice and the σ_H=0 blur destroys the on-axis H signal
the flatten preserves).  The pipeline prints the active background method
(`[background] step-4 radial flatten: ON | ΔPDF SUBTRACT_BG (legacy blur): off`)
and warns if both are on.  The ΔPDF stage is dataset-aware — it recomputes if
the cached `_delta_pdf.h5` came from a different input or transform config
(including switching the flatten on/off, or setting `SUBTRACT_BG`).  Every
individual stage's env vars still pass through and override the defaults.

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
python3 examples/run_pipeline.py
```

The individual stages can still be run by hand (the batch commands below).

## Batch Pipeline

Run from the repo root:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl RING_PRESET=cc_on \
python3 examples/remove_rings_3d.py

PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl PUNCH_PRESET=cc_on MODE=both \
MIN_I=0.8 MIN_PROM=0.8 INTEGER_FIT_POSITION=1 INTEGER_FIT_SHAPE=1 \
INTEGER_H_GUARD=0.12 \
SEARCH_EXCLUDE_H=-0.6667,-0.3333,0.3333,0.6667 SEARCH_EXCLUDE_H_WIDTH=0.08 \
PREVIEW=0 \
python3 examples/punch_bragg_3d.py

PYTHONPATH=src METHOD=q_shell \
python3 examples/backfill_bragg_3d.py
```

Outputs are written under `data/processed/`:

- `*_ringremoved.h5`
- `*_braggpunched.h5`
- `*_braggpunched_backfilled.h5`

## Algorithm Decisions

Ring removal:

- Use `PatchedRadialRingModel` slice-by-slice over H.
- Keep `q_step=0.02` as the default. `0.015` can reduce ring leftovers but may
  eat broad diffuse on rich-diffuse slices.
- Keep `profile_method="median"` and `texture_q_smooth=0.0` for the current
  class defaults. The `cc_on` 3D script preset uses a slightly smoother texture
  (`n_fourier=6`, `texture_q_smooth=0.02`, `texture_ridge=0.08`) for the cleaner
  Mantid-background-subtracted data.
- Ring removal is subtractive only. The removed mask-and-replace experiment was
  invalid because radial excess is also real diffuse scattering.

Bragg punch:

- The direct beam is not treated as a Bragg peak. It is punched separately as an
  origin ellipsoid, currently `INCIDENT_ELLIPSOID_R_HKL=0.15,0.50,1.00`.
- Integer-node Bragg logic enumerates integer `(h,k,l)` nodes, decides whether a
  nearby peak exists, recentres to the measured peak, and can fit local position
  and anisotropic shape.
- `MODE=both` runs integer detection first, then search on the integer-punched
  residual.
- `SEARCH_EXCLUDE_H` protects explicit fractional-H diffuse planes from the
  hkl-agnostic search stage. `SEARCH_EXCLUDE_H_FRACTIONS=0.3333,0.6667`
  (cc_on default) protects the whole q=1/3 family *periodically* — every
  integer±1/3 plane (±1/3, ±2/3, ±4/3, ±5/3 …), not just a fixed list.
- `INTEGER_H_GUARD` prevents integer-H Bragg punch ellipsoids from extending
  into fractional-H diffuse planes.
- `INTEGER_LOCAL_NMAD=8` (cc_on default) catches small-but-sharp weak Bragg at
  integer nodes via local-MAD prominence (below the absolute floors). It is
  position-locked to integer nodes, so it never touches the q=1/3 diffuse.
  Validated on 45K: catches ~372k extra Bragg voxels and spares ~178k on the
  higher-order thirds planes, with thirds-plane diffuse punching unchanged.
- **Q-space punch migration underway (ROADMAP Phase 6, Phase 0 done).** The punch
  ellipsoid is defined in HKL today; it is being moved to a Q-space
  resolution-ellipsoid (one quadratic form `δhklᵀ A δhkl ≤ 1`) because the peak
  profile is a function of Q, not the lattice constants. On TbTi3Bi4 the metric
  is diagonal to ~0.5%, so the default radii are already near-isotropic in Q
  (`(0.09,0.12,0.45)` rlu ≈ `(0.097,0.072,0.115)` Å⁻¹). `punch_radii` stays
  supported (maps to a diagonal `A`); defaults are unchanged. Phase 0 is the
  characterization/spec suite `tests/test_bragg_qspace_phase0.py`. See
  `docs/algorithms/bragg_cleanup.md` → "Punch Coordinate Space".

Backfill:

- `backfill_bragg(method="q_shell")` fills ordinary Bragg holes from the robust
  background at the same `|Q|`.
- The direct beam keeps its special just-outside-`|Q|` fill, because the adjacent
  local shell can sit inside the direct-beam over-subtraction halo.

## Resolved Issues

- Ring off-centering was re-investigated and is not a blocker on testable data.
  At H=0 the fitted centre is at the origin to numerical precision. Apparent
  offsets at nonzero H are projection/fit artifacts and do not materially change
  residual metrics.
- The mask-and-replace ring cleanup was removed. It masked real diffuse signal
  because it keyed on radial excess rather than azimuthal smoothness.
- The direct-beam fill originally bled into neighbouring H planes when using a
  solid `|Q|` ball. It now fills only the connected beam footprint and enclosed
  shadow.
- The 3D-ΔPDF Fourier-centring bug (missing `ifftshift`, one-sided padding) is
  fixed. Symptom was each atom-like feature splitting into mixed +/- lobes and a
  scrambled `x_H=0` plane. See `docs/algorithms/delta_pdf.md`.
- The axis **cross** along `y_K=0` / `z_L=0` is diagnosed (2026-06-05): it is
  the FT of the **residual smooth diffuse envelope** that survives ring removal
  + Bragg punch + backfill — NOT Bragg leakage or masking. It appears even on
  Bragg-free planes (`H=1/3`), with 0 % masking on the axis lines. The envelope
  is ≈ separable, so its FT lands on the axes; `subtract_mean` only removes the
  scalar DC term, not the envelope shape. Fix (implemented): subtract a smooth
  background (Gaussian blur `σ≈1.5 r.l.u.`) before windowing — shipped as
  `subtract_smooth_bg` in `compute_delta_pdf` and `SUBTRACT_BG=<σ rlu>` in the
  `delta_pdf.py` / `delta_pdf_plane.py` drivers. A threshold-clip alternative
  (`I_new=max(I−c,0)`) was tested and rejected: it sparsifies the input but does
  NOT remove the cross (the bright central envelope survives any threshold) and
  adds hard-edge ripple. See `docs/algorithms/delta_pdf.md` → "The axis cross is
  the residual diffuse background" / "Methods compared", and the side-by-side
  driver `examples/compare_delta_pdf_methods.py`.

## Tests

Latest full suite:

```bash
PYTHONPATH=src python3 \
  -m pytest -o addopts=''
```

Expected current result: `140 passed` (includes the ΔPDF centring guard
`test_delta_pdf_centring_positive_peak`, the `peak_profile` diagnostic tests,
the flatten robustness guards in `test_radial_flatten.py`, and the Q-space punch
Phase 0 characterization/spec suite `test_bragg_qspace_phase0.py`).

`scripts/check.sh` runs the same three checks as GitHub CI (`.github/workflows/ci.yml`)
— pytest, `ruff check src/ tests/`, and `mypy src/ndiff` — and is the recommended
pre-push gate (`PY=/path/to/python bash scripts/check.sh`, or symlink it as
`.git/hooks/pre-push`). A check whose tool is not installed is skipped, not
failed, so a bare clone without the dev extras still works.

## Next Step

The 3D-ΔPDF now produces a physically sensible map (coherent single-sign
correlation peaks). Remaining tuning / inspection:

- Reduce the near-origin artifact: residual high-`|Q|` Bragg leakage, backfill
  discontinuities at punch boundaries, and the direct-beam punch all feed it.
  Consider tapered punch boundaries or a softer high-`|Q|` window.
- The cross along the `y_K=0` / `z_L=0` axes is diagnosed and fixed: it is the
  FT of the residual smooth diffuse background, removed by `SUBTRACT_BG=<σ rlu>`
  (smooth-bg subtraction; threshold-clip was tested and rejected — see Resolved
  Issues). Remaining: re-interpret the cleaned H=1/3 / H=2/3 K–L correlation
  lattice against the TbTi3Bi4 structure, and pick a default `σ` for batch runs.
- Compare `apodization` (`hann` vs `gaussian` vs `none`) for peak sharpness vs
  termination ripple.
- Interpret the K-L correlation lattice against the TbTi3Bi4 structure and the
  H=±1/3 modulation.
- The Bragg/diffuse diagnostic (see Additional Workflows) confirms broad diffuse
  co-located at the q=1/3 satellites at 22/45 K. "Phase B" — subtract the sharp
  resolution core but KEEP the broad diffuse at those satellites, rather than
  punching them — is the gated next experiment for the magnetic signal.
