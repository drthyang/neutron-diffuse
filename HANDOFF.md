# Hand-off Notes — neutron-diffuse

**Date:** 2026-06-05
**Repo:** `neutron-diffuse`
**Current branch:** `main`

## Current State

The full pipeline now runs end to end through the 3D-ΔPDF:

1. Full-3D powder-ring subtraction via `examples/remove_rings_3d.py`.
2. Guarded Bragg/satellite punch via `examples/punch_bragg_3d.py`.
3. Bragg-hole backfill via `examples/backfill_bragg_3d.py`.
4. 3D-ΔPDF transform via `examples/delta_pdf.py` (full 3D),
   `examples/delta_pdf_plane.py` (single reciprocal H-plane 2D), and
   `examples/explore_delta_pdf.py` (interactive y_K–z_L viewer, x_H slider).

A **Fourier-centring bug** in `compute_delta_pdf` was found and fixed
(2026-06-05): the transform was missing `ifftshift` on the centred input and
used one-sided zero-padding, which flipped real-space peak signs by pixel
parity (each correlation split into +/- lobes; `x_H=0` looked scrambled). The
correct recipe is `fftshift(fftn(ifftshift(·)))` with symmetric padding. See
`docs/algorithms/delta_pdf.md` and regression test
`test_delta_pdf_centring_positive_peak`. After the fix the ΔPDF shows coherent
single-sign correlation peaks on the expected b/c lattice.

Preferred input on this machine is the Mantid-background-subtracted file:

```text
data/raw/*_cc_sub_bkg.nxs
```

Use the bundled conda environment:

```bash
/Users/tt9/miniforge3/envs/rmc-discord/bin/python3
```

There is no `rmc-discord` env on this machine.

## Recommended QA Command

Interactive all-H cleanup viewer:

```bash
env PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl USE_BACKGROUND=0 \
PUNCH_PRESET=cc_on MODE=both MIN_I=0.8 MIN_PROM=0.8 \
INTEGER_FIT_POSITION=1 INTEGER_FIT_SHAPE=1 INTEGER_H_GUARD=0.12 \
SEARCH_EXCLUDE_H=-0.6667,-0.3333,0.3333,0.6667 SEARCH_EXCLUDE_H_WIDTH=0.08 \
BACKFILL_METHOD=q_shell H_VALUE=0.3333 \
/Users/tt9/miniforge3/envs/rmc-discord/bin/python3 examples/explore_slice.py
```

This opens four panels: `data`, `Removed ring`, `Punched`, `Backfilled`.

Latest visual QA preference:

- `MODE=both` looked better than integer-only because search catches residual
  off-integer/satellite peaks.
- Protect fractional-H diffuse planes from search with `SEARCH_EXCLUDE_H`.
- Keep integer punches confined near integer-H planes with `INTEGER_H_GUARD=0.12`.
- The last guarded `MODE=both` viewer punched about `6.02%` of valid voxels
  versus `7.02%` for unrestricted `MODE=both` and `2.27%` for integer-only.

## Batch Pipeline

Run from the repo root:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl RING_PRESET=cc_on \
/Users/tt9/miniforge3/envs/rmc-discord/bin/python3 examples/remove_rings_3d.py

PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl PUNCH_PRESET=cc_on MODE=both \
MIN_I=0.8 MIN_PROM=0.8 INTEGER_FIT_POSITION=1 INTEGER_FIT_SHAPE=1 \
INTEGER_H_GUARD=0.12 \
SEARCH_EXCLUDE_H=-0.6667,-0.3333,0.3333,0.6667 SEARCH_EXCLUDE_H_WIDTH=0.08 \
PREVIEW=0 \
/Users/tt9/miniforge3/envs/rmc-discord/bin/python3 examples/punch_bragg_3d.py

PYTHONPATH=src METHOD=q_shell \
/Users/tt9/miniforge3/envs/rmc-discord/bin/python3 examples/backfill_bragg_3d.py
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
- `SEARCH_EXCLUDE_H` protects known fractional-H diffuse planes from the
  hkl-agnostic search stage.
- `INTEGER_H_GUARD` prevents integer-H Bragg punch ellipsoids from extending
  into fractional-H diffuse planes.

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
PYTHONPATH=src /Users/tt9/miniforge3/envs/rmc-discord/bin/python3 \
  -m pytest -o addopts=''
```

Expected current result: `74 passed` (includes the ΔPDF centring guard
`test_delta_pdf_centring_positive_peak`).

`ruff` is not installed in the `sci-general` env. Use `git diff --check` plus
`py_compile` locally, or install/use a dev env for full linting.

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
