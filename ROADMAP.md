# Roadmap

## Pipeline

```text
[ Mantid / symmetrised HKL volume ]
        |
        v
  1. Powder-ring subtraction        implemented, real-data QA active
  2. Bragg/satellite punch          implemented, real-data QA active
  3. Bragg-hole backfill            implemented, real-data QA active
  4. 3D-DeltaPDF Fourier transform  implemented, centring bug fixed
```

The current development goal is to enter the final 3D-DeltaPDF stage with a
well-organized cleanup stack and clear defaults.

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

## Phase 4 — 3D-DeltaPDF  Implemented

Implemented API and drivers:

- `ndiff.analysis.compute_delta_pdf` (algorithm: `docs/algorithms/delta_pdf.md`)
- apodization: `hann`, `gaussian`, `none`; optional mean subtraction; symmetric
  zero padding; real-space axes from the UB matrix
- `examples/delta_pdf.py` — full 3D transform; slice/line-cut/radial PNGs and a
  `_delta_pdf.h5` cache
- `examples/delta_pdf_plane.py` — single reciprocal H-plane 2D-ΔPDF
- `examples/explore_delta_pdf.py` — interactive y_K–z_L viewer with x_H slider

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

## Phase 5 — Release Hygiene  In Progress

Before treating the pipeline as a stable release candidate:

- Keep README, ROADMAP, HANDOFF, and algorithm docs aligned with current
  defaults.
- Keep generated data and preview images out of commits unless they are explicit
  reference artifacts.
- Add a lightweight CLI or config-file runner if env-var scripts become hard to
  reproduce.
- Add CI coverage for the Bragg guard/exclusion behavior.
- Revisit `ruff`/`mypy` availability in the development environment.
