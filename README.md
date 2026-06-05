# neutron-diffuse

**3D diffuse neutron scattering cleanup and 3D-DeltaPDF preparation.**

`neutron-diffuse` works on symmetrised 3D HKL volumes from Mantid or equivalent
pipelines. The current real-data workflow removes powder rings, punches Bragg and
satellite peaks, backfills the holes, and prepares the cleaned diffuse volume for
3D-DeltaPDF Fourier transformation.

```text
[ Mantid / symmetrised HKL volume ]
        |
        v
  (1) Powder-ring subtraction          examples/remove_rings_3d.py
  (2) Bragg/satellite punch            examples/punch_bragg_3d.py
  (3) Bragg-hole backfill              examples/backfill_bragg_3d.py
  (4) 3D-DeltaPDF Fourier transform    ndiff.analysis.compute_delta_pdf
```

## Current Real-Data Workflow

Run from the repository root. The preferred input on this machine is the
Mantid-background-subtracted `*_cc_sub_bkg.nxs` file in `data/raw/`.

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl RING_PRESET=cc_on \
/opt/homebrew/Caskroom/miniforge/base/envs/sci-general/bin/python3 examples/remove_rings_3d.py

PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl PUNCH_PRESET=cc_on MODE=both \
MIN_I=0.8 MIN_PROM=0.8 INTEGER_FIT_POSITION=1 INTEGER_FIT_SHAPE=1 \
INTEGER_H_GUARD=0.12 \
SEARCH_EXCLUDE_H=-0.6667,-0.3333,0.3333,0.6667 SEARCH_EXCLUDE_H_WIDTH=0.08 \
PREVIEW=0 \
/opt/homebrew/Caskroom/miniforge/base/envs/sci-general/bin/python3 examples/punch_bragg_3d.py

PYTHONPATH=src METHOD=q_shell \
/opt/homebrew/Caskroom/miniforge/base/envs/sci-general/bin/python3 examples/backfill_bragg_3d.py
```

The three scripts write:

- `data/processed/*_ringremoved.h5`
- `data/processed/*_braggpunched.h5`
- `data/processed/*_braggpunched_backfilled.h5`

Use the interactive all-H viewer for visual QA:

```bash
env PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl USE_BACKGROUND=0 \
PUNCH_PRESET=cc_on MODE=both MIN_I=0.8 MIN_PROM=0.8 \
INTEGER_FIT_POSITION=1 INTEGER_FIT_SHAPE=1 INTEGER_H_GUARD=0.12 \
SEARCH_EXCLUDE_H=-0.6667,-0.3333,0.3333,0.6667 SEARCH_EXCLUDE_H_WIDTH=0.08 \
BACKFILL_METHOD=q_shell H_VALUE=0.3333 \
/opt/homebrew/Caskroom/miniforge/base/envs/sci-general/bin/python3 examples/explore_slice.py
```

The viewer shows `data`, `Removed ring`, `Punched`, and `Backfilled`, with an H
slider for scrubbing the full volume.

## Algorithms

### Powder Rings

The current production path uses `PatchedRadialRingModel`, a non-parametric
radial-background model fit independently on each H plane. It estimates the
azimuthally smooth powder-ring contribution and subtracts it, preserving
structured diffuse residuals by construction.

Important defaults/knobs:

- `profile_method="median"` for robust per-bin ring level.
- `texture_q_smooth=0.0` in the class default to preserve azimuthally varying
  ring width.
- `RING_PRESET=cc_on` in the 3D driver for the cleaner `*_cc_sub_bkg.nxs`
  workflow.
- Ring removal is subtractive only. The removed mask-and-replace experiment was
  rejected because radial excess can be real diffuse scattering.

See [docs/algorithms/powder_rings.md](docs/algorithms/powder_rings.md).

### Bragg Punch And Backfill

`BraggRemover` supports:

- `mode="integer"`: enumerate integer HKL nodes and decide per node whether a
  nearby peak exists.
- `mode="search"` / `mode="auto"`: hkl-agnostic per-`|Q|` high-tail outlier
  search for off-integer satellites.
- `mode="both"`: integer first, then search on the residual.

The current visual QA preference is guarded `MODE=both`: integer-node punches are
confined near integer-H planes with `INTEGER_H_GUARD`, while `SEARCH_EXCLUDE_H`
protects known fractional-H diffuse planes from the hkl-agnostic search stage.

`backfill_bragg(method="q_shell")` fills ordinary Bragg holes from the robust
background level at the same `|Q|`. The direct beam is handled separately with a
just-outside-`|Q|` fill.

See [docs/algorithms/bragg_cleanup.md](docs/algorithms/bragg_cleanup.md).

## Python API Sketch

```python
import ndiff
from ndiff.analysis import BraggRemover, backfill_bragg, compute_delta_pdf

vol = ndiff.load("data/processed/sample_ringremoved.h5")

remover = BraggRemover(
    mode="both",
    punch_radii=(0.09, 0.12, 0.45),
    min_intensity=0.8,
    min_prominence=0.8,
    integer_optimize_position=True,
    integer_optimize_shape=True,
    integer_h_guard_hkl=0.12,
    search_n_mad=4.0,
    search_min_intensity=0.8,
    search_min_prominence=0.8,
    search_exclude_h_centers=(-2 / 3, -1 / 3, 1 / 3, 2 / 3),
    search_exclude_h_half_width=0.08,
    incident_beam_ellipsoid_radii_hkl=(0.15, 0.50, 1.00),
)

punched = remover.apply(vol)
filled = backfill_bragg(punched, method="q_shell")
dpdf = compute_delta_pdf(filled, apodization="hann")
```

## Installation

```bash
git clone https://github.com/drthyang/neutron-diffuse
cd neutron-diffuse
pip install -e ".[dev]"
```

## Module Overview

```text
src/ndiff/
├── core.py                     HKLVolume: 3D array + UB matrix + mask + sigma
├── io/                         Mantid NeXus, HDF5, and legacy HKL I/O
├── preprocessing/              empty subtraction, powder-ring models, ring fill
├── analysis/                   Bragg punch/fill and 3D-DeltaPDF
├── inpainting/                 symmetry, TV, RBF, biharmonic fallbacks
└── visualization/              slices, profiles, overview, interactive viewer
```

## Testing

```bash
PYTHONPATH=src /opt/homebrew/Caskroom/miniforge/base/envs/sci-general/bin/python3 \
  -m pytest -o addopts=''
```

Current expected result: `73 passed`.

## Status

Ring removal and Bragg cleanup are in active real-data QA. The next development
stage is the final 3D-DeltaPDF Fourier transform inspection and tuning. See
[HANDOFF.md](HANDOFF.md) for the current operational state and
[ROADMAP.md](ROADMAP.md) for the phase plan.
