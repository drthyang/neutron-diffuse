# neutron-diffuse

`neutron-diffuse` is a Python toolkit for cleaning 3D reciprocal-space neutron
diffuse scattering volumes and preparing them for 3D-DeltaPDF analysis.

The current workflow is built around symmetrised Mantid HKL volumes. It removes
powder-ring backgrounds, punches sharp Bragg and satellite peaks, fills the
punched holes with a diffuse-background estimate, Fourier-transforms the cleaned
volume into a real-space 3D-DeltaPDF, and ends with a back-FFT consistency check.

```text
Mantid / symmetrised HKL volume
        |
        v
  1. powder-ring subtraction        examples/remove_rings_3d.py
  2. Bragg/satellite punch          examples/punch_bragg_3d.py
  3. Bragg-hole backfill            examples/backfill_bragg_3d.py
  4. radial-background flatten      examples/flatten_background_3d.py
  5. 3D-DeltaPDF transform          examples/delta_pdf.py
  6. back-FFT consistency check     examples/delta_pdf_consistency.py
  7. cleanup / DeltaPDF viewers     examples/explore_slice.py, examples/explore_delta_pdf_ortho.py
```

For a **3D-PDF** (total scattering with the Bragg peaks *kept* — a Patterson-like
map) instead of the ΔPDF, use `examples/run_pipeline_pdf.py`, which skips the
punch and backfill stages. To characterise magnetic diffuse that sits *at* the
Bragg satellites, use `examples/investigate_bragg_diffuse.py`.

## Web UI

`neutron-diffuse` ships a single browser console (React) that unifies the
cleanup and 3D-ΔPDF viewers and drives the whole pipeline from a raw `.nxs` with
live progress. It has two interchangeable run modes that share the same UI:

- **Native** — `pip install -e ".[web]"` then `ndiff-web` (serves
  http://127.0.0.1:8000); full-resolution local work.
- **In-browser** — the hosted GitHub Pages build runs the real `ndiff` pipeline
  client-side via Pyodide on your own data, no install (modest volumes).

A sidebar console with five views — a pipeline runner (the default landing
view), reciprocal-space cleanup, 3D-ΔPDF orthoslices, multi-temperature
comparison, and the consistency check — replaces the standalone
`examples/explore_*.py` viewers (which remain as a fallback). See
[QUICKSTART.md](QUICKSTART.md) to get running and [docs/web.md](docs/web.md) for
the reference, architecture, and development workflow.

## Install

Requires Python 3.10 or newer.

```bash
git clone https://github.com/drthyang/neutron-diffuse
cd neutron-diffuse
pip install -e ".[dev]"
```

For local scripts run from the repository root, set:

```bash
export PYTHONPATH=src
export MPLCONFIGDIR=/tmp/mpl
```

`MPLCONFIGDIR` keeps Matplotlib cache files out of the repository.

## Development

Before pushing, run the same checks as CI (tests, lint, type check):

```bash
bash scripts/check.sh
```

It runs `pytest`, `ruff check src/ tests/`, and `mypy src/ndiff`. Set
`PY=/path/to/python` to choose the interpreter. To run it automatically on every
push, install it as a git hook in your clone:

```bash
ln -s ../../scripts/check.sh .git/hooks/pre-push
```

## Input Data

The preferred input is a Mantid-background-subtracted NeXus file in
`data/raw/`:

```text
*_cc_sub_bkg.nxs
```

Here `cc` means CORELLI correlation chopper, and `sub_bkg` means the empty-can
background has already been subtracted. You can also load ndiff HDF5 files
written by the package itself.

## Quick Start

To run the app, see [QUICKSTART.md](QUICKSTART.md). For the current TbTi3Bi4
CLI command set (concise commands for the 22 K, 45 K, and 100 K workflows and
viewers), see [docs/commands.md](docs/commands.md).

Run the complete pipeline:

```bash
DATA_FILE=data/raw/your_volume_cc_sub_bkg.nxs \
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
python3 examples/run_pipeline.py
```

`examples/run_pipeline.py` runs all compute stages, skips stages whose outputs
already exist, writes the 3D-DeltaPDF, runs the consistency check, then opens the
cleanup and DeltaPDF viewers. The stages are: (1) ring removal, (2) Bragg punch,
(3) Bragg backfill, (4) **radial-background flatten** — the explicit
background-removal step (default on; `FLATTEN=0` to skip), (5) 3D-DeltaPDF FFT,
(6) back-FFT consistency check. The background is removed at step 4, not by a
hidden blur inside the FFT: the transform's own Gaussian `SUBTRACT_BG` is off by
default because it is the *alternative* remover (see step 5 below).

Useful overrides:

| Variable | Effect |
| --- | --- |
| `DATA_FILE=/path/to/file.nxs` | Use a specific input file. |
| `NO_VIEWER=1` | Stop after writing outputs and the consistency check; do not open GUI viewers. |
| `FORCE=1` | Recompute every stage. |
| `FORCE_FROM=rings|punch|backfill|flatten|pdf|check` | Recompute from one stage onward. |
| `FLATTEN=0` | Skip the radial-background flatten (step 4). |
| `CONSISTENCY=0` | Skip the final back-FFT consistency check. |

## Run Stages Manually

Run these from the repository root when you want to inspect or tune individual
stages.

### 1. Remove Powder Rings

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl RING_PRESET=cc_on \
python3 examples/remove_rings_3d.py
```

Output:

```text
data/processed/*_ringremoved.h5
```

### 2. Punch Bragg And Satellite Peaks

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl PUNCH_PRESET=cc_on MODE=both \
MIN_I=0.8 MIN_PROM=0.8 INTEGER_FIT_POSITION=1 INTEGER_FIT_SHAPE=1 \
INTEGER_H_GUARD=0.12 \
SEARCH_EXCLUDE_H=-0.6667,-0.3333,0.3333,0.6667 SEARCH_EXCLUDE_H_WIDTH=0.08 \
PREVIEW=0 \
python3 examples/punch_bragg_3d.py
```

Output:

```text
data/processed/*_braggpunched.h5
```

### 3. Backfill Bragg Holes

```bash
PYTHONPATH=src METHOD=q_shell \
python3 examples/backfill_bragg_3d.py
```

Output:

```text
data/processed/*_braggpunched_backfilled.h5
```

### 4. Flatten The Radial Background (Background Removal)

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
python3 examples/flatten_background_3d.py
```

Output:

```text
data/processed/*_backfilled_flattened.h5
```

Sweeps spherical `|Q|` shells and subtracts a smooth, continuous per-shell
background floor (default estimator `floor`/p25), so the isotropic radial
pedestal flattens to ≈0 while the anisotropic diffuse and Bragg residuals are
preserved. This is the explicit background-removal step. Its robustness is
validated across 22/45/100 K by `examples/validate_flatten.py` (background is
isotropic, strong-feature contrast 100% retained, no over-subtraction).

### 5. Compute The 3D-DeltaPDF

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
CROP_H=4 CROP_K=8 CROP_L=15 APODIZE=gaussian \
python3 examples/delta_pdf.py
```

Outputs:

```text
examples/_delta_pdf.h5
examples/_delta_pdf_hk0.png
examples/_delta_pdf_h0l.png
examples/_delta_pdf_0kl.png
```

The standard 3D workflow crops reciprocal space to `|H| <= 4`, `|K| <= 8`, and
`|L| <= 15` before the transform. The background has already been removed at
step 4, so the transform's own smooth-background subtraction is **off**.
`SUBTRACT_BG=0,sigma,sigma` is the *legacy alternative*: it subtracts a smooth
per-H-plane Gaussian background inside the FFT. Use the flatten **or**
`SUBTRACT_BG`, never both — running both removes the background twice, and the
per-H-plane blur (σ_H=0) destroys the on-axis H-direction signal that the
flatten preserves.

### 6. Check DeltaPDF Consistency

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
DATA_FILE=data/processed/your_volume_ringremoved_braggpunched_backfilled_flattened.h5 \
python3 examples/delta_pdf_consistency.py
```

The check computes the ΔPDF with the standard settings, inverse-transforms it
back to reciprocal space, and compares **data | back-FFT | residual**. A faithful
transform gives Pearson `r` close to 1 with a small normalised RMS residual. The
web UI exposes the same check interactively, including adjustable `|Q|` and
real-space `r` bands.

## Visual QA

Inspect cleanup across H:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl USE_BACKGROUND=0 \
PUNCH_PRESET=cc_on MODE=both MIN_I=0.8 MIN_PROM=0.8 \
INTEGER_FIT_POSITION=1 INTEGER_FIT_SHAPE=1 INTEGER_H_GUARD=0.12 \
SEARCH_EXCLUDE_H=-0.6667,-0.3333,0.3333,0.6667 SEARCH_EXCLUDE_H_WIDTH=0.08 \
BACKFILL_METHOD=q_shell H_VALUE=0.3333 \
python3 examples/explore_slice.py
```

The viewer shows raw data, removed ring intensity, punched data, and backfilled
data, with an H slider for scrubbing through the volume.

Inspect the real-space 3D-DeltaPDF:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl RMAX=50 \
python3 examples/explore_delta_pdf_ortho.py
```

This opens three linked orthogonal real-space cuts with movable cut sliders,
contrast control, and unit-cell gridlines.

## Python API

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

## Documentation

Start with [docs/README.md](docs/README.md).

Key pages:

| Page | Purpose |
| --- | --- |
| [docs/algorithms/powder_rings.md](docs/algorithms/powder_rings.md) | Powder-ring model and subtraction strategy. |
| [docs/algorithms/bragg_cleanup.md](docs/algorithms/bragg_cleanup.md) | Bragg/satellite detection, punching, and backfill. |
| [docs/algorithms/delta_pdf.md](docs/algorithms/delta_pdf.md) | 3D-DeltaPDF transform, centring, background subtraction, and consistency checks. |
| [QUICKSTART.md](QUICKSTART.md) | Get the app running (native or in-browser) in a few commands. |
| [docs/commands.md](docs/commands.md) | Concise CLI command recipes for 22 K, 45 K, and 100 K data. |
| [docs/web.md](docs/web.md) | Browser console: run modes, viewers, architecture, dev workflow. |
| [docs/interactive.md](docs/interactive.md) | Matplotlib viewer usage and visualization API. |
| [CHANGELOG.md](CHANGELOG.md) | Release notes and version history. |

## Package Layout

```text
src/ndiff/
├── core.py              HKLVolume: 3D array, HKL axes, mask, sigma, UB matrix
├── io/                  Mantid NeXus, ndiff HDF5, and ASCII HKL I/O
├── preprocessing/       powder-ring models, background handling, sampling
├── analysis/            Bragg punch/fill and 3D-DeltaPDF
├── inpainting/          symmetry, TV, RBF, and biharmonic fallbacks
└── visualization/       slices, profiles, overview plots, interactive viewers
```

## Tests And CI

Run locally in a Python 3.10+ environment with dev dependencies:

```bash
PYTHONPATH=src python3 -m pytest -o addopts=''
python3 -m ruff check src/ tests/
python3 -m mypy src/ndiff --ignore-missing-imports
```

GitHub Actions runs the same checks on Python 3.10, 3.11, and 3.12.

## Status

Version 0.2.0. The recommended workflow is operational and now ends with the
back-FFT consistency check: powder-ring removal, Bragg cleanup, Bragg-hole
backfill, radial flatten, 3D-DeltaPDF transform, consistency QA, and interactive
viewers. The package remains pre-1.0/alpha while the public API and file formats
continue to evolve.
