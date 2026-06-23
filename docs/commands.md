# Command Recipes

Concise CLI commands for the current TbTi3Bi4 workflow — for scripting and batch
runs. For the interactive browser console (the quickest way to run and inspect
the pipeline) see [../QUICKSTART.md](../QUICKSTART.md) and [web.md](web.md). Run
these from the repository root.

## Environment

```bash
export PY=/opt/homebrew/Caskroom/miniforge/base/envs/sci-general/bin/python
export PYTHONPATH=src
export MPLCONFIGDIR=/private/tmp/nebula3d-mpl
```

If you use a different Python 3.10+ environment with the package dependencies,
replace `$PY` with that interpreter.

## Full Workflow

`examples/run_pipeline.py` runs:

```text
ring removal -> Bragg punch -> Bragg backfill -> radial-background flatten -> DeltaPDF -> consistency check
```

Step 4, the **radial-background flatten, is the explicit background-removal step
and is on by default** (disable with `FLATTEN=0`). The DeltaPDF's own Gaussian
`SUBTRACT_BG` blur is the alternative remover and defaults off — never run both.

It skips stages whose output files already exist. Add `FORCE=1` to recompute
everything, or `FORCE_FROM=punch` to recompute from one stage onward. Valid
`FORCE_FROM` stages are `rings`, `punch`, `backfill`, `flatten`, `pdf`, and
`check`. Set `CONSISTENCY=0` only when you intentionally want to skip the final
round-trip QA.

Ring removal processes H-slices/KL planes by default. To process the same volume
as K-slices/HL planes or L-slices/HK planes, add `SLICE_AXIS=K` or
`SLICE_AXIS=L` to the pipeline command. The default is `SLICE_AXIS=H`. The
ring-removed filename does not encode the axis, so when re-running the same
dataset with a different `SLICE_AXIS`, add `FORCE_FROM=rings` to recompute
instead of reusing the cached output.

```bash
# 22 K
NO_VIEWER=1 \
DATA_FILE="data/raw/TbTi3Bi4_22K_mmm_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg.nxs" \
$PY examples/run_pipeline.py

# 45 K
NO_VIEWER=1 \
DATA_FILE="data/raw/TbTi3Bi4_45K_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg.nxs" \
$PY examples/run_pipeline.py

# 100 K
NO_VIEWER=1 \
DATA_FILE="data/raw/TbTi3Bi4_100K_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg.nxs" \
$PY examples/run_pipeline.py
```

## Per-Temperature DeltaPDF Files

Use this when the reciprocal-space stages already exist and you want persistent
DeltaPDF files in `data/processed`. The input is the **flattened** (step-4,
background-removed) volume, so the transform's own `SUBTRACT_BG` is left off. (To
use the legacy in-FFT Gaussian blur instead, point `PROC_FILE` at the
`*_backfilled.h5` and add `SUBTRACT_BG="0,1.5,1.5"` — but not both.)

The `CROP_H=4 CROP_K=8 CROP_L=15` below is an **optional band-limit**. The
default (omit them) is the full `|Q|` range, which gives a finer real-space grid
that matches the web 3D-ΔPDF page and the consistency view — at the cost of a
larger transform (~4× the voxels) that includes the noisier outer `|Q|` shells.

```bash
# 22 K
PROC_FILE="data/processed/TbTi3Bi4_22K_mmm_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched_backfilled_flattened.h5" \
OUT_FILE="data/processed/TbTi3Bi4_22K_mmm_delta_pdf.h5" \
CROP_H=4 CROP_K=8 CROP_L=15 APODIZE=gaussian GAUSSIAN_SIGMA=0.4 \
$PY examples/delta_pdf.py

# 45 K
PROC_FILE="data/processed/TbTi3Bi4_45K_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched_backfilled_flattened.h5" \
OUT_FILE="data/processed/TbTi3Bi4_45K_delta_pdf.h5" \
CROP_H=4 CROP_K=8 CROP_L=15 APODIZE=gaussian GAUSSIAN_SIGMA=0.4 \
$PY examples/delta_pdf.py

# 100 K
PROC_FILE="data/processed/TbTi3Bi4_100K_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched_backfilled_flattened.h5" \
OUT_FILE="data/processed/TbTi3Bi4_100K_delta_pdf.h5" \
CROP_H=4 CROP_K=8 CROP_L=15 APODIZE=gaussian GAUSSIAN_SIGMA=0.4 \
$PY examples/delta_pdf.py
```

## View Processed Reciprocal-Space Data

These commands open the four-panel cleanup QA viewer:

```text
raw -> ring removed -> Bragg punched -> backfilled
```

```bash
# 22 K
DATA_FILE="data/raw/TbTi3Bi4_22K_mmm_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg.nxs" \
RING_FILE="data/processed/TbTi3Bi4_22K_mmm_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved.h5" \
PUNCH_FILE="data/processed/TbTi3Bi4_22K_mmm_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched.h5" \
BACKFILL_FILE="data/processed/TbTi3Bi4_22K_mmm_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched_backfilled.h5" \
$PY examples/explore_slice.py

# 45 K
DATA_FILE="data/raw/TbTi3Bi4_45K_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg.nxs" \
RING_FILE="data/processed/TbTi3Bi4_45K_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved.h5" \
PUNCH_FILE="data/processed/TbTi3Bi4_45K_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched.h5" \
BACKFILL_FILE="data/processed/TbTi3Bi4_45K_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched_backfilled.h5" \
$PY examples/explore_slice.py

# 100 K
DATA_FILE="data/raw/TbTi3Bi4_100K_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg.nxs" \
RING_FILE="data/processed/TbTi3Bi4_100K_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved.h5" \
PUNCH_FILE="data/processed/TbTi3Bi4_100K_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched.h5" \
BACKFILL_FILE="data/processed/TbTi3Bi4_100K_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched_backfilled.h5" \
$PY examples/explore_slice.py
```

## View DeltaPDF

```bash
# 22 K
PDF_FILE="data/processed/TbTi3Bi4_22K_mmm_delta_pdf.h5" \
$PY examples/explore_delta_pdf_ortho.py

# 45 K
PDF_FILE="data/processed/TbTi3Bi4_45K_delta_pdf.h5" \
$PY examples/explore_delta_pdf_ortho.py

# 100 K
PDF_FILE="data/processed/TbTi3Bi4_100K_delta_pdf.h5" \
$PY examples/explore_delta_pdf_ortho.py
```

## Check DeltaPDF Consistency

Use this as the endpoint of a single-temperature workflow. It starts from the
flattened diffuse volume (or a backfilled volume if flattening was disabled),
computes the ΔPDF, inverse-transforms it back to reciprocal space, and writes a
`data | back-FFT | residual` figure plus metrics on stdout.

```bash
# auto-detects a 22 K flattened/backfilled input when DATA_FILE is omitted
$PY examples/delta_pdf_consistency.py

# explicit input/output
DATA_FILE="data/processed/TbTi3Bi4_22K_mmm_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched_backfilled_flattened.h5" \
OUT_PNG="data/processed/TbTi3Bi4_22K_mmm_delta_pdf_consistency.png" \
$PY examples/delta_pdf_consistency.py
```

A faithful transform should give Pearson `r` close to 1 and a small normalised
RMS residual over the reliably recovered reciprocal-space region. For interactive
band-limited checks, use the web UI's **Consistency check** view.

## Compare DeltaPDF Across Temperatures

With the pipeline outputs in `data/processed`, the viewer auto-detects each
temperature's ΔPDF — no paths needed:

```bash
$PY examples/explore_delta_pdf_multi.py
```

It shows the three temperatures × three orthoslice planes on a per-plane colour
scale (temperatures comparable within each column). Drag the cut sliders to move
the planes; the `contrast ×` slider rescales. Pass
`PDF_22K=… PDF_45K=… PDF_100K=…` to override the auto-detected files.

## 3D-PDF (Bragg Kept)

The DeltaPDF workflow removes the Bragg peaks (punch -> backfill) to isolate the
diffuse. The **3D-PDF** instead keeps the Bragg peaks and Fourier-transforms the
total scattering, giving a Patterson-like map of the average-structure
correlations. `examples/run_pipeline_pdf.py` runs `ring removal -> 3D-PDF` with
**no punch and no backfill**; the smooth-background subtraction is off. The
ring-removed files are shared with the DeltaPDF workflow, so stage 1 is skipped
if it already ran.

```bash
# 22 K (45 K / 100 K: same with the matching raw file)
NO_VIEWER=1 \
DATA_FILE="data/raw/TbTi3Bi4_22K_mmm_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg.nxs" \
$PY examples/run_pipeline_pdf.py
```

Output: `data/processed/*_ringremoved_3dpdf.h5`. Set `RING_REMOVAL=0` to transform
the raw data directly. View it with the orthoslice viewer (the window title shows
the kind, 3D-PDF vs 3D-DeltaPDF, and the temperature):

```bash
PDF_FILE="data/processed/TbTi3Bi4_22K_mmm_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_3dpdf.h5" \
$PY examples/explore_delta_pdf_ortho.py
```

## Bragg / Diffuse Separation Diagnostic

When magnetic diffuse sits *at* the q=1/3 magnetic satellites, characterise the
peak shape before separating it. `examples/investigate_bragg_diffuse.py` calibrates
the resolution on nuclear Bragg, then decomposes each satellite into a sharp
(resolution-limited) core plus a broad diffuse component on H/K/L line cuts.

```bash
# all temperatures + the temperature-series overlay
T_SERIES=1 $PY examples/investigate_bragg_diffuse.py
```

Writes per-temperature cut/fit figures, a summary plot, and a CSV with resolution
`σ(|Q|)`, points-across-FWHM (which axes resolve the core), correlation length ξ,
and the diffuse fraction.
