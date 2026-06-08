# Quick Start

Concise commands for the current TbTi3Bi4 workflow. Run them from the repository
root.

## Environment

```bash
export PY=/opt/homebrew/Caskroom/miniforge/base/envs/sci-general/bin/python
export PYTHONPATH=src
export MPLCONFIGDIR=/private/tmp/ndiff-mpl
```

If you use a different Python 3.10+ environment with the package dependencies,
replace `$PY` with that interpreter.

## Full Workflow

`examples/run_pipeline.py` runs:

```text
ring removal -> Bragg punch -> Bragg backfill -> DeltaPDF
```

It skips stages whose output files already exist. Add `FORCE=1` to recompute
everything, or `FORCE_FROM=punch` to recompute from one stage onward. Valid
`FORCE_FROM` stages are `rings`, `punch`, `backfill`, and `pdf`.

Ring removal processes H-slices/KL planes by default. To process the same volume
as K-slices/HL planes or L-slices/HK planes, add `SLICE_AXIS=K` or
`SLICE_AXIS=L` to the pipeline command. The default is `SLICE_AXIS=H`.

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
DeltaPDF files in `data/processed`.

```bash
# 22 K
PROC_FILE="data/processed/TbTi3Bi4_22K_mmm_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched_backfilled.h5" \
OUT_FILE="data/processed/TbTi3Bi4_22K_mmm_delta_pdf.h5" \
SUBTRACT_BG="0,1.5,1.5" CROP_H=4 CROP_K=8 CROP_L=15 APODIZE=gaussian GAUSSIAN_SIGMA=0.4 \
$PY examples/delta_pdf.py

# 45 K
PROC_FILE="data/processed/TbTi3Bi4_45K_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched_backfilled.h5" \
OUT_FILE="data/processed/TbTi3Bi4_45K_delta_pdf.h5" \
SUBTRACT_BG="0,1.5,1.5" CROP_H=4 CROP_K=8 CROP_L=15 APODIZE=gaussian GAUSSIAN_SIGMA=0.4 \
$PY examples/delta_pdf.py

# 100 K
PROC_FILE="data/processed/TbTi3Bi4_100K_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched_backfilled.h5" \
OUT_FILE="data/processed/TbTi3Bi4_100K_delta_pdf.h5" \
SUBTRACT_BG="0,1.5,1.5" CROP_H=4 CROP_K=8 CROP_L=15 APODIZE=gaussian GAUSSIAN_SIGMA=0.4 \
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

## Compare DeltaPDF Across Temperatures

```bash
PDF_22K="data/processed/TbTi3Bi4_22K_mmm_delta_pdf.h5" \
PDF_45K="data/processed/TbTi3Bi4_45K_delta_pdf.h5" \
PDF_100K="data/processed/TbTi3Bi4_100K_delta_pdf.h5" \
$PY examples/explore_delta_pdf_multi.py
```

Set `SHARED_SCALE=1` to lock all panels to the 22 K color scale.

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
