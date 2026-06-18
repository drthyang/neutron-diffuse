# Plotting And Reproducibility Commands

This page records the command set used for the current TbTi3Bi4 processing and
visual QA work. It is intentionally more dataset-specific than the repository
README.

Run commands from the repository root. Set `PYTHONPATH=src` so scripts import the
local checkout, and set `MPLCONFIGDIR=/tmp/mpl` so Matplotlib cache files stay
out of the repo.

The examples below use a local conda environment named `sci-general`:

```bash
PY=/opt/homebrew/Caskroom/miniforge/base/envs/sci-general/bin/python3
```

If your Python 3.10+ environment is already active, replace `$PY` with
`python3`.

---

## Full Pipeline: Raw Volume To DeltaPDF

Runs the full workflow (ring removal → Bragg punch → backfill →
radial-background flatten → 3D-DeltaPDF → consistency check) and then opens the
cleanup QA viewer and the DeltaPDF orthoslice viewer. Already-computed stages are
skipped automatically. Background removal is step 4 (the flatten, on by
default); the transform's own Gaussian `SUBTRACT_BG` blur defaults off.

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
  $PY examples/run_pipeline.py
```

Key environment overrides:

| Variable | Effect |
|---|---|
| `DATA_FILE=/path/to/file.nxs` | Override auto-detected input (default: 22 K `cc_sub_bkg`) |
| `NO_VIEWER=1` | Stop after writing `_delta_pdf.h5` and the consistency figure; skip GUI stages |
| `FORCE=1` | Recompute every stage even if output exists |
| `FORCE_FROM=rings\|punch\|backfill\|flatten\|pdf\|check` | Recompute from that stage onward |
| `FLATTEN=0` | Skip the radial-background flatten (step 4) |
| `CONSISTENCY=0` | Skip the final back-FFT consistency check |

The script auto-detects the 22 K dataset in `data/raw/`. For other temperatures pass
`DATA_FILE` explicitly:

```bash
# 45 K
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl NO_VIEWER=1 \
  DATA_FILE="data/raw/TbTi3Bi4_45K_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg.nxs" \
  $PY examples/run_pipeline.py

# 100 K
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl NO_VIEWER=1 \
  DATA_FILE="data/raw/TbTi3Bi4_100K_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg.nxs" \
  $PY examples/run_pipeline.py
```

---

## Compute Per-Temperature DeltaPDF Files

Writes `examples/_delta_pdf_{T}.h5` using the **flattened** (step-4,
background-removed) volumes, so the transform's own `SUBTRACT_BG` is left off.
The `OUT_FILE` env var overrides the default `examples/_delta_pdf.h5` output.
(To use the legacy in-FFT Gaussian blur instead, point `PROC_FILE` at the
`*_backfilled.h5` and add `SUBTRACT_BG="0,1.5,1.5"` — never both.)

```bash
# 22 K
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
  PROC_FILE="data/processed/TbTi3Bi4_22K_mmm_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched_backfilled_flattened.h5" \
  OUT_FILE="examples/_delta_pdf_22K.h5" \
  CROP_H=4 CROP_K=8 CROP_L=15 APODIZE=gaussian GAUSSIAN_SIGMA=0.4 \
  $PY examples/delta_pdf.py

# 45 K
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
  PROC_FILE="data/processed/TbTi3Bi4_45K_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched_backfilled_flattened.h5" \
  OUT_FILE="examples/_delta_pdf_45K.h5" \
  CROP_H=4 CROP_K=8 CROP_L=15 APODIZE=gaussian GAUSSIAN_SIGMA=0.4 \
  $PY examples/delta_pdf.py

# 100 K
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
  PROC_FILE="data/processed/TbTi3Bi4_100K_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched_backfilled_flattened.h5" \
  OUT_FILE="examples/_delta_pdf_100K.h5" \
  CROP_H=4 CROP_K=8 CROP_L=15 APODIZE=gaussian GAUSSIAN_SIGMA=0.4 \
  $PY examples/delta_pdf.py
```

---

## Interactive Viewers

### DeltaPDF Consistency Check

Use this as the endpoint for a single-temperature ΔPDF workflow. It recomputes
the forward transform from the cleaned diffuse input, inverse-transforms it back
to reciprocal space, and writes a `data | back-FFT | residual` figure while
printing Pearson `r`, normalised RMS, and per-plane correlations.

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
  DATA_FILE="data/processed/TbTi3Bi4_22K_mmm_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched_backfilled_flattened.h5" \
  OUT_PNG="data/processed/TbTi3Bi4_22K_mmm_delta_pdf_consistency.png" \
  $PY examples/delta_pdf_consistency.py
```

The browser **Consistency check** view serves the same round trip interactively
through `/api/consistency/{dataset_id}`, with adjustable `|Q|` and real-space
`r` bands.

### Single-Temperature DeltaPDF Orthoslice Viewer

Three orthogonal real-space cuts (x_H–y_K, x_H–z_L, y_K–z_L) with cut
sliders, contrast control, and unit-cell gridlines.

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
  PDF_FILE=examples/_delta_pdf_22K.h5 \
  $PY examples/explore_delta_pdf_ortho.py
```

| Variable | Default | Effect |
|---|---|---|
| `PDF_FILE` | `examples/_delta_pdf.h5` | ΔPDF file to load |
| `RMAX` | `50` | Display half-window in Å |
| `PERCENTILE` | `98` | Colour-scale percentile (r > 3 Å) |
| `CONTRAST_MIN` / `CONTRAST_MAX` | `0.1` / `20` | Contrast slider range |

### Multi-Temperature DeltaPDF Viewer (22 K / 45 K / 100 K)

3 x 3 grid: rows = temperatures, columns = orthogonal cuts. All cut sliders are
shared so the same real-space slice is shown for every temperature. Each column
uses a per-plane colour scale (temperatures comparable within a column) and the
`contrast ×` slider rescales. With the pipeline outputs in `data/processed` the
files are auto-detected — no paths needed:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
  $PY examples/explore_delta_pdf_multi.py
```

| Variable | Default | Effect |
|---|---|---|
| `PDF_22K` / `PDF_45K` / `PDF_100K` | auto-detect `data/processed/*{T}*_delta_pdf.h5` | override each ΔPDF file |
| `RMAX` | `50` | Display half-window in Å |
| `PERCENTILE` | `98` | Per-plane colour-scale percentile at r>3 Å |
| `CONTRAST_MIN` / `CONTRAST_MAX` | `0.1` / `20` | contrast-× slider range |

### Processed-Data QA Viewer

4-panel slice viewer (raw → ring-removed → punched → backfilled) with an
**H/K/L plane selector**, a cut-position slider, and vmin/vmax sliders. The
selector retargets the slider to the chosen fixed axis and redraws the panels as
0kl, h0l, or hk0 slices. Set the initial orientation with `VIEW_AXIS=H|K|L`
(default `H`) and the initial cut with `{H,K,L}_VALUE`.

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
  DATA_FILE="data/raw/TbTi3Bi4_22K_mmm_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg.nxs" \
  RING_FILE="data/processed/TbTi3Bi4_22K_mmm_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved.h5" \
  PUNCH_FILE="data/processed/TbTi3Bi4_22K_mmm_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched.h5" \
  BACKFILL_FILE="data/processed/TbTi3Bi4_22K_mmm_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched_backfilled.h5" \
  $PY examples/explore_slice.py
```
