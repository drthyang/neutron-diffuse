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

Runs all four processing stages (ring removal → Bragg punch → backfill →
3D-DeltaPDF) and then opens the cleanup QA viewer and the DeltaPDF orthoslice
viewer. Already-computed stages are skipped automatically.

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
  $PY examples/run_pipeline.py
```

Key environment overrides:

| Variable | Effect |
|---|---|
| `DATA_FILE=/path/to/file.nxs` | Override auto-detected input (default: 22 K `cc_sub_bkg`) |
| `NO_VIEWER=1` | Stop after writing `_delta_pdf.h5`; skip GUI stages |
| `FORCE=1` | Recompute every stage even if output exists |
| `FORCE_FROM=rings\|punch\|backfill\|pdf` | Recompute from that stage onward |

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

Writes `examples/_delta_pdf_{T}.h5` using the already-backfilled volumes.
The `OUT_FILE` env var overrides the default `examples/_delta_pdf.h5` output.

```bash
# 22 K
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
  PROC_FILE="data/processed/TbTi3Bi4_22K_mmm_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched_backfilled.h5" \
  OUT_FILE="examples/_delta_pdf_22K.h5" \
  SUBTRACT_BG="0,1.5,1.5" CROP_K=8 CROP_L=15 APODIZE=gaussian GAUSSIAN_SIGMA=0.4 \
  $PY examples/delta_pdf.py

# 45 K
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
  PROC_FILE="data/processed/TbTi3Bi4_45K_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched_backfilled.h5" \
  OUT_FILE="examples/_delta_pdf_45K.h5" \
  SUBTRACT_BG="0,1.5,1.5" CROP_K=8 CROP_L=15 APODIZE=gaussian GAUSSIAN_SIGMA=0.4 \
  $PY examples/delta_pdf.py

# 100 K
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
  PROC_FILE="data/processed/TbTi3Bi4_100K_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched_backfilled.h5" \
  OUT_FILE="examples/_delta_pdf_100K.h5" \
  SUBTRACT_BG="0,1.5,1.5" CROP_K=8 CROP_L=15 APODIZE=gaussian GAUSSIAN_SIGMA=0.4 \
  $PY examples/delta_pdf.py
```

---

## Interactive Viewers

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

3 x 3 grid: rows = temperatures, columns = orthogonal cuts. All cut
sliders are shared so the same real-space slice is shown for every temperature.

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
  $PY examples/explore_delta_pdf_multi.py
```

| Variable | Default | Effect |
|---|---|---|
| `PDF_22K` | `examples/_delta_pdf_22K.h5` | 22 K ΔPDF file |
| `PDF_45K` | `examples/_delta_pdf_45K.h5` | 45 K ΔPDF file |
| `PDF_100K` | `examples/_delta_pdf_100K.h5` | 100 K ΔPDF file |
| `RMAX` | `50` | Display half-window in Å |
| `SHARED_SCALE` | `0` | `1` = lock all panels to the 22 K colour scale |

### Processed-Data QA Viewer

4-panel K-L slice viewer (raw → ring-removed → punched → backfilled) with
H-value and vmin/vmax sliders.

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
  DATA_FILE="data/raw/TbTi3Bi4_22K_mmm_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg.nxs" \
  RING_FILE="data/processed/TbTi3Bi4_22K_mmm_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved.h5" \
  PUNCH_FILE="data/processed/TbTi3Bi4_22K_mmm_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched.h5" \
  BACKFILL_FILE="data/processed/TbTi3Bi4_22K_mmm_(0,k,l)_[h,0,0]_[-12.0,12.0]_[-30.0,30.0]_[-5.0,5.0]_401x401x301_mmm_cc_sub_bkg_ringremoved_braggpunched_backfilled.h5" \
  $PY examples/explore_slice.py
```
