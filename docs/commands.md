# Command Recipes

Concise CLI commands for scripting and batch runs. For the interactive browser
console, see [../QUICKSTART.md](../QUICKSTART.md) and [web.md](web.md). Run these
from the repository root and replace the placeholder paths with your own data.

## Environment

```bash
export PY=/path/to/python
export PYTHONPATH=src
export MPLCONFIGDIR=/private/tmp/nebula3d-mpl
```

If you use an activated Python 3.10+ environment with the package dependencies,
replace `$PY` with `python`.

## Full Workflow

`examples/run_pipeline.py` runs:

```text
ring removal -> Bragg punch -> Bragg backfill -> radial-background flatten -> DeltaPDF -> consistency check
```

Step 4, the **radial-background flatten, is the explicit background-removal step
and is on by default** (disable with `FLATTEN=0`). The DeltaPDF's own Gaussian
`SUBTRACT_BG` blur is the alternative remover and defaults off; do not run both.

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
NO_VIEWER=1 \
DATA_FILE="data/raw/example_volume_cc_sub_bkg.nxs" \
$PY examples/run_pipeline.py
```

For several related volumes, run the same command once per input file:

```bash
for DATA_FILE in data/raw/condition_*.nxs; do
  NO_VIEWER=1 DATA_FILE="$DATA_FILE" $PY examples/run_pipeline.py
done
```

## DeltaPDF Files

Use this when the reciprocal-space cleanup stages already exist and you want a
persistent DeltaPDF file in `data/processed`. The input is the **flattened**
(step-4, background-removed) volume, so the transform's own `SUBTRACT_BG` is left
off. To use the legacy in-FFT Gaussian blur instead, point `PROC_FILE` at a
`*_backfilled.h5` file and add `SUBTRACT_BG="0,1.5,1.5"`, but not both.

The `CROP_H=4 CROP_K=8 CROP_L=15` below is an **optional band-limit**. The
default (omit them) is the full `|Q|` range, which gives a finer real-space grid
that matches the web 3D-DeltaPDF page and the consistency view, at the cost of a
larger transform that includes noisier outer `|Q|` shells.

```bash
PROC_FILE="data/processed/example_volume_ringremoved_braggpunched_backfilled_flattened.h5" \
OUT_FILE="data/processed/example_volume_delta_pdf.h5" \
CROP_H=4 CROP_K=8 CROP_L=15 APODIZE=gaussian GAUSSIAN_SIGMA=0.4 \
$PY examples/delta_pdf.py
```

## View Processed Reciprocal-Space Data

These commands open the four-panel cleanup QA viewer:

```text
raw -> ring removed -> Bragg punched -> backfilled
```

```bash
DATA_FILE="data/raw/example_volume_cc_sub_bkg.nxs" \
RING_FILE="data/processed/example_volume_ringremoved.h5" \
PUNCH_FILE="data/processed/example_volume_ringremoved_braggpunched.h5" \
BACKFILL_FILE="data/processed/example_volume_ringremoved_braggpunched_backfilled.h5" \
$PY examples/explore_slice.py
```

## View DeltaPDF

```bash
PDF_FILE="data/processed/example_volume_delta_pdf.h5" \
$PY examples/explore_delta_pdf_ortho.py
```

## Check DeltaPDF Consistency

Use this as the endpoint of a single-volume workflow. It starts from the
flattened diffuse volume (or a backfilled volume if flattening was disabled),
computes the DeltaPDF, inverse-transforms it back to reciprocal space, and
writes a `data | back-FFT | residual` figure plus metrics on stdout.

```bash
DATA_FILE="data/processed/example_volume_ringremoved_braggpunched_backfilled_flattened.h5" \
OUT_PNG="data/processed/example_volume_delta_pdf_consistency.png" \
$PY examples/delta_pdf_consistency.py
```

A faithful transform should give Pearson `r` close to 1 and a small normalised
RMS residual over the reliably recovered reciprocal-space region. For interactive
band-limited checks, use the web UI's **Consistency check** view.

## Compare DeltaPDF Files

With pipeline outputs in `data/processed`, use the web UI's multi-volume view for
general comparisons. The script accepts a comma-separated list of DeltaPDF paths:

```bash
PDF_FILES="data/processed/condition_a_delta_pdf.h5,data/processed/condition_b_delta_pdf.h5,data/processed/condition_c_delta_pdf.h5" \
PDF_LABELS="condition A,condition B,condition C" \
$PY examples/explore_delta_pdf_multi.py
```

The viewer shows three orthoslice planes for each file on a per-plane colour
scale.

## 3D-PDF (Bragg Kept)

The DeltaPDF workflow removes the Bragg peaks (punch -> backfill) to isolate the
diffuse. The **3D-PDF** instead keeps the Bragg peaks and Fourier-transforms the
total scattering, giving a Patterson-like map of the average-structure
correlations. `examples/run_pipeline_pdf.py` runs `ring removal -> 3D-PDF` with
**no punch and no backfill**; the smooth-background subtraction is off. The
ring-removed files are shared with the DeltaPDF workflow, so stage 1 is skipped
if it already ran.

```bash
NO_VIEWER=1 \
DATA_FILE="data/raw/example_volume_cc_sub_bkg.nxs" \
$PY examples/run_pipeline_pdf.py
```

Output: `data/processed/*_ringremoved_3dpdf.h5`. Set `RING_REMOVAL=0` to
transform the raw data directly. View it with the orthoslice viewer:

```bash
PDF_FILE="data/processed/example_volume_ringremoved_3dpdf.h5" \
$PY examples/explore_delta_pdf_ortho.py
```

## Bragg / Diffuse Separation Diagnostic

When diffuse scattering sits *at* sharp Bragg or satellite positions,
characterise the peak shape before separating it.
`examples/investigate_bragg_diffuse.py` calibrates the resolution on sharp Bragg
features, then decomposes each selected feature into a sharp
(resolution-limited) core plus a broad diffuse component on H/K/L line cuts.

```bash
$PY examples/investigate_bragg_diffuse.py
```

Writes cut/fit figures, a summary plot, and a CSV with resolution `sigma(|Q|)`,
points-across-FWHM, correlation length, and the diffuse fraction.
