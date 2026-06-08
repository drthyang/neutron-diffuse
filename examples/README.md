# Examples

This directory contains runnable scripts for the current `neutron-diffuse`
workflow. The scripts are meant to be run from the repository root with
`PYTHONPATH=src`.

Generated outputs are intentionally excluded from the repository:

- `examples/_*.png`
- `examples/_*.h5`

Regenerate those files by rerunning the scripts below.

## Main Workflow

| Script | Purpose |
| --- | --- |
| `run_pipeline.py` | End-to-end 3D-DeltaPDF workflow: ring removal, Bragg punch, backfill, 3D-DeltaPDF, and viewers. |
| `run_pipeline_pdf.py` | End-to-end **3D-PDF** workflow — KEEP Bragg (no punch, no backfill): ring removal → 3D-PDF → viewer. |
| `remove_rings_3d.py` | Remove powder rings from a raw Mantid HKL volume. |
| `punch_bragg_3d.py` | Punch Bragg and satellite peaks from a ring-removed volume. |
| `backfill_bragg_3d.py` | Fill Bragg-punched holes before the DeltaPDF transform. |
| `delta_pdf.py` | Compute and save the full 3D-DeltaPDF (Bragg removed). |
| `pdf_3d.py` | Compute and save the total-scattering 3D-PDF (Bragg kept). |

## Viewers

| Script | Purpose |
| --- | --- |
| `explore_slice.py` | Reciprocal-space cleanup QA: raw, ring removed, punched, and backfilled slices. |
| `explore_delta_pdf_ortho.py` | Single-temperature 3D-DeltaPDF orthoslice viewer. |
| `explore_delta_pdf_multi.py` | Multi-temperature 3D-DeltaPDF comparison viewer. |
| `explore_delta_pdf.py` | Single-plane DeltaPDF viewer with an `x_H` slider. |
| `explore_volume.py` | Raw versus processed reciprocal-space volume viewer. |
| `explore.py` | IPython preamble for ad hoc plotting with `ndiff.visualization`. |

## Diagnostics And Comparisons

| Script | Purpose |
| --- | --- |
| `delta_pdf_plane.py` | 2D DeltaPDF for one reciprocal H plane. |
| `compare_delta_pdf_methods.py` | Compare DeltaPDF background-removal methods. |
| `ring_linecut.py` | Inspect a Bragg-free linecut and ring positions. |
| `investigate_bragg_diffuse.py` | Separate Bragg from co-located magnetic diffuse at the q=1/3 satellites: line-cut peak-shape fits, resolution `σ(\|Q\|)`, correlation length ξ, diffuse fraction, and a temperature series. |

## Typical Commands

Run the complete workflow:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python3 examples/run_pipeline.py
```

Run without opening viewers:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl NO_VIEWER=1 python3 examples/run_pipeline.py
```

Force recomputation from the DeltaPDF stage:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl FORCE_FROM=pdf python3 examples/run_pipeline.py
```

Run the 3D-PDF workflow instead (keeps the Bragg peaks — no punch, no backfill):

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python3 examples/run_pipeline_pdf.py
```
