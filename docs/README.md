# Documentation Guide

This directory contains the deeper notes for `neutron-diffuse`. The README at
the repository root is the quick entry point; these pages explain the algorithms,
viewers, and reproducible command recipes in more detail.

## Recommended Reading Order

1. [Repository README](../README.md) - installation and package overview.
2. [Quick Start](quick_start.md) - concise 22 K, 45 K, and 100 K workflow,
   the browser-UI workflow, and plotting commands.
3. [Web UI](web.md) - the browser console (FastAPI + React) that runs the
   pipeline and unifies the cleanup, DeltaPDF, multi-temperature, and
   consistency views.
4. [Powder ring removal](algorithms/powder_rings.md) - how smooth powder-ring
   intensity is estimated and subtracted.
5. [Bragg cleanup](algorithms/bragg_cleanup.md) - integer-node Bragg punching,
   search-mode satellite punching, direct-beam handling, and backfill.
6. [3D-DeltaPDF transform](algorithms/delta_pdf.md) - centred FFT recipe, the
   radial-background flatten (the default step-4 background removal), the
   alternative smooth-background subtraction, real-space viewer assumptions, and
   the back-FFT consistency check.
7. [Interactive exploration](interactive.md) - the standalone matplotlib cleanup
   QA viewers, DeltaPDF viewers, and plotting primitives.

## Pages

| Page | Use it when you need to... |
| --- | --- |
| [quick_start.md](quick_start.md) | Run or view the current 22 K, 45 K, and 100 K workflow (CLI or browser UI). |
| [web.md](web.md) | Launch, use, or develop the browser UI (viewers + pipeline runner + consistency check). |
| [algorithms/powder_rings.md](algorithms/powder_rings.md) | Understand or tune powder-ring subtraction. |
| [algorithms/bragg_cleanup.md](algorithms/bragg_cleanup.md) | Tune Bragg punching, satellite search, or q-shell backfill. |
| [algorithms/delta_pdf.md](algorithms/delta_pdf.md) | Understand the FFT, centring, apodization, background subtraction, and round-trip consistency check. |
| [algorithms/inpainting.md](algorithms/inpainting.md) | Compare general inpainting fallbacks such as symmetry and TV. |
| [interactive.md](interactive.md) | Use the cleanup and DeltaPDF viewers or the plotting API. |
| [plotting_commands.md](plotting_commands.md) | Reproduce the current plotting and multi-temperature command set. |
| [../CHANGELOG.md](../CHANGELOG.md) | Review release notes and version history. |

## Output Locations

Common generated files:

| Path | Contents |
| --- | --- |
| `data/processed/*_ringremoved.h5` | Powder-ring-subtracted reciprocal-space volume. |
| `data/processed/*_braggpunched.h5` | Ring-removed volume with Bragg/satellite holes punched. |
| `data/processed/*_braggpunched_backfilled.h5` | Cleaned diffuse volume after Bragg-hole backfill. |
| `data/processed/*_backfilled_flattened.h5` | Background-removed diffuse volume (step-4 radial flatten; feeds the DeltaPDF). |
| `data/processed/*_delta_pdf.h5` | Per-temperature 3D-DeltaPDF output. |
| `data/processed/*_delta_pdf_consistency.json` | Back-FFT consistency metrics from the library/web pipeline. |
| `data/processed/*_delta_pdf_consistency.png` | `data | back-FFT | residual` consistency figure. |
| `data/processed/*_3dpdf.h5` | Total-scattering 3D-PDF output (Bragg kept; `run_pipeline_pdf.py`). |
| `examples/_delta_pdf.h5` | Default cached 3D-DeltaPDF output when `OUT_FILE` is not set. |
| `examples/_*.png` | Generated preview figures. |

Generated data and figures are intentionally ignored by Git.

## Terminology

| Term | Meaning |
| --- | --- |
| HKL volume | A regular 3D grid in reciprocal-lattice coordinates H, K, and L. |
| UB matrix | Matrix converting HKL coordinates to Cartesian reciprocal-space Q. |
| Powder ring | Smooth azimuthal ring intensity from polycrystalline material in the beam path. |
| Bragg punch | Masking sharp Bragg or satellite peaks so they do not dominate the DeltaPDF. |
| Backfill | Replacing punched voxels with a local or radial diffuse-background estimate. |
| 3D-DeltaPDF | Fourier transform of cleaned diffuse scattering into real-space pair correlations. |
| Consistency check | Inverse-transforming the ΔPDF back to reciprocal space and comparing it with the diffuse input. |
