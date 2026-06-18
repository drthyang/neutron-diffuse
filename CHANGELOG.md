# Changelog

## 0.2.0 - 2026-06-18

- Promoted the consistency check to the endpoint of the recommended 3D-DeltaPDF
  workflow.
- Added the FastAPI/React consistency viewer and `/api/consistency` endpoints
  for reciprocal-space back-FFT comparison with optional `|Q|` and real-space
  bands.
- Updated `examples/run_pipeline.py` to run the back-FFT consistency check by
  default after the DeltaPDF stage.
- Updated documentation around the full workflow, web UI, reproducibility
  commands, and output artifacts.
- Aligned package, API, and web app version metadata at `0.2.0`.

## 0.1.0 - Initial alpha

- Initial alpha toolkit for reciprocal-space neutron diffuse cleanup and
  3D-DeltaPDF exploration.
