"""FastAPI backend for the neutron-diffuse web UI.

The server is a thin presentation/orchestration layer over the ``ndiff`` library:
it discovers datasets and their pipeline-stage outputs, serves 2D slices of the
reciprocal-space volumes (reusing :func:`ndiff.visualization.extract_slice`),
drives :func:`ndiff.pipeline.run_pipeline` as a background job, and exposes the
3D-ΔPDF plus back-FFT consistency viewers.

Use :func:`ndiff.server.app.create_app` to build the ASGI app, or the
``ndiff-web`` console script to launch it.
"""

from ndiff.server.app import create_app

__all__ = ["create_app"]
