# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tsung-han Yang

"""FastAPI backend for the nebula3d web UI.

The server is a thin presentation/orchestration layer over the ``nebula3d`` library:
it discovers datasets and their pipeline-stage outputs, serves 2D slices of the
reciprocal-space volumes (reusing :func:`nebula3d.visualization.extract_slice`),
drives :func:`nebula3d.pipeline.run_pipeline` as a background job, and exposes the
3D-ΔPDF plus back-FFT consistency viewers.

Use :func:`nebula3d.server.app.create_app` to build the ASGI app, or the
``nebula3d-web`` console script to launch it.

``create_app`` is imported lazily (via ``__getattr__``) so that importing a
FastAPI-free helper submodule — e.g. ``from nebula3d.server import volumes`` — does
not pull in FastAPI.  This lets the in-browser bridge (:mod:`nebula3d.webbridge`)
reuse the slicing/discovery helpers under Pyodide, where FastAPI is not
installed.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nebula3d.server.app import create_app

__all__ = ["create_app"]


def __getattr__(name: str) -> object:
    if name == "create_app":
        from nebula3d.server.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
