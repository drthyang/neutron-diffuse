"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from nebula3d import __version__
from nebula3d.server.config import ServerConfig, config_from_env
from nebula3d.server.jobs import JobManager
from nebula3d.server.routers import bragg as bragg_router
from nebula3d.server.routers import consistency as consistency_router
from nebula3d.server.routers import datasets as datasets_router
from nebula3d.server.routers import deltapdf as deltapdf_router
from nebula3d.server.routers import pipeline as pipeline_router
from nebula3d.server.routers import slices as slices_router

#: Vite dev-server origins allowed during local development.
DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

#: Directory holding the built single-page app (populated at release/build time).
STATIC_DIR = Path(__file__).parent / "static"


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Build the ASGI app.

    Parameters
    ----------
    config:
        Server configuration; defaults to :func:`config_from_env`.
    """
    app = FastAPI(title="nebula3d", version=__version__)
    app.state.config = config or config_from_env()
    app.state.jobs = JobManager()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(datasets_router.router)
    app.include_router(slices_router.router)
    app.include_router(deltapdf_router.router)
    app.include_router(pipeline_router.router)
    app.include_router(consistency_router.router)
    app.include_router(bragg_router.router)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    # Serve the built SPA when present (production / installed wheel).  Mounted
    # last so it never shadows the /api routes.
    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="spa")

    return app
