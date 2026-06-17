"""Dataset listing endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from ndiff.server import consistency as consistency_mod
from ndiff.server import datasets as ds_mod
from ndiff.server import deltapdf as dpdf_mod
from ndiff.server import volumes as vol_mod
from ndiff.server.config import ServerConfig
from ndiff.server.deps import get_config
from ndiff.server.schemas import DataRootIn, DataRootOut, DatasetOut, StageStatusOut

router = APIRouter(prefix="/api", tags=["datasets"])


def _data_root_out(cfg: ServerConfig) -> DataRootOut:
    return DataRootOut(
        data_root=str(cfg.data_root),
        raw_exists=cfg.raw_dir.is_dir(),
        processed_exists=cfg.processed_dir.is_dir(),
        n_datasets=len(ds_mod.discover_datasets(cfg)),
    )


@router.get("/data-root", response_model=DataRootOut)
def get_data_root(cfg: ServerConfig = Depends(get_config)) -> DataRootOut:
    """Return the active data root scanned by the web server."""
    return _data_root_out(cfg)


@router.put("/data-root", response_model=DataRootOut)
def set_data_root(req: DataRootIn, request: Request) -> DataRootOut:
    """Switch the active data root for this running server process."""
    root = Path(req.data_root).expanduser().resolve()
    if not root.is_dir():
        raise HTTPException(status_code=400, detail=f"Folder does not exist: {root}")

    cfg = ServerConfig(data_root=root)
    request.app.state.config = cfg
    vol_mod.clear_cache()
    dpdf_mod.clear_cache()
    consistency_mod.clear_cache()
    return _data_root_out(cfg)


@router.get("/datasets", response_model=list[DatasetOut])
def list_datasets(cfg: ServerConfig = Depends(get_config)) -> list[DatasetOut]:
    """List datasets grouped by raw input, with per-stage output status."""
    out: list[DatasetOut] = []
    for ds in ds_mod.discover_datasets(cfg):
        stages = [
            StageStatusOut(name=s.name, exists=s.exists, kind=s.kind,
                           volume_id=f"{ds.id}.{s.name}")
            for s in ds.stages
        ]
        out.append(DatasetOut(
            id=ds.id, temperature=ds.temperature, raw_name=ds.raw_name,
            stem=ds.stem, stages=stages,
        ))
    return out
