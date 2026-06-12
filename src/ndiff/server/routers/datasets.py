"""Dataset listing endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ndiff.server import datasets as ds_mod
from ndiff.server.config import ServerConfig
from ndiff.server.deps import get_config
from ndiff.server.schemas import DatasetOut, StageStatusOut

router = APIRouter(prefix="/api", tags=["datasets"])


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
