"""Real-space 3D-ΔPDF metadata and orthoslice endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from ndiff.server.config import ServerConfig
from ndiff.server.datasets import StageStatus, resolve_volume
from ndiff.server.deltapdf import DPDF_PLANES, dpdf_meta, dpdf_slice_envelope
from ndiff.server.deps import get_config
from ndiff.server.schemas import DeltaPdfMetaOut, LatticeOut

router = APIRouter(prefix="/api/deltapdf", tags=["deltapdf"])


def _resolve_dpdf(cfg: ServerConfig, volume_id: str) -> StageStatus:
    stage = resolve_volume(cfg, volume_id)
    if stage is None:
        raise HTTPException(404, f"unknown volume id {volume_id!r}")
    if not stage.path.exists():
        raise HTTPException(404, f"ΔPDF not found for {volume_id!r}")
    if stage.kind != "delta_pdf":
        raise HTTPException(400, f"{volume_id!r} is not a ΔPDF volume")
    return stage


@router.get("/{volume_id}/meta", response_model=DeltaPdfMetaOut)
def meta(volume_id: str, cfg: ServerConfig = Depends(get_config)) -> DeltaPdfMetaOut:
    stage = _resolve_dpdf(cfg, volume_id)
    m = dpdf_meta(stage.path)
    return DeltaPdfMetaOut(
        id=volume_id, shape=m["shape"], x_range=m["x_range"], y_range=m["y_range"],
        z_range=m["z_range"], lattice=LatticeOut(**m["lattice"]), q_max=m["q_max"],
        planes=m["planes"],
    )


@router.get("/{volume_id}/slice")
def slice_(
    volume_id: str,
    plane: str = Query("yz"),
    value: float = Query(0.0),
    cfg: ServerConfig = Depends(get_config),
) -> Response:
    stage = _resolve_dpdf(cfg, volume_id)
    if plane not in DPDF_PLANES:
        raise HTTPException(400, f"unknown plane {plane!r}; choose one of {DPDF_PLANES}")
    body = dpdf_slice_envelope(stage.path, plane=plane, value=value)
    return Response(content=body, media_type="application/octet-stream")
