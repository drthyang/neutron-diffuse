# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tsung-han Yang

"""Volume metadata and 2D-slice endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from nebula3d.server.config import ServerConfig
from nebula3d.server.datasets import StageStatus, resolve_volume
from nebula3d.server.deps import get_config
from nebula3d.server.schemas import LatticeOut, VolumeMetaOut
from nebula3d.server.volumes import PLANES, slice_envelope, volume_meta

router = APIRouter(prefix="/api/volumes", tags=["volumes"])


def _resolve_hkl(cfg: ServerConfig, volume_id: str) -> StageStatus:
    """Resolve a volume id to an existing HKLVolume stage, or raise HTTP errors."""
    stage = resolve_volume(cfg, volume_id)
    if stage is None:
        raise HTTPException(404, f"unknown volume id {volume_id!r}")
    if not stage.path.exists():
        raise HTTPException(404, f"stage output not found for {volume_id!r}")
    if stage.kind != "hkl":
        raise HTTPException(
            400, f"{volume_id!r} is a {stage.kind} volume; use the ΔPDF endpoints")
    return stage


@router.get("/{volume_id}/meta", response_model=VolumeMetaOut)
def meta(volume_id: str, cfg: ServerConfig = Depends(get_config)) -> VolumeMetaOut:
    stage = _resolve_hkl(cfg, volume_id)
    m = volume_meta(stage.path)
    return VolumeMetaOut(
        id=volume_id, stage=stage.name, kind=stage.kind,
        shape=m["shape"], h_range=m["h_range"], k_range=m["k_range"],
        l_range=m["l_range"], lattice=LatticeOut(**m["lattice"]),
        ub_matrix=m.get("ub_matrix"),
        planes=m["planes"],
    )


@router.get("/{volume_id}/slice")
def slice_(
    volume_id: str,
    plane: str = Query("hk"),
    value: float = Query(0.0),
    interp: bool = Query(False),
    cfg: ServerConfig = Depends(get_config),
) -> Response:
    stage = _resolve_hkl(cfg, volume_id)
    if plane not in PLANES:
        raise HTTPException(400, f"unknown plane {plane!r}; choose one of {PLANES}")
    body = slice_envelope(stage.path, plane=plane, value=value, interp=interp)
    return Response(content=body, media_type="application/octet-stream")
