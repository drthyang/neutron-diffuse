"""Back-FFT consistency endpoints: |Q|-band-limited ΔPDF round trip + slices."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from ndiff.server.config import ServerConfig
from ndiff.server.consistency import (
    PANELS,
    consistency_meta,
    consistency_slice_envelope,
    pdf_input_path,
)
from ndiff.server.deps import get_config
from ndiff.server.volumes import PLANES

router = APIRouter(prefix="/api/consistency", tags=["consistency"])


def _band(val_min: float | None, val_max: float | None) -> tuple[float, float] | None:
    """Assemble the band limits; None when neither bound is set (full data)."""
    if val_min is None and val_max is None:
        return None
    lo = 0.0 if val_min is None else float(val_min)
    hi = float("inf") if val_max is None else float(val_max)
    if hi <= lo:
        raise HTTPException(400, f"max limit ({hi}) must exceed min limit ({lo})")
    return (lo, hi)


@router.get("/{dataset_id}/meta")
def meta(
    dataset_id: str,
    q_min: float | None = Query(None),
    q_max: float | None = Query(None),
    r_min: float | None = Query(None),
    r_max: float | None = Query(None),
    cfg: ServerConfig = Depends(get_config),
) -> dict:
    path = pdf_input_path(cfg, dataset_id)
    if path is None:
        raise HTTPException(
            404, f"no flattened/backfilled volume for {dataset_id!r}; "
                 "run the pipeline first")
    return consistency_meta(path, _band(q_min, q_max), _band(r_min, r_max))


@router.get("/{dataset_id}/slice")
def slice_(
    dataset_id: str,
    panel: str = Query("data"),
    plane: str = Query("kl"),
    value: float = Query(0.0),
    q_min: float | None = Query(None),
    q_max: float | None = Query(None),
    r_min: float | None = Query(None),
    r_max: float | None = Query(None),
    cfg: ServerConfig = Depends(get_config),
) -> Response:
    if panel not in PANELS:
        raise HTTPException(400, f"unknown panel {panel!r}; choose one of {PANELS}")
    # Note: PLANES are reciprocal, but if panel is dpdf, plane will be real-space (xy, yz, zx)
    # We bypass the PLANES check for dpdf.
    if panel != "dpdf" and plane not in PLANES:
        raise HTTPException(400, f"unknown plane {plane!r}; choose one of {PLANES}")
    path = pdf_input_path(cfg, dataset_id)
    if path is None:
        raise HTTPException(404, f"no flattened/backfilled volume for {dataset_id!r}")
    body = consistency_slice_envelope(
        path, _band(q_min, q_max), _band(r_min, r_max), panel, plane, value)
    return Response(content=body, media_type="application/octet-stream")
