# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tsung-han Yang

"""Bragg peak profile review endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException

from nebula3d.pipeline import pipeline_paths
from nebula3d.server.config import ServerConfig
from nebula3d.server.datasets import find_dataset
from nebula3d.server.deps import get_config
from nebula3d.server.schemas import BraggProfileOut

router = APIRouter(prefix="/api/bragg", tags=["bragg"])


@router.get("/{dataset_id}/profile", response_model=BraggProfileOut)
def profile(
    dataset_id: str,
    cfg: ServerConfig = Depends(get_config),
) -> BraggProfileOut:
    """Return per-peak Bragg punch width metadata for one dataset."""
    ds = find_dataset(cfg, dataset_id)
    if ds is None:
        raise HTTPException(404, f"unknown dataset id {dataset_id!r}")

    paths = pipeline_paths(ds.raw_path, proc_dir=cfg.processed_dir)
    path = paths.bragg_profile_json
    if not path.exists():
        return BraggProfileOut(
            dataset_id=dataset_id,
            profile_path=str(path),
            has_profile=False,
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(
            500, f"could not read Bragg profile {path.name}: {exc}") from exc

    return BraggProfileOut(
        dataset_id=dataset_id,
        profile_path=str(path),
        has_profile=True,
        schema_version=int(data.get("schema_version", 1)),
        width_labels=list(data.get("width_labels", [])),
        hkl_width_labels=list(data.get("hkl_width_labels", [])),
        width_units=dict(data.get("width_units", {})),
        n_peaks=int(data.get("n_peaks", 0)),
        fit_covariance=bool(data.get("fit_covariance", False)),
        punch_frame=data.get("punch_frame"),
        peaks=list(data.get("peaks", [])),
    )
