"""Pipeline execution endpoints: run, stream progress (SSE), cancel."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ndiff.pipeline import STAGES, PipelineParams
from ndiff.server.config import ServerConfig
from ndiff.server.datasets import find_dataset
from ndiff.server.deps import get_config
from ndiff.server.jobs import JobManager
from ndiff.server.schemas import JobOut, PipelineRunRequest

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


def build_params(req: PipelineRunRequest) -> PipelineParams:
    """Apply the curated request overrides onto the validated defaults."""
    p = PipelineParams(flatten_enabled=req.flatten_enabled)
    sp = req.params

    if sp.punch_min_intensity is not None:
        p.punch = dataclasses.replace(p.punch, min_intensity=sp.punch_min_intensity)
    if sp.punch_search_n_mad is not None:
        p.punch = dataclasses.replace(p.punch, search_n_mad=sp.punch_search_n_mad)
    if sp.backfill_method is not None:
        p.backfill = dataclasses.replace(p.backfill, method=sp.backfill_method)
    if sp.flatten_estimator is not None:
        p.flatten = dataclasses.replace(p.flatten, estimator=sp.flatten_estimator)
    if sp.flatten_floor_percentile is not None:
        p.flatten = dataclasses.replace(
            p.flatten, floor_percentile=sp.flatten_floor_percentile)

    dp_kw: dict = {}
    if sp.pdf_apodization is not None:
        dp_kw["apodization"] = sp.pdf_apodization
    if sp.pdf_gaussian_sigma is not None:
        dp_kw["gaussian_sigma"] = sp.pdf_gaussian_sigma
    if any(v is not None for v in (sp.pdf_crop_h, sp.pdf_crop_k, sp.pdf_crop_l)):
        cur = p.delta_pdf.crop_hkl or (4.0, 8.0, 15.0)
        dp_kw["crop_hkl"] = (
            sp.pdf_crop_h if sp.pdf_crop_h is not None else cur[0],
            sp.pdf_crop_k if sp.pdf_crop_k is not None else cur[1],
            sp.pdf_crop_l if sp.pdf_crop_l is not None else cur[2],
        )
    if dp_kw:
        p.delta_pdf = dataclasses.replace(p.delta_pdf, **dp_kw)
    return p


def _jobs(request: Request) -> JobManager:
    return request.app.state.jobs  # type: ignore[no-any-return]


@router.post("/run", response_model=JobOut)
def run(req: PipelineRunRequest, request: Request,
        cfg: ServerConfig = Depends(get_config)) -> JobOut:
    ds = find_dataset(cfg, req.dataset_id)
    if ds is None:
        raise HTTPException(404, f"unknown dataset {req.dataset_id!r}")
    if not ds.raw_path.exists():
        raise HTTPException(400, f"raw input not found for {req.dataset_id!r}; "
                                 "cannot run the pipeline")
    if req.force_from is not None and req.force_from not in STAGES:
        raise HTTPException(400, f"force_from must be one of {STAGES}")
    params = build_params(req)
    job = _jobs(request).start(
        ds.raw_path, params, proc_dir=cfg.processed_dir,
        force=req.force, force_from=req.force_from)
    return JobOut(**job.snapshot())


@router.get("/jobs/{jid}", response_model=JobOut)
def job_status(jid: str, request: Request) -> JobOut:
    job = _jobs(request).get(jid)
    if job is None:
        raise HTTPException(404, f"unknown job {jid!r}")
    return JobOut(**job.snapshot())


@router.get("/jobs/{jid}/events")
async def job_events(jid: str, request: Request) -> StreamingResponse:
    job = _jobs(request).get(jid)
    if job is None:
        raise HTTPException(404, f"unknown job {jid!r}")

    async def stream() -> AsyncIterator[str]:
        idx = 0
        while True:
            events, status = job.events_since(idx)
            for ev in events:
                yield f"data: {json.dumps(ev)}\n\n"
            idx += len(events)
            if status != "running":
                yield f"data: {json.dumps({'type': status})}\n\n"
                break
            if await request.is_disconnected():
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.post("/jobs/{jid}/cancel", response_model=JobOut)
def job_cancel(jid: str, request: Request) -> JobOut:
    jobs = _jobs(request)
    job = jobs.get(jid)
    if job is None:
        raise HTTPException(404, f"unknown job {jid!r}")
    jobs.cancel(jid)
    return JobOut(**job.snapshot())
