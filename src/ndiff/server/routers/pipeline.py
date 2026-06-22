"""Pipeline execution endpoints: run, stream progress (SSE), cancel."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ndiff.pipeline import STAGES
from ndiff.server.config import ServerConfig
from ndiff.server.datasets import find_dataset
from ndiff.server.deps import get_config
from ndiff.server.jobs import JobManager
from ndiff.server.params import build_params
from ndiff.server.schemas import JobOut, PipelineRunRequest

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

# build_params is re-exported (kept importable from this router for back-compat);
# it lives in ndiff.server.params so the FastAPI-free in-browser bridge can reuse
# it.  Invalid-band errors surface as ValueError there and become HTTP 400 here.
__all__ = ["router", "build_params"]


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
    try:
        params = build_params(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
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
