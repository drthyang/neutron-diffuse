"""Background pipeline-job execution.

The pipeline is CPU-bound and minutes long, so each run executes in a **separate
process** (a spawn-context :class:`multiprocessing.Process`) and reports progress
back over a queue.  A per-job daemon thread drains the queue into the job's event
list and status; the SSE endpoint then streams those events to the browser
without touching the queue directly (so status tracking is independent of whether
a client is connected, and multiple clients can follow the same job).

Cancellation terminates the worker process.
"""

from __future__ import annotations

import multiprocessing as mp
import multiprocessing.queues
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ndiff.pipeline import STAGES, PipelineParams, run_pipeline

_ctx = mp.get_context("spawn")  # robust across macOS/Linux (avoids fork+threads)


def _worker(
    queue: mp.queues.Queue,
    input_path: str,
    params: PipelineParams,
    proc_dir: str | None,
    stages: Sequence[str],
    force: bool,
    force_from: str | None,
) -> None:
    """Process entry point: run the pipeline, streaming progress onto the queue."""
    def progress(stage: str, status: str, fraction: float | None,
                 message: str) -> None:
        queue.put({"type": "progress", "stage": stage, "status": status,
                   "fraction": fraction, "message": message})

    try:
        run_pipeline(input_path, params, proc_dir=proc_dir, stages=tuple(stages),
                     force=force, force_from=force_from, progress=progress)
        queue.put({"type": "done"})
    except Exception as exc:  # noqa: BLE001 - report any failure to the client
        queue.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})


@dataclass
class Job:
    id: str
    input_name: str
    status: str = "running"  # running | done | error | cancelled
    error: str | None = None
    events: list[dict] = field(default_factory=list)
    _process: mp.process.BaseProcess | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "input": self.input_name,
                "status": self.status,
                "error": self.error,
                "n_events": len(self.events),
            }

    def events_since(self, idx: int) -> tuple[list[dict], str]:
        with self._lock:
            return self.events[idx:], self.status


class JobManager:
    """In-process registry of pipeline jobs (one worker process each)."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(
        self,
        input_path: str | Path,
        params: PipelineParams,
        *,
        proc_dir: str | Path | None = None,
        stages: Sequence[str] = STAGES,
        force: bool = False,
        force_from: str | None = None,
    ) -> Job:
        jid = uuid.uuid4().hex[:12]
        queue = _ctx.Queue()
        job = Job(id=jid, input_name=Path(input_path).name)
        proc = _ctx.Process(
            target=_worker,
            args=(queue, str(input_path), params,
                  str(proc_dir) if proc_dir is not None else None,
                  tuple(stages), force, force_from),
            daemon=True,
        )
        job._process = proc
        with self._lock:
            self._jobs[jid] = job
        proc.start()
        threading.Thread(target=self._drain, args=(job, queue), daemon=True).start()
        return job

    def _drain(self, job: Job, queue: mp.queues.Queue) -> None:
        while True:
            ev = queue.get()
            with job._lock:
                if ev["type"] == "progress":
                    job.events.append(ev)
                    continue
                if ev["type"] == "error":
                    job.status = "error"
                    job.error = ev.get("message")
                elif ev["type"] == "done":
                    if job.status == "running":
                        job.status = "done"
                break

    def get(self, jid: str) -> Job | None:
        with self._lock:
            return self._jobs.get(jid)

    def cancel(self, jid: str) -> bool:
        job = self.get(jid)
        if job is None or job.status != "running":
            return False
        proc = job._process
        if proc is not None and proc.is_alive():
            proc.terminate()
        with job._lock:
            job.status = "cancelled"
        return True
