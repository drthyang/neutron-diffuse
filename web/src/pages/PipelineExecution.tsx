// Pipeline execution page — the stage stepper and live log for the job started
// from the Configure page.  Job state (and the SSE stream feeding it) lives in
// the pipeline store, so this page reflects an in-flight job even if it was
// mounted after navigating away and back.

import { useEffect, useRef } from "react";
import { useShallow } from "zustand/react/shallow";

import { STAGES, STAGE_LABELS, usePipelineStore } from "../state/pipelineStore";

interface StageInfo {
  status: string;
  fraction: number | null;
}

function stepClass(status: string | undefined, off: boolean): string {
  if (off) return "off";
  if (!status) return "";
  if (status === "start" || status === "progress") return "active";
  return status; // done | skip | error
}

function stepSub(info: StageInfo | undefined, off: boolean): string {
  if (off) return "disabled";
  if (!info) return "pending";
  switch (info.status) {
    case "start":
      return "starting…";
    case "progress":
      return info.fraction != null
        ? `running · ${Math.round(info.fraction * 100)}%`
        : "running…";
    case "done":
      return "done";
    case "skip":
      return "skipped — output exists";
    case "error":
      return "failed";
    default:
      return info.status;
  }
}

export function PipelineExecution() {
  const { events, terminal, running, flatten, jobId } = usePipelineStore(
    useShallow((s) => ({
      events: s.events,
      terminal: s.terminal,
      running: s.running,
      flatten: s.flatten,
      jobId: s.jobId,
    })),
  );
  const cancel = usePipelineStore((s) => s.cancel);
  const logRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    logRef.current?.scrollTo(0, logRef.current.scrollHeight);
  }, [events]);

  // latest status (+ fraction) per stage from the event stream
  const stageInfo: Record<string, StageInfo> = {};
  for (const ev of events) {
    if (ev.stage && ev.status) {
      stageInfo[ev.stage] = { status: ev.status, fraction: ev.fraction ?? null };
    }
  }

  const started = running || terminal != null || events.length > 0;

  return (
    <div className="exec-page">
      {!started && (
        <div className="banner muted">
          No job yet — set parameters on the Configure page and press Run.
        </div>
      )}
      {terminal === "done" && (
        <div className="banner ok">
          Job finished — open the 3D-ΔPDF tab to view the result.
        </div>
      )}
      {terminal === "cancelled" && <div className="banner warn">Job cancelled.</div>}
      {terminal === "error" && (
        <div className="banner err">Job failed — see the log below.</div>
      )}

      <div className="exec-grid">
        <div className="card">
          <div className="card-head">
            <h3>Stages</h3>
            <div className="card-head-actions">
              {running && <span className="spin" />}
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                onClick={cancel}
                disabled={!running}
              >
                Cancel
              </button>
            </div>
          </div>
          <div className="card-body">
            <ol className="stepper">
              {STAGES.map((s) => {
                const off = !flatten && s === "flatten";
                const info = stageInfo[s];
                const cls = stepClass(info?.status, off);
                return (
                  <li key={s} className={cls}>
                    <span className="step-rail">
                      <span className="step-dot" />
                    </span>
                    <span className="step-main">
                      <span className="step-name">{STAGE_LABELS[s]}</span>
                      <span className="step-sub">{stepSub(info, off)}</span>
                      {cls === "active" && info?.fraction != null && (
                        <span className="step-bar">
                          <span style={{ width: `${info.fraction * 100}%` }} />
                        </span>
                      )}
                    </span>
                  </li>
                );
              })}
            </ol>
          </div>
        </div>

        <div className="card exec-log-card">
          <div className="card-head">
            <h3>Log</h3>
            <span className="log-count">
              {jobId ? `job ${jobId.slice(0, 8)} · ` : ""}
              {events.length > 0 ? `${events.length} events` : ""}
            </span>
          </div>
          <div className="log" ref={logRef}>
            {events.length === 0 && !running && (
              <div className="log-empty">
                Stages with existing outputs are skipped — enable “force” on the
                Configure page to recompute them.
              </div>
            )}
            {events.map((ev, i) => (
              <div key={i} className={`log-line ${ev.status ?? ""}`}>
                <span className="log-stage">
                  {ev.stage ? STAGE_LABELS[ev.stage] ?? ev.stage : ""}
                </span>
                <span className="log-msg">{ev.message}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
