// Pipeline execution page — a Siri/HomePod-style glowing orb is the focal point,
// flanked by the stage stepper and a condensed live log for the job started from
// the Configure page.  Job state (and the SSE stream feeding it) lives in the
// pipeline store, so this page reflects an in-flight job even if it was mounted
// after navigating away and back.

import { useEffect, useMemo, useRef } from "react";
import { useShallow } from "zustand/react/shallow";

import type { JobEvent } from "../api/types";
import { STAGES, STAGE_LABELS, usePipelineStore } from "../state/pipelineStore";

type Phase = "idle" | "running" | "done" | "cancelled" | "error";

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

// Collapse the raw event stream into one line per stage-phase: consecutive
// updates from the same stage (e.g. a stage streaming "running 12% / 13% / …")
// fold into a single line whose text is the latest update, so the log reads as a
// short, scannable history instead of a wall of repeated progress messages.
function condense(events: JobEvent[]): JobEvent[] {
  const out: JobEvent[] = [];
  for (const ev of events) {
    const prev = out[out.length - 1];
    if (prev && ev.stage && prev.stage === ev.stage) out[out.length - 1] = ev;
    else out.push(ev);
  }
  return out;
}

function logText(ev: JobEvent): string {
  if (ev.message) return ev.message;
  if (ev.status === "progress" && ev.fraction != null)
    return `running · ${Math.round(ev.fraction * 100)}%`;
  return ev.status ?? ev.type;
}

// The glowing orb.  Pure CSS plasma (animated, blurred, screen-blended colour
// blobs + a breathing halo) themed per phase; an SVG ring around it tracks
// overall stage completion.
function Orb({ phase, progress }: { phase: Phase; progress: number }) {
  const r = 78;
  const circumference = 2 * Math.PI * r;
  const p = Math.max(0, Math.min(1, progress));
  return (
    <div className={`orb-stage orb-${phase}`} aria-hidden>
      <div className="orb-glow" />
      <div className="orb-core">
        <span className="orb-blob orb-blob-1" />
        <span className="orb-blob orb-blob-2" />
        <span className="orb-blob orb-blob-3" />
        <span className="orb-sheen" />
      </div>
      <svg className="orb-ring" viewBox="0 0 168 168">
        <circle className="orb-ring-track" cx="84" cy="84" r={r} />
        <circle
          className="orb-ring-fill"
          cx="84"
          cy="84"
          r={r}
          style={{
            strokeDasharray: circumference,
            strokeDashoffset: circumference * (1 - p),
          }}
        />
      </svg>
    </div>
  );
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

  const condensed = useMemo(() => condense(events), [events]);

  useEffect(() => {
    logRef.current?.scrollTo(0, logRef.current.scrollHeight);
  }, [condensed]);

  // latest status (+ fraction) per stage from the event stream
  const stageInfo: Record<string, StageInfo> = {};
  for (const ev of events) {
    if (ev.stage && ev.status) {
      stageInfo[ev.stage] = { status: ev.status, fraction: ev.fraction ?? null };
    }
  }

  const visibleStages = STAGES.filter((s) => flatten || s !== "flatten");

  // Phase drives every visual: orb colour/tempo, headline, ring.
  const started = running || terminal != null || events.length > 0;
  let phase: Phase = "idle";
  if (terminal === "done") phase = "done";
  else if (terminal === "cancelled") phase = "cancelled";
  else if (terminal === "error") phase = "error";
  else if (running || (started && terminal == null)) phase = "running";

  // Currently-running stage + smooth overall completion for the ring.
  let activeStage: string | null = null;
  let activeFrac: number | null = null;
  let doneCount = 0;
  visibleStages.forEach((s) => {
    const info = stageInfo[s];
    if (!info) return;
    if (info.status === "start" || info.status === "progress") {
      activeStage = s;
      activeFrac = info.fraction;
    } else if (info.status === "done" || info.status === "skip") {
      doneCount += 1;
    }
  });
  const n = visibleStages.length || 1;
  const overall =
    phase === "done"
      ? 1
      : (doneCount + (activeStage && activeFrac != null ? activeFrac : 0)) / n;

  // Headline + subline per phase.
  const activeNo = activeStage ? visibleStages.indexOf(activeStage) + 1 : 0;
  const pct = activeFrac != null ? `${Math.round(activeFrac * 100)}%` : null;
  const lastError = [...events].reverse().find((e) => e.status === "error")?.message;

  let title: string;
  let sub: string;
  switch (phase) {
    case "running":
      title = activeStage ? STAGE_LABELS[activeStage] : "Starting…";
      sub = activeStage
        ? `Step ${activeNo} of ${n}${pct ? ` · ${pct}` : ""}`
        : "Spinning up the pipeline";
      break;
    case "done":
      title = "Complete";
      sub = "Open the 3D-ΔPDF tab to view the result.";
      break;
    case "cancelled":
      title = "Cancelled";
      sub = "The job was stopped before finishing.";
      break;
    case "error":
      title = "Failed";
      sub = lastError ?? "See the log below for details.";
      break;
    default:
      title = "Ready";
      sub = "Set parameters on the Configure page and press Run.";
  }

  return (
    <div className="exec-page">
      <div className="exec-grid">
        <div className={`card exec-status-card phase-${phase}`}>
          <div className="status-hero">
            <Orb phase={phase} progress={overall} />
            <div className="hero-text">
              <h2 className="hero-title">{title}</h2>
              <p className="hero-sub">{sub}</p>
            </div>
          </div>

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

          <div className="status-foot">
            <span className="hero-pct">{Math.round(overall * 100)}%</span>
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

        <div className="card exec-log-card">
          <div className="card-head">
            <h3>Log</h3>
            <span className="log-count">
              {jobId ? `job ${jobId.slice(0, 8)}` : ""}
              {jobId && condensed.length > 0 ? " · " : ""}
              {condensed.length > 0 ? `${condensed.length} steps` : ""}
            </span>
          </div>
          <div className="log" ref={logRef}>
            {condensed.length === 0 && !running && (
              <div className="log-empty">
                Stages with existing outputs are skipped — enable “force” on the
                Configure page to recompute them.
              </div>
            )}
            {condensed.map((ev, i) => (
              <div key={i} className={`log-line ${ev.status ?? ""}`}>
                <span className="log-glyph" />
                <span className="log-stage">
                  {ev.stage ? STAGE_LABELS[ev.stage] ?? ev.stage : ""}
                </span>
                <span className="log-msg">{logText(ev)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
