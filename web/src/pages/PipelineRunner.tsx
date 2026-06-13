// Pipeline runner — drives ndiff.pipeline.run_pipeline as a background job and
// streams progress over SSE.  Configuration form on the left; stage stepper and
// live log on the right.

import { useEffect, useMemo, useRef, useState } from "react";

import { cancelJob, runPipeline } from "../api/client";
import { useDatasets } from "../api/hooks";
import type { JobEvent, StageParamsIn } from "../api/types";
import { Field, Switch } from "../components/ui";

const STAGES = ["rings", "punch", "backfill", "flatten", "pdf"];
const STAGE_LABELS: Record<string, string> = {
  rings: "Ring removal",
  punch: "Bragg punch",
  backfill: "Backfill",
  flatten: "Flatten",
  pdf: "3D-ΔPDF",
};

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

export function PipelineRunner() {
  const datasetsQ = useDatasets();
  const datasets = useMemo(() => datasetsQ.data ?? [], [datasetsQ.data]);

  const [datasetId, setDatasetId] = useState("");
  const [flatten, setFlatten] = useState(true);
  const [force, setForce] = useState(false);
  const [ringNPatches, setRingNPatches] = useState("");
  const [ringNFourier, setRingNFourier] = useState("");
  const [punchMinI, setPunchMinI] = useState("");
  const [backfillMethod, setBackfillMethod] = useState("");
  const [flattenEstimator, setFlattenEstimator] = useState("");
  const [pdfApod, setPdfApod] = useState("");

  const [jobId, setJobId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [terminal, setTerminal] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!datasetId && datasets.length) setDatasetId(datasets[0].id);
  }, [datasetId, datasets]);

  // stream job progress over SSE
  useEffect(() => {
    if (!jobId) return;
    const es = new EventSource(`/api/pipeline/jobs/${jobId}/events`);
    es.onmessage = (e) => {
      const ev = JSON.parse(e.data) as JobEvent;
      if (["done", "error", "cancelled"].includes(ev.type)) {
        setTerminal(ev.type);
        setRunning(false);
        es.close();
      } else {
        setEvents((prev) => [...prev, ev]);
      }
    };
    es.onerror = () => {
      es.close();
      setRunning(false);
    };
    return () => es.close();
  }, [jobId]);

  useEffect(() => {
    logRef.current?.scrollTo(0, logRef.current.scrollHeight);
  }, [events]);

  const onRun = async () => {
    setEvents([]);
    setTerminal(null);
    setRunning(true);
    const params: StageParamsIn = {};
    if (ringNPatches) params.rings_n_patches = Number(ringNPatches);
    if (ringNFourier) params.rings_n_fourier = Number(ringNFourier);
    if (punchMinI) params.punch_min_intensity = Number(punchMinI);
    if (backfillMethod) params.backfill_method = backfillMethod;
    if (flattenEstimator) params.flatten_estimator = flattenEstimator;
    if (pdfApod) params.pdf_apodization = pdfApod;
    try {
      const job = await runPipeline({
        dataset_id: datasetId,
        flatten_enabled: flatten,
        force,
        params,
      });
      setJobId(job.id);
    } catch (e) {
      setTerminal("error");
      setRunning(false);
      setEvents([{ type: "progress", status: "error", message: (e as Error).message }]);
    }
  };

  const onCancel = async () => {
    if (jobId) await cancelJob(jobId).catch(() => undefined);
  };

  // latest status (+ fraction) per stage from the event stream
  const stageInfo: Record<string, StageInfo> = {};
  for (const ev of events) {
    if (ev.stage && ev.status) {
      stageInfo[ev.stage] = { status: ev.status, fraction: ev.fraction ?? null };
    }
  }

  return (
    <div className="pipeline-layout">
      {/* ------------------------------------------------ configuration */}
      <div className="card">
        <div className="card-head">
          <h3>Configuration</h3>
        </div>
        <div className="card-body">
          <div className="config-section">
            <span className="config-section-title">Input</span>
            <Field label="Dataset">
              <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
                {datasets.map((d) => (
                  <option key={d.id} value={d.id} title={d.raw_name}>
                    {d.temperature ?? d.stem}
                  </option>
                ))}
              </select>
            </Field>
          </div>

          <div className="config-section">
            <span className="config-section-title">Ring removal</span>
            <div className="config-grid">
              <Field label="Patches (n)">
                <input
                  type="number"
                  min="4"
                  step="1"
                  placeholder="36"
                  value={ringNPatches}
                  title="Number of azimuthal patches the powder rings are divided into"
                  onChange={(e) => setRingNPatches(e.target.value)}
                />
              </Field>
              <Field label="Fourier order">
                <input
                  type="number"
                  min="0"
                  step="1"
                  placeholder="6"
                  value={ringNFourier}
                  title="Fourier order of the azimuthal texture T(φ) modelling the Al powder rings"
                  onChange={(e) => setRingNFourier(e.target.value)}
                />
              </Field>
            </div>
          </div>

          <div className="config-section">
            <span className="config-section-title">Stage parameters</span>
            <div className="config-grid">
              <Field label="Punch min I">
                <input
                  type="number"
                  step="0.1"
                  placeholder="0.8"
                  value={punchMinI}
                  onChange={(e) => setPunchMinI(e.target.value)}
                />
              </Field>
              <Field label="Backfill">
                <select
                  value={backfillMethod}
                  onChange={(e) => setBackfillMethod(e.target.value)}
                >
                  <option value="">q_shell (default)</option>
                  <option value="local">local</option>
                  <option value="tv">tv</option>
                  <option value="symmetry+tv">symmetry+tv</option>
                </select>
              </Field>
              <Field label="Flatten estimator">
                <select
                  value={flattenEstimator}
                  onChange={(e) => setFlattenEstimator(e.target.value)}
                >
                  <option value="">floor (default)</option>
                  <option value="median">median</option>
                  <option value="mode">mode</option>
                  <option value="snip">snip</option>
                </select>
              </Field>
              <Field label="ΔPDF apodization">
                <select value={pdfApod} onChange={(e) => setPdfApod(e.target.value)}>
                  <option value="">gaussian (default)</option>
                  <option value="hann">hann</option>
                  <option value="none">none</option>
                </select>
              </Field>
            </div>
          </div>

          <div className="config-section">
            <span className="config-section-title">Options</span>
            <Switch label="Flatten stage" checked={flatten} onChange={setFlatten} />
            <Switch
              label="Force — recompute existing outputs"
              checked={force}
              onChange={setForce}
            />
          </div>

          <div className="config-actions">
            <button
              type="button"
              className="btn btn-primary"
              onClick={onRun}
              disabled={running || !datasetId}
            >
              {running && <span className="spin" />}
              {running ? "Running…" : "Run pipeline"}
            </button>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={onCancel}
              disabled={!running}
            >
              Cancel
            </button>
          </div>
        </div>
      </div>

      {/* --------------------------------------------------- execution */}
      <div className="exec-col">
        <div className="card">
          <div className="card-head">
            <h3>Execution</h3>
            {running && <span className="spin" />}
          </div>
          <div className="card-body">
            {terminal === "done" && (
              <div className="banner ok">
                Job finished — open the 3D-ΔPDF tab to view the result.
              </div>
            )}
            {terminal === "cancelled" && (
              <div className="banner warn">Job cancelled.</div>
            )}
            {terminal === "error" && (
              <div className="banner err">Job failed — see the log below.</div>
            )}

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

        <div className="card">
          <div className="card-head">
            <h3>Log</h3>
            <span className="log-count">
              {events.length > 0 ? `${events.length} events` : ""}
            </span>
          </div>
          <div className="log" ref={logRef}>
            {events.length === 0 && !running && (
              <div className="log-empty">
                Pick a dataset and press Run. Stages with existing outputs are
                skipped — enable “force” to recompute them.
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
