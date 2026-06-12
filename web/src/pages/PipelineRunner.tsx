// Pipeline runner — drives ndiff.pipeline.run_pipeline as a background job and
// streams progress over SSE.  Turns the long env-var command lines into a form.

import { useEffect, useMemo, useRef, useState } from "react";

import { cancelJob, runPipeline } from "../api/client";
import { useDatasets } from "../api/hooks";
import type { JobEvent, StageParamsIn } from "../api/types";

const STAGES = ["rings", "punch", "backfill", "flatten", "pdf"];
const STAGE_LABELS: Record<string, string> = {
  rings: "ring removal",
  punch: "Bragg punch",
  backfill: "backfill",
  flatten: "flatten",
  pdf: "3D-ΔPDF",
};

export function PipelineRunner() {
  const datasetsQ = useDatasets();
  const datasets = useMemo(() => datasetsQ.data ?? [], [datasetsQ.data]);

  const [datasetId, setDatasetId] = useState("");
  const [flatten, setFlatten] = useState(true);
  const [force, setForce] = useState(false);
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

  // latest status per stage from the event stream
  const stageStatus: Record<string, string> = {};
  for (const ev of events) {
    if (ev.stage && ev.status) stageStatus[ev.stage] = ev.status;
  }

  return (
    <div className="viewer">
      <div className="controls">
        <label>
          dataset
          <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.temperature ?? d.stem}
              </option>
            ))}
          </select>
        </label>
        <label>
          punch min I
          <input
            type="number"
            step="0.1"
            placeholder="0.8"
            value={punchMinI}
            onChange={(e) => setPunchMinI(e.target.value)}
          />
        </label>
        <label>
          backfill
          <select value={backfillMethod} onChange={(e) => setBackfillMethod(e.target.value)}>
            <option value="">q_shell (default)</option>
            <option value="local">local</option>
            <option value="tv">tv</option>
            <option value="symmetry+tv">symmetry+tv</option>
          </select>
        </label>
        <label>
          flatten est.
          <select
            value={flattenEstimator}
            onChange={(e) => setFlattenEstimator(e.target.value)}
          >
            <option value="">floor (default)</option>
            <option value="median">median</option>
            <option value="mode">mode</option>
            <option value="snip">snip</option>
          </select>
        </label>
        <label>
          ΔPDF apod.
          <select value={pdfApod} onChange={(e) => setPdfApod(e.target.value)}>
            <option value="">gaussian (default)</option>
            <option value="hann">hann</option>
            <option value="none">none</option>
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={flatten} onChange={(e) => setFlatten(e.target.checked)} />
          flatten
        </label>
        <label className="check">
          <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
          force
        </label>
        <div className="run-buttons">
          <button className="run" onClick={onRun} disabled={running || !datasetId}>
            {running ? "running…" : "Run pipeline"}
          </button>
          <button className="cancel" onClick={onCancel} disabled={!running}>
            Cancel
          </button>
        </div>
      </div>

      <div className="stage-chips">
        {STAGES.map((s) => {
          const st = !flatten && s === "flatten" ? "off" : stageStatus[s] ?? "";
          return (
            <div key={s} className={`stage-chip ${st}`}>
              <span className="dot" />
              {STAGE_LABELS[s]}
              {st && <span className="st"> · {st}</span>}
            </div>
          );
        })}
      </div>

      {terminal && (
        <div className={`status ${terminal === "done" ? "" : "error"}`}>
          job {terminal}
          {terminal === "done" && " — open the ΔPDF tab to view the result."}
        </div>
      )}

      <div className="log" ref={logRef}>
        {events.length === 0 && !running && (
          <div className="log-empty">
            Pick a dataset and press Run. Stages with existing outputs are skipped
            (tick “force” to recompute).
          </div>
        )}
        {events.map((ev, i) => (
          <div key={i} className={`log-line ${ev.status ?? ""}`}>
            <span className="log-stage">{ev.stage ? STAGE_LABELS[ev.stage] ?? ev.stage : ""}</span>
            <span className="log-msg">{ev.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
