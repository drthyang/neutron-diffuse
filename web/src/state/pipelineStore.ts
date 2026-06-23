// Pipeline configuration + execution state (zustand).
//
// This store owns *both* the configuration form values and the running job so
// they survive navigation between the Configure and Execution pages — the pages
// mount/unmount, the store does not.  The SSE EventSource is held at module
// scope (not in a React effect) for the same reason: switching pages must not
// tear down the live progress stream.

import { create } from "zustand";

import { cancelJob, runPipeline } from "../api/client";
import { cancelPipeline, engine, PYODIDE_MODE } from "../api/pyodideEngine";
import { queryClient } from "../api/queryClient";
import type { JobEvent, StageParamsIn } from "../api/types";
import { useDatasetStore } from "./datasetStore";

// Mirrors nebula3d.pipeline.STAGES (incl. the 6th back-FFT consistency check) so the
// Execution stepper and log show every stage the backend streams progress for.
export const STAGES = [
  "rings", "punch", "backfill", "flatten", "pdf", "pdf_check",
] as const;

export const STAGE_LABELS: Record<string, string> = {
  rings: "Ring removal",
  punch: "Bragg punch",
  backfill: "Backfill",
  flatten: "Flatten",
  pdf: "3D-ΔPDF",
  pdf_check: "Consistency check",
};

// step number (1-based) of each stage, to tie config groups to the stepper
export const STAGE_NO: Record<string, number> = Object.fromEntries(
  STAGES.map((s, i) => [s, i + 1]),
);

export type PunchPlane = "hk" | "hl" | "kl";

// All editable configuration form values.  Strings mirror the raw <input>
// values (empty = "use the backend default"); the run action converts them.
interface PipelineConfig {
  flatten: boolean;
  force: boolean;
  ringModel: string; // "patched" | "parametric"
  ringRadialMode: string; // parametric: "rolling" | "peaks"
  ringNPatches: string;
  ringNFourier: string;
  ringSliceAxis: string;
  ringWidth: string; // parametric: ring width / rolling window (Å⁻¹)
  punchMinI: string;
  punchMethod: string;
  punchMode: string;
  // Q-space resolution floor along a*, b*, c* (Å⁻¹); blank = backend default
  punchQA: string;
  punchQB: string;
  punchQC: string;
  punchFitCovariance: boolean; // fit a tilted 3×3 resolution ellipsoid per peak
  punchMargin: string;
  punchPhiTail: string;
  incidentBeamQA: string;
  incidentBeamQB: string;
  incidentBeamQC: string;
  incidentBeamMargin: string;
  incidentBeamFitCovariance: boolean; // fit a tilted ellipsoid to the direct beam
  punchSliceZoom: number;
  punchSliceContrast: number;
  punchCutH: number;
  punchCutK: number;
  punchCutL: number;
  backfillMethod: string;
  flattenEstimator: string;
  pdfApod: string;
  pdfQMin: string;
  pdfQMax: string;
}

interface PipelineState extends PipelineConfig {
  // job execution
  jobId: string | null;
  running: boolean;
  events: JobEvent[];
  terminal: string | null;
  // actions
  patch: (p: Partial<PipelineConfig>) => void;
  run: () => Promise<void>;
  cancel: () => Promise<void>;
}

// Live progress stream — module scope so it is independent of any component's
// lifecycle.  Closed on terminal events / errors and replaced on each run.
let es: EventSource | null = null;

function closeStream() {
  es?.close();
  es = null;
}

export const usePipelineStore = create<PipelineState>((set, get) => ({
  flatten: true,
  force: false,
  ringModel: "parametric",
  ringRadialMode: "rolling",
  ringNPatches: "",
  ringNFourier: "8",
  ringSliceAxis: "H",
  ringWidth: "",
  punchMinI: "",
  punchMethod: "ellipsoid",
  punchMode: "",
  punchQA: "",
  punchQB: "",
  punchQC: "",
  punchFitCovariance: false,
  punchMargin: "",
  punchPhiTail: "",
  incidentBeamQA: "",
  incidentBeamQB: "",
  incidentBeamQC: "",
  incidentBeamMargin: "",
  incidentBeamFitCovariance: false,
  punchSliceZoom: 1,
  punchSliceContrast: 1.35,
  punchCutH: 0,
  punchCutK: 0,
  punchCutL: 0,
  backfillMethod: "",
  flattenEstimator: "",
  pdfApod: "",
  pdfQMin: "",
  pdfQMax: "",

  jobId: null,
  running: false,
  events: [],
  terminal: null,

  patch: (p) => set(p),

  run: async () => {
    const s = get();
    const datasetId = useDatasetStore.getState().datasetId ?? "";
    closeStream();
    set({ events: [], terminal: null, running: true, jobId: null });

    const params = formToParams(s);

    if (PYODIDE_MODE) {
      await runInBrowser(params, s.flatten, s.force, set, get);
      return;
    }

    try {
      const job = await runPipeline({
        dataset_id: datasetId,
        flatten_enabled: s.flatten,
        force: s.force,
        params,
      });
      set({ jobId: job.id });

      es = new EventSource(`/api/pipeline/jobs/${job.id}/events`);
      es.onmessage = (e) => {
        const ev = JSON.parse(e.data) as JobEvent;
        if (["done", "error", "cancelled"].includes(ev.type)) {
          set({ terminal: ev.type, running: false });
          closeStream();
        } else {
          set({ events: [...get().events, ev] });
        }
      };
      es.onerror = () => {
        closeStream();
        set({ running: false });
      };
    } catch (e) {
      set({
        terminal: "error",
        running: false,
        events: [
          { type: "progress", status: "error", message: (e as Error).message },
        ],
      });
    }
  },

  cancel: async () => {
    if (PYODIDE_MODE) {
      cancelPipeline();
      set({ terminal: "cancelled", running: false });
      return;
    }
    const { jobId } = get();
    if (jobId) await cancelJob(jobId).catch(() => undefined);
  },
}));

// Convert the editable form values into the curated StageParamsIn the pipeline
// accepts (empty fields stay unset → the backend/bridge default is used).
function formToParams(s: PipelineConfig): StageParamsIn {
  const params: StageParamsIn = {};
  if (s.ringModel) params.rings_model = s.ringModel;
  if (s.ringNPatches) params.rings_n_patches = Number(s.ringNPatches);
  if (s.ringNFourier) params.rings_n_fourier = Number(s.ringNFourier);
  if (s.ringSliceAxis) params.rings_slice_axis = s.ringSliceAxis;
  if (s.ringModel === "parametric") {
    params.rings_radial_mode = s.ringRadialMode;
    if (s.ringWidth) params.rings_ring_width = Number(s.ringWidth);
  }
  if (s.punchMinI) params.punch_min_intensity = Number(s.punchMinI);
  if (s.punchMode) params.punch_mode = s.punchMode;
  if (s.punchMargin) params.punch_margin = Number(s.punchMargin);
  if (s.punchPhiTail) params.punch_phi_tail_hkl = Number(s.punchPhiTail);
  // Q-space is the web UI's punch frame.
  params.punch_frame = "q";
  if (s.punchQA) params.punch_q_radius_a = Number(s.punchQA);
  if (s.punchQB) params.punch_q_radius_b = Number(s.punchQB);
  if (s.punchQC) params.punch_q_radius_c = Number(s.punchQC);
  if (s.incidentBeamQA) params.incident_beam_q_radius_a = Number(s.incidentBeamQA);
  if (s.incidentBeamQB) params.incident_beam_q_radius_b = Number(s.incidentBeamQB);
  if (s.incidentBeamQC) params.incident_beam_q_radius_c = Number(s.incidentBeamQC);
  if (s.incidentBeamMargin) params.incident_beam_q_margin = Number(s.incidentBeamMargin);
  if (s.punchFitCovariance) params.punch_fit_covariance = true;
  if (s.incidentBeamFitCovariance) params.incident_beam_fit_covariance = true;
  if (s.backfillMethod) params.backfill_method = s.backfillMethod;
  if (s.flattenEstimator) params.flatten_estimator = s.flattenEstimator;
  if (s.pdfApod) params.pdf_apodization = s.pdfApod;
  if (s.pdfQMin || s.pdfQMax) {
    params.pdf_q_min = s.pdfQMin ? Number(s.pdfQMin) : 0;
    if (s.pdfQMax) params.pdf_q_max = Number(s.pdfQMax);
  }
  return params;
}

type Setter = (p: Partial<PipelineState>) => void;
type Getter = () => PipelineState;

// Drive the pipeline locally via Pyodide (Worker).  Boot progress appears in
// the Configure page's dedicated boot panel; stage progress streams into the
// Execution log.  The Worker is never blocked from the main thread's view, so
// the UI repaints freely throughout.
async function runInBrowser(
  params: StageParamsIn,
  flatten: boolean,
  force: boolean,
  set: Setter,
  get: Getter,
): Promise<void> {
  const log = (ev: JobEvent) => set({ events: [...get().events, ev] });
  try {
    await engine.runPipeline({
      paramsJson: JSON.stringify(params),
      flattenEnabled: flatten,
      force,
      onProgress: (ev) =>
        log({
          type: "progress",
          stage: ev.stage,
          status: ev.status,
          fraction: ev.fraction ?? null,
          message: ev.message,
        }),
    });
    await queryClient.invalidateQueries();
    set({ terminal: "done", running: false });
  } catch (e) {
    log({ type: "progress", status: "error", message: (e as Error).message });
    set({ terminal: "error", running: false });
  }
}
