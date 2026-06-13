// Pipeline configuration + execution state (zustand).
//
// This store owns *both* the configuration form values and the running job so
// they survive navigation between the Configure and Execution pages — the pages
// mount/unmount, the store does not.  The SSE EventSource is held at module
// scope (not in a React effect) for the same reason: switching pages must not
// tear down the live progress stream.

import { create } from "zustand";

import { cancelJob, runPipeline } from "../api/client";
import type { JobEvent, StageParamsIn } from "../api/types";

export const STAGES = ["rings", "punch", "backfill", "flatten", "pdf"] as const;

export const STAGE_LABELS: Record<string, string> = {
  rings: "Ring removal",
  punch: "Bragg punch",
  backfill: "Backfill",
  flatten: "Flatten",
  pdf: "3D-ΔPDF",
};

// step number (1-based) of each stage, to tie config groups to the stepper
export const STAGE_NO: Record<string, number> = Object.fromEntries(
  STAGES.map((s, i) => [s, i + 1]),
);

export type PunchPlane = "hk" | "hl" | "kl";

// All editable configuration form values.  Strings mirror the raw <input>
// values (empty = "use the backend default"); the run action converts them.
interface PipelineConfig {
  datasetId: string;
  flatten: boolean;
  force: boolean;
  ringNPatches: string;
  ringNFourier: string;
  ringSliceAxis: string;
  punchMinI: string;
  punchMethod: string;
  punchMode: string;
  punchRH: string;
  punchRK: string;
  punchRL: string;
  punchMargin: string;
  punchPhiTail: string;
  punchPlane: PunchPlane;
  backfillMethod: string;
  flattenEstimator: string;
  pdfApod: string;
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
  datasetId: "",
  flatten: true,
  force: false,
  ringNPatches: "",
  ringNFourier: "",
  ringSliceAxis: "H",
  punchMinI: "",
  punchMethod: "ellipsoid",
  punchMode: "",
  punchRH: "",
  punchRK: "",
  punchRL: "",
  punchMargin: "",
  punchPhiTail: "",
  punchPlane: "hl",
  backfillMethod: "",
  flattenEstimator: "",
  pdfApod: "",

  jobId: null,
  running: false,
  events: [],
  terminal: null,

  patch: (p) => set(p),

  run: async () => {
    const s = get();
    closeStream();
    set({ events: [], terminal: null, running: true, jobId: null });

    const params: StageParamsIn = {};
    if (s.ringNPatches) params.rings_n_patches = Number(s.ringNPatches);
    if (s.ringNFourier) params.rings_n_fourier = Number(s.ringNFourier);
    if (s.ringSliceAxis) params.rings_slice_axis = s.ringSliceAxis;
    if (s.punchMinI) params.punch_min_intensity = Number(s.punchMinI);
    if (s.punchMode) params.punch_mode = s.punchMode;
    if (s.punchRH) params.punch_radius_h = Number(s.punchRH);
    if (s.punchRK) params.punch_radius_k = Number(s.punchRK);
    if (s.punchRL) params.punch_radius_l = Number(s.punchRL);
    if (s.punchMargin) params.punch_margin = Number(s.punchMargin);
    if (s.punchPhiTail) params.punch_phi_tail_hkl = Number(s.punchPhiTail);
    if (s.backfillMethod) params.backfill_method = s.backfillMethod;
    if (s.flattenEstimator) params.flatten_estimator = s.flattenEstimator;
    if (s.pdfApod) params.pdf_apodization = s.pdfApod;

    try {
      const job = await runPipeline({
        dataset_id: s.datasetId,
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
    const { jobId } = get();
    if (jobId) await cancelJob(jobId).catch(() => undefined);
  },
}));
