// In-browser compute engine (Pyodide) — Web Worker edition.
//
// When VITE_DATA_MODE === "pyodide" there is no FastAPI server: the *real* nebula3d
// reduction pipeline runs locally in the user's browser via Pyodide hosted in a
// Web Worker.  Moving Pyodide off the main thread means the UI stays responsive
// during the ~3-minute pipeline run: progress events stream in as Worker messages
// while React can repaint freely.
//
// Architecture
// ────────────
//  pyodideWorker.ts  — classic Worker; boots Pyodide, dispatches RPC calls.
//  pyodideEngine.ts  — Worker lifecycle + RPC layer + typed public API.
//
//  Main thread ↔ Worker protocol:
//    request:  { id: number, type: string, ...payload }
//    response: { id: number|null, type: "result"|"result_binary"|"error"|..., ...payload }
//
//  Binary slice envelopes ([uint32 hdr_len][JSON hdr][float32 data]) are
//  transferred as Transferable ArrayBuffer (zero-copy Worker → main thread).
//
// See docs/web.md ("In-browser run" / "Architecture") for context.

import type {
  ConsistencyMeta,
  Dataset,
  DeltaPdfMeta,
  Slice,
  SliceHeader,
  VolumeMeta,
} from "./types";

export const PYODIDE_MODE = import.meta.env.VITE_DATA_MODE === "pyodide";

// Pipeline stages — exposed so callers can iterate them for display purposes.
export const ENGINE_STAGES = [
  "rings", "punch", "backfill", "flatten", "pdf", "pdf_check",
] as const;

// ---------------------------------------------------------------------------
// Boot status (observable — drives the boot progress panel in the Configure UI)
// ---------------------------------------------------------------------------
export interface BootStatus {
  phase: "idle" | "runtime" | "packages" | "wheel" | "ready" | "error";
  message: string;
  ready: boolean;
  error?: string;
}

let bootStatus: BootStatus = { phase: "idle", message: "not started", ready: false };
const bootListeners = new Set<(s: BootStatus) => void>();

export function getBootStatus(): BootStatus {
  return bootStatus;
}
export function subscribeBoot(fn: (s: BootStatus) => void): () => void {
  bootListeners.add(fn);
  return () => bootListeners.delete(fn);
}
function setBoot(s: BootStatus): void {
  bootStatus = s;
  for (const fn of bootListeners) fn(s);
}

// ---------------------------------------------------------------------------
// Pipeline progress (fire-and-forget events from the Worker during a run)
// ---------------------------------------------------------------------------
export interface PipelineProgressEvent {
  stage: string;
  status: string;
  fraction: number | null;
  message: string;
}

const progressListeners = new Set<(ev: PipelineProgressEvent) => void>();

function subscribeProgress(fn: (ev: PipelineProgressEvent) => void): () => void {
  progressListeners.add(fn);
  return () => progressListeners.delete(fn);
}

// ---------------------------------------------------------------------------
// Worker lifecycle
// ---------------------------------------------------------------------------
let workerInstance: Worker | null = null;
let bootPromise: Promise<void> | null = null;
let idCounter = 0;
const pending = new Map<number, {
  resolve: (v: unknown) => void;
  reject: (e: Error) => void;
}>();

function getOrCreateWorker(): Worker {
  if (!workerInstance) {
    workerInstance = new Worker(
      new URL("../workers/pyodideWorker.ts", import.meta.url),
      { type: "classic" },
    );
    workerInstance.addEventListener("message", handleWorkerMessage);
    workerInstance.addEventListener("error", (ev: ErrorEvent) => {
      const msg = `Worker error: ${ev.message}`;
      setBoot({ phase: "error", message: msg, ready: false, error: msg });
      rejectAllPending(msg);
      workerInstance = null;
      bootPromise = null;
    });
  }
  return workerInstance;
}

function rejectAllPending(msg: string): void {
  for (const p of pending.values()) p.reject(new Error(msg));
  pending.clear();
}

function handleWorkerMessage(ev: MessageEvent): void {
  const msg = ev.data as {
    id: number | null;
    type: string;
    [k: string]: unknown;
  };

  // Fire-and-forget events (id is null).
  if (msg.id === null) {
    if (msg.type === "boot_status") {
      setBoot({
        phase: msg.phase as BootStatus["phase"],
        message: msg.message as string,
        ready: msg.ready as boolean,
        error: msg.error as string | undefined,
      });
    } else if (msg.type === "progress") {
      const ev: PipelineProgressEvent = {
        stage: msg.stage as string,
        status: msg.status as string,
        fraction: msg.fraction as number | null,
        message: msg.message as string,
      };
      for (const fn of progressListeners) fn(ev);
    }
    return;
  }

  // RPC response.
  const p = pending.get(msg.id);
  if (!p) return;
  pending.delete(msg.id);

  if (msg.type === "error") {
    p.reject(new Error(msg.message as string));
  } else if (msg.type === "result_binary") {
    // Received transferred ArrayBuffer — wrap in Uint8Array for decoding.
    p.resolve(new Uint8Array(msg.payload as ArrayBuffer));
  } else {
    p.resolve(msg.payload);
  }
}

function rpc(
  type: string,
  data: Record<string, unknown> = {},
  transfer: Transferable[] = [],
): Promise<unknown> {
  const id = ++idCounter;
  const w = getOrCreateWorker();
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    w.postMessage({ id, type, ...data }, transfer);
  });
}

// Trigger Pyodide boot (one-time ~15–25 MB WASM download + package install).
// Idempotent: returns the same promise on concurrent calls; retries on error.
export function ensureBooted(): Promise<void> {
  if (!bootPromise) {
    bootPromise = (async () => {
      getOrCreateWorker();
      const base = new URL(
        import.meta.env.BASE_URL ?? "/",
        window.location.origin,
      ).href;
      await rpc("boot", { wheelBase: base });
    })().catch((e: unknown) => {
      bootPromise = null;
      throw e;
    });
  }
  return bootPromise;
}

// Terminate the Worker (stops an in-progress pipeline run) and reset state.
// A new Worker will be created on the next ensureBooted() call.
export function cancelPipeline(): void {
  if (workerInstance) {
    workerInstance.terminate();
    workerInstance = null;
  }
  bootPromise = null;
  rejectAllPending("Pipeline cancelled");
  setBoot({ phase: "idle", message: "not started", ready: false });
}

// ---------------------------------------------------------------------------
// Binary slice envelope decoder (same format the FastAPI backend produces)
// ---------------------------------------------------------------------------
function decodeSliceBytes(u8: Uint8Array): Slice {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const headerLen = dv.getUint32(0, true);
  const header = JSON.parse(
    new TextDecoder().decode(u8.subarray(4, 4 + headerLen)),
  ) as SliceHeader;
  // .slice() to a 0-offset buffer so the Float32Array view is correctly aligned.
  const data = new Float32Array(u8.slice(4 + headerLen).buffer);
  return { header, data };
}

async function jsonCall<T>(method: string, args: unknown[]): Promise<T> {
  await ensureBooted();
  return JSON.parse(
    (await rpc("json_call", { method, args })) as string,
  ) as T;
}

async function sliceCall(method: string, args: unknown[]): Promise<Slice> {
  await ensureBooted();
  const u8 = (await rpc("slice_call", { method, args })) as Uint8Array;
  return decodeSliceBytes(u8);
}

// ---------------------------------------------------------------------------
// Public engine API (mirrors the FastAPI endpoints; same return types)
// ---------------------------------------------------------------------------
export const engine = {
  async loadFile(file: File): Promise<string> {
    await ensureBooted();
    const buffer = await file.arrayBuffer();
    return (await rpc("load_file", { name: file.name, buffer }, [buffer])) as string;
  },

  async loadDemo(): Promise<string> {
    await ensureBooted();
    return (await rpc("load_demo")) as string;
  },

  async runPipeline(opts: {
    paramsJson: string;
    flattenEnabled: boolean;
    force: boolean;
    forceFrom?: string | null;
    onProgress?: (ev: PipelineProgressEvent) => void;
  }): Promise<Dataset[]> {
    await ensureBooted();
    const { paramsJson, flattenEnabled, force, forceFrom, onProgress } = opts;
    const unsub = onProgress ? subscribeProgress(onProgress) : (): void => {};
    try {
      const json = (await rpc("run_pipeline", {
        paramsJson,
        flattenEnabled,
        force,
        forceFrom: forceFrom ?? null,
      })) as string;
      return JSON.parse(json) as Dataset[];
    } finally {
      unsub();
    }
  },

  datasets(): Promise<Dataset[]> {
    return jsonCall<Dataset[]>("datasets_json", []);
  },
  volumeMeta(volumeId: string): Promise<VolumeMeta> {
    return jsonCall<VolumeMeta>("volume_meta_json", [volumeId]);
  },
  volumeSlice(volumeId: string, plane: string, value: number, interp: boolean): Promise<Slice> {
    return sliceCall("volume_slice", [volumeId, plane, value, interp]);
  },
  dpdfMeta(volumeId: string): Promise<DeltaPdfMeta> {
    return jsonCall<DeltaPdfMeta>("dpdf_meta_json", [volumeId]);
  },
  dpdfSlice(volumeId: string, plane: string, value: number): Promise<Slice> {
    return sliceCall("dpdf_slice", [volumeId, plane, value]);
  },
  consistencyMeta(
    datasetId: string,
    qMin?: number,
    qMax?: number,
    rMin?: number,
    rMax?: number,
  ): Promise<ConsistencyMeta> {
    return jsonCall<ConsistencyMeta>("consistency_meta_json", [
      datasetId, qMin ?? null, qMax ?? null, rMin ?? null, rMax ?? null,
    ]);
  },
  consistencySlice(
    datasetId: string,
    panel: string,
    plane: string,
    value: number,
    qMin?: number,
    qMax?: number,
    rMin?: number,
    rMax?: number,
  ): Promise<Slice> {
    return sliceCall("consistency_slice", [
      datasetId, panel, plane, value,
      qMin ?? null, qMax ?? null, rMin ?? null, rMax ?? null,
    ]);
  },
};
