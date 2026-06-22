// In-browser compute engine (Pyodide) for the backend-less GitHub Pages build.
//
// When VITE_DATA_MODE === "pyodide" there is no FastAPI server: the *real* ndiff
// reduction pipeline runs locally in the user's browser via Pyodide (CPython +
// numpy/scipy/h5py compiled to WebAssembly), driven by the ndiff.webbridge module
// in the installed wheel.  This module owns the singleton runtime — lazy boot,
// wheel install, and typed wrappers that return the same Slice / Meta shapes the
// API client would, so the React viewers are unchanged apart from the source of
// the bytes.  Nothing is uploaded; the user's file and all computation stay on
// their machine.
//
// See docs/browser-hosted-app-plan.md (P1–P4) for the architecture and rationale.

import type {
  ConsistencyMeta,
  Dataset,
  DeltaPdfMeta,
  Slice,
  SliceHeader,
  VolumeMeta,
} from "./types";

export const PYODIDE_MODE = import.meta.env.VITE_DATA_MODE === "pyodide";

const PYODIDE_VERSION = "0.26.2";
const PYODIDE_INDEX = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const WHEEL = "neutron_diffuse-0.2.0-py3-none-any.whl";
const BASE = import.meta.env.BASE_URL ?? "/";

// Pipeline stages, mirroring ndiff.pipeline.STAGES — driven one at a time so the
// UI can repaint per-stage progress between the (main-thread-blocking) calls.
export const ENGINE_STAGES = [
  "rings", "punch", "backfill", "flatten", "pdf", "pdf_check",
] as const;

// ---------------------------------------------------------------------------
// Minimal Pyodide typings (the CDN module ships no types).
// ---------------------------------------------------------------------------
interface PyProxy {
  toJs(opts?: { create_proxies?: boolean }): unknown;
  destroy(): void;
  // Bridge methods are exposed as callable properties on the module proxy.
  [k: string]: unknown;
}
interface PyodideAPI {
  loadPackage(names: string[]): Promise<void>;
  runPythonAsync(code: string): Promise<unknown>;
  pyimport(name: string): PyProxy;
  FS: { writeFile(path: string, data: Uint8Array): void; mkdirTree(path: string): void };
  globals: { set(k: string, v: unknown): void; delete(k: string): void };
}
declare global {
  interface Window {
    loadPyodide?: (opts: { indexURL: string }) => Promise<PyodideAPI>;
  }
}

// ---------------------------------------------------------------------------
// Boot progress (observable so the UI can show the one-time WASM download).
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
// Lazy boot (singleton) — inject the CDN runtime, load packages, install wheel.
// ---------------------------------------------------------------------------
let pyodide: PyodideAPI | null = null;
let bridge: PyProxy | null = null;
let bootPromise: Promise<PyProxy> | null = null;

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) return resolve();
    const el = document.createElement("script");
    el.src = src;
    el.onload = () => resolve();
    el.onerror = () => reject(new Error(`failed to load ${src}`));
    document.head.appendChild(el);
  });
}

async function boot(): Promise<PyProxy> {
  try {
    setBoot({ phase: "runtime", message: "Downloading Python runtime (~10 MB)…", ready: false });
    await loadScript(`${PYODIDE_INDEX}pyodide.js`);
    if (!window.loadPyodide) throw new Error("Pyodide failed to register on window");
    pyodide = await window.loadPyodide({ indexURL: PYODIDE_INDEX });

    setBoot({ phase: "packages", message: "Loading numpy, scipy, h5py, matplotlib…", ready: false });
    await pyodide.loadPackage(["numpy", "scipy", "h5py", "matplotlib", "micropip"]);

    setBoot({ phase: "wheel", message: "Installing the ndiff reduction package…", ready: false });
    const wheelUrl = new URL(`${BASE}wheels/${WHEEL}`, window.location.origin).href;
    pyodide.globals.set("_ndiff_wheel_url", wheelUrl);
    // deps=False: numpy/scipy/h5py/matplotlib are already provided by loadPackage;
    // skip micropip's version check (Pyodide ships matplotlib < the wheel's pin).
    await pyodide.runPythonAsync(
      "import micropip\n" +
      "await micropip.install(_ndiff_wheel_url, deps=False)\n",
    );
    pyodide.globals.delete("_ndiff_wheel_url");

    bridge = pyodide.pyimport("ndiff.webbridge");
    (bridge.setup as () => unknown)();

    setBoot({ phase: "ready", message: "Ready — compute runs locally in your browser.", ready: true });
    return bridge;
  } catch (e) {
    const error = (e as Error).message || String(e);
    setBoot({ phase: "error", message: `Boot failed: ${error}`, ready: false, error });
    bootPromise = null; // allow a retry
    throw e;
  }
}

export function ensureBooted(): Promise<PyProxy> {
  if (!bootPromise) bootPromise = boot();
  return bootPromise;
}

// ---------------------------------------------------------------------------
// Result decoding
// ---------------------------------------------------------------------------
// Decode the binary slice envelope: [uint32 LE header_len][JSON header][float32].
function decodeSliceBytes(u8: Uint8Array): Slice {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const headerLen = dv.getUint32(0, true);
  const header = JSON.parse(
    new TextDecoder().decode(u8.subarray(4, 4 + headerLen)),
  ) as SliceHeader;
  // .slice() copies to a fresh 0-offset buffer so the Float32 view is aligned.
  const data = new Float32Array(u8.slice(4 + headerLen).buffer);
  return { header, data };
}

// Call a bridge method that returns Python `bytes`, decode to a Slice, and free
// the proxy (WASM-side bytes must be released explicitly).
async function sliceCall(method: string, args: unknown[]): Promise<Slice> {
  const b = await ensureBooted();
  const proxy = (b[method] as (...a: unknown[]) => PyProxy)(...args);
  try {
    return decodeSliceBytes(proxy.toJs() as Uint8Array);
  } finally {
    proxy.destroy();
  }
}

// Call a bridge method that returns a JSON string (auto-converted to a JS string).
async function jsonCall<T>(method: string, args: unknown[]): Promise<T> {
  const b = await ensureBooted();
  const s = (b[method] as (...a: unknown[]) => string)(...args);
  return JSON.parse(s) as T;
}

// ---------------------------------------------------------------------------
// Public engine API (mirrors the FastAPI endpoints)
// ---------------------------------------------------------------------------
export const engine = {
  /** Write the user's uploaded file into Pyodide and register it; returns the id. */
  async loadFile(file: File): Promise<string> {
    const b = await ensureBooted();
    const bytes = new Uint8Array(await file.arrayBuffer());
    pyodide!.FS.mkdirTree("/uploads");
    const tmp = `/uploads/${Date.now()}`;
    pyodide!.FS.writeFile(tmp, bytes);
    return (b.load_input as (n: string, p: string) => string)(file.name, tmp);
  },

  /** Generate a small synthetic volume in the workspace (demo / smoke test). */
  async loadDemo(): Promise<string> {
    const b = await ensureBooted();
    return (b.make_demo_input as (n: number) => string)(24);
  },

  /** Drive the pipeline one stage at a time, streaming per-stage progress. */
  async runPipeline(opts: {
    paramsJson: string;
    flattenEnabled: boolean;
    force: boolean;
    forceFrom?: string | null;
    onProgress?: (ev: {
      stage: string;
      status: string;
      fraction: number | null;
      message: string;
    }) => void;
  }): Promise<Dataset[]> {
    const b = await ensureBooted();
    const { paramsJson, flattenEnabled, force, forceFrom, onProgress } = opts;
    for (const stage of ENGINE_STAGES) {
      const cb = (s: string, status: string, fraction: number | null, message: string) =>
        onProgress?.({ stage: s, status, fraction: fraction ?? null, message });
      (b.run as (...a: unknown[]) => string)(
        stage, paramsJson, flattenEnabled, force, forceFrom ?? null, cb,
      );
      // Yield to the event loop so React flushes the just-emitted progress.
      await new Promise((r) => setTimeout(r, 0));
    }
    return engine.datasets();
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
    datasetId: string, qMin?: number, qMax?: number, rMin?: number, rMax?: number,
  ): Promise<ConsistencyMeta> {
    return jsonCall<ConsistencyMeta>("consistency_meta_json",
      [datasetId, qMin ?? null, qMax ?? null, rMin ?? null, rMax ?? null]);
  },
  consistencySlice(
    datasetId: string, panel: string, plane: string, value: number,
    qMin?: number, qMax?: number, rMin?: number, rMax?: number,
  ): Promise<Slice> {
    return sliceCall("consistency_slice",
      [datasetId, panel, plane, value, qMin ?? null, qMax ?? null, rMin ?? null, rMax ?? null]);
  },
};
