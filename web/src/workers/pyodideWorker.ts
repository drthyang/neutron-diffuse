// Web Worker: hosts the Pyodide runtime so the nebula3d pipeline runs off the
// main thread.  Built as a classic worker (importScripts is available), driven
// by message-passing from the main thread.
//
// Message protocol
// ─────────────────
//  Main → Worker   { id: number, type: string, ...payload }
//  Worker → Main   { id: number|null, type: string, ...payload }
//
//  id: null  — fire-and-forget events (boot_status, pipeline progress)
//  id: number — RPC responses (result / result_binary / error)
//
// Binary results (slice envelopes) are posted with an ArrayBuffer in the
// transfer list (zero-copy from Worker to main thread).
//
// See docs/web.md ("In-browser run" / "Architecture") for the rationale.

// importScripts is only available in classic workers; declare so TS is happy.
declare function importScripts(...urls: string[]): void;

const PYODIDE_VERSION = "0.26.2";
const PYODIDE_INDEX = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const STAGES = ["rings", "punch", "backfill", "flatten", "pdf", "pdf_check"] as const;

// Minimal Pyodide typings (the CDN script ships no TypeScript types).
interface PyProxy {
  toJs(opts?: { create_proxies?: boolean }): unknown;
  destroy(): void;
  [k: string]: unknown;
}
interface PyodideAPI {
  loadPackage(names: string[]): Promise<void>;
  runPythonAsync(code: string): Promise<unknown>;
  pyimport(name: string): PyProxy;
  FS: {
    writeFile(path: string, data: Uint8Array): void;
    mkdirTree(path: string): void;
    unlink(path: string): void;
  };
  globals: { set(k: string, v: unknown): void; delete(k: string): void };
}
// loadPyodide is injected into the Worker scope by importScripts(pyodide.js).
declare function loadPyodide(opts: { indexURL: string }): Promise<PyodideAPI>;

// Typed postMessage bypassing the DOM Window vs DedicatedWorkerGlobalScope mismatch.
type PostFn = (data: unknown, transfer?: Transferable[]) => void;
const post: PostFn = (
  self as unknown as { postMessage: PostFn }
).postMessage.bind(self);

// Discriminated union for all messages the Worker receives from the main thread.
type WorkerRequest =
  | { id: number; type: "boot"; wheelBase: string }
  | { id: number; type: "load_file"; name: string; buffer: ArrayBuffer }
  | { id: number; type: "load_demo" }
  | {
      id: number;
      type: "run_pipeline";
      paramsJson: string;
      flattenEnabled: boolean;
      force: boolean;
      forceFrom: string | null;
    }
  | { id: number; type: "json_call"; method: string; args: unknown[] }
  | { id: number; type: "slice_call"; method: string; args: unknown[] };

// ---------------------------------------------------------------------------
// Module state
// ---------------------------------------------------------------------------
let py: PyodideAPI | null = null;
let bridge: PyProxy | null = null;

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
function postBoot(phase: string, message: string, ready: boolean, error?: string): void {
  post({ id: null, type: "boot_status", phase, message, ready, error });
}

async function boot(wheelBase: string): Promise<void> {
  postBoot("runtime", "Downloading Python runtime (~10 MB)…", false);
  importScripts(`${PYODIDE_INDEX}pyodide.js`);
  py = await loadPyodide({ indexURL: PYODIDE_INDEX });

  postBoot("packages", "Loading numpy, scipy, h5py, matplotlib…", false);
  await py.loadPackage(["numpy", "scipy", "h5py", "matplotlib", "micropip"]);

  postBoot("wheel", "Installing the nebula3d reduction package…", false);
  const wheelUrl = `${wheelBase}wheels/nebula3d-0.2.0-py3-none-any.whl`;
  py.globals.set("_nebula3d_wheel_url", wheelUrl);
  await py.runPythonAsync(
    "import micropip\nawait micropip.install(_nebula3d_wheel_url, deps=False)\n",
  );
  py.globals.delete("_nebula3d_wheel_url");

  bridge = py.pyimport("nebula3d.webbridge");
  (bridge.setup as () => unknown)();

  postBoot("ready", "Ready — compute runs locally in your browser.", true);
}

// ---------------------------------------------------------------------------
// Message dispatch
// ---------------------------------------------------------------------------
async function dispatch(req: WorkerRequest): Promise<void> {
  const { id } = req;

  const reply = (payload: unknown): void => post({ id, type: "result", payload });
  const replyBinary = (buf: ArrayBuffer): void =>
    post({ id, type: "result_binary", payload: buf }, [buf]);
  const replyError = (e: unknown): void =>
    post({ id, type: "error", message: (e as Error).message ?? String(e) });

  try {
    switch (req.type) {
      case "boot":
        await boot(req.wheelBase);
        reply(null);
        break;

      case "load_file": {
        const bytes = new Uint8Array(req.buffer);
        py!.FS.mkdirTree("/uploads");
        const tmp = `/uploads/${Date.now()}`;
        py!.FS.writeFile(tmp, bytes);
        // Pre-flight: a metadata-only size check (reads the HDF5 shape, not the
        // arrays) so an oversized volume is rejected with a clear message rather
        // than crashing the reduction with a numpy MemoryError.
        const report = JSON.parse(
          (bridge!.inspect_input as (n: string, p: string) => string)(req.name, tmp),
        ) as { ok: boolean; message: string };
        if (!report.ok) {
          py!.FS.unlink(tmp);
          replyError(new Error(report.message));
          break;
        }
        const dsId = (bridge!.load_input as (n: string, p: string) => string)(req.name, tmp);
        reply(dsId);
        break;
      }

      case "load_demo": {
        const dsId = (bridge!.make_demo_input as (n: number) => string)(24);
        reply(dsId);
        break;
      }

      case "run_pipeline": {
        const { paramsJson, flattenEnabled, force, forceFrom } = req;
        const progress = (
          stage: string,
          status: string,
          fraction: number | null,
          message: string,
        ): void => {
          post({ id: null, type: "progress", stage, status, fraction, message });
        };
        for (const stage of STAGES) {
          (bridge!.run as (...a: unknown[]) => string)(
            stage,
            paramsJson,
            flattenEnabled,
            force,
            forceFrom ?? null,
            progress,
          );
        }
        reply((bridge!.datasets_json as () => string)());
        break;
      }

      case "json_call": {
        const result = (bridge![req.method] as (...a: unknown[]) => string)(...req.args);
        reply(result);
        break;
      }

      case "slice_call": {
        const proxy = (bridge![req.method] as (...a: unknown[]) => PyProxy)(...req.args);
        try {
          const u8 = proxy.toJs() as Uint8Array;
          // Slice out of the WASM ArrayBuffer so it can be transferred without
          // detaching the entire WASM memory.  Pyodide uses plain ArrayBuffer
          // (not SharedArrayBuffer), so the cast is safe.
          const buf = u8.buffer.slice(
            u8.byteOffset,
            u8.byteOffset + u8.byteLength,
          ) as ArrayBuffer;
          replyBinary(buf);
        } finally {
          proxy.destroy();
        }
        break;
      }
    }
  } catch (e) {
    replyError(e);
  }
}

self.addEventListener("message", ((ev: MessageEvent<WorkerRequest>) => {
  void dispatch(ev.data);
}) as EventListener);
