// Typed fetch wrappers for the nebula3d API, including the binary slice envelope.

import { engine, PYODIDE_MODE } from "./pyodideEngine";
import type {
  BraggProfile,
  ConsistencyMeta,
  DataRoot,
  Dataset,
  DeltaPdfMeta,
  JobOut,
  PipelineRunRequest,
  Slice,
  SliceHeader,
  VolumeMeta,
} from "./types";

function bandParams(qMin?: number, qMax?: number, rMin?: number, rMax?: number): string {
  const p = new URLSearchParams();
  if (qMin != null) p.set("q_min", String(qMin));
  if (qMax != null) p.set("q_max", String(qMax));
  if (rMin != null) p.set("r_min", String(rMin));
  if (rMax != null) p.set("r_max", String(rMax));
  return p.toString();
}

async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${url}`);
  return (await r.json()) as T;
}

// Decode the binary slice envelope: [uint32 LE header_len][JSON header][float32 data].
async function fetchEnvelope(url: string): Promise<Slice> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  const buf = await r.arrayBuffer();
  const headerLen = new DataView(buf).getUint32(0, true);
  const headerBytes = new Uint8Array(buf, 4, headerLen);
  const header = JSON.parse(new TextDecoder().decode(headerBytes)) as SliceHeader;
  // Copy to a fresh 0-offset buffer (float32 view needs 4-byte alignment, which
  // an arbitrary header length does not guarantee).
  const data = new Float32Array(buf.slice(4 + headerLen));
  return { header, data };
}

export function fetchHealth(): Promise<{ status: string }> {
  // Backend-less builds report healthy so the shell renders (Pyodide boots
  // lazily on the first compute, not here).
  if (PYODIDE_MODE) return Promise.resolve({ status: "pyodide" });
  return getJSON<{ status: string }>("/api/health");
}

export function fetchDatasets(): Promise<Dataset[]> {
  if (PYODIDE_MODE) return engine.datasets();
  return getJSON<Dataset[]>("/api/datasets");
}

export async function fetchDataRoot(): Promise<DataRoot> {
  if (PYODIDE_MODE) {
    const n = (await engine.datasets()).length;
    return {
      data_root: "in-browser (Pyodide)",
      raw_exists: true, processed_exists: true, n_datasets: n,
    };
  }
  return getJSON<DataRoot>("/api/data-root");
}

export async function setDataRoot(dataRoot: string): Promise<DataRoot> {
  const r = await fetch("/api/data-root", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data_root: dataRoot }),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return (await r.json()) as DataRoot;
}

export async function browseDataRoot(): Promise<DataRoot> {
  const r = await fetch("/api/data-root/browse", {
    method: "POST",
    headers: { "X-Nebula3d-Local": "1" },
  });
  if (!r.ok) {
    const body = await r.text();
    let detail = body;
    try {
      detail = (JSON.parse(body) as { detail?: string }).detail ?? body;
    } catch {
      // Plain text error body.
    }
    throw new Error(`${r.status} ${detail}`);
  }
  return (await r.json()) as DataRoot;
}

export function fetchMeta(volumeId: string): Promise<VolumeMeta> {
  if (PYODIDE_MODE) return engine.volumeMeta(volumeId);
  return getJSON<VolumeMeta>(`/api/volumes/${encodeURIComponent(volumeId)}/meta`);
}

export function fetchSlice(
  volumeId: string,
  plane: string,
  value: number,
  interp = false,
): Promise<Slice> {
  if (PYODIDE_MODE) return engine.volumeSlice(volumeId, plane, value, interp);
  const params = new URLSearchParams({
    plane,
    value: String(value),
    interp: String(interp),
  });
  return fetchEnvelope(
    `/api/volumes/${encodeURIComponent(volumeId)}/slice?${params.toString()}`,
  );
}

export function fetchDpdfMeta(volumeId: string): Promise<DeltaPdfMeta> {
  if (PYODIDE_MODE) return engine.dpdfMeta(volumeId);
  return getJSON<DeltaPdfMeta>(`/api/deltapdf/${encodeURIComponent(volumeId)}/meta`);
}

export function fetchDpdfSlice(
  volumeId: string,
  plane: string,
  value: number,
): Promise<Slice> {
  if (PYODIDE_MODE) return engine.dpdfSlice(volumeId, plane, value);
  const params = new URLSearchParams({ plane, value: String(value) });
  return fetchEnvelope(
    `/api/deltapdf/${encodeURIComponent(volumeId)}/slice?${params.toString()}`,
  );
}

export function fetchConsistencyMeta(
  datasetId: string,
  qMin?: number,
  qMax?: number,
  rMin?: number,
  rMax?: number,
): Promise<ConsistencyMeta> {
  if (PYODIDE_MODE) return engine.consistencyMeta(datasetId, qMin, qMax, rMin, rMax);
  const qs = bandParams(qMin, qMax, rMin, rMax);
  const url = `/api/consistency/${encodeURIComponent(datasetId)}/meta${qs ? `?${qs}` : ""}`;
  return getJSON<ConsistencyMeta>(url);
}

export function fetchBraggProfile(datasetId: string): Promise<BraggProfile> {
  if (PYODIDE_MODE) {
    return Promise.resolve({
      dataset_id: datasetId,
      profile_path: null,
      has_profile: false,
      schema_version: 1,
      width_labels: ["Qx", "Qy", "Qz"],
      hkl_width_labels: ["H", "K", "L"],
      width_units: { hkl: "r.l.u.", q: "Å⁻¹" },
      n_peaks: 0,
      fit_covariance: false,
      punch_frame: null,
      peaks: [],
    });
  }
  return getJSON<BraggProfile>(`/api/bragg/${encodeURIComponent(datasetId)}/profile`);
}

export async function saveConsistencyDpdf(
  datasetId: string,
  qMin?: number,
  qMax?: number,
  rMin?: number,
  rMax?: number,
): Promise<{ saved: boolean; path: string; filename: string }> {
  if (PYODIDE_MODE) {
    throw new Error(
      "Saving the ΔPDF to disk requires the desktop / server app (not the browser build).",
    );
  }
  const qs = bandParams(qMin, qMax, rMin, rMax);
  const url = `/api/consistency/${encodeURIComponent(datasetId)}/save${qs ? `?${qs}` : ""}`;
  const r = await fetch(url, { method: "POST" });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${url}`);
  return (await r.json()) as { saved: boolean; path: string; filename: string };
}

export function fetchConsistencySlice(
  datasetId: string,
  panel: string,
  plane: string,
  value: number,
  qMin?: number,
  qMax?: number,
  rMin?: number,
  rMax?: number,
): Promise<Slice> {
  if (PYODIDE_MODE) {
    return engine.consistencySlice(datasetId, panel, plane, value, qMin, qMax, rMin, rMax);
  }
  const params = new URLSearchParams({ panel, plane, value: String(value) });
  if (qMin != null) params.set("q_min", String(qMin));
  if (qMax != null) params.set("q_max", String(qMax));
  if (rMin != null) params.set("r_min", String(rMin));
  if (rMax != null) params.set("r_max", String(rMax));
  return fetchEnvelope(
    `/api/consistency/${encodeURIComponent(datasetId)}/slice?${params.toString()}`,
  );
}

export async function runPipeline(req: PipelineRunRequest): Promise<JobOut> {
  const r = await fetch("/api/pipeline/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return (await r.json()) as JobOut;
}

export async function cancelJob(jobId: string): Promise<JobOut> {
  const r = await fetch(`/api/pipeline/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST",
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return (await r.json()) as JobOut;
}
