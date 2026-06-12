// Typed fetch wrappers for the ndiff API, including the binary slice envelope.

import type {
  Dataset,
  DeltaPdfMeta,
  JobOut,
  PipelineRunRequest,
  Slice,
  SliceHeader,
  VolumeMeta,
} from "./types";

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

export function fetchDatasets(): Promise<Dataset[]> {
  return getJSON<Dataset[]>("/api/datasets");
}

export function fetchMeta(volumeId: string): Promise<VolumeMeta> {
  return getJSON<VolumeMeta>(`/api/volumes/${encodeURIComponent(volumeId)}/meta`);
}

export function fetchSlice(
  volumeId: string,
  plane: string,
  value: number,
  interp = false,
): Promise<Slice> {
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
  return getJSON<DeltaPdfMeta>(`/api/deltapdf/${encodeURIComponent(volumeId)}/meta`);
}

export function fetchDpdfSlice(
  volumeId: string,
  plane: string,
  value: number,
): Promise<Slice> {
  const params = new URLSearchParams({ plane, value: String(value) });
  return fetchEnvelope(
    `/api/deltapdf/${encodeURIComponent(volumeId)}/slice?${params.toString()}`,
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
