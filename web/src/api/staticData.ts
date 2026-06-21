// Static (backend-less) data layer for the GitHub Pages build.
//
// When VITE_DATA_MODE === "static" the SPA has no FastAPI backend, so the data
// layer reads a pre-baked manifest + downsampled ΔPDF volumes (written by
// scripts/export_web_assets.py) and slices them client-side, producing the same
// `Slice` / `Dataset` / `DeltaPdfMeta` shapes the API would return.  The 3D-ΔPDF
// and Multi-temperature viewers therefore work unchanged and stay interactive.
//
// Volume binary format (little-endian), matching the exporter:
//   [uint32 header_len][JSON header][float16 data, C-order [ix, iy, iz]]

import type { Dataset, DeltaPdfMeta, Lattice, Slice } from "./types";

export const STATIC_MODE = import.meta.env.VITE_DATA_MODE === "static";

const BASE = import.meta.env.BASE_URL ?? "/";

interface DpdfManifestMeta {
  shape: [number, number, number];
  x_range: [number, number];
  y_range: [number, number];
  z_range: [number, number];
  lattice: Lattice;
  q_max: number | null;
  planes: string[];
}

interface ManifestDataset {
  id: string;
  temperature: string | null;
  raw_name: string;
  stem: string;
  stages: Dataset["stages"];
  dpdf: { asset: string; meta: DpdfManifestMeta };
}

interface Manifest {
  version: number;
  mode: string;
  stride: number;
  datasets: ManifestDataset[];
}

interface Volume {
  nx: number;
  ny: number;
  nz: number;
  xs: Float64Array;
  ys: Float64Array;
  zs: Float64Array;
  data: Float32Array; // length nx*ny*nz, C-order [ix, iy, iz]
}

let manifestPromise: Promise<Manifest> | null = null;
const volumeCache = new Map<string, Promise<Volume>>();

function loadManifest(): Promise<Manifest> {
  if (!manifestPromise) {
    manifestPromise = fetch(`${BASE}data/manifest.json`).then((r) => {
      if (!r.ok) throw new Error(`manifest ${r.status} ${r.statusText}`);
      return r.json() as Promise<Manifest>;
    });
  }
  return manifestPromise;
}

// IEEE-754 half-float (uint16) → float32, filling `out` from `u16`.
function decodeHalf(u16: Uint16Array, out: Float32Array): void {
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const sign = (h & 0x8000) >> 15;
    const exp = (h & 0x7c00) >> 10;
    const frac = h & 0x03ff;
    let val: number;
    if (exp === 0) {
      val = frac * 2 ** -24; // subnormal
    } else if (exp === 0x1f) {
      val = frac ? NaN : Infinity; // inf/nan
    } else {
      val = (1 + frac / 1024) * 2 ** (exp - 15);
    }
    out[i] = sign ? -val : val;
  }
}

async function loadVolume(asset: string): Promise<Volume> {
  let p = volumeCache.get(asset);
  if (p) return p;
  p = (async () => {
    const r = await fetch(`${BASE}${asset}`);
    if (!r.ok) throw new Error(`volume ${r.status} ${r.statusText}`);
    const buf = await r.arrayBuffer();
    const headerLen = new DataView(buf).getUint32(0, true);
    const header = JSON.parse(
      new TextDecoder().decode(new Uint8Array(buf, 4, headerLen)),
    ) as { nx: number; ny: number; nz: number; x_axis: number[]; y_axis: number[]; z_axis: number[] };
    const { nx, ny, nz } = header;
    // Half-float payload begins after the header; copy to a 2-byte-aligned buffer.
    const u16 = new Uint16Array(buf.slice(4 + headerLen));
    const data = new Float32Array(nx * ny * nz);
    decodeHalf(u16, data);
    return {
      nx, ny, nz,
      xs: Float64Array.from(header.x_axis),
      ys: Float64Array.from(header.y_axis),
      zs: Float64Array.from(header.z_axis),
      data,
    };
  })();
  volumeCache.set(asset, p);
  return p;
}

function nearest(axis: Float64Array, value: number): number {
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < axis.length; i++) {
    const d = Math.abs(axis[i] - value);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

// p98 of |ΔPDF| at in-plane radius > 3 Å (skip the near-origin spike), matching
// the server's _robust_far so client slices self-scale like the matplotlib path.
function robustFar(data: Float32Array, xs: Float64Array, ys: Float64Array, nx: number): number {
  const vals: number[] = [];
  for (let iy = 0; iy < ys.length; iy++) {
    const y = ys[iy];
    for (let ix = 0; ix < nx; ix++) {
      const x = xs[ix];
      if (x * x + y * y <= 9) continue; // r <= 3 Å
      const v = data[iy * nx + ix];
      if (Number.isFinite(v)) vals.push(Math.abs(v));
    }
  }
  if (vals.length === 0) return 1;
  vals.sort((a, b) => a - b);
  const idx = Math.min(vals.length - 1, Math.floor(0.98 * (vals.length - 1)));
  return vals[idx] || 1;
}

const idx3 = (ix: number, iy: number, iz: number, ny: number, nz: number): number =>
  (ix * ny + iy) * nz + iz;

async function findDpdf(volumeId: string): Promise<ManifestDataset> {
  const man = await loadManifest();
  const ds = man.datasets.find((d) => d.stages.some((s) => s.volume_id === volumeId));
  if (!ds) throw new Error(`no static ΔPDF volume for ${volumeId}`);
  return ds;
}

/* ------------------------------------------------------------ public API */

export async function staticDatasets(): Promise<Dataset[]> {
  const man = await loadManifest();
  return man.datasets.map((d) => ({
    id: d.id,
    temperature: d.temperature,
    raw_name: d.raw_name,
    stem: d.stem,
    stages: d.stages,
  }));
}

export async function staticDpdfMeta(volumeId: string): Promise<DeltaPdfMeta> {
  const ds = await findDpdf(volumeId);
  const m = ds.dpdf.meta;
  return { id: volumeId, shape: m.shape, x_range: m.x_range, y_range: m.y_range,
    z_range: m.z_range, lattice: m.lattice, q_max: m.q_max, planes: m.planes };
}

export async function staticDpdfSlice(
  volumeId: string,
  plane: string,
  value: number,
): Promise<Slice> {
  const ds = await findDpdf(volumeId);
  const vol = await loadVolume(ds.dpdf.asset);
  const { nx, ny, nz, xs, ys, zs, data } = vol;

  let outNx: number;
  let outNy: number;
  let out: Float32Array;
  let ax: Float64Array;
  let ay: Float64Array;
  let xLabel: string;
  let yLabel: string;
  let cut: string;

  if (plane === "xy") {
    // x_H–y_K, fix z_L. out[iy*nx + ix] = vol[ix, iy, iz]
    const iz = nearest(zs, value);
    outNx = nx; outNy = ny; ax = xs; ay = ys;
    xLabel = "x_H (Å)"; yLabel = "y_K (Å)";
    cut = `z_L = ${zs[iz].toPrecision(3)} Å`;
    out = new Float32Array(nx * ny);
    for (let iy = 0; iy < ny; iy++)
      for (let ix = 0; ix < nx; ix++) out[iy * nx + ix] = data[idx3(ix, iy, iz, ny, nz)];
  } else if (plane === "xz") {
    // x_H–z_L, fix y_K. out[iz*nx + ix] = vol[ix, iy, iz]
    const iy = nearest(ys, value);
    outNx = nx; outNy = nz; ax = xs; ay = zs;
    xLabel = "x_H (Å)"; yLabel = "z_L (Å)";
    cut = `y_K = ${ys[iy].toPrecision(3)} Å`;
    out = new Float32Array(nx * nz);
    for (let iz = 0; iz < nz; iz++)
      for (let ix = 0; ix < nx; ix++) out[iz * nx + ix] = data[idx3(ix, iy, iz, ny, nz)];
  } else {
    // yz: y_K–z_L, fix x_H. out[iz*ny + iy] = vol[ix, iy, iz]
    const ix = nearest(xs, value);
    outNx = ny; outNy = nz; ax = ys; ay = zs;
    xLabel = "y_K (Å)"; yLabel = "z_L (Å)";
    cut = `x_H = ${xs[ix].toPrecision(3)} Å`;
    out = new Float32Array(ny * nz);
    for (let iz = 0; iz < nz; iz++)
      for (let iy = 0; iy < ny; iy++) out[iz * ny + iy] = data[idx3(ix, iy, iz, ny, nz)];
  }

  return {
    header: {
      nx: outNx,
      ny: outNy,
      x_axis: Array.from(ax),
      y_axis: Array.from(ay),
      x_label: xLabel,
      y_label: yLabel,
      cut_label: cut,
      robust_max: robustFar(out, ax, ay, outNx),
    },
    data: out,
  };
}
