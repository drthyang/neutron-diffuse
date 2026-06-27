// One intensity-slice tile for the Bragg-profile selected-peak detail.
//
// SliceCanvas only crops symmetrically about the origin, but a Bragg peak sits at
// an arbitrary center_hkl — so this renders a square window *centred on the peak*
// by resampling the fetched slice in r.l.u. coordinates, then overlays the punch
// fit ellipses (fitted / measured / |Q|-floor) and a centre crosshair as SVG.

import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, type ReactNode } from "react";

import { fetchSlice } from "../api/client";
import type { Slice } from "../api/types";

export interface Ellipse {
  rx: number; // semi-axis along the tile x (r.l.u.)
  ry: number; // semi-axis along the tile y (r.l.u.)
}

interface Props {
  volumeId: string | undefined;
  plane: string; // "hk0" | "h0l" | "0kl"
  value: number; // fixed third-axis coordinate (r.l.u.)
  cx: number; // peak centre along the tile x axis (r.l.u.)
  cy: number; // peak centre along the tile y axis (r.l.u.)
  half: number; // window half-extent (r.l.u.)
  lut: Uint8ClampedArray;
  contrast: number; // 0.5–3.0; higher = brighter
  fitted: Ellipse;
  measured?: Ellipse | null;
  floor?: Ellipse | null;
  axisLabel: ReactNode;
  axisColor: string;
}

const RASTER = 140; // square raster; CSS scales it to the tile

function nearest(axis: number[], v: number): number {
  const n = axis.length;
  if (n < 2) return 0;
  const t = ((v - axis[0]) / (axis[n - 1] - axis[0])) * (n - 1);
  return Math.max(0, Math.min(n - 1, Math.round(t)));
}

// Robust upper colour limit from the window samples (p99 keeps a single hot voxel
// from blowing out the scale); contrast brightens by lowering vmax.
function windowVmax(
  slice: Slice,
  cx: number,
  cy: number,
  half: number,
  contrast: number,
): number {
  const { nx, x_axis: xs, y_axis: ys } = slice.header;
  const data = slice.data;
  const vals: number[] = [];
  for (let py = 0; py < RASTER; py += 2) {
    const yr = cy - half + ((py + 0.5) / RASTER) * 2 * half;
    const iy = nearest(ys, yr);
    for (let px = 0; px < RASTER; px += 2) {
      const xr = cx - half + ((px + 0.5) / RASTER) * 2 * half;
      const v = data[iy * nx + nearest(xs, xr)];
      if (Number.isFinite(v) && v > 0) vals.push(v);
    }
  }
  if (vals.length === 0) return 1;
  vals.sort((a, b) => a - b);
  const p99 = vals[Math.min(vals.length - 1, Math.floor(vals.length * 0.99))];
  return Math.max(p99 / Math.max(contrast, 0.01), 1e-9);
}

function SliceTile({
  slice,
  cx,
  cy,
  half,
  lut,
  contrast,
  fitted,
  measured,
  floor,
  axisLabel,
  axisColor,
}: Omit<Props, "volumeId" | "plane" | "value"> & { slice: Slice }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { nx, x_axis: xs, y_axis: ys } = slice.header;
    const data = slice.data;
    const vmax = windowVmax(slice, cx, cy, half, contrast);

    canvas.width = RASTER;
    canvas.height = RASTER;
    const img = ctx.createImageData(RASTER, RASTER);
    const out = img.data;
    for (let py = 0; py < RASTER; py++) {
      // y flipped so +axis points up
      const yr = cy + half - ((py + 0.5) / RASTER) * 2 * half;
      const iy = nearest(ys, yr);
      for (let px = 0; px < RASTER; px++) {
        const xr = cx - half + ((px + 0.5) / RASTER) * 2 * half;
        const v = data[iy * nx + nearest(xs, xr)];
        const o = (py * RASTER + px) * 4;
        if (!Number.isFinite(v)) {
          out[o] = out[o + 1] = out[o + 2] = 40;
          out[o + 3] = 255;
          continue;
        }
        const t = Math.max(0, Math.min(1, v / vmax));
        const li = (t * 255) | 0;
        out[o] = lut[li * 4];
        out[o + 1] = lut[li * 4 + 1];
        out[o + 2] = lut[li * 4 + 2];
        out[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [slice, cx, cy, half, lut, contrast]);

  // SVG overlay in a 0..200 box centred at 100; r.l.u. → px scale = 100/half.
  const C = 100;
  const s = 100 / half;
  const ell = (e: Ellipse) => ({ rx: Math.max(0.5, e.rx * s), ry: Math.max(0.5, e.ry * s) });
  const f = ell(fitted);
  const m = measured ? ell(measured) : null;
  const fl = floor ? ell(floor) : null;

  return (
    <div className="bragg-tile">
      <div className="bragg-tile-canvas">
        <canvas ref={ref} />
        <svg viewBox="0 0 200 200" preserveAspectRatio="none">
          <line x1={C} y1="0" x2={C} y2="200" className="bragg-cross" />
          <line x1="0" y1={C} x2="200" y2={C} className="bragg-cross" />
          {fl && (
            <ellipse cx={C} cy={C} rx={fl.rx} ry={fl.ry} className="bragg-ell-floor" />
          )}
          {m && (
            <ellipse cx={C} cy={C} rx={m.rx} ry={m.ry} className="bragg-ell-meas" />
          )}
          <ellipse cx={C} cy={C} rx={f.rx} ry={f.ry} className="bragg-ell-fit" />
        </svg>
      </div>
      <span className="bragg-tile-label" style={{ color: axisColor }}>
        {axisLabel}
      </span>
    </div>
  );
}

export function BraggPeakSlice({ volumeId, plane, value, ...rest }: Props) {
  const q = useQuery({
    queryKey: ["braggSlice", volumeId, plane, value],
    queryFn: () => fetchSlice(volumeId as string, plane, value, true),
    enabled: !!volumeId,
    staleTime: 5 * 60 * 1000,
  });

  if (!volumeId || q.isError) {
    return (
      <div className="bragg-tile">
        <div className="bragg-tile-canvas bragg-tile-empty">
          <span>{!volumeId ? "no volume" : "slice failed"}</span>
        </div>
        <span className="bragg-tile-label" style={{ color: rest.axisColor }}>
          {rest.axisLabel}
        </span>
      </div>
    );
  }
  if (!q.data) {
    return (
      <div className="bragg-tile">
        <div className="bragg-tile-canvas bragg-tile-empty">
          <span>loading…</span>
        </div>
        <span className="bragg-tile-label" style={{ color: rest.axisColor }}>
          {rest.axisLabel}
        </span>
      </div>
    );
  }
  return <SliceTile slice={q.data} {...rest} />;
}
