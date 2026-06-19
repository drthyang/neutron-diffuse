// Renders a 2D slice to a canvas via a colormap LUT.  Colour/contrast/log are
// applied here client-side, so changing them re-renders instantly with no refetch.
//
// The canvas raster is the slice's native resolution (or the cropped window);
// CSS scales it for display.  Display modes:
//   • width  — fixed display width, height follows the data aspect ratio
//   • fit    — letterbox to fill the parent box (preserves aspect)
//   • windowA + size — crop to a square coordinate window [-windowA, +windowA]
//     on both axes and draw it into a square `size` px box (used by the ΔPDF
//     viewer so all three orthoslices share one square real-space window)
//   • windowX/windowY — crop each axis independently, still drawn into a square
//     box after row-resampling preserves equal physical units per pixel
//
// Colour mapping: sequential data maps [vmin, vmax] → LUT; `diverging` data maps
// symmetrically about 0 over ±vmax; `log` uses log10(v+1)/log10(vmax+1).

import { useEffect, useRef } from "react";

import type { Slice } from "../api/types";

interface Props {
  slice: Slice;
  lut: Uint8ClampedArray; // 256 * 4 RGBA
  vmax: number; // upper colour limit (contrast x robust scale)
  vmin?: number; // lower colour limit for sequential data (default 0)
  log: boolean;
  diverging?: boolean; // signed data centred at 0 (ΔPDF)
  width?: number; // fixed display width in CSS px
  fit?: boolean; // letterbox to fill the parent box (preserves aspect)
  windowA?: number; // half-extent in Å — crop to a square physical window
  windowX?: number; // half-extent along x in the slice's coordinate units
  windowY?: number; // half-extent along y in the slice's coordinate units
  size?: number; // square display size in px (used with windowA)
  bands?: [number, number]; // [min, max] band for circle overlays
  cutDistance?: number; // distance from origin for intersection
  reciprocalAxes?: boolean; // x/y/cut coordinates are r.l.u.; convert to Å^-1
  latX?: number;
  latY?: number;
  latCut?: number;
}

export function SliceCanvas({
  slice,
  lut,
  vmax,
  vmin = 0,
  log,
  diverging = false,
  width = 340,
  fit = false,
  windowA,
  windowX,
  windowY,
  size,
  bands,
  cutDistance,
  reciprocalAxes = false,
  latX,
  latY,
  latCut,
}: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const { nx, ny, x_axis: xs, y_axis: ys } = slice.header;

  let ix0 = 0;
  let ix1 = nx - 1;
  let iy0 = 0;
  let iy1 = ny - 1;
  const xWindow = windowX ?? windowA;
  const yWindow = windowY ?? windowA;
  if (xWindow != null) {
    while (ix0 < ix1 && xs[ix0] < -xWindow) ix0++;
    while (ix1 > ix0 && xs[ix1] > xWindow) ix1--;
  }
  if (yWindow != null) {
    while (iy0 < iy1 && ys[iy0] < -yWindow) iy0++;
    while (iy1 > iy0 && ys[iy1] > yWindow) iy1--;
  }
  const cw = ix1 - ix0 + 1;
  const ch_raw = iy1 - iy0 + 1;
  
  const dx = nx > 1 ? (xs[nx - 1] - xs[0]) / (nx - 1) : 1;
  const dy = ny > 1 ? (ys[ny - 1] - ys[0]) / (ny - 1) : 1;
  const qScaleX = reciprocalAxes && latX ? 2 * Math.PI / latX : 1;
  const qScaleY = reciprocalAxes && latY ? 2 * Math.PI / latY : 1;
  const dx_Q = dx * qScaleX;
  const dy_Q = dy * qScaleY;
  const qScaleCut = reciprocalAxes && latCut ? 2 * Math.PI / latCut : 1;
  const ch = Math.max(1, Math.round(ch_raw * Math.abs(dy_Q / dx_Q)));

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = cw;
    canvas.height = ch;
    const img = ctx.createImageData(cw, ch);
    const out = img.data;
    const data = slice.data;
    const vmaxSafe = vmax > 0 ? vmax : 1;
    const span = vmaxSafe - vmin > 0 ? vmaxSafe - vmin : 1;
    const logMax = Math.log10(vmaxSafe + 1) || 1;

    for (let rr = 0; rr < ch; rr++) {
      // nearest-neighbor scale rows to match physical aspect ratio
      const srcRow = (iy1 - Math.floor(rr * (ch_raw / ch))) * nx;
      const dstRow = rr * cw;
      for (let cc = 0; cc < cw; cc++) {
        const v = data[srcRow + ix0 + cc];
        const o = (dstRow + cc) * 4;
        if (!Number.isFinite(v)) {
          out[o] = 128;
          out[o + 1] = 128;
          out[o + 2] = 128;
          out[o + 3] = 255;
          continue;
        }
        let t: number;
        if (diverging) {
          t = 0.5 + 0.5 * Math.max(-1, Math.min(1, v / vmaxSafe));
        } else if (log) {
          t = Math.max(0, Math.min(1, Math.log10(Math.max(v, 0) + 1) / logMax));
        } else {
          t = Math.max(0, Math.min(1, (v - vmin) / span));
        }
        const li = (t * 255) | 0;
        out[o] = lut[li * 4];
        out[o + 1] = lut[li * 4 + 1];
        out[o + 2] = lut[li * 4 + 2];
        out[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [slice, lut, vmax, vmin, log, diverging, nx, ny, xWindow, yWindow, cw, ch, ch_raw, ix0, ix1, iy0, iy1, xs, ys]);

  // A windowed crop is always shown as a physical square: either at a fixed
  // `size` (single ΔPDF viewer) or filling its square parent cell (multi-temp
  // grid / Q-equal previews).
  let wrapperStyle: React.CSSProperties;
  if (xWindow != null || yWindow != null) {
    wrapperStyle =
      size != null
        ? { width: size, height: size, position: "relative" }
        : { width: "100%", aspectRatio: "1 / 1", display: "block", position: "relative" };
  } else if (fit) {
    wrapperStyle = { maxWidth: "100%", maxHeight: "100%", width: "auto", height: "auto", position: "relative", display: "inline-block" };
  } else {
    wrapperStyle = { width, height: "auto", position: "relative", display: "inline-block" };
  }

  const vX = (xs[ix0] - dx / 2) * qScaleX;
  const vW = cw * dx_Q;
  const vH = ch_raw * dy_Q;

  const circles: number[] = [];
  if (bands && cutDistance != null) {
    const [bMin, bMax] = bands;
    const cutPhys = cutDistance * qScaleCut;
    const cutSq = cutPhys * cutPhys;

    if (bMin > 0) {
      const rSq1 = bMin * bMin - cutSq;
      if (rSq1 > 0) circles.push(Math.sqrt(rSq1));
    }
    if (bMax > 0) {
      const rSq2 = bMax * bMax - cutSq;
      if (rSq2 > 0) circles.push(Math.sqrt(rSq2));
    }
  }

  return (
    <div style={wrapperStyle}>
      <canvas ref={ref} className="slice-canvas" style={{ width: "100%", height: "100%", display: "block", imageRendering: "auto" }} />
      {circles.length > 0 && (
        <svg
          style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}
          viewBox={`${vX} ${-((ys[iy1] + Math.abs(dy) / 2) * qScaleY)} ${vW} ${Math.abs(vH)}`}
          preserveAspectRatio="none"
        >
          <g transform="scale(1, -1)">
            {circles.map((r, i) => (
              <g key={i}>
                <circle
                  cx={0}
                  cy={0}
                  r={r}
                  fill="none"
                  stroke="rgba(0, 0, 0, 0.8)"
                  strokeWidth={vW / 100}
                />
                <circle
                  cx={0}
                  cy={0}
                  r={r}
                  fill="none"
                  stroke="rgba(255, 255, 255, 0.9)"
                  strokeWidth={vW / 150}
                />
              </g>
            ))}
          </g>
        </svg>
      )}
    </div>
  );
}
