// Renders a 2D slice to a canvas via a colormap LUT.  Colour/contrast/log are
// applied here client-side, so changing them re-renders instantly with no refetch.
//
// The canvas raster is the slice's native resolution (or the cropped window);
// CSS scales it for display.  Display modes:
//   • width  — fixed display width, height follows the data aspect ratio
//   • fit    — letterbox to fill the parent box (preserves aspect)
//   • windowA + size — crop to a square physical window [-windowA, +windowA] Å on
//     both axes and draw it into a square `size` px box (used by the ΔPDF viewer
//     so all three orthoslices share one square real-space window)
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
  size?: number; // square display size in px (used with windowA)
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
  size,
}: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const { nx, ny } = slice.header;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Crop to the square physical window [-windowA, +windowA] on both axes.
    const xs = slice.header.x_axis;
    const ys = slice.header.y_axis;
    let ix0 = 0;
    let ix1 = nx - 1;
    let iy0 = 0;
    let iy1 = ny - 1;
    if (windowA != null) {
      while (ix0 < ix1 && xs[ix0] < -windowA) ix0++;
      while (ix1 > ix0 && xs[ix1] > windowA) ix1--;
      while (iy0 < iy1 && ys[iy0] < -windowA) iy0++;
      while (iy1 > iy0 && ys[iy1] > windowA) iy1--;
    }
    const cw = ix1 - ix0 + 1;
    const ch = iy1 - iy0 + 1;

    canvas.width = cw;
    canvas.height = ch;
    const img = ctx.createImageData(cw, ch);
    const out = img.data;
    const data = slice.data;
    const vmaxSafe = vmax > 0 ? vmax : 1;
    const span = vmaxSafe - vmin > 0 ? vmaxSafe - vmin : 1;
    const logMax = Math.log10(vmaxSafe + 1) || 1;

    for (let rr = 0; rr < ch; rr++) {
      // flip vertically: top output row is the largest y.
      const srcRow = (iy1 - rr) * nx;
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
  }, [slice, lut, vmax, vmin, log, diverging, nx, ny, windowA]);

  // A windowA crop is always shown as a physical square: either at a fixed `size`
  // (single ΔPDF viewer) or filling its square parent cell (multi-temp grid).
  let style: React.CSSProperties;
  if (windowA != null) {
    style =
      size != null
        ? { width: size, height: size, imageRendering: "auto" }
        : {
            // Responsive square: fill the column width and force a 1:1 box
            // (overriding the raster's native aspect) so the physical window
            // renders square and in-flow — same region as the fixed-size ΔPDF
            // panels, without collapsing the grid track.
            width: "100%",
            height: "auto",
            aspectRatio: "1 / 1",
            display: "block",
            imageRendering: "auto",
          };
  } else if (fit) {
    style = { maxWidth: "100%", maxHeight: "100%", width: "auto", height: "auto" };
  } else {
    style = { width, height: "auto" };
  }

  return <canvas ref={ref} className="slice-canvas" style={style} />;
}
