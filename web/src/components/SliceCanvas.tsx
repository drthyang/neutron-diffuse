// Renders a 2D slice to a canvas via a colormap LUT.  Colour/contrast/log are
// applied here client-side, so changing them re-renders instantly with no refetch.

import { useEffect, useRef } from "react";

import type { Slice } from "../api/types";

interface Props {
  slice: Slice;
  lut: Uint8ClampedArray; // 256 * 4 RGBA
  vmax: number; // upper colour limit (contrast x robust scale)
  log: boolean;
  diverging?: boolean; // signed data centred at 0 (ΔPDF)
  width?: number; // display width in CSS px
}

export function SliceCanvas({
  slice,
  lut,
  vmax,
  log,
  diverging = false,
  width = 340,
}: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const { nx, ny } = slice.header;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;

    const off = document.createElement("canvas");
    off.width = nx;
    off.height = ny;
    const octx = off.getContext("2d");
    if (!octx) return;
    const img = octx.createImageData(nx, ny);
    const out = img.data;
    const data = slice.data;
    const vmaxSafe = vmax > 0 ? vmax : 1;
    const logMax = Math.log10(vmaxSafe + 1) || 1;

    for (let r = 0; r < ny; r++) {
      const srcRow = r * nx;
      // flip vertically: data row 0 (smallest y) goes to the canvas bottom.
      const dstRow = (ny - 1 - r) * nx;
      for (let c = 0; c < nx; c++) {
        const v = data[srcRow + c];
        const o = (dstRow + c) * 4;
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
          t = Math.max(0, Math.min(1, v / vmaxSafe));
        }
        const li = (t * 255) | 0;
        out[o] = lut[li * 4];
        out[o + 1] = lut[li * 4 + 1];
        out[o + 2] = lut[li * 4 + 2];
        out[o + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const height = Math.round((width * ny) / nx);
    canvas.width = width;
    canvas.height = height;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, width, height);
  }, [slice, lut, vmax, log, diverging, nx, ny, width]);

  return (
    <canvas
      ref={ref}
      style={{ width, imageRendering: "pixelated", background: "#222", borderRadius: 4 }}
    />
  );
}
