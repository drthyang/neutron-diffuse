// Shared view state for the ΔPDF viewers (single + multi-temperature).
// Cut positions are indices along x_H / y_K / z_L; all temperatures share the
// same grid shape, so the indices are reused across the multi-temp comparison.

import { create } from "zustand";

interface DpdfState {
  cutX: number;
  cutY: number;
  cutZ: number;
  contrast: number;
  gridlines: boolean;
  colormap: string;
  windowFull: number; // square real-space window width in Å (shared by both viewers)
  centered: boolean;
  setCutX: (i: number) => void;
  setCutY: (i: number) => void;
  setCutZ: (i: number) => void;
  setContrast: (c: number) => void;
  setGridlines: (b: boolean) => void;
  setColormap: (c: string) => void;
  setWindowFull: (w: number) => void;
  setCentered: (c: boolean) => void;
  center: (x: number, y: number, z: number) => void;
}

export const useDpdfStore = create<DpdfState>((set) => ({
  cutX: 0,
  cutY: 0,
  cutZ: 0,
  contrast: 1,
  gridlines: false,
  colormap: "RdBu_r",
  windowFull: 80,
  centered: false,
  setCutX: (cutX) => set({ cutX }),
  setCutY: (cutY) => set({ cutY }),
  setCutZ: (cutZ) => set({ cutZ }),
  setContrast: (contrast) => set({ contrast }),
  setGridlines: (gridlines) => set({ gridlines }),
  setColormap: (colormap) => set({ colormap }),
  setWindowFull: (windowFull) => set({ windowFull }),
  setCentered: (centered) => set({ centered }),
  center: (cutX, cutY, cutZ) => set({ cutX, cutY, cutZ, centered: true }),
}));
