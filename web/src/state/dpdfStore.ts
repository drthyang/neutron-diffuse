// Shared view state for the ΔPDF viewers (single + multi-temperature).
// Cut positions are indices along x_H / y_K / z_L; all temperatures share the
// same grid shape, so the indices are reused across the multi-temp comparison.

import { create } from "zustand";

interface DpdfState {
  datasetId?: string;
  cutX: number;
  cutY: number;
  cutZ: number;
  contrast: number;
  gridlines: boolean;
  centered: boolean;
  setDataset: (id: string) => void;
  setCutX: (i: number) => void;
  setCutY: (i: number) => void;
  setCutZ: (i: number) => void;
  setContrast: (c: number) => void;
  setGridlines: (b: boolean) => void;
  center: (x: number, y: number, z: number) => void;
}

export const useDpdfStore = create<DpdfState>((set) => ({
  cutX: 0,
  cutY: 0,
  cutZ: 0,
  contrast: 1,
  gridlines: false,
  centered: false,
  setDataset: (datasetId) => set({ datasetId }),
  setCutX: (cutX) => set({ cutX }),
  setCutY: (cutY) => set({ cutY }),
  setCutZ: (cutZ) => set({ cutZ }),
  setContrast: (contrast) => set({ contrast }),
  setGridlines: (gridlines) => set({ gridlines }),
  center: (cutX, cutY, cutZ) => set({ cutX, cutY, cutZ, centered: true }),
}));
