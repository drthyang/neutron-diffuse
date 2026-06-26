// Shared view state for the reciprocal-space viewer (zustand).

import { create } from "zustand";

export type FixedAxis = "H" | "K" | "L";
export type RealAxis = "X" | "Y" | "Z";

// Which fixed axis maps to which Mantid plane alias (cut perpendicular to it).
export const AXIS_TO_PLANE: Record<FixedAxis, string> = {
  H: "0kl",
  K: "h0l",
  L: "hk0",
};

export const REAL_AXIS_TO_PLANE: Record<RealAxis, string> = {
  X: "yz",
  Y: "zx",
  Z: "xy",
};

export const AXIS_INDEX: Record<FixedAxis, 0 | 1 | 2> = { H: 0, K: 1, L: 2 };
export const REAL_AXIS_INDEX: Record<RealAxis, 0 | 1 | 2> = { X: 0, Y: 1, Z: 2 };

interface ViewerState {
  fixedAxis: FixedAxis;
  cutIndex: number; // index along the fixed axis
  contrast: number; // multiplies the robust auto colour scale
  zoom: number; // ≥1 — crops the reciprocal slice symmetrically about Q=0
  log: boolean;
  colormap: string;
  divColormap: string;
  setFixedAxis: (a: FixedAxis) => void;
  setCutIndex: (i: number) => void;
  setContrast: (c: number) => void;
  setZoom: (z: number) => void;
  setLog: (b: boolean) => void;
  setColormap: (c: string) => void;
  setDivColormap: (c: string) => void;
}

export const useViewerStore = create<ViewerState>((set) => ({
  fixedAxis: "H",
  cutIndex: 0,
  contrast: 1,
  zoom: 1,
  log: false,
  colormap: "inferno",
  divColormap: "RdBu_r",
  setFixedAxis: (fixedAxis) => set({ fixedAxis }),
  setCutIndex: (cutIndex) => set({ cutIndex }),
  setContrast: (contrast) => set({ contrast }),
  setZoom: (zoom) => set({ zoom }),
  setLog: (log) => set({ log }),
  setColormap: (colormap) => set({ colormap }),
  setDivColormap: (divColormap) => set({ divColormap }),
}));
