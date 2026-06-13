// TypeScript mirrors of the FastAPI response models (ndiff/server/schemas.py).

export type VolumeKind = "hkl" | "delta_pdf";

export interface StageStatus {
  name: string;
  exists: boolean;
  kind: VolumeKind;
  volume_id: string;
}

export interface Dataset {
  id: string;
  temperature: string | null;
  raw_name: string;
  stem: string;
  stages: StageStatus[];
}

export interface Lattice {
  a: number | null;
  b: number | null;
  c: number | null;
}

export interface VolumeMeta {
  id: string;
  stage: string;
  kind: string;
  shape: [number, number, number];
  h_range: [number, number];
  k_range: [number, number];
  l_range: [number, number];
  lattice: Lattice;
  planes: string[];
}

export interface DeltaPdfMeta {
  id: string;
  shape: [number, number, number];
  x_range: [number, number];
  y_range: [number, number];
  z_range: [number, number];
  lattice: Lattice;
  q_max: number | null;
  planes: string[];
}

// Header decoded from the binary slice envelope.
export interface SliceHeader {
  ny: number;
  nx: number;
  x_axis: number[];
  y_axis: number[];
  x_label: string;
  y_label: string;
  cut_label: string;
  robust_max: number;
}

export interface Slice {
  header: SliceHeader;
  // length ny*nx, row-major; row index = y (ascending), NaN = masked.
  data: Float32Array;
}

export interface StageParamsIn {
  rings_n_patches?: number;
  rings_n_fourier?: number;
  rings_slice_axis?: string;
  punch_min_intensity?: number;
  punch_search_n_mad?: number;
  punch_mode?: string;
  punch_radius_h?: number;
  punch_radius_k?: number;
  punch_radius_l?: number;
  punch_margin?: number;
  punch_phi_tail_hkl?: number;
  backfill_method?: string;
  flatten_estimator?: string;
  flatten_floor_percentile?: number;
  pdf_apodization?: string;
  pdf_gaussian_sigma?: number;
  pdf_crop_h?: number;
  pdf_crop_k?: number;
  pdf_crop_l?: number;
}

export interface PipelineRunRequest {
  dataset_id: string;
  flatten_enabled: boolean;
  force: boolean;
  force_from?: string | null;
  params: StageParamsIn;
}

export interface JobOut {
  id: string;
  input: string;
  status: string;
  error: string | null;
  n_events: number;
}

// One Server-Sent-Event payload from a running job.
export interface JobEvent {
  type: string; // "progress" | "done" | "error" | "cancelled"
  stage?: string;
  status?: string;
  fraction?: number | null;
  message?: string;
}
