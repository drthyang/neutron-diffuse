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

export interface ConsistencyMetrics {
  pearson_r: number;
  normalized_rms: number;
  rms: number;
  n_voxels: number;
  per_plane_r: Record<string, number>;
  q_band: [number, number] | null;
  r_band: [number, number] | null;
  q_data_max: number;
  r_data_max: number;
  crop_hkl: number[] | null;
  apodization: string;
}

export interface ConsistencyMeta {
  shape: [number, number, number];
  h_range: [number, number];
  k_range: [number, number];
  l_range: [number, number];
  dpdf_shape: [number, number, number];
  x_range: [number, number];
  y_range: [number, number];
  z_range: [number, number];
  lattice: Lattice;
  planes: string[];
  q_data_max: number;
  r_data_max: number;
  metrics: ConsistencyMetrics;
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
  // "patched" (per-patch) | "parametric" (separable Ring(|Q|) × per-shell texture)
  rings_model?: string;
  rings_ring_width?: number;
  // "rolling" (continuous Ring(|Q|)) | "peaks" (discrete pseudo-Voigt)
  rings_radial_mode?: string;
  punch_min_intensity?: number;
  punch_search_n_mad?: number;
  punch_mode?: string;
  punch_radius_h?: number;
  punch_radius_k?: number;
  punch_radius_l?: number;
  punch_margin?: number;
  punch_phi_tail_hkl?: number;
  // Q-space punch (opt-in): frame "q" + isotropic / per-a*,b*,c* radius (Å⁻¹)
  punch_frame?: string;
  punch_q_radius?: number;
  punch_q_radius_a?: number;
  punch_q_radius_b?: number;
  punch_q_radius_c?: number;
  punch_fit_covariance?: boolean;
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
