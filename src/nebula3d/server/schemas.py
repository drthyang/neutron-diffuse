"""Pydantic response models for the API."""

from __future__ import annotations

from pydantic import BaseModel


class StageStatusOut(BaseModel):
    name: str
    exists: bool
    kind: str            # "hkl" | "delta_pdf"
    volume_id: str       # "<dataset_id>.<stage>"


class DatasetOut(BaseModel):
    id: str
    temperature: str | None
    raw_name: str
    stem: str
    stages: list[StageStatusOut]


class DataRootIn(BaseModel):
    data_root: str


class DataRootOut(BaseModel):
    data_root: str
    raw_exists: bool
    processed_exists: bool
    n_datasets: int


class LatticeOut(BaseModel):
    a: float | None
    b: float | None
    c: float | None


class VolumeMetaOut(BaseModel):
    id: str
    stage: str
    kind: str
    shape: list[int]
    h_range: list[float]
    k_range: list[float]
    l_range: list[float]
    lattice: LatticeOut
    ub_matrix: list[list[float]] | None = None
    planes: list[str]


class DeltaPdfMetaOut(BaseModel):
    id: str
    shape: list[int]
    x_range: list[float]
    y_range: list[float]
    z_range: list[float]
    lattice: LatticeOut
    q_max: float | None
    planes: list[str]


class BraggPeakWidthOut(BaseModel):
    index: int
    source_node_hkl: list[int] | None = None
    center_hkl: list[float]
    q_abs: float
    intensity: float | None = None
    local_background: float | None = None
    width_hkl: list[float]
    width_q: list[float]
    measured_width_hkl: list[float] | None = None
    measured_width_q: list[float] | None = None
    resolution_limited: list[bool] | None = None
    principal_directions_hkl: list[list[float]]
    fit_kind: str


class BraggProfileOut(BaseModel):
    dataset_id: str
    profile_path: str | None = None
    has_profile: bool
    schema_version: int = 1
    width_labels: list[str] = []
    hkl_width_labels: list[str] = []
    width_units: dict[str, str] = {}
    n_peaks: int = 0
    fit_covariance: bool = False
    punch_frame: str | None = None
    peaks: list[BraggPeakWidthOut] = []


class StageParamsIn(BaseModel):
    """Curated, optional per-stage overrides (None = use the validated default)."""

    rings_n_patches: int | None = None
    rings_n_fourier: int | None = None
    rings_slice_axis: str | None = None
    # "patched" (non-parametric per-patch) | "parametric" (separable Ring(|Q|) ×
    # per-shell Fourier texture); rings_ring_width is the ring-width / rolling
    # window (Å⁻¹); rings_radial_mode is "rolling" (continuous) | "peaks".
    rings_model: str | None = None
    rings_ring_width: float | None = None
    rings_radial_mode: str | None = None
    punch_min_intensity: float | None = None
    punch_search_n_mad: float | None = None
    punch_mode: str | None = None
    punch_radius_h: float | None = None
    punch_radius_k: float | None = None
    punch_radius_l: float | None = None
    punch_margin: float | None = None
    punch_phi_tail_hkl: float | None = None
    # Q-space punch (opt-in): frame "q" + isotropic or per-a*,b*,c* radius (Å^-1)
    punch_frame: str | None = None
    punch_q_radius: float | None = None
    punch_q_radius_a: float | None = None
    punch_q_radius_b: float | None = None
    punch_q_radius_c: float | None = None
    incident_beam_q_radius_a: float | None = None
    incident_beam_q_radius_b: float | None = None
    incident_beam_q_radius_c: float | None = None
    incident_beam_q_margin: float | None = None
    # Legacy HKL direct-beam overrides kept for API compatibility.
    incident_beam_radius_h: float | None = None
    incident_beam_radius_k: float | None = None
    incident_beam_radius_l: float | None = None
    incident_beam_margin: float | None = None
    # Phase 3: fit a tilted 3×3 resolution ellipsoid (covariance) per Bragg peak
    punch_fit_covariance: bool | None = None
    # Experimental diagnostic: do not floor/cap Bragg covariance-fit radii.
    punch_fit_unconstrained: bool | None = None
    # Fit a tilted covariance ellipsoid to the direct-beam remnant at the origin
    incident_beam_fit_covariance: bool | None = None
    backfill_method: str | None = None
    flatten_estimator: str | None = None
    flatten_floor_percentile: float | None = None
    pdf_apodization: str | None = None
    pdf_gaussian_sigma: float | None = None
    pdf_crop_h: float | None = None
    pdf_crop_k: float | None = None
    pdf_crop_l: float | None = None
    pdf_q_min: float | None = None
    pdf_q_max: float | None = None


class PipelineRunRequest(BaseModel):
    dataset_id: str
    flatten_enabled: bool = True
    force: bool = False
    force_from: str | None = None
    stages: list[str] | None = None  # enabled-stage subset; None → all stages
    params: StageParamsIn = StageParamsIn()


class JobOut(BaseModel):
    id: str
    input: str
    status: str
    error: str | None = None
    n_events: int
