"""Library pipeline orchestration: raw .nxs/HKL volume → 3D-ΔPDF.

This is the in-process, testable home for the five-stage workflow that
``examples/run_pipeline.py`` previously drove by shelling out to per-stage example
scripts with environment-variable parameters.  Each stage here calls the existing
library primitives directly:

    1. rings    → :class:`~ndiff.preprocessing.PatchedRadialRingModel`
                  (+ :func:`~ndiff.preprocessing.confirm_ring_shells_across_h`)
    2. punch    → :class:`~ndiff.analysis.BraggRemover`
    3. backfill → :func:`~ndiff.analysis.backfill_bragg`
    4. flatten  → :func:`~ndiff.preprocessing.flatten_radial_background`
    5. pdf      → :func:`~ndiff.analysis.compute_delta_pdf`

:func:`run_pipeline` chains the stages with the same on-disk file naming, the
skip-if-exists resume behaviour, and the stale-ΔPDF guard as the original script,
and reports progress through an optional callback so a server (or CLI) can stream
it.  Stage parameter defaults mirror the validated ``cc_on`` presets.

The heavy diffraction maths is **not** reimplemented here — only the orchestration
and the per-stage glue (build the model/params, call it, write the output).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

import ndiff
from ndiff.analysis import (
    BraggRemover,
    DeltaPDF,
    backfill_bragg,
    compute_delta_pdf,
    invert_delta_pdf,
)
from ndiff.core import HKLVolume
from ndiff.preprocessing import (
    ParametricRingModel,
    PatchedRadialRingModel,
    azimuthal_sampling_mask,
    confirm_ring_shells_across_h,
    flatten_radial_background,
)

__all__ = [
    "RingParams",
    "PunchParams",
    "BackfillParams",
    "FlattenParams",
    "DeltaPdfParams",
    "PipelineParams",
    "PipelinePaths",
    "STAGES",
    "remove_rings",
    "punch_bragg",
    "backfill",
    "flatten",
    "delta_pdf",
    "delta_pdf_transform_config",
    "pdf_consistency_check",
    "pipeline_paths",
    "run_pipeline",
]

# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------
# progress(stage, status, fraction, message)
#   stage    : one of STAGES ("rings", "punch", ...)
#   status   : "start" | "progress" | "done" | "skip" | "error"
#   fraction : 0..1 within the stage, or None when not measurable
#   message  : human-readable line (also suitable for a log stream)
Status = Literal["start", "progress", "done", "skip", "error"]
ProgressFn = Callable[[str, Status, float | None, str], None]

STAGES: tuple[str, ...] = (
    "rings", "punch", "backfill", "flatten", "pdf", "pdf_check")


def _emit(progress: ProgressFn | None, stage: str, status: Status,
          fraction: float | None, message: str) -> None:
    if progress is not None:
        progress(stage, status, fraction, message)


# ---------------------------------------------------------------------------
# Stage parameters (defaults mirror the validated cc_on presets)
# ---------------------------------------------------------------------------
@dataclass
class RingParams:
    """Powder-ring removal (per-slice ``PatchedRadialRingModel`` or
    ``ParametricRingModel``, selected by ``ring_model``)."""

    q_min: float = 1.5
    q_max: float = 10.5
    slice_axis: str = "H"          # H fits 0kl/KL slices; K → h0l; L → hk0
    profile_method: str = "median"
    n_fourier: int = 6
    n_patches: int = 36
    q_step: float = 0.02
    texture_q_smooth: float = 0.02
    texture_ridge: float = 0.08
    ring_amp_cap: float = 3.0       # per-shell amplitude ceiling × cross-stack norm
    confirm_rings: bool = True      # confirm real |Q| shells across the stack axis
    # "patched" (default, non-parametric per-patch) | "parametric" (separable
    # Ring(|Q|) × per-shell Fourier texture — binning-free azimuthal LS, so the
    # statistics don't vary with |Q|).
    ring_model: str = "patched"
    ring_width: float = 0.24        # parametric: ring width / rolling window (Å⁻¹)
    ring_eta0: float = 0.5          # parametric peaks: initial pseudo-Voigt Lorentzian frac
    # parametric radial model: "rolling" (continuous Ring(|Q|), thick window swept
    # Qmin→Qmax) | "peaks" (discrete pseudo-Voigt rings)
    ring_radial_mode: str = "rolling"
    ring_roll_step: float = 0.04    # parametric rolling: |Q| spacing of window centres


@dataclass
class PunchParams:
    """Bragg / satellite punch (``BraggRemover``).

    ``min_prominence`` defaults to 1.0 to match the original
    ``examples/punch_bragg_3d.py`` (which hardcodes it; its ``MIN_PROM`` env var
    was never read).
    """

    mode: str = "both"
    punch_radii: tuple[float, float, float] = (0.09, 0.12, 0.45)
    min_intensity: float = 0.8
    min_prominence: float = 1.0
    integer_n_mad: float | None = None
    integer_q_step: float | None = None
    integer_optimize_position: bool = True
    integer_optimize_shape: bool = True
    integer_fit_covariance: bool = False
    integer_fit_threshold_frac: float = 0.35
    integer_fit_radius_n_sigma: float = 2.5
    integer_fit_max_radius_hkl: tuple[float, float, float] | None = None
    integer_h_guard_hkl: float | None = 0.12
    integer_local_prominence_n_mad: float | None = 8.0
    integer_local_min_prominence: float = 0.0
    search_n_mad: float = 4.0
    search_min_intensity: float = 0.8
    search_min_prominence: float = 0.8
    search_exclude_h_centers: tuple[float, ...] | None = None
    search_exclude_h_half_width: float = 0.08
    search_exclude_h_fractions: tuple[float, ...] | None = (0.3333, 0.6667)
    margin: float = 0.02
    max_radius_scale: float = 2.0
    phi_tail_hkl: float = 0.12
    incident_beam_radii: tuple[float, float, float] = (0.24, 0.24, 0.90)
    incident_beam_margin: float = 0.12
    incident_beam_phi_tail_hkl: float = 0.0
    incident_beam_ellipsoid_radii_hkl: tuple[float, float, float] | None = (
        0.15, 0.50, 1.00,
    )
    incident_beam_sphere_radius_hkl: float | None = None
    # Q-space punch (ROADMAP Phase 6, default since Phase 4).  The punch footprint
    # is the reciprocal-Å⁻¹ resolution floor (per a*,b*,c*); the per-peak fit +
    # φ-tail still modulate it (adaptive), so this reproduces the legacy HKL punch
    # while being lattice/temperature-portable.  ΔPDF A/B vs the old HKL default:
    # r=0.9998 (22K).  Set punch_frame="hkl" + punch_radii to restore the legacy
    # rlu footprint; punch_q_radii ≈ HKL radii (0.09,0.12,0.45) × b*(22K).
    punch_frame: str = "q"
    punch_q_radius: float | None = None
    punch_q_radii: tuple[float, float, float] | None = (0.097, 0.072, 0.115)


@dataclass
class BackfillParams:
    """Backfill of punched Bragg holes (``backfill_bragg``)."""

    method: str = "q_shell"
    laue_class: str = "mmm"
    local_radius: int = 2
    local_min_count: int = 8
    q_shell_step: float = 0.05
    q_shell_min_count: int = 20
    tv_lam: float = 0.2
    tv_iter: int = 80


@dataclass
class FlattenParams:
    """Isotropic radial-background flatten (``flatten_radial_background``)."""

    estimator: str = "floor"
    floor_percentile: float = 25.0
    q_step: float = 0.05
    smooth: float = 0.10
    snip_width: float = 0.3
    min_count: int = 20
    q_range: tuple[float, float] | None = None


@dataclass
class DeltaPdfParams:
    """3D-ΔPDF transform (``compute_delta_pdf``)."""

    apodization: str = "gaussian"
    gaussian_sigma: float = 0.4
    zero_pad: bool = True
    subtract_mean: bool = True
    crop_hkl: tuple[float, float, float] | None = (4.0, 8.0, 15.0)
    # None = off; float = isotropic blur σ; (σ_H, σ_K, σ_L) = per-axis.
    subtract_smooth_bg: float | tuple[float, float, float] | None = None


@dataclass
class PipelineParams:
    """All stage parameters plus the flatten on/off toggle."""

    rings: RingParams = field(default_factory=RingParams)
    punch: PunchParams = field(default_factory=PunchParams)
    backfill: BackfillParams = field(default_factory=BackfillParams)
    flatten: FlattenParams = field(default_factory=FlattenParams)
    delta_pdf: DeltaPdfParams = field(default_factory=DeltaPdfParams)
    flatten_enabled: bool = True
    # Stage 6 — back-FFT round-trip consistency check (inverse-transform the
    # ΔPDF and compare to the diffuse data it came from); writes a metric JSON
    # and a comparison figure, no large volume.
    pdf_check_enabled: bool = True


# ---------------------------------------------------------------------------
# Stage 1 — powder-ring removal (per-slice driver)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _SliceConfig:
    axis_name: str
    axis_dim: int
    axis_attr: str
    plane: str


_SLICE_CONFIGS = {
    "H": _SliceConfig("H", 0, "h_axis", "0kl"),
    "K": _SliceConfig("K", 1, "k_axis", "h0l"),
    "L": _SliceConfig("L", 2, "l_axis", "hk0"),
}


def _slice_volume(v: HKLVolume, cfg: _SliceConfig, index: int) -> HKLVolume:
    """Return a 3D one-plane HKLVolume view along ``cfg.axis_dim``."""
    sl = [slice(None), slice(None), slice(None)]
    sl[cfg.axis_dim] = slice(index, index + 1)
    kwargs = {
        "data": v.data[tuple(sl)],
        "sigma": v.sigma[tuple(sl)],
        "mask": v.mask[tuple(sl)],
        cfg.axis_attr: getattr(v, cfg.axis_attr)[index:index + 1],
    }
    return dataclasses.replace(v, **kwargs)


def _take_plane(arr: np.ndarray, cfg: _SliceConfig, index: int) -> np.ndarray:
    return np.take(arr, index, axis=cfg.axis_dim)


def _assign_plane(dest: np.ndarray, cfg: _SliceConfig, index: int,
                  plane: np.ndarray) -> None:
    sl: list[slice | int] = [slice(None), slice(None), slice(None)]
    sl[cfg.axis_dim] = index
    dest[tuple(sl)] = plane


def remove_rings(vol: HKLVolume, params: RingParams | None = None, *,
                 progress: ProgressFn | None = None) -> HKLVolume:
    """Subtract powder rings from every plane along the stack axis independently.

    Ports the validated per-slice driver of ``examples/remove_rings_3d.py``:
    optionally confirm the real |Q| shells across the stack axis (so a
    Bragg-fed phantom on one plane washes out), cap each shell's per-plane
    amplitude, then fit and subtract ``PatchedRadialRingModel`` per plane.
    """
    p = params or RingParams()
    cfg = _SLICE_CONFIGS[p.slice_axis.strip().upper()]
    q_range = (p.q_min, p.q_max)
    axis_values = getattr(vol, cfg.axis_attr)

    model_tag = (f"{p.ring_model}:{p.ring_radial_mode}"
                 if p.ring_model.strip().lower() == "parametric" else p.ring_model)
    _emit(progress, "rings", "start", 0.0,
          f"ring removal [{model_tag}]: {axis_values.size} {cfg.axis_name} "
          f"planes (plane={cfg.plane}, |Q| {q_range})")

    ring_centers = ring_halfwidths = ring_ceilings = None
    if p.confirm_rings:
        ring_centers, ring_halfwidths, ring_amps = confirm_ring_shells_across_h(
            vol, plane=cfg.plane, q_range=q_range)
        if p.ring_amp_cap > 0 and ring_amps.size:
            ring_ceilings = p.ring_amp_cap * ring_amps
        _emit(progress, "rings", "progress", 0.0,
              f"confirmed {ring_centers.size} ring shells across "
              f"{cfg.axis_name} (amp cap {p.ring_amp_cap}×)")

    model: PatchedRadialRingModel | ParametricRingModel
    if p.ring_model.strip().lower() == "parametric":
        model = ParametricRingModel(
            plane=cfg.plane,
            q_step=p.q_step,
            n_fourier=p.n_fourier,
            profile_method=p.profile_method,
            texture_ridge=p.texture_ridge,
            ring_width=p.ring_width,
            eta0=p.ring_eta0,
            radial_mode=p.ring_radial_mode,
            roll_step=p.ring_roll_step,
            allowed_ring_centers=ring_centers,
            allowed_ring_halfwidths=ring_halfwidths,
            allowed_ring_ceilings=ring_ceilings,
        )
    else:
        model = PatchedRadialRingModel(
            plane=cfg.plane,
            q_step=p.q_step,
            n_patches=p.n_patches,
            n_fourier=p.n_fourier,
            profile_method=p.profile_method,
            texture_q_smooth=p.texture_q_smooth,
            texture_ridge=p.texture_ridge,
            allowed_ring_centers=ring_centers,
            allowed_ring_halfwidths=ring_halfwidths,
            allowed_ring_ceilings=ring_ceilings,
        )

    res_data = np.empty_like(vol.data)
    out_mask = vol.mask.copy()
    n_skipped = 0
    n = int(axis_values.size)

    for ip in range(n):
        sl = _slice_volume(vol, cfg, ip)
        valid = sl.mask & np.isfinite(sl.data)
        if int(valid.sum()) < model.min_voxels_per_patch:
            _assign_plane(res_data, cfg, ip, _take_plane(sl.data, cfg, 0))
            n_skipped += 1
            continue

        keep = azimuthal_sampling_mask(sl, plane=cfg.plane, min_count_frac=0.25,
                                       q_range=q_range)
        src = dataclasses.replace(sl, mask=keep)
        _assign_plane(out_mask, cfg, ip, _take_plane(keep, cfg, 0))

        try:
            model.fit(src, q_range=q_range)        # caches the per-plane fit
            _, I_ring = model.subtract(src)         # uses the cached fit
        except Exception as exc:  # numerical edge case on a plane — leave as-is
            _assign_plane(res_data, cfg, ip, _take_plane(sl.data, cfg, 0))
            n_skipped += 1
            _emit(progress, "rings", "progress", (ip + 1) / n,
                  f"{cfg.axis_name}[{ip}] fit failed ({exc}); left as-is")
            continue

        I_ring2d = _take_plane(I_ring, cfg, 0)
        sl_data2d = _take_plane(sl.data, cfg, 0)
        _assign_plane(res_data, cfg, ip, sl_data2d - I_ring2d)

        if ip % 30 == 0 or ip == n - 1:
            _emit(progress, "rings", "progress", (ip + 1) / n,
                  f"{cfg.axis_name}[{ip + 1}/{n}]")

    out_vol = dataclasses.replace(vol, data=res_data, mask=out_mask)
    _emit(progress, "rings", "done", 1.0,
          f"ring removal complete ({n_skipped} planes left unchanged)")
    return out_vol


# ---------------------------------------------------------------------------
# Stage 2 — Bragg / satellite punch
# ---------------------------------------------------------------------------
def punch_bragg(vol: HKLVolume, params: PunchParams | None = None, *,
                progress: ProgressFn | None = None) -> HKLVolume:
    """Detect and punch Bragg/satellite peaks; return the masked volume."""
    p = params or PunchParams()
    _emit(progress, "punch", "start", None, f"Bragg punch (mode={p.mode})")

    remover = BraggRemover(
        mode=p.mode, punch_radii=p.punch_radii, min_intensity=p.min_intensity,
        min_prominence=p.min_prominence,
        integer_n_mad=p.integer_n_mad, integer_q_step=p.integer_q_step,
        integer_optimize_position=p.integer_optimize_position,
        integer_optimize_shape=p.integer_optimize_shape,
        integer_fit_covariance=p.integer_fit_covariance,
        integer_fit_threshold_frac=p.integer_fit_threshold_frac,
        integer_fit_radius_n_sigma=p.integer_fit_radius_n_sigma,
        integer_fit_max_radius_hkl=p.integer_fit_max_radius_hkl,
        integer_h_guard_hkl=p.integer_h_guard_hkl,
        integer_local_prominence_n_mad=p.integer_local_prominence_n_mad,
        integer_local_min_prominence=p.integer_local_min_prominence,
        intensity_scale=True, max_radius_scale=p.max_radius_scale, margin=p.margin,
        punch_incident_beam=True, incident_beam_radii=p.incident_beam_radii,
        incident_beam_margin=p.incident_beam_margin,
        incident_beam_phi_tail_hkl=p.incident_beam_phi_tail_hkl,
        incident_beam_ellipsoid_radii_hkl=p.incident_beam_ellipsoid_radii_hkl,
        incident_beam_sphere_radius_hkl=p.incident_beam_sphere_radius_hkl,
        phi_tail_hkl=p.phi_tail_hkl,
        search_n_mad=p.search_n_mad, search_min_intensity=p.search_min_intensity,
        search_min_prominence=p.search_min_prominence,
        search_exclude_h_centers=p.search_exclude_h_centers,
        search_exclude_h_half_width=p.search_exclude_h_half_width,
        search_exclude_h_fractions=p.search_exclude_h_fractions,
        punch_frame=p.punch_frame, punch_q_radius=p.punch_q_radius,
        punch_q_radii=p.punch_q_radii,
    )
    peaks = remover.detect_peaks(vol)
    keep = remover.build_mask(vol)
    valid = vol.mask & np.isfinite(vol.data)
    punched = int((valid & ~keep).sum())
    out_vol = dataclasses.replace(vol, mask=vol.mask & keep)
    _emit(progress, "punch", "done", 1.0,
          f"detected {len(peaks)} peaks; punched {punched:,} voxels")
    return out_vol


# ---------------------------------------------------------------------------
# Stage 3 — backfill punched holes
# ---------------------------------------------------------------------------
def backfill(vol: HKLVolume, params: BackfillParams | None = None, *,
             progress: ProgressFn | None = None) -> HKLVolume:
    """Fill punched Bragg holes; return an all-valid volume for the FFT."""
    p = params or BackfillParams()
    _emit(progress, "backfill", "start", None, f"backfill (method={p.method})")
    filled = backfill_bragg(
        vol, method=p.method, laue_class=p.laue_class,  # type: ignore[arg-type]
        local_radius=p.local_radius, local_min_count=p.local_min_count,
        q_shell_step=p.q_shell_step, q_shell_min_count=p.q_shell_min_count,
        tv_lam=p.tv_lam, tv_iter=p.tv_iter,
    )
    _emit(progress, "backfill", "done", 1.0, "backfill complete")
    return filled


# ---------------------------------------------------------------------------
# Stage 4 — radial-background flatten (the explicit background remover)
# ---------------------------------------------------------------------------
def flatten(vol: HKLVolume, params: FlattenParams | None = None, *,
            progress: ProgressFn | None = None) -> HKLVolume:
    """Subtract the smooth isotropic radial pedestal; return the flattened volume."""
    p = params or FlattenParams()
    _emit(progress, "flatten", "start", None,
          f"radial-background flatten (estimator={p.estimator})")
    res = flatten_radial_background(
        vol, q_step=p.q_step, estimator=p.estimator,
        floor_percentile=p.floor_percentile, snip_width=p.snip_width,
        smooth=p.smooth, min_count=p.min_count, q_range=p.q_range,
    )
    _emit(progress, "flatten", "done", 1.0,
          f"flatten complete (bg max {float(np.nanmax(res.bg_curve)):.4g})")
    return res.volume


# ---------------------------------------------------------------------------
# Stage 5 — 3D-ΔPDF transform
# ---------------------------------------------------------------------------
def delta_pdf(vol: HKLVolume, params: DeltaPdfParams | None = None, *,
              progress: ProgressFn | None = None) -> DeltaPDF:
    """Compute the 3D-ΔPDF (FFT) from the cleaned diffuse volume."""
    p = params or DeltaPdfParams()
    _emit(progress, "pdf", "start", None, f"3D-ΔPDF FFT (apodize={p.apodization})")
    dpdf = compute_delta_pdf(
        vol, apodization=p.apodization,  # type: ignore[arg-type]
        gaussian_sigma=p.gaussian_sigma,
        zero_pad=p.zero_pad, subtract_mean=p.subtract_mean,
        real_space_angstrom=True, crop_hkl=p.crop_hkl,
        subtract_smooth_bg=p.subtract_smooth_bg,
    )
    _emit(progress, "pdf", "done", 1.0,
          f"ΔPDF complete (|Q|max {dpdf.q_max:.2f} Å⁻¹, shape {dpdf.data.shape})")
    return dpdf


def _param_string(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, tuple):
        return ",".join(f"{float(v):.12g}" for v in value)
    return f"{float(value):.12g}"  # type: ignore[arg-type]


def delta_pdf_transform_config(p: DeltaPdfParams) -> str:
    """Build the stable ``transform_config`` stamp used by the stale-ΔPDF guard.

    Matches the string written by ``examples/delta_pdf.py`` so caches written by
    either path are interchangeable.
    """
    return ";".join((
        f"apodize={p.apodization}",
        f"gaussian_sigma={p.gaussian_sigma:.12g}",
        f"zero_pad={int(p.zero_pad)}",
        f"subtract_mean={int(p.subtract_mean)}",
        f"crop_hkl={_param_string(p.crop_hkl)}",
        f"subtract_bg={_param_string(p.subtract_smooth_bg)}",
    ))


def write_delta_pdf_h5(dpdf: DeltaPDF, vol: HKLVolume, p: DeltaPdfParams,
                       source_name: str, out_path: Path) -> None:
    """Write the ΔPDF to the same HDF5 schema the viewers read.

    Mirrors ``examples/delta_pdf.py`` (data + x/y/z axes, provenance attrs, and
    the direct-lattice constants for unit-cell gridlines).
    """
    import h5py

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as fh:
        fh.create_dataset("data", data=dpdf.data, compression="gzip",
                          compression_opts=4)
        fh.create_dataset("x_axis", data=dpdf.x_axis)
        fh.create_dataset("y_axis", data=dpdf.y_axis)
        fh.create_dataset("z_axis", data=dpdf.z_axis)
        fh.attrs["q_max"] = dpdf.q_max
        fh.attrs["apodization"] = dpdf.apodization
        fh.attrs["source_file"] = source_name
        fh.attrs["crop_hkl"] = _param_string(p.crop_hkl)
        fh.attrs["subtract_smooth_bg"] = _param_string(p.subtract_smooth_bg)
        fh.attrs["gaussian_sigma"] = p.gaussian_sigma
        fh.attrs["zero_pad"] = int(p.zero_pad)
        fh.attrs["subtract_mean"] = int(p.subtract_mean)
        fh.attrs["transform_config"] = delta_pdf_transform_config(p)
        try:
            direct = 2 * np.pi * np.linalg.inv(vol.ub_matrix).T
            fh.attrs["lat_a"] = float(np.linalg.norm(direct[:, 0]))
            fh.attrs["lat_b"] = float(np.linalg.norm(direct[:, 1]))
            fh.attrs["lat_c"] = float(np.linalg.norm(direct[:, 2]))
        except np.linalg.LinAlgError:
            pass


def _crop_hkl(vol: HKLVolume, crop_hkl: tuple[float, float, float] | None
              ) -> HKLVolume:
    """Symmetric ±crop_hkl crop of *vol* (matches ``compute_delta_pdf``)."""
    if crop_hkl is None:
        return vol
    h_max, k_max, l_max = crop_hkl
    ih = np.where(np.abs(vol.h_axis) <= h_max)[0]
    ik = np.where(np.abs(vol.k_axis) <= k_max)[0]
    il = np.where(np.abs(vol.l_axis) <= l_max)[0]
    sl = (slice(ih[0], ih[-1] + 1), slice(ik[0], ik[-1] + 1),
          slice(il[0], il[-1] + 1))
    return dataclasses.replace(
        vol, data=vol.data[sl], sigma=vol.sigma[sl], mask=vol.mask[sl],
        h_axis=vol.h_axis[ih[0]:ih[-1] + 1], k_axis=vol.k_axis[ik[0]:ik[-1] + 1],
        l_axis=vol.l_axis[il[0]:il[-1] + 1])


def pdf_consistency_check(
    vol: HKLVolume,
    dpdf: DeltaPDF,
    p: DeltaPdfParams,
    *,
    h_values: Sequence[float] = (0.0, 1.0 / 3.0, 1.0),
    figure_path: Path | None = None,
) -> dict:
    """Back-FFT round-trip check: inverse-transform *dpdf* and compare to *vol*.

    Inverse-transforms the ΔPDF back to reciprocal space
    (:func:`~ndiff.analysis.invert_delta_pdf`) and measures how well it
    reproduces the cleaned diffuse data it was built from (cropped to the
    transform window).  Returns a metrics dict — Pearson ``r`` and normalised
    RMS residual over the reliably-recovered region, plus per-H-plane ``r`` —
    and, when *figure_path* is given, writes a ``data | back-FFT | residual``
    comparison PNG.  A faithful ΔPDF gives ``r ≈ 1``; a gross mismatch flags a
    transform bug or an over-aggressive ``crop_hkl`` / apodization.
    """
    recon = invert_delta_pdf(dpdf, deapodize=True)
    data_vol = _crop_hkl(vol, p.crop_hkl)
    data = np.where(np.isfinite(data_vol.masked_data()), data_vol.data, 0.0)
    rec = recon.data
    reliable = recon.mask & np.isfinite(data)

    def _r(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
        a, b = a[m], b[m]
        return float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else float("nan")

    resid = rec - data
    rms = float(np.sqrt(np.mean(resid[reliable] ** 2))) if reliable.any() else 0.0
    denom = (float(np.sqrt(np.mean(data[reliable] ** 2)))
             if reliable.any() else 0.0) or 1.0
    per_plane: dict[str, float] = {}
    rows = []
    for hv in h_values:
        ih = int(np.argmin(np.abs(recon.h_axis - hv)))
        h_actual = float(recon.h_axis[ih])
        r_plane = _r(rec[ih], data[ih], reliable[ih])
        per_plane[f"{h_actual:+.3f}"] = r_plane
        rows.append((h_actual, data[ih], rec[ih], reliable[ih], r_plane))

    metrics = {
        "pearson_r": _r(rec, data, reliable),
        "normalized_rms": rms / denom,
        "rms": rms,
        "n_voxels": int(reliable.sum()),
        "per_plane_r": per_plane,
        "crop_hkl": list(p.crop_hkl) if p.crop_hkl else None,
        "apodization": p.apodization,
    }

    if figure_path is not None:
        _write_consistency_figure(rows, Path(figure_path))
    return metrics


def _write_consistency_figure(rows: list, out_png: Path) -> None:
    """data | back-FFT | residual panels per H plane (lazy matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4 * n), squeeze=False)
    titles = ["data  I(Q)", "back-FFT  IFFT[ΔPDF]", "residual (data − recon)"]
    for r, (hv, d2, r2, m2, rho) in enumerate(rows):
        finite = d2[np.isfinite(d2)]
        vmax = float(np.nanpercentile(finite, 99)) if finite.size else 1.0
        rscale = float(np.nanpercentile(np.abs(d2[m2]), 99)) if m2.any() else 1.0
        panels = [(d2, "magma", 0.0, vmax), (r2, "magma", 0.0, vmax),
                  (d2 - r2, "RdBu_r", -rscale, rscale)]
        for c, (panel, cmap, vmin, vmx) in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(panel.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmx,
                      aspect="auto")
            extra = f"  r={rho:.4f}" if c == 1 else ""
            ax.set_title(f"H={hv:+.3f}  {titles[c]}{extra}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def _pdf_is_current(pdf_path: Path, expected_src: str, expected_config: str) -> bool:
    if not pdf_path.exists():
        return False
    try:
        import h5py

        with h5py.File(pdf_path, "r") as fh:
            return (
                fh.attrs.get("source_file", "") == expected_src
                and fh.attrs.get("transform_config", "") == expected_config
            )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclass
class PipelinePaths:
    """Chained on-disk artifacts for one input volume (names match run_pipeline)."""

    input: Path
    ringremoved: Path
    braggpunched: Path
    backfilled: Path
    flattened: Path
    delta_pdf: Path
    pdf_input: Path     # flattened if flatten enabled, else backfilled
    pdf_check_json: Path
    pdf_check_png: Path


def pipeline_paths(input_path: str | Path, *, proc_dir: str | Path | None = None,
                   flatten_enabled: bool = True) -> PipelinePaths:
    """Resolve the chained output paths for ``input_path``.

    Names match ``examples/run_pipeline.py`` exactly so the viewers and the
    multi-temperature auto-detection find the same ``*_delta_pdf.h5`` files.
    """
    inp = Path(input_path)
    proc = Path(proc_dir) if proc_dir is not None else Path("data/processed")
    ring = proc / f"{inp.stem}_ringremoved.h5"
    punch = proc / f"{ring.stem}_braggpunched.h5"
    fill = proc / f"{punch.stem}_backfilled.h5"
    flat = proc / f"{fill.stem}_flattened.h5"
    pdf = proc / f"{fill.stem}_delta_pdf.h5"
    return PipelinePaths(
        input=inp, ringremoved=ring, braggpunched=punch, backfilled=fill,
        flattened=flat, delta_pdf=pdf,
        pdf_input=flat if flatten_enabled else fill,
        pdf_check_json=proc / f"{pdf.stem}_consistency.json",
        pdf_check_png=proc / f"{pdf.stem}_consistency.png",
    )


def run_pipeline(
    input_path: str | Path,
    params: PipelineParams | None = None,
    *,
    proc_dir: str | Path | None = None,
    stages: Sequence[str] = STAGES,
    force: bool = False,
    force_from: str | None = None,
    progress: ProgressFn | None = None,
) -> PipelinePaths:
    """Run the full pipeline, resuming from existing outputs.

    Parameters
    ----------
    input_path:
        Raw ``.nxs`` (or ndiff ``.h5``) input volume.
    params:
        Stage parameters; defaults to the validated ``cc_on`` presets.
    proc_dir:
        Directory for the chained ``.h5`` outputs (default ``data/processed``).
    stages:
        Subset of :data:`STAGES` to consider (always run in canonical order).
        ``flatten`` is additionally gated by ``params.flatten_enabled``.
    force:
        Recompute every selected stage even if its output exists.
    force_from:
        Recompute from this stage onward (one of :data:`STAGES`).
    progress:
        Optional ``progress(stage, status, fraction, message)`` callback.

    Returns
    -------
    PipelinePaths
        The resolved artifact paths (whether freshly computed or pre-existing).
    """
    p = params or PipelineParams()
    if force_from is not None and force_from not in STAGES:
        raise ValueError(f"force_from={force_from!r}; choose one of {STAGES}")

    selected = [s for s in STAGES if s in set(stages)]
    paths = pipeline_paths(input_path, proc_dir=proc_dir,
                           flatten_enabled=p.flatten_enabled)
    paths.delta_pdf.parent.mkdir(parents=True, exist_ok=True)

    def forced(stage: str) -> bool:
        if force:
            return True
        if force_from is not None:
            return STAGES.index(stage) >= STAGES.index(force_from)
        return False

    def want(stage: str) -> bool:
        return stage in selected

    # --- stage 1: rings -----------------------------------------------------
    if want("rings"):
        if paths.ringremoved.exists() and not forced("rings"):
            _emit(progress, "rings", "skip", None,
                  f"{paths.ringremoved.name} exists")
        else:
            vol = ndiff.load(paths.input)
            out = remove_rings(vol, p.rings, progress=progress)
            ndiff.save(out, paths.ringremoved)

    # --- stage 2: punch -----------------------------------------------------
    if want("punch"):
        if paths.braggpunched.exists() and not forced("punch"):
            _emit(progress, "punch", "skip", None,
                  f"{paths.braggpunched.name} exists")
        else:
            vol = ndiff.load(paths.ringremoved)
            out = punch_bragg(vol, p.punch, progress=progress)
            ndiff.save(out, paths.braggpunched)

    # --- stage 3: backfill --------------------------------------------------
    if want("backfill"):
        if paths.backfilled.exists() and not forced("backfill"):
            _emit(progress, "backfill", "skip", None,
                  f"{paths.backfilled.name} exists")
        else:
            vol = ndiff.load(paths.braggpunched)
            out = backfill(vol, p.backfill, progress=progress)
            ndiff.save(out, paths.backfilled)

    # --- stage 4: flatten (optional) ---------------------------------------
    if want("flatten") and p.flatten_enabled:
        if paths.flattened.exists() and not forced("flatten"):
            _emit(progress, "flatten", "skip", None,
                  f"{paths.flattened.name} exists")
        else:
            vol = ndiff.load(paths.backfilled)
            out = flatten(vol, p.flatten, progress=progress)
            ndiff.save(out, paths.flattened)
    elif want("flatten"):
        _emit(progress, "flatten", "skip", None,
              "flatten disabled (no background removed)")

    # --- stage 5: ΔPDF (with stale-cache guard) -----------------------------
    dpdf_obj: DeltaPDF | None = None     # in-memory ΔPDF if (re)computed here
    pdf_vol: HKLVolume | None = None     # its input volume, for the check stage
    if want("pdf"):
        expected_config = delta_pdf_transform_config(p.delta_pdf)
        is_current = _pdf_is_current(paths.delta_pdf, paths.pdf_input.name,
                                     expected_config)
        if is_current and not forced("pdf"):
            _emit(progress, "pdf", "skip", None,
                  f"{paths.delta_pdf.name} is current")
        else:
            if paths.delta_pdf.exists() and not is_current:
                _emit(progress, "pdf", "progress", None,
                      f"{paths.delta_pdf.name} stale — recomputing")
            pdf_vol = ndiff.load(paths.pdf_input)
            dpdf_obj = delta_pdf(pdf_vol, p.delta_pdf, progress=progress)
            write_delta_pdf_h5(dpdf_obj, pdf_vol, p.delta_pdf,
                               paths.pdf_input.name, paths.delta_pdf)

    # --- stage 6: back-FFT round-trip consistency check ---------------------
    if want("pdf_check") and p.pdf_check_enabled:
        outputs_exist = (paths.pdf_check_json.exists()
                         and paths.pdf_check_png.exists())
        # Re-run if the ΔPDF was (re)computed this run, if outputs are missing,
        # or if explicitly forced; otherwise the cached check is still valid.
        if dpdf_obj is None and outputs_exist and not forced("pdf_check"):
            _emit(progress, "pdf_check", "skip", None,
                  f"{paths.pdf_check_json.name} exists")
        else:
            _emit(progress, "pdf_check", "start", None,
                  "back-FFT round-trip consistency check")
            if pdf_vol is None:
                pdf_vol = ndiff.load(paths.pdf_input)
            if dpdf_obj is None:
                dpdf_obj = delta_pdf(pdf_vol, p.delta_pdf)
            metrics = pdf_consistency_check(
                pdf_vol, dpdf_obj, p.delta_pdf,
                figure_path=paths.pdf_check_png)
            paths.pdf_check_json.parent.mkdir(parents=True, exist_ok=True)
            paths.pdf_check_json.write_text(json.dumps(metrics, indent=2))
            _emit(progress, "pdf_check", "done", 1.0,
                  f"back-FFT vs data: r={metrics['pearson_r']:.5f}, "
                  f"normalised RMS={metrics['normalized_rms']:.3e}")
    elif want("pdf_check"):
        _emit(progress, "pdf_check", "skip", None, "consistency check disabled")

    return paths
