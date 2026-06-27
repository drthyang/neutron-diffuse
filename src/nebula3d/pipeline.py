# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tsung-han Yang

"""Library pipeline orchestration: raw .nxs/HKL volume → 3D-ΔPDF.

This is the in-process, testable home for the six-stage workflow that
``examples/run_pipeline.py`` previously drove by shelling out to per-stage example
scripts with environment-variable parameters.  Each stage here calls the existing
library primitives directly:

    1. rings    → :class:`~nebula3d.preprocessing.PatchedRadialRingModel`
                  (+ :func:`~nebula3d.preprocessing.confirm_ring_shells_across_h`)
    2. punch    → :class:`~nebula3d.analysis.BraggRemover`
    3. backfill → :func:`~nebula3d.analysis.backfill_bragg`
    4. flatten  → :func:`~nebula3d.preprocessing.flatten_radial_background`
    5. pdf      → :func:`~nebula3d.analysis.compute_delta_pdf`
    6. pdf_check → :func:`pdf_consistency_check`

:func:`run_pipeline` chains the stages with the same on-disk file naming, the
skip-if-exists resume behaviour, and the stale-ΔPDF guard as the original script,
and reports progress through an optional callback so a server (or CLI) can stream
it.  Stage parameter defaults mirror the validated ``cc_on`` presets.

The heavy diffraction maths is **not** reimplemented here — only the orchestration
and the per-stage glue (build the model/params, call it, write the output).
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

import nebula3d
from nebula3d.analysis import (
    BraggRemover,
    DeltaPDF,
    backfill_bragg,
    compute_delta_pdf,
    invert_delta_pdf,
)
from nebula3d.analysis.delta_pdf import _q_max_from_axes
from nebula3d.core import HKLVolume
from nebula3d.preprocessing import (
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
    "consistency_reconstruction",
    "write_bragg_profile_json",
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


def _q_at(vol: HKLVolume, hkl: tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.asarray(hkl, dtype=float) @ vol.ub_matrix.T))


def _principal_widths_from_shape(
    vol: HKLVolume,
    shape_hkl: np.ndarray,
) -> tuple[list[float], list[float], list[list[float]]]:
    """Return ellipsoid semi-widths in HKL and Q units, sorted largest first."""
    lam, vecs = np.linalg.eigh(shape_hkl)
    radii = 1.0 / np.sqrt(np.clip(lam, 1e-300, None))
    order = np.argsort(radii)[::-1]
    widths_hkl = [float(radii[i]) for i in order]
    directions = [np.asarray(vecs[:, i], dtype=float) for i in order]
    widths_q = [
        float(widths_hkl[i] * np.linalg.norm(vol.ub_matrix @ directions[i]))
        for i in range(3)
    ]
    direction_rows = [directions[i].tolist() for i in range(3)]
    return widths_hkl, widths_q, direction_rows


def _axis_widths_from_shape(
    vol: HKLVolume,
    shape_hkl: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Return ellipsoid half-widths along HKL grid axes and Cartesian Q axes.

    The covariance is the (pseudo-)inverse of the shape matrix; the Q-space
    covariance is the HKL covariance pushed through the UB matrix
    (``cov_q = UB · cov_hkl · UBᵀ``).  Both use ``pinv`` / a direct congruence so a
    degenerate peak shape (e.g. one extremely large measured width, giving a
    near-singular shape matrix) yields a finite width rather than raising
    ``LinAlgError: Singular matrix``.
    """
    cov_hkl = np.linalg.pinv(shape_hkl)
    widths_hkl = [
        float(np.sqrt(max(float(cov_hkl[i, i]), 0.0))) for i in range(3)
    ]
    cov_q = vol.ub_matrix @ cov_hkl @ vol.ub_matrix.T
    widths_q = [
        float(np.sqrt(max(float(cov_q[i, i]), 0.0))) for i in range(3)
    ]
    return widths_hkl, widths_q


def _shape_from_radii(radii: tuple[float, float, float]) -> np.ndarray:
    return np.diag([1.0 / (float(r) * float(r)) for r in radii])


def bragg_profile_from_records(
    vol: HKLVolume,
    remover: BraggRemover,
    peaks: Sequence,
) -> dict:
    """Summarise detected Bragg peak ellipsoid widths for review charts."""
    base = remover._fit_base_radii(vol)  # noqa: SLF001 - profile mirrors punch internals
    n_sigma = max(float(remover.integer_fit_radius_n_sigma), 0.0)
    steps = tuple(abs(s) for s in remover._steps(vol))  # noqa: SLF001
    pad_hkl = tuple(0.5 * s for s in steps)  # half-voxel: the punch-width pad floor
    rows = []
    for idx, peak in enumerate(peaks):
        if peak.shape_hkl is not None:
            shape = np.asarray(peak.shape_hkl, dtype=float)
            fit_kind = "tilted"
        elif peak.radii_hkl is None and str(remover.punch_frame).lower() == "spherical":
            # Fixed spherical punch: report the *real* per-peak ellipsoid so the
            # principal widths/directions follow Q̂ (ρ) and the two transverse axes.
            sm = remover._spherical_shape_matrix(vol, tuple(peak.center_hkl))  # noqa: SLF001
            if sm is not None:
                shape = sm
                fit_kind = "spherical"
            else:
                shape = _shape_from_radii(base)
                fit_kind = "axis_aligned"
        else:
            shape = _shape_from_radii(peak.radii_hkl or base)
            fit_kind = "axis_aligned"
        widths_hkl_principal, widths_q_principal, directions = (
            _principal_widths_from_shape(vol, shape)
        )
        widths_hkl, widths_q = _axis_widths_from_shape(vol, shape)
        # Measured (pad-free, floor-free) per-axis width from a local moment fit.
        # This is the *data* width that the width histogram should plot; the
        # ``width_*`` above are punch radii and pile resolution-limited peaks onto
        # the half-voxel pad / base floor.  ``None`` when the peak is unmeasurable.
        sigmas = remover.measure_peak_sigmas(vol, tuple(peak.center_hkl))
        if sigmas is not None:
            measured_width_hkl = [n_sigma * s for s in sigmas]
            measured_radii: tuple[float, float, float] = (
                max(measured_width_hkl[0], 1e-9),
                max(measured_width_hkl[1], 1e-9),
                max(measured_width_hkl[2], 1e-9),
            )
            _, measured_width_q = _axis_widths_from_shape(
                vol, _shape_from_radii(measured_radii))
            # An axis is resolution-limited when its measured width falls below
            # the half-voxel pad — i.e. the punch radius there is set by the pad,
            # not the data.  Those are exactly the peaks that form the spike.
            resolution_limited = [
                bool(mw < pad) for mw, pad in zip(measured_width_hkl, pad_hkl)
            ]
        else:
            measured_width_hkl = None
            measured_width_q = None
            resolution_limited = None
        rows.append({
            "index": idx,
            "source_node_hkl": (
                list(peak.source_node_hkl) if peak.source_node_hkl is not None else None
            ),
            "center_hkl": [float(v) for v in peak.center_hkl],
            "q_abs": _q_at(vol, peak.center_hkl),
            "intensity": (
                float(peak.intensity) if np.isfinite(peak.intensity) else None
            ),
            "local_background": (
                float(peak.local_background)
                if np.isfinite(peak.local_background) else None
            ),
            "width_hkl": widths_hkl,
            "width_q": widths_q,
            "measured_width_hkl": measured_width_hkl,
            "measured_width_q": measured_width_q,
            "resolution_limited": resolution_limited,
            "principal_width_hkl": widths_hkl_principal,
            "principal_width_q": widths_q_principal,
            "principal_directions_hkl": directions,
            "fit_kind": fit_kind,
        })
    return {
        "schema_version": 1,
        "width_labels": ["Qx", "Qy", "Qz"],
        "hkl_width_labels": ["H", "K", "L"],
        "width_units": {"hkl": "r.l.u.", "q": "Å⁻¹"},
        "n_peaks": len(rows),
        "fit_covariance": bool(remover.integer_fit_covariance),
        "punch_frame": str(remover.punch_frame),
        "peaks": rows,
    }


def write_bragg_profile_json(profile: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")


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
    integer_fit_unconstrained: bool = False
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
    incident_beam_q_radii: tuple[float, float, float] | None = (
        0.16, 0.30, 0.25,
    )
    incident_beam_q_margin: float = 0.0
    incident_beam_ellipsoid_radii_hkl: tuple[float, float, float] | None = (
        0.15, 0.50, 1.00,
    )
    incident_beam_sphere_radius_hkl: float | None = None
    incident_beam_fit_covariance: bool = False
    # Spherical-frame punch (default): the ellipsoid axes follow the *local*
    # spherical frame at each peak — (rρ, rθ, rφ) in Å⁻¹ with rρ radial (along Q̂),
    # rφ azimuthal (a*–b* ring tangent), rθ polar (c* pole).  This orients every
    # peak's footprint by construction (no tilt angle).  Set punch_frame="q" to use
    # the fixed a*/b*/c* radii (``punch_q_radii``), or "hkl" + ``punch_radii`` for
    # the legacy rlu footprint.
    punch_frame: str = "spherical"
    punch_q_radius: float | None = None
    punch_q_radii: tuple[float, float, float] | None = (0.097, 0.072, 0.115)
    # (rρ, rθ, rφ) Å⁻¹ — seeded from the q radii so the default punch volume is
    # comparable to the previous a*/b*/c* default.
    punch_spherical_radii: tuple[float, float, float] | None = (0.097, 0.072, 0.115)


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
    # Full |Q| range by default (no crop) so the saved/display ΔPDF has the same
    # real-space resolution as the back-FFT consistency view, which always uses the
    # full range.  Real-space pixel size is fixed by the |Q| extent kept, not by
    # padding (see docs/algorithms/delta_pdf.md); cropping coarsens the grid.  Pass
    # a (h, k, l) tuple to band-limit (smaller files, trims the noisier outer |Q|).
    crop_hkl: tuple[float, float, float] | None = None
    q_band: tuple[float, float] | None = None
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

    def make_model() -> PatchedRadialRingModel | ParametricRingModel:
        if p.ring_model.strip().lower() == "parametric":
            return ParametricRingModel(
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
            return PatchedRadialRingModel(
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

    dummy_model = make_model()
    min_voxels = dummy_model.min_voxels_per_patch

    res_data = np.empty_like(vol.data)
    out_mask = vol.mask.copy()
    n = int(axis_values.size)

    def process_slice(ip: int) -> tuple[int, np.ndarray, np.ndarray | None, bool, str | None]:
        sl = _slice_volume(vol, cfg, ip)
        valid = sl.mask & np.isfinite(sl.data)
        if int(valid.sum()) < min_voxels:
            return ip, _take_plane(sl.data, cfg, 0), None, True, None

        keep = azimuthal_sampling_mask(sl, plane=cfg.plane, min_count_frac=0.25,
                                       q_range=q_range)
        src = dataclasses.replace(sl, mask=keep)
        out_mask_2d = _take_plane(keep, cfg, 0)

        local_model = make_model()
        try:
            local_model.fit(src, q_range=q_range)
            _, I_ring = local_model.subtract(src)
        except Exception as exc:
            return ip, _take_plane(sl.data, cfg, 0), out_mask_2d, True, str(exc)

        I_ring2d = _take_plane(I_ring, cfg, 0)
        sl_data2d = _take_plane(sl.data, cfg, 0)
        return ip, sl_data2d - I_ring2d, out_mask_2d, False, None

    n_skipped = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_slice, ip): ip for ip in range(n)}
        
        done_count = 0
        for future in concurrent.futures.as_completed(futures):
            ip, data_2d, mask_2d, skipped, err_msg = future.result()
            
            _assign_plane(res_data, cfg, ip, data_2d)
            if mask_2d is not None:
                _assign_plane(out_mask, cfg, ip, mask_2d)
                
            if skipped:
                n_skipped += 1
            if err_msg:
                _emit(progress, "rings", "progress", (done_count + 1) / n,
                      f"{cfg.axis_name}[{ip}] fit failed ({err_msg}); left as-is")

            done_count += 1
            if done_count % 30 == 0 or done_count == n:
                _emit(progress, "rings", "progress", done_count / n,
                      f"{cfg.axis_name}[{done_count}/{n}] (parallel)")

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
        integer_fit_unconstrained=p.integer_fit_unconstrained,
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
        incident_beam_q_radii=p.incident_beam_q_radii,
        incident_beam_q_margin=p.incident_beam_q_margin,
        incident_beam_ellipsoid_radii_hkl=p.incident_beam_ellipsoid_radii_hkl,
        incident_beam_sphere_radius_hkl=p.incident_beam_sphere_radius_hkl,
        incident_beam_fit_covariance=p.incident_beam_fit_covariance,
        phi_tail_hkl=p.phi_tail_hkl,
        search_n_mad=p.search_n_mad, search_min_intensity=p.search_min_intensity,
        search_min_prominence=p.search_min_prominence,
        search_exclude_h_centers=p.search_exclude_h_centers,
        search_exclude_h_half_width=p.search_exclude_h_half_width,
        search_exclude_h_fractions=p.search_exclude_h_fractions,
        punch_frame=p.punch_frame, punch_q_radius=p.punch_q_radius,
        punch_q_radii=p.punch_q_radii,
        punch_spherical_radii=p.punch_spherical_radii,
    )
    peak_records = remover._detect_peak_records(vol)  # noqa: SLF001 - avoid refitting
    keep = remover._punch_centers(
        vol, np.ones(vol.shape, dtype=bool), peak_records)  # noqa: SLF001
    keep = remover._punch_incident_beam(vol, keep)  # noqa: SLF001
    profile = bragg_profile_from_records(vol, remover, peak_records)
    valid = vol.mask & np.isfinite(vol.data)
    punched = int((valid & ~keep).sum())
    out_vol = dataclasses.replace(vol, mask=vol.mask & keep)
    _emit(progress, "punch", "done", 1.0,
          f"detected {len(peak_records)} peaks; punched {punched:,} voxels")
    setattr(out_vol, "_bragg_profile", profile)
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
        real_space_angstrom=True, crop_hkl=p.crop_hkl, q_band=p.q_band,
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
        f"q_band={_param_string(p.q_band)}",
        f"subtract_bg={_param_string(p.subtract_smooth_bg)}",
    ))


def write_delta_pdf_h5(dpdf: DeltaPDF, vol: HKLVolume, p: DeltaPdfParams,
                       source_name: str, out_path: Path,
                       r_band: tuple[float, float] | None = None) -> None:
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
        fh.attrs["q_band"] = _param_string(p.q_band)
        fh.attrs["r_band"] = _param_string(r_band)
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


_CHECK_H_VALUES: tuple[float, ...] = (0.0, 1.0 / 3.0, 1.0)


def _consistency_metrics(
    rec: np.ndarray, data: np.ndarray, region: np.ndarray,
    h_axis: np.ndarray, h_values: Sequence[float],
) -> tuple[dict, list]:
    """Pearson r + normalised RMS over *region*, plus per-H-plane r and figure rows."""
    def _r(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
        a, b = a[m], b[m]
        return float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else float("nan")

    rms = float(np.sqrt(np.mean((rec - data)[region] ** 2))) if region.any() else 0.0
    denom = (float(np.sqrt(np.mean(data[region] ** 2)))
             if region.any() else 0.0) or 1.0
    per_plane: dict[str, float] = {}
    rows = []
    for hv in h_values:
        ih = int(np.argmin(np.abs(h_axis - hv)))
        h_actual = float(h_axis[ih])
        r_plane = _r(rec[ih], data[ih], region[ih])
        per_plane[f"{h_actual:+.3f}"] = r_plane
        rows.append((h_actual, data[ih], rec[ih], region[ih], r_plane))
    metrics = {
        "pearson_r": _r(rec, data, region),
        "normalized_rms": rms / denom,
        "rms": rms,
        "n_voxels": int(region.sum()),
        "per_plane_r": per_plane,
    }
    return metrics, rows


def pdf_consistency_check(
    vol: HKLVolume,
    dpdf: DeltaPDF,
    p: DeltaPdfParams,
    *,
    h_values: Sequence[float] = _CHECK_H_VALUES,
    figure_path: Path | None = None,
) -> dict:
    """Back-FFT round-trip check: inverse-transform *dpdf* and compare to *vol*.

    Inverse-transforms the ΔPDF back to reciprocal space
    (:func:`~nebula3d.analysis.invert_delta_pdf`) and measures how well it
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
    region = recon.mask & np.isfinite(data)
    if p.q_band is not None:
        region &= _band_limit_q(data_vol, p.q_band)[1]
    metrics, rows = _consistency_metrics(
        recon.data, data, region, recon.h_axis, h_values)
    metrics["crop_hkl"] = list(p.crop_hkl) if p.crop_hkl else None
    metrics["q_band"] = list(p.q_band) if p.q_band else None
    metrics["apodization"] = p.apodization
    if figure_path is not None:
        _write_consistency_figure(rows, Path(figure_path))
    return metrics


def _band_limit_q(
    vol: HKLVolume, q_band: tuple[float, float],
    q_mag: np.ndarray | None = None,
) -> tuple[HKLVolume, np.ndarray]:
    """Mask voxels outside the spherical |Q| shell [qmin, qmax] (Å⁻¹).

    ``q_mag`` (Å⁻¹, voxel |Q|) may be supplied to reuse an already-computed
    magnitude grid — recomputing it rebuilds a full meshgrid + UB matmul.
    """
    qmin, qmax = q_band
    if q_mag is None:
        q_mag = vol.q_magnitude()
    in_band = (q_mag >= qmin) & (q_mag <= qmax)
    return dataclasses.replace(vol, mask=vol.mask & in_band), in_band


def consistency_reconstruction(
    vol: HKLVolume,
    p: DeltaPdfParams,
    *,
    q_band: tuple[float, float] | None = None,
    r_band: tuple[float, float] | None = None,
    h_values: Sequence[float] = _CHECK_H_VALUES,
) -> dict:
    """Band-limit (optional) → ΔPDF → inverse-FFT → compare; return sliceable volumes.

    Powers the interactive consistency view.  When *q_band* is given, keeps only
    the spherical |Q| shell ``[qmin, qmax]`` (Å⁻¹) of the diffuse data before the
    forward+inverse round trip, so the caller can see which ΔPDF features and how
    much signal come from low- vs high-|Q| data.  Returns reciprocal-space
    ``HKLVolume``s (``recon``, ``data``, ``residual``) on the cropped grid — ready
    for :func:`nebula3d.visualization.extract_slice` — plus the metrics dict.
    """
    vol_c = _crop_hkl(vol, p.crop_hkl)
    in_band = np.ones(vol_c.data.shape, dtype=bool)
    # The full per-voxel |Q| grid is only needed to build the band mask.  When no
    # |Q| band is requested, skip it entirely and get the q_data_max scalar from
    # the 8 box corners (exact — |Q| is convex), avoiding a ~48M-voxel meshgrid.
    if q_band is not None:
        q_mag_c = vol_c.q_magnitude()
        vol_c, in_band = _band_limit_q(vol_c, q_band, q_mag=q_mag_c)
        q_data_max = float(q_mag_c.max())
    else:
        q_data_max = _q_max_from_axes(
            vol_c.h_axis, vol_c.k_axis, vol_c.l_axis, vol_c.ub_matrix)
    dpdf = compute_delta_pdf(
        vol_c, apodization=p.apodization,  # type: ignore[arg-type]
        gaussian_sigma=p.gaussian_sigma, zero_pad=p.zero_pad,
        subtract_mean=p.subtract_mean, real_space_angstrom=True,
        crop_hkl=None, subtract_smooth_bg=p.subtract_smooth_bg)

    # max R for the UI scale — farthest real-space corner (Å)
    r_data_max = float(np.sqrt(
        max(dpdf.x_axis[0]**2, dpdf.x_axis[-1]**2) +
        max(dpdf.y_axis[0]**2, dpdf.y_axis[-1]**2) +
        max(dpdf.z_axis[0]**2, dpdf.z_axis[-1]**2)
    ))

    if r_band is not None:
        rmin, rmax = r_band
        # Broadcast the separable real-space axes rather than materialising the
        # full X/Y/Z meshgrids — same R, one array instead of four.
        R = np.sqrt(dpdf.x_axis[:, None, None] ** 2
                    + dpdf.y_axis[None, :, None] ** 2
                    + dpdf.z_axis[None, None, :] ** 2)
        r_mask = (R >= rmin) & (R <= rmax)
        dpdf.data = np.where(r_mask, dpdf.data, 0.0)

    recon = invert_delta_pdf(dpdf, deapodize=True)
    data = np.where(np.isfinite(vol_c.masked_data()), vol_c.data, 0.0)
    region = recon.mask & np.isfinite(data) & in_band
    metrics, _rows = _consistency_metrics(
        recon.data, data, region, recon.h_axis, h_values)
    metrics["q_data_max"] = q_data_max
    metrics["q_band"] = list(q_band) if q_band else None
    metrics["r_data_max"] = r_data_max
    metrics["r_band"] = list(r_band) if r_band else None
    metrics["crop_hkl"] = list(p.crop_hkl) if p.crop_hkl else None
    metrics["apodization"] = p.apodization

    zeros = np.zeros(recon.data.shape, dtype=np.float64)
    ones = np.ones(recon.data.shape, dtype=bool)
    data_vol = dataclasses.replace(vol_c, data=data, sigma=zeros, mask=ones)
    resid_vol = dataclasses.replace(recon, data=data - recon.data)
    return {"metrics": metrics, "recon": recon, "data": data_vol,
            "residual": resid_vol, "dpdf": dpdf}


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
    bragg_profile_json: Path
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
    multi-volume auto-detection find the same ``*_delta_pdf.h5`` files.
    """
    inp = Path(input_path)
    proc = Path(proc_dir) if proc_dir is not None else Path("data/processed")
    ring = proc / f"{inp.stem}_ringremoved.h5"
    punch = proc / f"{ring.stem}_braggpunched.h5"
    fill = proc / f"{punch.stem}_backfilled.h5"
    flat = proc / f"{fill.stem}_flattened.h5"
    pdf = proc / f"{fill.stem}_delta_pdf.h5"
    return PipelinePaths(
        input=inp, ringremoved=ring, braggpunched=punch,
        bragg_profile_json=proc / f"{punch.stem}_profile.json",
        backfilled=fill, flattened=flat, delta_pdf=pdf,
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
        Raw ``.nxs`` (or nebula3d ``.h5``) input volume.
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

    # Pass-through chaining: a disabled cleanup stage is skipped and its input
    # flows to the next enabled stage.  Each stage reads the output of the most
    # recent *enabled* stage before it (raw input if none upstream are enabled).
    _stage_output = {
        "rings": paths.ringremoved,
        "punch": paths.braggpunched,
        "backfill": paths.backfilled,
        "flatten": paths.flattened,
    }
    _cleanup_chain = ("rings", "punch", "backfill", "flatten")

    def _produces(stage: str) -> bool:
        if stage == "flatten":
            return "flatten" in selected and p.flatten_enabled
        return stage in selected

    def stage_input(stage: str) -> Path:
        # Walk back from the immediate predecessor: take the most recent stage
        # that is enabled this run (its output is produced), or — for a partial
        # re-run where an upstream output already exists on disk — that file;
        # otherwise keep passing through, falling back to the raw input.
        upto = (_cleanup_chain.index(stage)
                if stage in _cleanup_chain else len(_cleanup_chain))
        for prev in reversed(_cleanup_chain[:upto]):
            out = _stage_output[prev]
            if _produces(prev) or out.exists():
                return out
        return paths.input

    pdf_input = stage_input("pdf")

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
            vol = nebula3d.load(paths.input)
            out = remove_rings(vol, p.rings, progress=progress)
            nebula3d.save(out, paths.ringremoved)

    # --- stage 2: punch -----------------------------------------------------
    if want("punch"):
        if paths.braggpunched.exists() and not forced("punch"):
            _emit(progress, "punch", "skip", None,
                  f"{paths.braggpunched.name} exists")
        else:
            vol = nebula3d.load(stage_input("punch"))
            out = punch_bragg(vol, p.punch, progress=progress)
            nebula3d.save(out, paths.braggpunched)
            profile = getattr(out, "_bragg_profile", None)
            if profile is not None:
                write_bragg_profile_json(profile, paths.bragg_profile_json)

    # --- stage 3: backfill --------------------------------------------------
    if want("backfill"):
        if paths.backfilled.exists() and not forced("backfill"):
            _emit(progress, "backfill", "skip", None,
                  f"{paths.backfilled.name} exists")
        else:
            vol = nebula3d.load(stage_input("backfill"))
            out = backfill(vol, p.backfill, progress=progress)
            nebula3d.save(out, paths.backfilled)

    # --- stage 4: flatten (optional) ---------------------------------------
    if want("flatten") and p.flatten_enabled:
        if paths.flattened.exists() and not forced("flatten"):
            _emit(progress, "flatten", "skip", None,
                  f"{paths.flattened.name} exists")
        else:
            vol = nebula3d.load(stage_input("flatten"))
            out = flatten(vol, p.flatten, progress=progress)
            nebula3d.save(out, paths.flattened)
    elif want("flatten"):
        _emit(progress, "flatten", "skip", None,
              "flatten disabled (no background removed)")

    # --- stage 5: ΔPDF (with stale-cache guard) -----------------------------
    dpdf_obj: DeltaPDF | None = None     # in-memory ΔPDF if (re)computed here
    pdf_vol: HKLVolume | None = None     # its input volume, for the check stage
    if want("pdf"):
        expected_config = delta_pdf_transform_config(p.delta_pdf)
        is_current = _pdf_is_current(paths.delta_pdf, pdf_input.name,
                                     expected_config)
        if is_current and not forced("pdf"):
            _emit(progress, "pdf", "skip", None,
                  f"{paths.delta_pdf.name} is current")
        else:
            if paths.delta_pdf.exists() and not is_current:
                _emit(progress, "pdf", "progress", None,
                      f"{paths.delta_pdf.name} stale — recomputing")
            pdf_vol = nebula3d.load(pdf_input)
            dpdf_obj = delta_pdf(pdf_vol, p.delta_pdf, progress=progress)
            write_delta_pdf_h5(dpdf_obj, pdf_vol, p.delta_pdf,
                               pdf_input.name, paths.delta_pdf)

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
                pdf_vol = nebula3d.load(pdf_input)
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
