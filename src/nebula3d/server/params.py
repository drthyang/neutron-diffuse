# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tsung-han Yang

"""Map a curated pipeline-run request onto the validated stage defaults.

This is the FastAPI-free home of :func:`build_params` so that both the API
router (:mod:`nebula3d.server.routers.pipeline`) and the in-browser bridge
(:mod:`nebula3d.webbridge`, under Pyodide) can share one request → ``PipelineParams``
translation.  It raises :class:`ValueError` on invalid bands; the router wraps
that into an HTTP 400.

The request object is duck-typed: it only needs a ``flatten_enabled`` attribute
and a ``params`` attribute whose fields default to ``None`` when unset (both the
Pydantic ``PipelineRunRequest`` and a plain namespace built from a JSON dict
satisfy this), so this module imports neither FastAPI nor Pydantic at runtime.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from nebula3d.pipeline import PipelineParams

if TYPE_CHECKING:
    from nebula3d.server.schemas import PipelineRunRequest


def build_params(req: PipelineRunRequest) -> PipelineParams:
    """Apply the curated request overrides onto the validated defaults."""
    p = PipelineParams(flatten_enabled=req.flatten_enabled)
    sp = req.params

    if sp.rings_n_patches is not None:
        p.rings = dataclasses.replace(p.rings, n_patches=sp.rings_n_patches)
    if sp.rings_n_fourier is not None:
        p.rings = dataclasses.replace(p.rings, n_fourier=sp.rings_n_fourier)
    if sp.rings_slice_axis is not None:
        p.rings = dataclasses.replace(p.rings, slice_axis=sp.rings_slice_axis)
    if sp.rings_model is not None:
        p.rings = dataclasses.replace(p.rings, ring_model=sp.rings_model)
    if sp.rings_ring_width is not None:
        p.rings = dataclasses.replace(p.rings, ring_width=sp.rings_ring_width)
    if sp.rings_radial_mode is not None:
        p.rings = dataclasses.replace(p.rings, ring_radial_mode=sp.rings_radial_mode)
    if sp.punch_min_intensity is not None:
        p.punch = dataclasses.replace(p.punch, min_intensity=sp.punch_min_intensity)
    if sp.punch_search_n_mad is not None:
        p.punch = dataclasses.replace(p.punch, search_n_mad=sp.punch_search_n_mad)
    if sp.punch_mode is not None:
        p.punch = dataclasses.replace(p.punch, mode=sp.punch_mode)
    if any(v is not None for v in
           (sp.punch_radius_h, sp.punch_radius_k, sp.punch_radius_l)):
        cur = p.punch.punch_radii
        p.punch = dataclasses.replace(p.punch, punch_radii=(
            sp.punch_radius_h if sp.punch_radius_h is not None else cur[0],
            sp.punch_radius_k if sp.punch_radius_k is not None else cur[1],
            sp.punch_radius_l if sp.punch_radius_l is not None else cur[2],
        ))
    if sp.punch_margin is not None:
        p.punch = dataclasses.replace(p.punch, margin=sp.punch_margin)
    if sp.punch_phi_tail_hkl is not None:
        p.punch = dataclasses.replace(p.punch, phi_tail_hkl=sp.punch_phi_tail_hkl)
    if sp.punch_frame is not None:
        p.punch = dataclasses.replace(p.punch, punch_frame=sp.punch_frame)
    if sp.punch_q_radius is not None:
        p.punch = dataclasses.replace(p.punch, punch_q_radius=sp.punch_q_radius)
    if any(v is not None for v in
           (sp.punch_q_radius_a, sp.punch_q_radius_b, sp.punch_q_radius_c)):
        cur = p.punch.punch_q_radii or (0.1, 0.1, 0.1)
        p.punch = dataclasses.replace(p.punch, punch_q_radii=(
            sp.punch_q_radius_a if sp.punch_q_radius_a is not None else cur[0],
            sp.punch_q_radius_b if sp.punch_q_radius_b is not None else cur[1],
            sp.punch_q_radius_c if sp.punch_q_radius_c is not None else cur[2],
        ))
    if any(v is not None for v in
           (sp.incident_beam_q_radius_a, sp.incident_beam_q_radius_b,
            sp.incident_beam_q_radius_c)):
        cur = p.punch.incident_beam_q_radii or (0.16, 0.30, 0.25)
        p.punch = dataclasses.replace(p.punch, incident_beam_q_radii=(
            sp.incident_beam_q_radius_a
            if sp.incident_beam_q_radius_a is not None else cur[0],
            sp.incident_beam_q_radius_b
            if sp.incident_beam_q_radius_b is not None else cur[1],
            sp.incident_beam_q_radius_c
            if sp.incident_beam_q_radius_c is not None else cur[2],
        ))
    if sp.incident_beam_q_margin is not None:
        p.punch = dataclasses.replace(
            p.punch, incident_beam_q_margin=sp.incident_beam_q_margin)
    if any(v is not None for v in
           (sp.incident_beam_radius_h, sp.incident_beam_radius_k,
            sp.incident_beam_radius_l)):
        cur = p.punch.incident_beam_ellipsoid_radii_hkl or p.punch.incident_beam_radii
        p.punch = dataclasses.replace(p.punch, incident_beam_ellipsoid_radii_hkl=(
            sp.incident_beam_radius_h if sp.incident_beam_radius_h is not None else cur[0],
            sp.incident_beam_radius_k if sp.incident_beam_radius_k is not None else cur[1],
            sp.incident_beam_radius_l if sp.incident_beam_radius_l is not None else cur[2],
        ))
    if sp.incident_beam_margin is not None:
        p.punch = dataclasses.replace(p.punch, incident_beam_margin=sp.incident_beam_margin)
    if sp.punch_fit_covariance is not None:
        p.punch = dataclasses.replace(
            p.punch, integer_fit_covariance=sp.punch_fit_covariance)
    if sp.punch_fit_unconstrained is not None:
        p.punch = dataclasses.replace(
            p.punch, integer_fit_unconstrained=sp.punch_fit_unconstrained)
    if sp.incident_beam_fit_covariance is not None:
        p.punch = dataclasses.replace(
            p.punch, incident_beam_fit_covariance=sp.incident_beam_fit_covariance)
    if sp.backfill_method is not None:
        p.backfill = dataclasses.replace(p.backfill, method=sp.backfill_method)
    if sp.flatten_estimator is not None:
        p.flatten = dataclasses.replace(p.flatten, estimator=sp.flatten_estimator)
    if sp.flatten_floor_percentile is not None:
        p.flatten = dataclasses.replace(
            p.flatten, floor_percentile=sp.flatten_floor_percentile)

    dp_kw: dict = {}
    if sp.pdf_apodization is not None:
        dp_kw["apodization"] = sp.pdf_apodization
    if sp.pdf_gaussian_sigma is not None:
        dp_kw["gaussian_sigma"] = sp.pdf_gaussian_sigma
    if any(v is not None for v in (sp.pdf_crop_h, sp.pdf_crop_k, sp.pdf_crop_l)):
        cur = p.delta_pdf.crop_hkl or (4.0, 8.0, 15.0)
        dp_kw["crop_hkl"] = (
            sp.pdf_crop_h if sp.pdf_crop_h is not None else cur[0],
            sp.pdf_crop_k if sp.pdf_crop_k is not None else cur[1],
            sp.pdf_crop_l if sp.pdf_crop_l is not None else cur[2],
        )
    if sp.pdf_q_min is not None or sp.pdf_q_max is not None:
        qmin = sp.pdf_q_min if sp.pdf_q_min is not None else 0.0
        if sp.pdf_q_max is None:
            raise ValueError("pdf_q_max is required when setting a |Q| band")
        if sp.pdf_q_max <= qmin:
            raise ValueError("pdf_q_max must be greater than pdf_q_min")
        dp_kw["q_band"] = (qmin, sp.pdf_q_max)
    if dp_kw:
        p.delta_pdf = dataclasses.replace(p.delta_pdf, **dp_kw)
    return p
