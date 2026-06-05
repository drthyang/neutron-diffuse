"""Backfill Bragg-punched holes — step 5 of the further analysis pipeline.

The preferred real-data fill is local-background replacement: each punched
Bragg/satellite hole is filled from the nearby unpunched shell around that hole.
Generic TV inpainting remains available as an option, but it can introduce
slice-scale staircase / smoothing artefacts in structured diffuse scattering.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ndiff.core import HKLVolume
from ndiff.inpainting.pipeline import fill, Method

BraggFillMethod = Method | Literal["local"]


def backfill_bragg(
    vol: HKLVolume,
    method: BraggFillMethod = "local",
    laue_class: str = "m3m",
    symmetry_ops: Optional[Sequence[NDArray]] = None,
    tv_lam: float = 0.2,
    tv_iter: int = 300,
    local_radius: int = 2,
    local_min_count: int = 8,
    direct_beam_fill: bool = True,
    direct_beam_q_gap: float = 0.05,
    direct_beam_q_width: float = 0.15,
) -> HKLVolume:
    """Fill Bragg-punched voxels in *vol*.

    ``method="local"`` fills each connected punched region with the median of
    nearby valid voxels in a dilated shell around that region.  This estimates
    the local background level near the Bragg peak and avoids inventing a smooth
    TV surface through real diffuse texture.  TV/symmetry methods are retained
    for explicit comparisons.

    The **direct beam** (the punched hole at the origin) is filled differently
    from ordinary Bragg holes: a generic dilated shell around that large,
    elongated hole straddles the over-subtracted halo that hugs the beam, so the
    fill is biased.  Instead the whole beam region is filled with the diffuse
    background measured in a thin ``|Q|`` shell *just outside* it
    (``|Q| > beam edge``), which is the physically meaningful low-|Q| diffuse
    level.  Controlled by ``direct_beam_fill`` / ``direct_beam_q_gap`` /
    ``direct_beam_q_width``; it falls back to the generic local fill when no
    clean outside shell is available (e.g. tiny synthetic volumes).

    Parameters
    ----------
    vol:
        Volume after Bragg punching (``vol.mask`` marks valid voxels).
    method:
        Inpainting strategy. Default ``"symmetry+tv"``.
    laue_class:
        Crystal Laue class for symmetry filling.
    tv_lam:
        TV regularisation weight (higher than for Al backfill).
    tv_iter:
        Maximum TV iterations.
    local_radius:
        Number of binary-dilation iterations used to form the local shell around
        each punched component.
    local_min_count:
        Minimum valid shell voxels required before using the local shell median.
        Components with fewer fall back to the global valid-data median.
    direct_beam_fill:
        If True (default), fill the origin hole from the ``|Q|``-just-outside
        diffuse background instead of the generic dilated shell.
    direct_beam_q_gap:
        Å⁻¹ offset from the beam's outer ``|Q|`` edge to the start of the
        sampling shell — pushes the sample past the beam halo / over-subtraction
        trough that hugs the direct beam.
    direct_beam_q_width:
        Å⁻¹ thickness of the ``|Q|`` shell sampled for the direct-beam fill.

    Returns
    -------
    HKLVolume with Bragg holes filled.
    """
    if method == "local":
        return _local_background_fill(
            vol, radius=local_radius, min_count=local_min_count,
            direct_beam_fill=direct_beam_fill,
            db_q_gap=direct_beam_q_gap, db_q_width=direct_beam_q_width,
        )
    return fill(
        vol,
        method=method,
        laue_class=laue_class,
        symmetry_ops=list(symmetry_ops) if symmetry_ops else None,
        tv_lam=tv_lam,
        tv_iter=tv_iter,
    )


def _local_background_fill(
    vol: HKLVolume,
    radius: int = 2,
    min_count: int = 8,
    direct_beam_fill: bool = True,
    db_q_gap: float = 0.05,
    db_q_width: float = 0.15,
) -> HKLVolume:
    """Fill each punched connected component from its local valid shell."""
    data = vol.data.copy()
    sigma = vol.sigma.copy()
    holes = (~vol.mask) & np.isfinite(vol.data)
    if not holes.any():
        return dataclasses.replace(vol, mask=np.ones(vol.shape, dtype=bool))

    valid = vol.mask & np.isfinite(vol.data)
    global_vals = data[valid]
    global_fill = float(np.median(global_vals)) if global_vals.size else 0.0
    global_sigma = float(np.median(sigma[valid])) if global_vals.size else 1.0

    # Direct beam first: fill the origin hole (and its interior) from the diffuse
    # background just outside it in |Q|, then exclude it from the generic loop.
    resolved = np.zeros(vol.shape, dtype=bool)
    if direct_beam_fill:
        resolved = _fill_direct_beam(
            vol, data, sigma, holes, valid, global_sigma,
            q_gap=db_q_gap, q_width=db_q_width, min_count=min_count,
        )

    remaining = holes & ~resolved
    labels, n_label = ndimage.label(remaining, structure=np.ones((3, 3, 3), dtype=bool))
    objects = ndimage.find_objects(labels)
    structure = np.ones((3, 3, 3), dtype=bool)
    pad = max(int(radius) + 1, 1)

    for lbl, obj in enumerate(objects, start=1):
        if obj is None:
            continue
        slices = []
        for s, n in zip(obj, vol.shape):
            slices.append(slice(max(0, s.start - pad), min(n, s.stop + pad)))
        region = tuple(slices)
        comp = labels[region] == lbl
        shell = ndimage.binary_dilation(
            comp, structure=structure, iterations=max(int(radius), 1)
        ) & ~comp
        shell_valid = shell & valid[region]
        if int(shell_valid.sum()) >= min_count:
            vals = data[region][shell_valid]
            fill_val = float(np.median(vals))
            fill_sig = float(np.std(vals)) if vals.size > 1 else global_sigma
        else:
            fill_val = global_fill
            fill_sig = global_sigma

        data_region = data[region]
        sigma_region = sigma[region]
        data_region[comp] = fill_val
        sigma_region[comp] = max(fill_sig, global_sigma)
        data[region] = data_region
        sigma[region] = sigma_region

    return dataclasses.replace(vol, data=data, sigma=sigma,
                               mask=np.ones(vol.shape, dtype=bool))


def _fill_direct_beam(
    vol: HKLVolume,
    data: NDArray,
    sigma: NDArray,
    holes: NDArray[np.bool_],
    valid: NDArray[np.bool_],
    global_sigma: float,
    q_gap: float,
    q_width: float,
    min_count: int,
) -> NDArray[np.bool_]:
    """Fill the origin (direct-beam) region from the |Q|-just-outside background.

    The beam region is the origin-connected blob of *punched holes* **and**
    *originally-unmeasured* voxels (the direct beam casts a detector shadow at
    |Q|≈0 that reads as unmeasured, not as a punched hole).  Both are filled so
    the centre is not left as a pit.  The blob is capped at the first ``|Q|`` gap
    in its voxels, so a punch that bridges toward a low-|Q| Bragg node (e.g.
    ``(0,0,2)``) does not drag that node into the beam region.  The fill value is
    the median diffuse level in a thin shell just outside that ``|Q|`` edge.

    Returns a boolean mask of the voxels resolved here (empty if no direct-beam
    region is found or no clean outside shell is available — the caller's generic
    per-component fill then handles those holes instead).
    """
    resolved = np.zeros(vol.shape, dtype=bool)
    nh, nk, nl = vol.shape

    ih = int(np.argmin(np.abs(vol.h_axis)))
    ik = int(np.argmin(np.abs(vol.k_axis)))
    il = int(np.argmin(np.abs(vol.l_axis)))

    # The direct beam is punched holes ∪ the unmeasured detector shadow at |Q|≈0.
    beam_like = holes | ~vol.mask
    structure = np.ones((3, 3, 3), dtype=bool)
    labels, _ = ndimage.label(beam_like, structure=structure)
    lbl = int(labels[ih, ik, il])
    if lbl == 0:
        box = (slice(max(0, ih - 2), min(nh, ih + 3)),
               slice(max(0, ik - 2), min(nk, ik + 3)),
               slice(max(0, il - 2), min(nl, il + 3)))
        nz = labels[box][labels[box] > 0]
        if nz.size == 0:
            return resolved  # no direct-beam region → generic fill handles all
        lbl = int(np.bincount(nz).argmax())

    obj = ndimage.find_objects(labels, max_label=lbl)[lbl - 1]
    if obj is None:
        return resolved
    pad = 6  # room for the outside |Q| shell beyond the beam edge
    region = tuple(slice(max(0, s.start - pad), min(n, s.stop + pad))
                   for s, n in zip(obj, vol.shape))

    comp_box = labels[region] == lbl
    q_box = _q_in_region(vol, region)

    # Outer |Q| edge of the beam = the first gap in the component's sorted |Q|.
    # Everything past that gap (a bridged Bragg node) is excluded from the beam.
    # The gap threshold adapts to the local |Q| sampling so it is not fooled by
    # the coarse spacing of a sparsely-sampled grid.
    qc = np.sort(q_box[comp_box])
    q_beam = float(qc[-1])
    if qc.size > 1:
        dqs = np.diff(qc)
        pos = dqs[dqs > 1e-9]
        step = float(np.median(pos)) if pos.size else 0.0
        brk = np.where(dqs > max(q_gap, 5.0 * step))[0]
        if brk.size:
            q_beam = float(qc[brk[0]])
    # Replace the whole beam ball — punched holes, the unmeasured central shadow,
    # *and* the over-subtracted halo voxels that survived the punch — so the
    # direct-beam spot becomes one clean patch of the just-outside background
    # rather than a flat disk pitted by a masked centre and a negative rim.
    solid_box = q_box <= q_beam

    valid_box = valid[region]
    shell = valid_box & (q_box > q_beam + q_gap) & (q_box <= q_beam + q_gap + q_width)
    if int(shell.sum()) < min_count:
        return resolved  # no clean outside shell → fall back to generic fill

    vals = data[region][shell]
    fill_val = float(np.median(vals))
    fill_sig = float(np.std(vals)) if vals.size > 1 else global_sigma

    data_region = data[region]
    sigma_region = sigma[region]
    data_region[solid_box] = fill_val
    sigma_region[solid_box] = max(fill_sig, global_sigma)
    data[region] = data_region
    sigma[region] = sigma_region

    resolved[region] = solid_box
    return resolved


def _q_in_region(vol: HKLVolume, region: tuple[slice, slice, slice]) -> NDArray:
    """|Q| (Å⁻¹) for a sub-box, built from the axes (cheap — no full-grid pass)."""
    H, K, L = np.meshgrid(
        vol.h_axis[region[0]], vol.k_axis[region[1]], vol.l_axis[region[2]],
        indexing="ij",
    )
    hkl = np.stack([H, K, L], axis=-1)
    return np.linalg.norm(hkl @ vol.ub_matrix.T, axis=-1)
