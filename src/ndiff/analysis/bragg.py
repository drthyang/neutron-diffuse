"""Bragg peak removal (punch step) for 3D-ΔPDF preparation.

Bragg peaks sit at (near-)integer (h, k, l) positions and are orders of magnitude
stronger than the diffuse signal. They must be excised ("punched") before
Fourier transforming to the 3D-ΔPDF.

Strategy
--------
1. Enumerate the integer (h,k,l) nodes within the HKL grid extent.
2. **Data-driven detection** (``min_intensity`` set): keep only nodes that carry
   a real peak (local max above ``min_intensity`` and above the local background
   by ``min_prominence``).  This crystal has many systematic absences — punching
   every node would gouge diffuse signal at the ~3/4 of nodes that are extinct.
   Each surviving peak is re-centred on its local argmax (peaks drift off the
   exact integer by thermal contraction etc.).
3. Punch a 3D ellipsoidal hole at each detected peak.  Radii are **anisotropic**
   (Bragg peaks are several-fold broader along the coarse axis — here L) and
   optionally **scale with intensity** (bright peaks have longer tails).
4. The mask is built on **local windows** around each peak, never a full-volume
   array per peak, so it scales to thousands of peaks on a 50M-voxel volume.

Punch → backfill (``ndiff.analysis.backfill_bragg``) → ΔPDF.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ndiff.core import HKLVolume


@dataclass(frozen=True)
class _PeakPunch:
    """Internal Bragg peak description used to build the punch mask.

    ``detect_peaks()`` keeps returning the historical ``(ih, ik, il, intensity)``
    tuples for callers/tests.  Internally we keep the richer fitted HKL centre
    and optional per-peak radii so integer-node detection can decide at the
    lattice node, recenter on the nearby maximum, and punch the fitted peak
    footprint rather than a one-size-fits-all voxel-centred ellipsoid.
    """

    ih: int
    ik: int
    il: int
    intensity: float
    center_hkl: tuple[float, float, float]
    radii_hkl: tuple[float, float, float] | None = None
    # Per-peak 3×3 HKL shape matrix A (punch where δᵀAδ ≤ 1).  Set by the
    # covariance fit (Phase 3); supersedes ``radii_hkl`` and folds in the φ-tail.
    shape_hkl: NDArray[np.float64] | None = None
    source_node_hkl: tuple[int, int, int] | None = None
    local_background: float = float("nan")

    def as_tuple(self) -> tuple[int, int, int, float]:
        return self.ih, self.ik, self.il, self.intensity


def _ellipsoid_inside(
    dh: NDArray[np.float64],
    dk: NDArray[np.float64],
    dl: NDArray[np.float64],
    *,
    radii: tuple[float, float, float] | None = None,
    shape_matrix: NDArray[np.float64] | None = None,
) -> NDArray[np.bool_]:
    """Boolean mask of voxels inside the punch quadratic form ``δᵀ A δ ≤ 1``.

    ``δ = (dh, dk, dl)`` are HKL offsets from the peak centre.  This is the
    single punch-shape kernel; the shape ``A`` is given one of two equivalent
    ways:

    - ``radii=(rh, rk, rl)`` — an axis-aligned ellipsoid, ``A = diag(1/r²)``.
      Evaluated with the historical ``(d/r)²`` arithmetic so the punch is
      *bit-identical* to the pre-Q-space kernel (the diagonal fast path).
    - ``shape_matrix`` — a general 3×3 symmetric (SPD) ``A`` in HKL coordinates,
      e.g. ``UBᵀ M UB`` for a Q-space resolution ellipsoid ``M`` (the
      forward-looking general path; see ``ROADMAP.md`` → Phase 6).

    Exactly one of ``radii`` / ``shape_matrix`` must be supplied.  The two
    descriptions agree to floating-point tolerance when ``A = diag(1/r²)``; the
    diagonal fast path is kept because it reproduces the old result exactly,
    while general ``A`` may flip a voxel sitting exactly on the ``quad == 1``
    boundary (different arithmetic path).
    """
    if (radii is None) == (shape_matrix is None):
        raise ValueError("supply exactly one of radii / shape_matrix")
    if radii is not None:
        rh, rk, rl = radii
        quad = (dh / rh) ** 2 + (dk / rk) ** 2 + (dl / rl) ** 2
    else:
        a = shape_matrix
        assert a is not None  # narrowed by the xor check above
        quad = (
            a[0, 0] * dh * dh + a[1, 1] * dk * dk + a[2, 2] * dl * dl
            + 2.0 * a[0, 1] * dh * dk
            + 2.0 * a[0, 2] * dh * dl
            + 2.0 * a[1, 2] * dk * dl
        )
    return quad <= 1.0


@dataclass
class BraggRemover:
    """Detect and punch Bragg reflections in an HKLVolume.

    Parameters
    ----------
    mode:
        ``"integer"`` (default) punches at integer (h,k,l) nodes (symmetry-based).
        ``"auto"`` / ``"search"`` finds *any* sharp peak as a high-tail outlier above the
        robust per-|Q|-shell diffuse level — catches off-integer satellites
        (small-domain / superlattice reflections) the integer mode misses, at the
        cost of also removing any sharp *structural* diffuse (acceptable when only
        magnetic diffuse is wanted).  ``"both"`` takes the union.
    punch_radius_hkl:
        Isotropic punch half-radius in fractional HKL units.  Used only when
        ``punch_radii`` is not given.
    punch_radii:
        ``(rh, rk, rl)`` per-axis half-radii (Å⁻¹-free, fractional HKL).  Bragg
        peaks are anisotropic — broad along a coarse axis — so prefer this over
        the isotropic radius (e.g. ``(0.12, 0.12, 0.45)`` here, L being broad).
    min_intensity:
        Detection threshold.  ``None`` (default) punches **every** integer node
        (legacy behaviour).  When set, only nodes whose local peak intensity
        exceeds this value (and the local background by ``min_prominence``) are
        punched — the data-driven path that skips systematic absences.
    min_prominence:
        A detected peak must exceed its local-window median by at least this.
    integer_n_mad:
        Optional per-|Q|-shell threshold for integer-node detection.  When set,
        each integer node's local peak must also exceed the robust diffuse level
        in its |Q| shell (``median + integer_n_mad * MAD``).  This keeps the
        lattice-aware search sensitive to weak high-|Q| Bragg peaks without
        punching extinct nodes from a flat global floor alone.
    integer_q_step:
        |Q| shell width for ``integer_n_mad``.  ``None`` reuses
        ``search_q_step``.
    integer_optimize_position:
        If True, refine accepted integer-node peaks by a positive-excess weighted
        centroid in the detection window.  This gives a continuous HKL punch
        centre instead of only the hottest voxel centre.
    integer_optimize_shape:
        If True, estimate per-peak anisotropic HKL radii from weighted second
        moments in the detection window.  The fitted radii are clipped between
        the configured base radii and ``integer_fit_max_radius_hkl``.
    integer_fit_threshold_frac:
        Fraction of peak excess above local background used to select voxels for
        the centroid/shape fit.
    integer_fit_radius_n_sigma:
        Convert fitted second-moment widths to punch half-radii by this factor.
        A small half-voxel padding is added before clipping.
    integer_fit_max_radius_hkl:
        Optional upper clamp for fitted integer-node radii.  ``None`` uses
        ``max_radius_scale * base_radius`` per axis.
    integer_h_guard_hkl:
        Optional half-width around the source integer-H plane for integer-node
        punches.  When set, fitted/scaled integer punches are clipped to
        ``|H - H_integer| <= integer_h_guard_hkl`` so strong Bragg peaks on
        integer-H planes cannot bleed into fractional-H diffuse planes.
    detect_window_hkl:
        Half-width (HKL) of the window used to locate/centre a peak and estimate
        its local background.
    intensity_scale:
        If True, multiply the punch radii by ``clip((I/intensity_ref)**(1/3), 1,
        max_radius_scale)`` so bright peaks (longer tails) get larger holes.
    intensity_ref:
        Reference intensity for the scaling.  ``None`` → the median detected-peak
        intensity (computed once).
    max_radius_scale:
        Upper clamp on the intensity radius multiplier.
    margin:
        Extra half-width (HKL) added to every punch radius — a guard band so the
        peak's faint wings are removed too.
    punch_incident_beam:
        Punch the nearest voxel to (0,0,0) as a separate incident-beam remnant,
        not as a Bragg reflection.  It is much brighter and broader than Bragg
        peaks, so it has independent radii / margin / tail settings.
    incident_beam_radii:
        Independent HKL half-radii for the incident-beam punch.  Defaults to
        twice the Bragg punch radii when unset.
    incident_beam_margin:
        Extra margin for the incident-beam punch.
    incident_beam_phi_tail_hkl:
        Extra K-L tangential half-width for the incident-beam remnant.
    incident_beam_q_radii:
        Q-space incident/direct-beam half-radii along a*, b*, c* in Å⁻¹.
        When set, this is converted through the UB matrix into the same general
        HKL shape-matrix punch used by the Q-space Bragg footprint.
    incident_beam_q_margin:
        Q-space margin in Å⁻¹ added to each ``incident_beam_q_radii`` component.
    incident_beam_ellipsoid_radii_hkl:
        If set, punch the incident beam as an origin-centred **anisotropic
        ellipsoid** ``(rh, rk, rl)`` in fractional HKL units.  Takes precedence
        over ``incident_beam_sphere_radius_hkl``.  Use this when the direct-beam
        footprint differs substantially along H, K, and L (size from linecuts
        through the origin).
    incident_beam_sphere_radius_hkl:
        If set (and ``incident_beam_ellipsoid_radii_hkl`` is *not* set), punch
        the incident beam as an isotropic HKL sphere around the origin.
    force_origin:
        Deprecated alias for ``punch_incident_beam``.
    phi_tail_hkl:
        Extra tangential half-width in the K-L plane, along the local powder-ring
        φ direction.  Use this when Bragg tails smear along rings rather than
        along the H/K/L grid axes.
    search_exclude_h_centers:
        Optional H-plane centres excluded from the hkl-agnostic search stage.
        Use this to protect known fractional-H diffuse planes while still using
        ``mode="both"`` for integer Bragg plus off-integer satellites elsewhere.
    search_exclude_h_half_width:
        Half-width in H around each protected search-exclusion centre.
    subtract_profile:
        Reserved (profile-subtraction path not implemented in this pass).
    """

    mode: str = "integer"
    punch_radius_hkl: float = 0.3
    punch_radii: tuple[float, float, float] | None = None
    min_intensity: float | None = None
    min_prominence: float = 1.0
    integer_n_mad: float | None = None
    integer_q_step: float | None = None
    integer_min_shell_size: int = 20
    # Local relative-prominence catch for *small but sharp* Bragg at integer
    # nodes: keep a node when (peak - local_bg) >= integer_local_prominence_n_mad
    # * (1.4826 * local MAD), measured in the detection window — even if it is
    # below the absolute min_intensity / min_prominence floors and the per-|Q|
    # shell threshold.  Position-locked to integer nodes (never a thirds plane),
    # so it is inherently safe for the fractional-H diffuse.  ``None`` disables.
    integer_local_prominence_n_mad: float | None = None
    integer_local_min_prominence: float = 0.0
    integer_optimize_position: bool = False
    integer_optimize_shape: bool = False
    # Phase 3: fit a full 3×3 HKL covariance (a tilted ellipsoid following the
    # peak's real orientation) instead of three axis-aligned radii, and fold the
    # φ-tail into it as a tangential inflation.  Requires ``integer_optimize_shape``.
    # Default False keeps the diagonal-radii fit + union φ-tail (bit-identical).
    integer_fit_covariance: bool = False
    integer_fit_threshold_frac: float = 0.35
    integer_fit_radius_n_sigma: float = 2.5
    integer_fit_max_radius_hkl: tuple[float, float, float] | None = None
    integer_h_guard_hkl: float | None = None
    detect_window_hkl: float = 0.2
    intensity_scale: bool = False
    intensity_ref: float | None = None
    max_radius_scale: float = 3.0
    margin: float = 0.0
    punch_incident_beam: bool = True
    incident_beam_radii: tuple[float, float, float] | None = None
    incident_beam_margin: float = 0.08
    incident_beam_phi_tail_hkl: float = 0.0
    incident_beam_q_radii: tuple[float, float, float] | None = None
    incident_beam_q_margin: float = 0.0
    incident_beam_ellipsoid_radii_hkl: tuple[float, float, float] | None = None
    incident_beam_sphere_radius_hkl: float | None = None
    force_origin: bool | None = None
    phi_tail_hkl: float = 0.0
    # --- Q-space punch (opt-in; ROADMAP Phase 6 / Phase 2) ---
    # ``punch_frame="q"`` describes the Bragg punch shape in reciprocal Å^-1
    # rather than fractional HKL, via the quadratic-form kernel
    # ``δhklᵀ A δhkl ≤ 1`` with ``A`` built from the UB metric (see
    # ``_q_shape_matrix``).  Default ``"hkl"`` keeps the legacy radii path
    # untouched.  In Q-mode the per-peak HKL shape-fit and the φ-tail are not
    # applied (that unification is Phase 3); intensity scaling still applies.
    punch_frame: str = "hkl"
    punch_q_radius: float | None = None  # isotropic, Å^-1  (A = g / ρ²)
    punch_q_radii: tuple[float, float, float] | None = None  # per a*,b*,c*, Å^-1
    # --- search mode (|Q|-shell outlier detection) ---
    search_q_step: float = 0.05
    search_n_mad: float = 8.0
    search_min_intensity: float = 2.0
    search_min_prominence: float = 0.0
    search_exclude_h_centers: tuple[float, ...] | None = None
    search_exclude_h_half_width: float = 0.0
    # Periodic H protection: fractional parts (mod 1, in [0,1)) of H to protect
    # across the WHOLE range, e.g. (1/3, 2/3) shields every integer±1/3 plane
    # (the q=1/3 satellite family) — not just a fixed centre list.  Uses the same
    # search_exclude_h_half_width.  ``None`` disables.
    search_exclude_h_fractions: tuple[float, ...] | None = None
    subtract_profile: bool = False

    def _radii(self) -> tuple[float, float, float]:
        if self.punch_radii is not None:
            return tuple(float(r) for r in self.punch_radii)  # type: ignore[return-value]
        r = float(self.punch_radius_hkl)
        return r, r, r

    @staticmethod
    def _shape_matrix_from_q_radii(
        vol: HKLVolume,
        radii_q: tuple[float, float, float],
    ) -> NDArray[np.float64]:
        """HKL shape matrix for Q half-radii along a*, b*, c*.

        The punch is ``δhklᵀ A δhkl ≤ 1`` (see :func:`_ellipsoid_inside`).  With
        ``punch_q_radii`` (ra, rb, rc) (Å^-1, along the reciprocal axes
        a*, b*, c*), ``A = Pᵀ diag(1/r²) P`` with ``P = ê·UB``
        (``ê`` = unit reciprocal-axis directions).
        """
        ra, rb, rc = (float(r) for r in radii_q)
        if min(ra, rb, rc) <= 0:
            raise ValueError("Q-space radii must be positive")
        ub = vol.ub_matrix
        unit = ub / np.linalg.norm(ub, axis=0)  # columns = unit recip-axis dirs
        p = unit.T @ ub
        d = np.diag([1.0 / ra**2, 1.0 / rb**2, 1.0 / rc**2])
        return p.T @ d @ p

    def _q_shape_matrix(
        self,
        vol: HKLVolume,
        *,
        scale: float = 1.0,
        margin_q: float = 0.0,
    ) -> NDArray[np.float64] | None:
        """HKL shape matrix ``A`` for the Q-space punch, or ``None`` in hkl mode.

        The punch is ``δhklᵀ A δhkl ≤ 1`` (see :func:`_ellipsoid_inside`).  With
        the metric ``g = UBᵀUB``:

        - ``punch_q_radius`` ρ (Å^-1, isotropic) → ``A = g / ρ²`` — a true Q-sphere
          ``|δQ| ≤ ρ`` for any crystal system.
        - ``punch_q_radii`` (ra, rb, rc) (Å^-1, along the reciprocal axes
          a*, b*, c*) → ``A = Pᵀ diag(1/r²) P`` with ``P = ê·UB`` (``ê`` = unit
          reciprocal-axis directions), the anisotropic generalisation.

        ``scale`` applies intensity scaling; ``margin_q`` is an additive Q-space
        guard band (Å^-1) applied after scaling, matching the web Bragg controls.
        """
        if str(self.punch_frame).lower() != "q":
            return None
        ub = vol.ub_matrix
        scale = max(0.0, float(scale))
        margin_q = max(0.0, float(margin_q))
        if self.punch_q_radii is not None:
            radii = tuple(
                float(r) * scale + margin_q for r in self.punch_q_radii
            )
            return self._shape_matrix_from_q_radii(vol, radii)  # type: ignore[arg-type]
        if self.punch_q_radius is not None:
            rho = float(self.punch_q_radius) * scale + margin_q
            if rho <= 0:
                raise ValueError("punch_q_radius must be positive")
            return (ub.T @ ub) / rho**2
        raise ValueError('punch_frame="q" requires punch_q_radius or punch_q_radii')

    @staticmethod
    def _ellipsoid_bounding_radii(
        shape_matrix: NDArray[np.float64],
    ) -> tuple[float, float, float]:
        """HKL half-extents of ``δᵀAδ ≤ 1`` (for local-window sizing).

        The extent along HKL axis ``i`` is ``sqrt((A⁻¹)_ii)``.
        """
        inv = np.linalg.inv(shape_matrix)
        return (
            float(np.sqrt(max(inv[0, 0], 0.0))),
            float(np.sqrt(max(inv[1, 1], 0.0))),
            float(np.sqrt(max(inv[2, 2], 0.0))),
        )

    def _h_guard_for(self, peak: _PeakPunch) -> tuple[float, float] | None:
        """Integer-H guard slab for a peak, or ``None`` if disabled."""
        if peak.source_node_hkl is None or self.integer_h_guard_hkl is None:
            return None
        return (float(peak.source_node_hkl[0]), float(self.integer_h_guard_hkl))

    def _fit_base_radii(self, vol: HKLVolume) -> tuple[float, float, float]:
        """Resolution-floor radii for the per-peak fit clip.

        In Q-mode the floor is the Q base ellipsoid's HKL bounding box (so the
        fitted punch is never smaller than the Å⁻¹ resolution and the floor is
        lattice-portable); in HKL mode it is the plain ``punch_radii`` — so the
        legacy fit is unchanged.
        """
        a = self._q_shape_matrix(vol)
        if a is not None:
            return self._ellipsoid_bounding_radii(a)
        return self._radii()

    @staticmethod
    def _shape_from_covariance(
        cov: NDArray[np.float64],
        steps: tuple[float, ...],
        base: tuple[float, ...],
        max_r: tuple[float, ...],
        n_sigma: float,
    ) -> NDArray[np.float64]:
        """HKL shape matrix ``A`` from a weighted second-moment covariance.

        The ellipsoid is built in the covariance eigenbasis: each eigen-radius is
        ``n_sigma·σ`` plus a half-voxel pad, then clipped to the base/max bounds
        *projected onto that eigenvector* (so the bound is anisotropy-aware).  For
        an axis-aligned peak the eigenvectors are the HKL axes and this reduces
        exactly to the diagonal ``_fit_integer_peak`` radii — the covariance fit
        is a strict generalisation.
        """
        lam, vecs = np.linalg.eigh(cov)  # ascending eigenvalues, orthonormal cols
        lam = np.clip(lam, 0.0, None)
        st = np.abs(np.asarray(steps, dtype=float))
        bs = np.asarray(base, dtype=float)
        mx = np.asarray(max_r, dtype=float)
        inv_r2 = np.empty(3)
        for k in range(3):
            v = vecs[:, k]
            pad = 0.5 * float(np.sqrt(np.sum((v * st) ** 2)))
            r = n_sigma * float(np.sqrt(lam[k])) + pad
            r_floor = float(np.sqrt(np.sum((v * bs) ** 2)))
            r_ceil = float(np.sqrt(np.sum((v * mx) ** 2)))
            r = min(max(r, r_floor), r_ceil)
            inv_r2[k] = 1.0 / (r * r)
        return np.asarray(vecs @ np.diag(inv_r2) @ vecs.T, dtype=np.float64)

    def _fold_phi_tail(
        self,
        vol: HKLVolume,
        shape_matrix: NDArray[np.float64],
        center_hkl: tuple[float, float, float],
        phi_tail: float,
    ) -> NDArray[np.float64]:
        """Inflate ``A`` along the local K-L ring tangent by ``phi_tail`` (a rank-1
        modification of the covariance), replacing the legacy union-of-ellipsoids.

        The half-extent of ``δᵀAδ ≤ 1`` along unit ``u`` is ``sqrt(uᵀ A⁻¹ u)``;
        adding ``(2·h_t·φ + φ²)·t̂t̂ᵀ`` to ``A⁻¹`` grows the tangential half-extent
        from ``h_t`` to ``h_t + φ`` and leaves orthogonal extents unchanged.
        """
        if phi_tail <= 0:
            return shape_matrix
        rt = self._kl_ring_directions(vol, center_hkl)
        if rt is None:
            return shape_matrix
        _, _, ktan, ltan = rt
        t = np.array([0.0, ktan, ltan])  # unit K-L tangent (H component 0)
        cov = np.linalg.inv(shape_matrix)
        h_t = float(np.sqrt(max(float(t @ cov @ t), 0.0)))
        tau = 2.0 * h_t * phi_tail + phi_tail * phi_tail
        return np.asarray(np.linalg.inv(cov + tau * np.outer(t, t)), dtype=np.float64)

    @staticmethod
    def _inflate_isotropic(
        shape_matrix: NDArray[np.float64], margin: float,
    ) -> NDArray[np.float64]:
        """Grow every half-radius of ``δᵀAδ ≤ 1`` by ``margin`` (guard band)."""
        if margin <= 0:
            return shape_matrix
        lam, vecs = np.linalg.eigh(shape_matrix)
        r = 1.0 / np.sqrt(np.clip(lam, 1e-300, None)) + margin
        return np.asarray(vecs @ np.diag(1.0 / (r * r)) @ vecs.T, dtype=np.float64)

    @staticmethod
    def _axis_hkl_margins_from_q_margin(
        vol: HKLVolume,
        margin_q: float,
    ) -> tuple[float, float, float]:
        """Axis-aligned HKL margins equivalent to a Q-space guard band."""
        margin_q = max(0.0, float(margin_q))
        if margin_q <= 0:
            return 0.0, 0.0, 0.0
        q_per_hkl = np.linalg.norm(vol.ub_matrix, axis=0)
        return tuple(
            float(margin_q / q) if q > 0 and np.isfinite(q) else 0.0
            for q in q_per_hkl
        )  # type: ignore[return-value]

    @staticmethod
    def _inflate_q_isotropic(
        vol: HKLVolume,
        shape_matrix: NDArray[np.float64],
        margin_q: float,
    ) -> NDArray[np.float64]:
        """Grow shape principal half-extents by a physical Q-space margin."""
        margin_q = max(0.0, float(margin_q))
        if margin_q <= 0:
            return shape_matrix
        lam, vecs = np.linalg.eigh(shape_matrix)
        radii = []
        for i, lam_i in enumerate(lam):
            v = vecs[:, i]
            q_per_hkl = float(np.linalg.norm(vol.ub_matrix @ v))
            r_hkl = 1.0 / np.sqrt(max(float(lam_i), 1e-300))
            if q_per_hkl > 0 and np.isfinite(q_per_hkl):
                r_hkl += margin_q / q_per_hkl
            radii.append(r_hkl)
        return np.asarray(
            vecs @ np.diag([1.0 / (r * r) for r in radii]) @ vecs.T,
            dtype=np.float64,
        )

    def _inflate_for_frame(
        self,
        vol: HKLVolume,
        shape_matrix: NDArray[np.float64],
        margin: float,
    ) -> NDArray[np.float64]:
        """Inflate a general punch shape using the active frame's margin units."""
        if str(self.punch_frame).lower() == "q":
            return self._inflate_q_isotropic(vol, shape_matrix, margin)
        return self._inflate_isotropic(shape_matrix, margin)

    @staticmethod
    def _steps(vol: HKLVolume) -> tuple[float, float, float]:
        def step(axis: NDArray) -> float:
            if axis.size < 2:
                return 1.0
            return float(axis[1] - axis[0])

        return (
            step(vol.h_axis),
            step(vol.k_axis),
            step(vol.l_axis),
        )

    def enumerate_bragg(self, vol: HKLVolume) -> list[tuple[int, int, int]]:
        """Integer (h,k,l) nodes within the grid extent."""
        hs = range(int(np.ceil(vol.h_axis.min())), int(np.floor(vol.h_axis.max())) + 1)
        ks = range(int(np.ceil(vol.k_axis.min())), int(np.floor(vol.k_axis.max())) + 1)
        ls = range(int(np.ceil(vol.l_axis.min())), int(np.floor(vol.l_axis.max())) + 1)
        return [(h, k, l) for h in hs for k in ks for l in ls
                if (h, k, l) != (0, 0, 0)]

    def detect_peaks(self, vol: HKLVolume) -> list[tuple[int, int, int, float]]:
        """Return ``(ih, ik, il, intensity)`` peak-centre voxels to punch.

        Dispatches on ``mode``:

        - ``"integer"`` — peaks at integer (h,k,l) nodes (symmetry-based; skips
          systematic absences when ``min_intensity`` is set).
        - ``"auto"`` / ``"search"`` — any sharp peak, found as a high-tail outlier above the
          robust per-|Q|-shell diffuse level.  Catches off-integer satellites
          (e.g. small-domain / superlattice reflections) the integer mode misses.
        - ``"both"`` — the union of the two (integer centres take precedence).
        """
        return [p.as_tuple() for p in self._detect_peak_records(vol)]

    def _detect_peak_records(self, vol: HKLVolume) -> list[_PeakPunch]:
        """Internal detector dispatch returning rich punch records."""
        if self.mode == "integer":
            return self._detect_integer(vol)
        if self.mode in {"auto", "search"}:
            return self._detect_search(vol)
        if self.mode == "both":
            # Sequential: punch the integer Bragg first, then search on the
            # residual.  With the strong integer peaks already masked out, the
            # per-|Q|-shell statistics are no longer inflated by them, so the
            # off-integer satellites stand out as clean outliers.
            integer = self._detect_integer(vol)
            keep = self._punch_centers(vol, np.ones(vol.shape, dtype=bool), integer)
            residual = dataclasses.replace(vol, mask=vol.mask & keep)
            return integer + self._detect_search(residual)
        raise ValueError(f"Unknown mode: {self.mode!r}")

    def _punches_incident_beam(self) -> bool:
        return self.punch_incident_beam if self.force_origin is None else bool(self.force_origin)

    def _incident_beam_center(self, vol: HKLVolume) -> tuple[int, int, int] | None:
        """Nearest valid voxel to the incident beam at (0,0,0)."""
        if not self._punches_incident_beam():
            return None
        ih = int(np.argmin(np.abs(vol.h_axis)))
        ik = int(np.argmin(np.abs(vol.k_axis)))
        il = int(np.argmin(np.abs(vol.l_axis)))
        if not (vol.mask[ih, ik, il] and np.isfinite(vol.data[ih, ik, il])):
            return None
        return ih, ik, il

    def _detect_integer(self, vol: HKLVolume) -> list[_PeakPunch]:
        """Peaks at integer (h,k,l) nodes.

        With ``min_intensity`` and ``integer_n_mad`` unset every node is returned
        at its nearest voxel (legacy punch-all).  When either is set, each node is
        examined in a local window: the peak is re-centred on the window argmax
        and kept only if it clears the requested absolute, local-prominence, and
        per-|Q|-shell thresholds — extinct nodes are dropped.
        """
        dh, dk, dl = self._steps(vol)
        nh, nk, nl = vol.shape
        data, valid = vol.data, (vol.mask & np.isfinite(vol.data))

        def nearest(axis: NDArray, val: int) -> int:
            return int(np.argmin(np.abs(axis - val)))

        out: list[_PeakPunch] = []
        if self.min_intensity is None and self.integer_n_mad is None:
            for h, k, l in self.enumerate_bragg(vol):
                ih = nearest(vol.h_axis, h)
                ik = nearest(vol.k_axis, k)
                il = nearest(vol.l_axis, l)
                out.append(
                    _PeakPunch(
                        ih=ih,
                        ik=ik,
                        il=il,
                        intensity=float("nan"),
                        center_hkl=(
                            float(vol.h_axis[ih]),
                            float(vol.k_axis[ik]),
                            float(vol.l_axis[il]),
                        ),
                        source_node_hkl=(h, k, l),
                    )
                )
            return out

        shell_thr = None
        shell_bins = None
        if self.integer_n_mad is not None:
            _, shell_bins, shell_thr = self._q_shell_thresholds(
                vol,
                q_step=self.integer_q_step or self.search_q_step,
                n_mad=float(self.integer_n_mad),
                min_intensity=(
                    -np.inf if self.min_intensity is None else float(self.min_intensity)
                ),
                min_shell_size=int(self.integer_min_shell_size),
            )

        wph = max(1, int(round(self.detect_window_hkl / abs(dh))))
        wpk = max(1, int(round(self.detect_window_hkl / abs(dk))))
        wpl = max(1, int(round(self.detect_window_hkl / abs(dl))))
        for h, k, l in self.enumerate_bragg(vol):
            ih, ik, il = (nearest(vol.h_axis, h), nearest(vol.k_axis, k),
                          nearest(vol.l_axis, l))
            hs, he = max(0, ih - wph), min(nh, ih + wph + 1)
            ks, ke = max(0, ik - wpk), min(nk, ik + wpk + 1)
            ls, le = max(0, il - wpl), min(nl, il + wpl + 1)
            win = data[hs:he, ks:ke, ls:le]
            wval = valid[hs:he, ks:ke, ls:le]
            if wval.sum() < 3:
                continue
            wv = np.where(wval, win, np.nan)
            peak = float(np.nanmax(wv))
            if not np.isfinite(peak):
                continue
            local_bg = float(np.nanmedian(wv))
            prom = peak - local_bg
            # re-centre on the true peak (thermal/lattice drift off the integer)
            off = np.unravel_index(int(np.nanargmax(wv)), wv.shape)
            ph = int(hs + int(off[0]))
            pk = int(ks + int(off[1]))
            pl = int(ls + int(off[2]))

            # Relative path: small-but-sharp peak, prominent in LOCAL-MAD units.
            # Catches weak Bragg at nodes that the absolute floors miss.
            ok_rel = False
            if self.integer_local_prominence_n_mad is not None and prom > 0:
                local_mad = float(np.nanmedian(np.abs(wv - local_bg))) * 1.4826
                ok_rel = (
                    local_mad > 0
                    and prom >= self.integer_local_prominence_n_mad * local_mad
                    and prom >= self.integer_local_min_prominence
                )

            # Absolute path: clears the configured floors and per-|Q| shell threshold.
            ok_abs = True
            if self.min_intensity is not None and peak < self.min_intensity:
                ok_abs = False
            if prom < self.min_prominence:
                ok_abs = False
            if ok_abs and shell_thr is not None and shell_bins is not None:
                if peak < float(shell_thr[shell_bins[ph, pk, pl]]):
                    ok_abs = False

            if not (ok_abs or ok_rel):
                continue
            center_hkl = (
                float(vol.h_axis[ph]),
                float(vol.k_axis[pk]),
                float(vol.l_axis[pl]),
            )
            radii_hkl = None
            shape_hkl = None
            if self.integer_optimize_position or self.integer_optimize_shape:
                center_hkl, radii_hkl, shape_hkl = self._fit_integer_peak(
                    vol, wv, wval, (hs, ks, ls), local_bg, peak,
                )
                ph = int(np.argmin(np.abs(vol.h_axis - center_hkl[0])))
                pk = int(np.argmin(np.abs(vol.k_axis - center_hkl[1])))
                pl = int(np.argmin(np.abs(vol.l_axis - center_hkl[2])))
                if not self.integer_optimize_shape:
                    radii_hkl = None
                    shape_hkl = None
            out.append(_PeakPunch(
                ih=ph, ik=pk, il=pl, intensity=peak,
                center_hkl=center_hkl, radii_hkl=radii_hkl, shape_hkl=shape_hkl,
                source_node_hkl=(h, k, l), local_background=local_bg,
            ))
        return out

    def _fit_integer_peak(
        self,
        vol: HKLVolume,
        window: NDArray[np.float64],
        valid: NDArray[np.bool_],
        origin: tuple[int, int, int],
        local_bg: float,
        peak: float,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float] | None,
        NDArray[np.float64] | None,
    ]:
        """Fit a local integer-node peak by robust moments in HKL coordinates.

        Returns ``(center, radii, shape)``.  In the default (diagonal) mode
        ``radii`` is the three axis-aligned half-radii and ``shape`` is ``None``;
        with ``integer_fit_covariance`` it returns ``radii=None`` and ``shape`` a
        full 3×3 HKL shape matrix (a tilted ellipsoid following the real peak).
        """
        hs, ks, ls = origin
        excess = np.where(valid, window - local_bg, 0.0)
        peak_excess = max(peak - local_bg, 0.0)

        def _argmax_center() -> tuple[float, float, float]:
            off = np.unravel_index(int(np.nanargmax(window)), window.shape)
            return (float(vol.h_axis[hs + off[0]]), float(vol.k_axis[ks + off[1]]),
                    float(vol.l_axis[ls + off[2]]))

        if peak_excess <= 0:
            return _argmax_center(), None, None

        threshold = max(0.0, float(self.integer_fit_threshold_frac)) * peak_excess
        fit_mask = valid & (excess >= threshold)
        if int(fit_mask.sum()) < 3:
            fit_mask = valid & (excess > 0)
        if int(fit_mask.sum()) < 3:
            return _argmax_center(), None, None

        H, K, L = np.meshgrid(
            vol.h_axis[hs:hs + window.shape[0]],
            vol.k_axis[ks:ks + window.shape[1]],
            vol.l_axis[ls:ls + window.shape[2]],
            indexing="ij",
        )
        weights = np.where(fit_mask, excess, 0.0)
        wsum = float(weights.sum())
        if wsum <= 0:
            return _argmax_center(), None, None

        center = (
            float((weights * H).sum() / wsum),
            float((weights * K).sum() / wsum),
            float((weights * L).sum() / wsum),
        )
        if not self.integer_optimize_shape:
            return center, None, None

        steps = tuple(abs(s) for s in self._steps(vol))
        base = self._fit_base_radii(vol)
        if self.integer_fit_max_radius_hkl is None:
            max_r = tuple(float(r) * float(self.max_radius_scale) for r in base)
        else:
            max_r = tuple(float(r) for r in self.integer_fit_max_radius_hkl)
        n_sigma = max(float(self.integer_fit_radius_n_sigma), 0.0)

        d = [H - center[0], K - center[1], L - center[2]]
        if self.integer_fit_covariance:
            cov = np.empty((3, 3))
            for i in range(3):
                for j in range(i, 3):
                    cov[i, j] = cov[j, i] = float((weights * d[i] * d[j]).sum() / wsum)
            shape = self._shape_from_covariance(cov, steps, base, max_r, n_sigma)
            return center, None, shape

        dh, dk, dl = steps
        sigmas = (
            float(np.sqrt(max((weights * d[0] ** 2).sum() / wsum, 0.0))),
            float(np.sqrt(max((weights * d[1] ** 2).sum() / wsum, 0.0))),
            float(np.sqrt(max((weights * d[2] ** 2).sum() / wsum, 0.0))),
        )
        fitted = (
            min(max(n_sigma * sigmas[0] + 0.5 * dh, base[0]), max_r[0]),
            min(max(n_sigma * sigmas[1] + 0.5 * dk, base[1]), max_r[1]),
            min(max(n_sigma * sigmas[2] + 0.5 * dl, base[2]), max_r[2]),
        )
        return center, fitted, None

    @staticmethod
    def _q_shell_thresholds(
        vol: HKLVolume,
        q_step: float,
        n_mad: float,
        min_intensity: float,
        min_shell_size: int = 20,
    ) -> tuple[NDArray[np.float64], NDArray[np.int_], NDArray[np.float64]]:
        """Robust per-|Q|-shell high-tail threshold arrays."""
        q = vol.q_magnitude()
        valid = vol.mask & np.isfinite(vol.data)
        if not valid.any():
            return q, np.zeros(vol.shape, dtype=int), np.full(1, np.inf)
        qs = float(q_step)
        qv = q[valid]
        edges = np.arange(qv.min(), qv.max() + qs, qs)
        nb = max(len(edges) - 1, 1)
        bin_idx = np.clip(np.digitize(q, edges) - 1, 0, nb - 1)

        # Per-shell robust threshold (median + n·MAD), computed once over the
        # sorted valid voxels so it is O(N log N), not O(N · n_bins).
        flat_b = bin_idx[valid]
        flat_I = vol.data[valid]
        order = np.argsort(flat_b, kind="stable")
        sb, sI = flat_b[order], flat_I[order]
        bounds = np.searchsorted(sb, np.arange(nb + 1))
        thr = np.full(nb, np.inf)
        for b in range(nb):
            seg = sI[bounds[b]:bounds[b + 1]]
            if seg.size < min_shell_size:
                continue
            med = float(np.median(seg))
            mad = float(np.median(np.abs(seg - med)))
            scale = 1.4826 * mad if mad > 0 else (float(np.std(seg)) or 1.0)
            thr[b] = med + n_mad * scale
        thr = np.maximum(thr, min_intensity)
        return q, bin_idx, thr

    def _detect_search(self, vol: HKLVolume) -> list[_PeakPunch]:
        """Peaks found as sharp |Q|-shell outliers (mode-agnostic to hkl).

        Reuses the ring-removal insight: at a given |Q| the diffuse is the bulk
        and any Bragg / satellite reflection is a sharp high-tail outlier.  For
        each |Q| shell (width ``search_q_step``) the robust level (median) and
        scale (MAD) are measured over the valid voxels; a voxel is a peak
        candidate when it exceeds ``median + search_n_mad · 1.4826·MAD`` (and an
        absolute floor).  Candidates are grouped into connected components and
        each component's brightest voxel is returned as a peak centre — so the
        shared ellipsoid punch removes the whole peak, not just its hottest voxel.
        """
        from scipy import ndimage

        valid = (vol.mask & np.isfinite(vol.data)) & ~self._search_excluded_h_mask(vol)
        if not valid.any():
            return []
        _, bin_idx, thr = self._q_shell_thresholds(
            vol,
            q_step=self.search_q_step,
            n_mad=self.search_n_mad,
            min_intensity=self.search_min_intensity,
        )

        cand = valid & (vol.data > thr[bin_idx])
        if not cand.any():
            return []
        # One centre per peak *summit*: a candidate voxel that is a local maximum
        # (≥ its 3×3×3 neighbours).  This catches every peak even when several are
        # joined into one above-threshold blob (taking a single max per connected
        # component would miss all but the brightest — e.g. satellites at the
        # measured-volume edge that touch a residual arc).
        scored = np.where(valid, vol.data, -np.inf)
        local_max = ndimage.maximum_filter(scored, size=3, mode="nearest")
        peaks = np.argwhere(cand & (scored >= local_max))
        if self.search_min_prominence > 0 and peaks.size:
            keep_peak = []
            nh, nk, nl = vol.shape
            for ih, ik, il in peaks:
                hs, he = max(0, ih - 1), min(nh, ih + 2)
                ks, ke = max(0, ik - 1), min(nk, ik + 2)
                ls, le = max(0, il - 1), min(nl, il + 2)
                w = vol.data[hs:he, ks:ke, ls:le]
                m = valid[hs:he, ks:ke, ls:le]
                if int(m.sum()) < 3:
                    keep_peak.append(False)
                    continue
                local_bg = float(np.median(w[m]))
                keep_peak.append(
                    float(vol.data[ih, ik, il]) - local_bg >= self.search_min_prominence
                )
            peaks = peaks[np.asarray(keep_peak, dtype=bool)]
        return [
            _PeakPunch(
                ih=int(ih),
                ik=int(ik),
                il=int(il),
                intensity=float(vol.data[ih, ik, il]),
                center_hkl=(
                    float(vol.h_axis[ih]),
                    float(vol.k_axis[ik]),
                    float(vol.l_axis[il]),
                ),
            )
            for ih, ik, il in peaks
        ]

    def _search_excluded_h_mask(self, vol: HKLVolume) -> NDArray[np.bool_]:
        """Return True for voxels protected from hkl-agnostic search punching.

        Two complementary mechanisms (combined), both using
        ``search_exclude_h_half_width``:
        - ``search_exclude_h_centers``: explicit H-plane centres.
        - ``search_exclude_h_fractions``: fractional parts mod 1 protected
          periodically across the whole range — e.g. ``(1/3, 2/3)`` shields
          every integer±1/3 plane (the q=1/3 satellite family).
        """
        half_width = max(float(self.search_exclude_h_half_width), 0.0)
        centers = self.search_exclude_h_centers
        fractions = self.search_exclude_h_fractions
        if half_width <= 0 or (not centers and not fractions):
            return np.zeros(vol.shape, dtype=bool)
        h_excluded = np.zeros(vol.h_axis.shape, dtype=bool)
        for h0 in centers or ():
            h_excluded |= np.abs(vol.h_axis - float(h0)) <= half_width
        if fractions:
            frac = np.mod(vol.h_axis, 1.0)  # [0,1); handles negative H naturally
            for f in fractions:
                f0 = float(f) % 1.0
                # circular distance on the unit interval
                d = np.abs(frac - f0)
                d = np.minimum(d, 1.0 - d)
                h_excluded |= d <= half_width
        return h_excluded[:, None, None]

    def _scale_factor(self, peak: float, ref: float) -> float:
        if not self.intensity_scale or not np.isfinite(peak) or ref <= 0:
            return 1.0
        return float(np.clip((peak / ref) ** (1.0 / 3.0), 1.0, self.max_radius_scale))

    def build_mask(self, vol: HKLVolume) -> NDArray[np.bool_]:
        """Return a keep-mask (True = valid, False = punched Bragg voxel).

        Built on local windows around each detected peak, so the cost is
        ``n_peaks × small_window`` rather than ``n_peaks × whole_volume``.
        """
        keep = self._punch_centers(
            vol, np.ones(vol.shape, dtype=bool), self._detect_peak_records(vol))
        return self._punch_incident_beam(vol, keep)

    def _punch_centers(
        self,
        vol: HKLVolume,
        keep: NDArray[np.bool_],
        peaks: list[_PeakPunch],
    ) -> NDArray[np.bool_]:
        """Punch an anisotropic, intensity-scaled ellipsoid at each peak centre,
        in place on *keep* (local windows only)."""
        # Base resolution radii: the Q ellipsoid's HKL bounding box in Q-mode
        # (lattice-portable floor), else the plain HKL radii — so the HKL path is
        # unchanged.  The diagonal per-peak fit is clipped to this same floor, so
        # in Q-mode it punches via the radii path below (identical mechanism to
        # HKL, incl. the union φ-tail) — the frame only relocates the floor.
        r_base = self._fit_base_radii(vol)
        q_shape = self._q_shape_matrix(vol)  # None in legacy (hkl) mode

        ref = self.intensity_ref
        if self.intensity_scale and ref is None:
            ints = np.array([p.intensity for p in peaks if np.isfinite(p.intensity)])
            ref = float(np.median(ints)) if ints.size else 1.0

        for peak_rec in peaks:
            s = self._scale_factor(peak_rec.intensity, ref if ref is not None else 1.0)
            center = (peak_rec.ih, peak_rec.ik, peak_rec.il)

            # (1) Covariance fit (Phase 3, either frame): tilted ellipsoid with the
            #     φ-tail and margin folded into the matrix.
            if peak_rec.shape_hkl is not None:
                a = self._inflate_for_frame(
                    vol,
                    self._fold_phi_tail(
                        vol, peak_rec.shape_hkl / (s * s), peak_rec.center_hkl,
                        max(0.0, float(self.phi_tail_hkl)) * s),
                    self.margin)
                self._punch_one(
                    vol, keep, center, self._ellipsoid_bounding_radii(a), 0.0,
                    center_hkl=peak_rec.center_hkl,
                    h_guard=self._h_guard_for(peak_rec), shape_matrix=a)
                continue

            # (2) Q-mode with no per-peak fit (search peaks / shape-fit off): the
            #     fixed Q base ellipsoid + Q-space margin + folded φ-tail.
            if q_shape is not None and peak_rec.radii_hkl is None:
                a = self._q_shape_matrix(vol, scale=s, margin_q=self.margin)
                assert a is not None
                a = self._fold_phi_tail(
                    vol, a, peak_rec.center_hkl,
                    max(0.0, float(self.phi_tail_hkl)) * s)
                self._punch_one(
                    vol, keep, center, self._ellipsoid_bounding_radii(a), 0.0,
                    center_hkl=peak_rec.center_hkl,
                    h_guard=self._h_guard_for(peak_rec), shape_matrix=a)
                continue

            # (3) Radii path: HKL adaptive, or Q-mode diagonal fit floored by the Q
            #     base — axis-aligned ellipsoid with the union φ-tail.
            rh_base, rk_base, rl_base = peak_rec.radii_hkl or r_base
            mh, mk, ml = (
                self._axis_hkl_margins_from_q_margin(vol, self.margin)
                if q_shape is not None
                else (float(self.margin), float(self.margin), float(self.margin))
            )
            radii = (
                rh_base * s + mh,
                rk_base * s + mk,
                rl_base * s + ml,
            )
            self._punch_one(
                vol, keep, center, radii,
                max(0.0, float(self.phi_tail_hkl)) * s,
                center_hkl=peak_rec.center_hkl,
                h_guard=self._h_guard_for(peak_rec),
            )
        return keep

    def _punch_incident_beam(self, vol: HKLVolume, keep: NDArray[np.bool_]) -> NDArray[np.bool_]:
        if not self._punches_incident_beam():
            return keep
        if self.incident_beam_q_radii is not None:
            radii_q = tuple(
                max(0.0, float(r) + max(0.0, float(self.incident_beam_q_margin)))
                for r in self.incident_beam_q_radii
            )
            if min(radii_q) <= 0:
                return keep
            shape = self._shape_matrix_from_q_radii(vol, radii_q)  # type: ignore[arg-type]
            return self._punch_origin_shape_matrix(vol, keep, shape)
        if self.incident_beam_ellipsoid_radii_hkl is not None:
            rh, rk, rl = (max(0.0, float(r))
                          for r in self.incident_beam_ellipsoid_radii_hkl)
            return self._punch_origin_ellipsoid(vol, keep, rh, rk, rl)
        if self.incident_beam_sphere_radius_hkl is not None:
            r = max(0.0, float(self.incident_beam_sphere_radius_hkl))
            return self._punch_origin_ellipsoid(vol, keep, r, r, r)
        center = self._incident_beam_center(vol)
        if center is None:
            return keep
        if self.incident_beam_radii is None:
            rh, rk, rl = self._radii()
            radii = (
                2.0 * rh + self.incident_beam_margin,
                2.0 * rk + self.incident_beam_margin,
                2.0 * rl + self.incident_beam_margin,
            )
        else:
            rh, rk, rl = (
                float(r) + self.incident_beam_margin for r in self.incident_beam_radii
            )
            radii = (rh, rk, rl)
        return self._punch_one(
            vol, keep, center, radii,
            max(0.0, float(self.incident_beam_phi_tail_hkl)),
        )

    def _punch_origin_shape_matrix(
        self,
        vol: HKLVolume,
        keep: NDArray[np.bool_],
        shape_matrix: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Punch an origin-centred ellipsoid described by ``δhklᵀAδhkl ≤ 1``."""
        ih = int(np.argmin(np.abs(vol.h_axis)))
        ik = int(np.argmin(np.abs(vol.k_axis)))
        il = int(np.argmin(np.abs(vol.l_axis)))
        radii = self._ellipsoid_bounding_radii(shape_matrix)
        return self._punch_one(
            vol,
            keep,
            (ih, ik, il),
            radii,
            0.0,
            center_hkl=(0.0, 0.0, 0.0),
            shape_matrix=shape_matrix,
        )

    def _punch_origin_ellipsoid(
        self,
        vol: HKLVolume,
        keep: NDArray[np.bool_],
        rh: float,
        rk: float,
        rl: float,
    ) -> NDArray[np.bool_]:
        """Punch an anisotropic HKL ellipsoid centred exactly at the origin."""
        if rh <= 0 or rk <= 0 or rl <= 0:
            return keep
        dh, dk, dl = self._steps(vol)
        nh, nk, nl = vol.shape
        ih = int(np.argmin(np.abs(vol.h_axis)))
        ik = int(np.argmin(np.abs(vol.k_axis)))
        il = int(np.argmin(np.abs(vol.l_axis)))
        wh = int(np.ceil(rh / abs(dh)))
        wk = int(np.ceil(rk / abs(dk)))
        wl = int(np.ceil(rl / abs(dl)))
        hs, he = max(0, ih - wh), min(nh, ih + wh + 1)
        ks, ke = max(0, ik - wk), min(nk, ik + wk + 1)
        ls, le = max(0, il - wl), min(nl, il + wl + 1)
        HH, KK, LL = np.meshgrid(vol.h_axis[hs:he], vol.k_axis[ks:ke],
                                  vol.l_axis[ls:le], indexing="ij")
        ellipsoid = _ellipsoid_inside(HH, KK, LL, radii=(rh, rk, rl))
        keep[hs:he, ks:ke, ls:le] &= ~ellipsoid
        return keep

    def _punch_one(
        self,
        vol: HKLVolume,
        keep: NDArray[np.bool_],
        center: tuple[int, int, int],
        radii: tuple[float, float, float],
        phi_tail: float,
        center_hkl: tuple[float, float, float] | None = None,
        h_guard: tuple[float, float] | None = None,
        shape_matrix: NDArray[np.float64] | None = None,
    ) -> NDArray[np.bool_]:
        """Punch one ellipsoid, optionally stretched along the local K-L tangent.

        ``shape_matrix`` (Q-space mode) overrides the axis-aligned ``radii``
        ellipsoid with the general quadratic form ``δhklᵀ A δhkl ≤ 1``; ``radii``
        is then only the HKL bounding box used to size the local window, and the
        φ-tail is not added.
        """
        dh, dk, dl = self._steps(vol)
        nh, nk, nl = vol.shape
        ih, ik, il = center
        rh, rk, rl = radii
        if center_hkl is None:
            ch, ck, cl = float(vol.h_axis[ih]), float(vol.k_axis[ik]), float(vol.l_axis[il])
        else:
            ch, ck, cl = center_hkl
        radial_tangent = self._kl_ring_directions(vol, (ch, ck, cl))
        if phi_tail > 0 and radial_tangent is not None:
            krad, lrad, ktan, ltan = radial_tangent
            wk_extra = abs(ktan) * phi_tail
            wl_extra = abs(ltan) * phi_tail
        else:
            wk_extra = wl_extra = 0.0
        wh, wk, wl = (
            int(np.ceil(rh / abs(dh))),
            int(np.ceil((rk + wk_extra) / abs(dk))),
            int(np.ceil((rl + wl_extra) / abs(dl))),
        )
        hs, he = max(0, ih - wh), min(nh, ih + wh + 1)
        ks, ke = max(0, ik - wk), min(nk, ik + wk + 1)
        ls, le = max(0, il - wl), min(nl, il + wl + 1)
        HH, KK, LL = np.meshgrid(vol.h_axis[hs:he], vol.k_axis[ks:ke],
                                 vol.l_axis[ls:le], indexing="ij")
        dH, dK, dL = HH - ch, KK - ck, LL - cl
        if shape_matrix is not None:
            punch = _ellipsoid_inside(dH, dK, dL, shape_matrix=shape_matrix)
        else:
            punch = _ellipsoid_inside(dH, dK, dL, radii=(rh, rk, rl))
        if shape_matrix is None and phi_tail > 0 and radial_tangent is not None:
            krad, lrad, ktan, ltan = radial_tangent
            d_rad = dK * krad + dL * lrad
            d_tan = dK * ktan + dL * ltan
            radial_half = max(float(np.hypot(krad * rk, lrad * rl)), 1e-12)
            tangent_half = max(float(np.hypot(ktan * rk, ltan * rl)) + phi_tail, 1e-12)
            phi_ell = (
                (dH / rh) ** 2
                + (d_rad / radial_half) ** 2
                + (d_tan / tangent_half) ** 2
            )
            punch |= phi_ell <= 1.0
        if h_guard is not None:
            h0, half_width = h_guard
            punch &= np.abs(HH - h0) <= max(float(half_width), 0.0)
        keep[hs:he, ks:ke, ls:le] &= ~punch
        return keep

    @staticmethod
    def _kl_ring_directions(
        vol: HKLVolume,
        hkl: tuple[float, float, float],
    ) -> tuple[float, float, float, float] | None:
        """Metric-aware radial and tangent unit vectors in the displayed K-L plane.

        Powder rings are constant-|Q| contours, with
        ``Q = UB @ hkl`` and ``|Q|² = hkl @ (UB.T @ UB) @ hkl``.  On a fixed-H
        ``0kl`` slice, the local radial direction in K-L coordinates is the K/L
        gradient of |Q|²; the ring tangent is perpendicular to that gradient.
        For an orthonormal UB this reduces to the familiar ``radial=(K,L)``,
        ``tangent=(-L,K)``.
        """
        metric = vol.ub_matrix.T @ vol.ub_matrix
        x = np.asarray(hkl, dtype=float)
        grad = 2.0 * (metric @ x)
        krad, lrad = float(grad[1]), float(grad[2])
        radial_norm = float(np.hypot(krad, lrad))
        if radial_norm <= 0:
            return None
        krad /= radial_norm
        lrad /= radial_norm
        ktan, ltan = -lrad, krad
        return krad, lrad, ktan, ltan

    def apply(self, vol: HKLVolume) -> HKLVolume:
        """Return a new volume with detected Bragg peaks masked out."""
        keep = self.build_mask(vol)
        return dataclasses.replace(vol, mask=vol.mask & keep)


def bragg_mask(
    vol: HKLVolume,
    punch_radius_hkl: float = 0.3,
    punch_radii: tuple[float, float, float] | None = None,
    min_intensity: float | None = None,
    min_prominence: float = 1.0,
    integer_n_mad: float | None = None,
    integer_q_step: float | None = None,
    integer_optimize_position: bool = False,
    integer_optimize_shape: bool = False,
    integer_fit_threshold_frac: float = 0.35,
    integer_fit_radius_n_sigma: float = 2.5,
    integer_fit_max_radius_hkl: tuple[float, float, float] | None = None,
    integer_h_guard_hkl: float | None = None,
    intensity_scale: bool = False,
    margin: float = 0.0,
    punch_incident_beam: bool = True,
    incident_beam_radii: tuple[float, float, float] | None = None,
    incident_beam_margin: float = 0.08,
    incident_beam_phi_tail_hkl: float = 0.0,
    incident_beam_q_radii: tuple[float, float, float] | None = None,
    incident_beam_q_margin: float = 0.0,
    incident_beam_ellipsoid_radii_hkl: tuple[float, float, float] | None = None,
    incident_beam_sphere_radius_hkl: float | None = None,
    force_origin: bool | None = None,
    phi_tail_hkl: float = 0.0,
    search_exclude_h_centers: tuple[float, ...] | None = None,
    search_exclude_h_half_width: float = 0.0,
) -> NDArray[np.bool_]:
    """Convenience wrapper.  Returns a keep-mask (True = valid)."""
    return BraggRemover(
        punch_radius_hkl=punch_radius_hkl,
        punch_radii=punch_radii,
        min_intensity=min_intensity,
        min_prominence=min_prominence,
        integer_n_mad=integer_n_mad,
        integer_q_step=integer_q_step,
        integer_optimize_position=integer_optimize_position,
        integer_optimize_shape=integer_optimize_shape,
        integer_fit_threshold_frac=integer_fit_threshold_frac,
        integer_fit_radius_n_sigma=integer_fit_radius_n_sigma,
        integer_fit_max_radius_hkl=integer_fit_max_radius_hkl,
        integer_h_guard_hkl=integer_h_guard_hkl,
        intensity_scale=intensity_scale,
        margin=margin,
        punch_incident_beam=punch_incident_beam,
        incident_beam_radii=incident_beam_radii,
        incident_beam_margin=incident_beam_margin,
        incident_beam_phi_tail_hkl=incident_beam_phi_tail_hkl,
        incident_beam_q_radii=incident_beam_q_radii,
        incident_beam_q_margin=incident_beam_q_margin,
        incident_beam_ellipsoid_radii_hkl=incident_beam_ellipsoid_radii_hkl,
        incident_beam_sphere_radius_hkl=incident_beam_sphere_radius_hkl,
        force_origin=force_origin,
        phi_tail_hkl=phi_tail_hkl,
        search_exclude_h_centers=search_exclude_h_centers,
        search_exclude_h_half_width=search_exclude_h_half_width,
    ).build_mask(vol)
