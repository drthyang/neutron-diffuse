"""Phase 0 — characterization & specification tests for the Q-space punch work.

These tests do **not** change any production behaviour.  They lock down the
*current* Bragg-punch output and pin the mathematical invariants that the
upcoming quadratic-form ("resolution-ellipsoid") punch kernel must honour, so
the later refactor can be proven to leave the working pipeline untouched.

Two groups:

1. **Golden masters** — snapshot the default ``punch_bragg`` keep-mask (count +
   content hash) and per-mechanism punch counts on a deterministic synthetic
   volume built on the *real* TbTi3Bi4 reciprocal metric.  Any drift in the
   detector/punch geometry trips these.

2. **Specification invariants** — reference implementations of the HKL-axis and
   Q-axis ellipsoids, proving on the grid that for a diagonal (orthogonal)
   metric the current anisotropic HKL punch is *identical* to a Q-space
   ellipsoid whose radii are expressed in Å^-1 (``r_rlu * b*``).  This is the
   contract the new kernel implements; it also documents that the punch is
   near-isotropic in Q (the headline finding motivating the move).
"""

import hashlib

import numpy as np

from ndiff.analysis.bragg import BraggRemover
from ndiff.core import HKLVolume
from ndiff.pipeline import PunchParams, punch_bragg

# Real 22 K UB matrix (columns = reciprocal-lattice vectors, Å^-1).  Rotated in
# the lab frame, but its metric g = UB^T UB is diagonal to ~0.5% → orthorhombic.
UB_22K = np.array([
    [-0.73475, -0.43626,  0.03571],
    [-0.75725,  0.41688,  0.03877],
    [-0.22948, -0.00539, -0.24912],
])


def _synthetic_vol() -> HKLVolume:
    """Deterministic diffuse background + planted peaks on the real metric.

    Kept CI-small (41^3) but faithful: the UB is the measured 22 K orientation,
    so the metric anisotropy (b* = 1.08, 0.60, 0.25 Å^-1) is real.  Peaks are
    planted at the origin (incident beam), two integer Bragg nodes, an integer
    node off the kz=0 plane, and one off-integer satellite at (0.5, 0, 0).
    """
    shape = (41, 41, 41)
    rng = np.random.default_rng(20240613)
    data = rng.uniform(0.4, 1.2, shape).astype(np.float64)
    vol = HKLVolume.from_arrays(
        data, (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), ub_matrix=UB_22K)

    def _set(h0: float, k0: float, l0: float, amp: float) -> None:
        ih = int(np.argmin(np.abs(vol.h_axis - h0)))
        ik = int(np.argmin(np.abs(vol.k_axis - k0)))
        il = int(np.argmin(np.abs(vol.l_axis - l0)))
        vol.data[ih, ik, il] = amp

    _set(0, 0, 0, 300.0)   # incident beam
    _set(1, 0, 0, 60.0)    # integer Bragg
    _set(0, 1, 0, 40.0)    # integer Bragg
    _set(1, 0, 1, 25.0)    # integer Bragg off the l=0 plane
    _set(0.5, 0, 0, 18.0)  # off-integer satellite (search mode)
    return vol


# ---------------------------------------------------------------------------
# Group 1 — golden masters (freeze current behaviour)
# ---------------------------------------------------------------------------

# Captured from the current implementation on 2026-06-13.  If a future change is
# *intended* to alter the punch geometry, regenerate these deliberately.
_GOLDEN_DEFAULT_PUNCHED = 821
_GOLDEN_DEFAULT_KEEP_SHA256 = (
    "7280be18cfc534f88c22a0b1a316582d29677e2ed1545ec1fbcd29de03b6732d"
)
_GOLDEN_INTEGER_ONLY_PUNCHED = 57
_GOLDEN_SEARCH_ONLY_PUNCHED = 76


def test_golden_default_punch_mask_is_unchanged():
    """The default-pipeline punch keep-mask is byte-for-byte stable."""
    vol = _synthetic_vol()
    out = punch_bragg(vol, PunchParams())
    keep = out.mask
    punched = int((~keep).sum())
    digest = hashlib.sha256(np.ascontiguousarray(keep).tobytes()).hexdigest()

    assert punched == _GOLDEN_DEFAULT_PUNCHED, (
        f"punched-voxel count drifted: {punched} != {_GOLDEN_DEFAULT_PUNCHED}"
    )
    assert digest == _GOLDEN_DEFAULT_KEEP_SHA256, (
        "default punch keep-mask changed; regenerate the golden only if "
        "the geometry change is intended"
    )


def test_golden_integer_mode_punch_count():
    """Integer-node punch count is frozen (origin excluded)."""
    vol = _synthetic_vol()
    rem = BraggRemover(mode="integer", punch_radii=(0.09, 0.12, 0.45),
                       min_intensity=10.0, force_origin=False)
    assert int((~rem.build_mask(vol)).sum()) == _GOLDEN_INTEGER_ONLY_PUNCHED


def test_golden_search_mode_punch_count():
    """Search-mode (off-integer satellite) punch count is frozen."""
    vol = _synthetic_vol()
    rem = BraggRemover(mode="search", punch_radii=(0.09, 0.12, 0.45),
                       search_min_intensity=10.0, search_n_mad=4.0,
                       force_origin=False)
    assert int((~rem.build_mask(vol)).sum()) == _GOLDEN_SEARCH_ONLY_PUNCHED


# ---------------------------------------------------------------------------
# Group 2 — specification invariants for the future quadratic-form kernel
# ---------------------------------------------------------------------------
#
# Reference implementations (independent of production code) of the two ways to
# describe an ellipsoidal punch.  The new kernel must reproduce these.


def _hkl_axis_ellipsoid(vol: HKLVolume, center, radii_rlu) -> np.ndarray:
    """Current punch shape: axis-aligned ellipsoid in fractional HKL."""
    H, K, L = vol.hkl_grid()
    dH, dK, dL = H - center[0], K - center[1], L - center[2]
    rh, rk, rl = radii_rlu
    return (dH / rh) ** 2 + (dK / rk) ** 2 + (dL / rl) ** 2 <= 1.0


def _q_axis_ellipsoid(vol: HKLVolume, center, radii_A) -> np.ndarray:
    """Ellipsoid aligned with the reciprocal axes, radii in Å^-1.

    δQ = UB · δhkl (q_cart = hkl @ UB.T); project onto the unit reciprocal-axis
    directions e_i = UB[:, i] / |UB[:, i]| and test the Å^-1 ellipsoid.
    """
    H, K, L = vol.hkl_grid()
    dhkl = np.stack([H - center[0], K - center[1], L - center[2]], axis=-1)
    dQ = dhkl @ vol.ub_matrix.T
    e = vol.ub_matrix / np.linalg.norm(vol.ub_matrix, axis=0)
    proj = dQ @ e  # components of δQ along each reciprocal-axis unit vector
    ra, rb, rc = radii_A
    return ((proj[..., 0] / ra) ** 2
            + (proj[..., 1] / rb) ** 2
            + (proj[..., 2] / rc) ** 2) <= 1.0


def _q_sphere(vol: HKLVolume, center, rho) -> np.ndarray:
    """Isotropic Q-sphere |δQ| <= rho, via the metric: δhkl^T g δhkl <= rho^2."""
    quad = _q_sphere_quad(vol, center)
    return quad <= rho ** 2


def _q_sphere_quad(vol: HKLVolume, center) -> np.ndarray:
    """The continuous form δhkl^T g δhkl (= |δQ|^2), tie-free for comparisons."""
    H, K, L = vol.hkl_grid()
    dhkl = np.stack([H - center[0], K - center[1], L - center[2]], axis=-1)
    g = vol.ub_matrix.T @ vol.ub_matrix
    return np.einsum("...i,ij,...j->...", dhkl, g, dhkl)


def _hkl_axis_quad(vol: HKLVolume, center, radii_rlu) -> np.ndarray:
    """Continuous HKL-axis ellipsoid value Σ (δ_i / r_i)^2 (punch where <= 1)."""
    H, K, L = vol.hkl_grid()
    dH, dK, dL = H - center[0], K - center[1], L - center[2]
    rh, rk, rl = radii_rlu
    return (dH / rh) ** 2 + (dK / rk) ** 2 + (dL / rl) ** 2


def _q_axis_quad(vol: HKLVolume, center, radii_A) -> np.ndarray:
    """Continuous Q-axis ellipsoid value (punch where <= 1)."""
    H, K, L = vol.hkl_grid()
    dhkl = np.stack([H - center[0], K - center[1], L - center[2]], axis=-1)
    dQ = dhkl @ vol.ub_matrix.T
    e = vol.ub_matrix / np.linalg.norm(vol.ub_matrix, axis=0)
    proj = dQ @ e
    ra, rb, rc = radii_A
    return ((proj[..., 0] / ra) ** 2
            + (proj[..., 1] / rb) ** 2
            + (proj[..., 2] / rc) ** 2)


def _diag_ub(vol: HKLVolume) -> np.ndarray:
    """An *exactly* diagonal UB with the fixture's reciprocal-axis lengths."""
    bstar = np.sqrt(np.diag(vol.ub_matrix.T @ vol.ub_matrix))
    return np.diag(bstar)


def test_fixture_metric_is_diagonal_to_half_percent():
    """Document the premise — and its limit.  The fixture's reciprocal metric is
    orthogonal only to ~0.5%; there is a small genuine shear.  So the Q-space
    equivalence is exact only for an *idealised* diagonal metric, not bit-for-bit
    on the real stored UB (see the boundary-difference test below)."""
    vol = _synthetic_vol()
    g = vol.ub_matrix.T @ vol.ub_matrix
    offdiag = np.abs(g - np.diag(np.diag(g))).max()
    diag = np.abs(np.diag(g)).max()
    assert 0.0 < offdiag / diag < 0.02


def test_q_and_hkl_quadratic_forms_match_on_exactly_diagonal_metric():
    """Tier-1 equivalence, robustly: on an *exactly* diagonal metric the
    continuous HKL-axis and Q-axis ellipsoid values are equal everywhere
    (radii_A = radii_rlu * b*).  Comparing the continuous forms — not the
    thresholded masks — avoids boundary-tie flips and is the contract the
    Phase-1 quadratic-form kernel must satisfy.
    """
    base = _synthetic_vol()
    vol = HKLVolume.from_arrays(
        base.data, (base.h_axis[0], base.h_axis[-1]),
        (base.k_axis[0], base.k_axis[-1]), (base.l_axis[0], base.l_axis[-1]),
        ub_matrix=_diag_ub(base))
    bstar = np.sqrt(np.diag(vol.ub_matrix.T @ vol.ub_matrix))
    for center in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                   (1.0, 0.0, 1.0)]:
        for radii_rlu in [(0.09, 0.12, 0.45), (0.2, 0.2, 0.2),
                          (0.15, 0.3, 0.6)]:
            radii_A = np.asarray(radii_rlu) * bstar
            q_hkl = _hkl_axis_quad(vol, center, radii_rlu)
            q_qax = _q_axis_quad(vol, center, radii_A)
            np.testing.assert_allclose(q_hkl, q_qax, rtol=1e-7, atol=1e-9)


def test_hkl_and_q_axis_masks_identical_on_exactly_diagonal_metric():
    """And the thresholded masks are bit-identical too, on an exactly diagonal
    metric — the literal 'identical masks' claim, valid only in the idealised
    orthogonal case."""
    base = _synthetic_vol()
    vol = HKLVolume.from_arrays(
        base.data, (base.h_axis[0], base.h_axis[-1]),
        (base.k_axis[0], base.k_axis[-1]), (base.l_axis[0], base.l_axis[-1]),
        ub_matrix=_diag_ub(base))
    bstar = np.sqrt(np.diag(vol.ub_matrix.T @ vol.ub_matrix))
    for center in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0)]:
        for radii_rlu in [(0.09, 0.12, 0.45), (0.2, 0.2, 0.2)]:
            radii_A = np.asarray(radii_rlu) * bstar
            hkl = _hkl_axis_ellipsoid(vol, center, radii_rlu)
            qax = _q_axis_ellipsoid(vol, center, radii_A)
            assert np.array_equal(hkl, qax)


def test_real_ub_hkl_vs_q_axis_differ_only_at_boundary():
    """Honesty test on the *real* (slightly sheared) 22 K UB: HKL-axis and
    Q-axis punches are not bit-identical, but disagree only at a handful of
    boundary voxels — and every disagreement sits within one grid step of the
    ellipsoid surface.  This bounds the real-world impact of the 0.5% shear."""
    vol = _synthetic_vol()  # carries the real UB_22K
    bstar = np.sqrt(np.diag(vol.ub_matrix.T @ vol.ub_matrix))
    for center in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                   (1.0, 0.0, 1.0)]:
        for radii_rlu in [(0.09, 0.12, 0.45), (0.2, 0.2, 0.2),
                          (0.15, 0.3, 0.6)]:
            radii_A = np.asarray(radii_rlu) * bstar
            hkl = _hkl_axis_ellipsoid(vol, center, radii_rlu)
            qax = _q_axis_ellipsoid(vol, center, radii_A)
            disagree = hkl ^ qax
            assert int(disagree.sum()) <= 16, (
                f"too many disagreements at center={center}, radii={radii_rlu}"
            )
            # every disagreeing voxel is a near-boundary tie (|quad - 1| small)
            if disagree.any():
                q_hkl = _hkl_axis_quad(vol, center, radii_rlu)
                assert np.abs(q_hkl[disagree] - 1.0).max() < 0.25


def test_default_punch_radii_are_near_isotropic_in_q():
    """The headline reveal: (0.09, 0.12, 0.45) rlu — a 5x HKL anisotropy — is
    ~0.07–0.11 Å^-1, i.e. near-isotropic in Q (max/min < 1.6)."""
    vol = _synthetic_vol()
    bstar = np.sqrt(np.diag(vol.ub_matrix.T @ vol.ub_matrix))
    radii_A = np.asarray((0.09, 0.12, 0.45)) * bstar
    np.testing.assert_allclose(radii_A, [0.0972, 0.0724, 0.1146], atol=2e-4)
    assert radii_A.max() / radii_A.min() < 1.6


def test_q_sphere_matches_q_axis_ellipsoid_when_radii_equal():
    """Sanity check on the reference forms: an isotropic Å^-1 ellipsoid and the
    metric Q-sphere of the same radius punch the same voxels."""
    vol = _synthetic_vol()
    rho = 0.1
    for center in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]:
        sphere = _q_sphere(vol, center, rho)
        ellip = _q_axis_ellipsoid(vol, center, (rho, rho, rho))
        assert np.array_equal(sphere, ellip)
