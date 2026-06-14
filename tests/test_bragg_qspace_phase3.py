"""Phase 3 — covariance fit + φ-tail folded into one resolution ellipsoid.

With ``integer_fit_covariance`` the integer-node fit returns a full 3×3 HKL shape
matrix (a tilted ellipsoid following the peak's real orientation) instead of three
axis-aligned radii, and the φ-tail is folded in as a rank-1 tangential inflation
rather than a union of two ellipsoids.  Default off → the diagonal-radii fit +
union φ-tail is unchanged (Phase 0 golden masters still pass).
"""

import numpy as np
import pytest

from ndiff.analysis.bragg import BraggRemover
from ndiff.core import HKLVolume

UB_DIAG = np.diag([1.07979, 0.60344, 0.25464])  # ~TbTi3Bi4 |a*|,|b*|,|c*|

_STEPS = (0.05, 0.05, 0.05)
_BASE = (0.001, 0.001, 0.001)   # tiny → clipping never interferes in unit tests
_MAX = (5.0, 5.0, 5.0)
_NSIG = 2.5


def _extent(shape_matrix, u):
    """Half-extent of δᵀAδ ≤ 1 along unit vector u: sqrt(uᵀ A⁻¹ u)."""
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    return float(np.sqrt(u @ np.linalg.inv(shape_matrix) @ u))


# --------------------------------------------------------------------------- #
# _shape_from_covariance: the diagonal reduction and the tilted generalisation
# --------------------------------------------------------------------------- #

def test_shape_from_covariance_diagonal_reduces_to_radii():
    """A diagonal covariance gives exactly the diagonal-fit radii: the covariance
    path is a strict generalisation of the legacy per-axis fit."""
    sig = np.array([0.05, 0.06, 0.07])
    cov = np.diag(sig**2)
    a = BraggRemover._shape_from_covariance(cov, _STEPS, _BASE, _MAX, _NSIG)
    r = _NSIG * sig + 0.5 * np.array(_STEPS)
    np.testing.assert_allclose(a, np.diag(1.0 / r**2), atol=1e-9)


def test_shape_from_covariance_tilted_follows_orientation():
    """An off-diagonal covariance elongated along the K=L diagonal yields a shape
    whose long axis is that diagonal (H-component ≈ 0, |k| ≈ |l|)."""
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    cov = rot @ np.diag([0.05**2, 0.05**2, 0.20**2]) @ rot.T
    assert abs(cov[1, 2]) > 1e-3  # genuinely off-diagonal

    a = BraggRemover._shape_from_covariance(cov, _STEPS, _BASE, _MAX, _NSIG)
    # longest punch axis = eigenvector of A with the smallest eigenvalue
    lam, vecs = np.linalg.eigh(a)
    long_axis = vecs[:, int(np.argmin(lam))]
    assert abs(long_axis[0]) < 0.05                      # lies in the K-L plane
    assert abs(abs(long_axis[1]) - abs(long_axis[2])) < 0.05  # the K=L diagonal


def test_fold_phi_tail_inflates_tangent_only():
    """Folding the φ-tail grows the half-extent along the local ring tangent by
    ≈ φ and leaves the orthogonal extents unchanged (a rank-1 modification)."""
    vol = HKLVolume.from_arrays(np.zeros((11, 11, 11)), (-1, 1), (-1, 1), (-1, 1),
                                ub_matrix=UB_DIAG)
    rem = BraggRemover()
    a = np.diag([1 / 0.10**2, 1 / 0.10**2, 1 / 0.10**2])  # isotropic r = 0.10
    # at hkl = (0, 1, 0) the K-L ring tangent is the L direction (diagonal metric)
    phi = 0.06
    a2 = rem._fold_phi_tail(vol, a, (0.0, 1.0, 0.0), phi)

    assert _extent(a2, (0, 0, 1)) == pytest.approx(0.10 + phi, abs=1e-6)  # tangent +φ
    assert _extent(a2, (0, 1, 0)) == pytest.approx(0.10, abs=1e-6)        # radial same
    assert _extent(a2, (1, 0, 0)) == pytest.approx(0.10, abs=1e-6)        # H same


# --------------------------------------------------------------------------- #
# integration through the fit / punch
# --------------------------------------------------------------------------- #

def _gaussian_peak_vol(cov, center=(1.0, 0.0, 0.0), amp=100.0):
    """Flat background with one anisotropic Gaussian Bragg peak."""
    shape = (41, 41, 41)
    h = np.linspace(0.0, 2.0, shape[0])
    k = np.linspace(-1.0, 1.0, shape[1])
    l = np.linspace(-1.0, 1.0, shape[2])
    H, K, L = np.meshgrid(h, k, l, indexing="ij")
    d = np.stack([H - center[0], K - center[1], L - center[2]], axis=-1)
    quad = np.einsum("...i,ij,...j->...", d, np.linalg.inv(cov), d)
    data = 0.5 + amp * np.exp(-0.5 * quad)
    return HKLVolume.from_arrays(data, (0.0, 2.0), (-1.0, 1.0), (-1.0, 1.0),
                                 ub_matrix=UB_DIAG)


def _cov_remover(**kw):
    base = dict(
        mode="integer", min_intensity=10.0, min_prominence=0.5,
        punch_incident_beam=False, intensity_scale=False, margin=0.0,
        phi_tail_hkl=0.0, detect_window_hkl=0.4,
        integer_optimize_position=True, integer_optimize_shape=True,
        punch_radii=(0.01, 0.01, 0.01), integer_fit_max_radius_hkl=(2.0, 2.0, 2.0),
    )
    base.update(kw)
    return BraggRemover(**base)


def test_default_fit_returns_radii_not_shape():
    """With covariance off (default), the integer fit yields radii and no shape."""
    cov = np.diag([0.06**2, 0.06**2, 0.06**2])
    vol = _gaussian_peak_vol(cov)
    rec = next(r for r in _cov_remover()._detect_peak_records(vol)
               if abs(r.center_hkl[0] - 1.0) < 0.2)
    assert rec.radii_hkl is not None
    assert rec.shape_hkl is None


def test_covariance_fit_records_a_shape_matrix():
    """With covariance on, the integer fit yields a 3×3 shape and no radii."""
    cov = np.diag([0.06**2, 0.06**2, 0.06**2])
    vol = _gaussian_peak_vol(cov)
    rec = next(r for r in
               _cov_remover(integer_fit_covariance=True)._detect_peak_records(vol)
               if abs(r.center_hkl[0] - 1.0) < 0.2)
    assert rec.radii_hkl is None
    assert rec.shape_hkl is not None and rec.shape_hkl.shape == (3, 3)


def test_covariance_punch_follows_tilted_peak():
    """A peak elongated along the K=L diagonal is punched further along (0,1,1)
    than along the orthogonal (0,1,-1) — the tilt the diagonal fit cannot see."""
    # long axis = rot[:,1] = (0, 1, 1)/√2 (put the large variance on that axis)
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    cov = rot @ np.diag([0.05**2, 0.18**2, 0.05**2]) @ rot.T
    vol = _gaussian_peak_vol(cov)
    keep = _cov_remover(integer_fit_covariance=True).build_mask(vol)
    punched = ~keep

    ih = int(np.argmin(np.abs(vol.h_axis - 1.0)))

    def _ray(dk, dl):
        n = 0
        for t in np.arange(-0.6, 0.6, 0.02):
            ik = int(np.argmin(np.abs(vol.k_axis - t * dk)))
            il = int(np.argmin(np.abs(vol.l_axis - t * dl)))
            n += int(punched[ih, ik, il])
        return n

    c2, s2 = 1 / np.sqrt(2), 1 / np.sqrt(2)
    assert _ray(c2, s2) > _ray(c2, -s2)   # long diagonal punched further


def test_covariance_default_off_is_unchanged():
    """Toggling covariance off reproduces the legacy diagonal-radii punch mask."""
    cov = np.diag([0.07**2, 0.05**2, 0.05**2])
    vol = _gaussian_peak_vol(cov)
    keep_off = _cov_remover().build_mask(vol)
    # sanity: the default path still punches the peak
    assert (~keep_off).any()
