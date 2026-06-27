"""Spherical-frame Q-space punch (``punch_frame="spherical"``).

The spherical frame describes each Bragg peak's punch ellipsoid in the *local*
spherical basis ``(ρ̂, θ̂, φ̂)`` at that peak — rρ radial (along Q̂), rφ azimuthal
(a*–b* ring tangent, c* pole), rθ polar — so the ellipsoid is correctly oriented
for every reflection with no tilt angle.

These tests prove: the per-peak shape matrix is built correctly (radial axis
along Q̂, two transverse axes ⊥ Q̂); the isotropic case reduces to the metric
Q-sphere and matches the q-frame punch; the orientation tracks the peak position;
the pole is handled gracefully; and the anisotropy goes the expected way.
"""

import numpy as np
import pytest

from nebula3d.analysis.bragg import BraggRemover
from nebula3d.core import HKLVolume

# Reuse the Phase-2 reference UB (metric diagonal to ~0.5%) and its clean diagonal.
UB_REFERENCE = np.array([
    [-0.73475, -0.43626,  0.03571],
    [-0.75725,  0.41688,  0.03877],
    [-0.22948, -0.00539, -0.24912],
])
BSTAR = np.sqrt(np.diag(UB_REFERENCE.T @ UB_REFERENCE))
UB_DIAG = np.diag(BSTAR)


def _single_peak_vol(ub, center=(1.0, 0.0, 0.0), shape=(61, 61, 61),
                     rng=(-1.5, 1.5)):
    data = np.full(shape, 0.5, dtype=np.float64)
    vol = HKLVolume.from_arrays(data, rng, rng, rng, ub_matrix=ub)
    ih = int(np.argmin(np.abs(vol.h_axis - center[0])))
    ik = int(np.argmin(np.abs(vol.k_axis - center[1])))
    il = int(np.argmin(np.abs(vol.l_axis - center[2])))
    vol.data[ih, ik, il] = 100.0
    return vol


def _remover(**kw):
    base = dict(
        mode="integer", min_intensity=10.0, min_prominence=0.5,
        punch_incident_beam=False, intensity_scale=False, margin=0.0,
        phi_tail_hkl=0.0, integer_optimize_position=False,
        integer_optimize_shape=False,
    )
    base.update(kw)
    return BraggRemover(**base)


# --------------------------------------------------------------------------- #
# frame construction
# --------------------------------------------------------------------------- #

def test_spherical_frame_is_orthonormal_with_radial_along_q():
    vol = _single_peak_vol(UB_REFERENCE)
    center = (1.0, 0.4, -0.3)
    rho_hat, theta_hat, phi_hat = BraggRemover._spherical_frame(vol, center)
    q = vol.ub_matrix @ np.asarray(center)
    # radial axis is exactly Q̂
    np.testing.assert_allclose(rho_hat, q / np.linalg.norm(q), atol=1e-12)
    # orthonormal right-handed frame
    R = np.column_stack([rho_hat, theta_hat, phi_hat])
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)
    # azimuth lies in the a*–b* plane (⊥ c*): φ̂ · ẑ ≈ 0
    z = vol.ub_matrix[:, 2] / np.linalg.norm(vol.ub_matrix[:, 2])
    assert abs(float(phi_hat @ z)) < 1e-10


def test_spherical_frame_none_at_origin():
    vol = _single_peak_vol(UB_REFERENCE)
    assert BraggRemover._spherical_frame(vol, (0.0, 0.0, 0.0)) is None


def test_spherical_shape_radial_axis_is_softest_or_stiffest_as_set():
    """The Q-space shape A_Q has eigen-pairs (1/r², axis) — radial eigenvalue is
    1/rρ² with eigenvector Q̂, transverse eigenvalues 1/rθ², 1/rφ² ⊥ Q̂."""
    vol = _single_peak_vol(UB_REFERENCE)
    center = (1.0, 0.0, 0.5)
    rrho, rtheta, rphi = 0.20, 0.06, 0.10
    rem = _remover(punch_frame="spherical",
                   punch_spherical_radii=(rrho, rtheta, rphi))
    a_hkl = rem._spherical_shape_matrix(vol, center)
    ub = vol.ub_matrix
    ub_inv = np.linalg.inv(ub)
    a_q = ub_inv.T @ a_hkl @ ub_inv  # A_hkl = UBᵀ A_Q UB → A_Q = (UB⁻ᵀ) A_hkl (UB⁻¹)
    q = ub @ np.asarray(center)
    qhat = q / np.linalg.norm(q)
    # radial quadratic value along Q̂ is 1/rρ²
    np.testing.assert_allclose(float(qhat @ a_q @ qhat), 1.0 / rrho**2, rtol=1e-10)
    # eigenvalues of A_Q are exactly {1/r²}
    lam = np.sort(np.linalg.eigvalsh(a_q))
    np.testing.assert_allclose(
        lam, np.sort([1 / rrho**2, 1 / rtheta**2, 1 / rphi**2]), rtol=1e-10)


def test_spherical_requires_radii():
    vol = _single_peak_vol(UB_DIAG)
    with pytest.raises(ValueError):
        _remover(punch_frame="spherical")._spherical_shape_matrix(vol, (1.0, 0, 0))
    with pytest.raises(ValueError):
        _remover(punch_frame="spherical",
                 punch_spherical_radii=(0.1, 0.0, 0.1))._spherical_shape_matrix(
            vol, (1.0, 0, 0))


def test_default_library_frame_unchanged():
    # The library default stays hkl; only PunchParams/UI flip to spherical.
    assert BraggRemover().punch_frame == "hkl"


# --------------------------------------------------------------------------- #
# orientation tracks the peak position
# --------------------------------------------------------------------------- #

def test_orientation_differs_between_peaks_with_same_radii():
    vol = _single_peak_vol(UB_REFERENCE)
    rem = _remover(punch_frame="spherical",
                   punch_spherical_radii=(0.20, 0.06, 0.10))
    a1 = rem._spherical_shape_matrix(vol, (1.0, 0.0, 0.0))
    a2 = rem._spherical_shape_matrix(vol, (0.0, 1.0, 0.3))
    # same radii, different Q directions → genuinely different oriented ellipsoids
    assert not np.allclose(a1, a2, atol=1e-6)


def test_pole_peak_returns_finite_spd_matrix():
    """A peak with Q̂ ∥ c* (degenerate azimuth) still yields a valid SPD shape."""
    vol = _single_peak_vol(UB_DIAG)
    rem = _remover(punch_frame="spherical",
                   punch_spherical_radii=(0.20, 0.06, 0.10))
    a = rem._spherical_shape_matrix(vol, (0.0, 0.0, 1.0))  # along c* for diagonal UB
    assert np.all(np.isfinite(a))
    assert np.all(np.linalg.eigvalsh(a) > 0)


# --------------------------------------------------------------------------- #
# punch behaviour
# --------------------------------------------------------------------------- #

def test_isotropic_spherical_equals_q_sphere():
    """rρ=rθ=rφ=ρ → A = (UBᵀUB)/ρ² and the mask equals the q-frame Q-sphere."""
    rho = 0.12
    vol = _single_peak_vol(UB_REFERENCE)
    a = _remover(punch_frame="spherical",
                 punch_spherical_radii=(rho, rho, rho))._spherical_shape_matrix(
        vol, (1.0, 0.0, 0.0))
    np.testing.assert_allclose(a, (UB_REFERENCE.T @ UB_REFERENCE) / rho**2, rtol=1e-10)

    keep_sph = _remover(punch_frame="spherical",
                        punch_spherical_radii=(rho, rho, rho)).build_mask(vol)
    keep_q = _remover(punch_frame="q", punch_q_radius=rho).build_mask(
        _single_peak_vol(UB_REFERENCE))
    np.testing.assert_array_equal(keep_sph, keep_q)


def test_radial_punch_reaches_further_when_rho_is_largest():
    """With rρ ≫ rθ,rφ the hole is elongated along Q̂ (radial), not transverse."""
    vol = _single_peak_vol(UB_REFERENCE, center=(1.0, 0.0, 0.0))
    keep = _remover(punch_frame="spherical",
                    punch_spherical_radii=(0.30, 0.05, 0.05)).build_mask(vol)
    punched = ~keep
    assert punched.any()
    H, K, L = vol.hkl_grid()
    d = np.stack([H - 1.0, K, L], axis=-1)
    dq = d @ vol.ub_matrix.T            # δQ Cartesian
    q1 = vol.ub_matrix @ np.array([1.0, 0.0, 0.0])
    qhat = q1 / np.linalg.norm(q1)
    proj = dq @ qhat                                   # signed radial projection
    transverse = np.linalg.norm(dq - proj[..., None] * qhat, axis=-1)
    radial = np.abs(proj)
    # the most radial-displaced punched voxel reaches much further than the most
    # transverse-displaced one
    assert radial[punched].max() > 3.0 * transverse[punched].max()
