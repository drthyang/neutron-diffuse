"""Phase 2 — the opt-in Q-space punch (``punch_frame="q"``).

Phase 2 lets the Bragg punch be specified in reciprocal Å^-1 instead of
fractional HKL, building the quadratic-form shape matrix ``A`` from the UB metric
and feeding it to the Phase-1 kernel.  Default ``punch_frame="hkl"`` is untouched
(the Phase 0 golden masters still pass).

These tests prove: the shape matrix is built correctly; the isotropic Q punch is
a true metric sphere around each peak; on a diagonal metric it reproduces the
equivalent HKL punch; and per-axis Q radii give the expected anisotropy.
"""

import numpy as np
import pytest

from nebula3d.analysis.bragg import BraggRemover
from nebula3d.core import HKLVolume

# Real 22 K UB (metric diagonal to ~0.5%); and an exactly diagonal UB built from
# its reciprocal-axis lengths for the clean-equivalence tests.
UB_22K = np.array([
    [-0.73475, -0.43626,  0.03571],
    [-0.75725,  0.41688,  0.03877],
    [-0.22948, -0.00539, -0.24912],
])
BSTAR = np.sqrt(np.diag(UB_22K.T @ UB_22K))         # |a*|,|b*|,|c*| in Å^-1
UB_DIAG = np.diag(BSTAR)


def _single_peak_vol(ub, center=(1.0, 0.0, 0.0), shape=(61, 61, 61),
                     rng=(-1.5, 1.5)):
    """Flat background with one strong Bragg peak at an integer node."""
    data = np.full(shape, 0.5, dtype=np.float64)
    vol = HKLVolume.from_arrays(data, rng, rng, rng, ub_matrix=ub)
    ih = int(np.argmin(np.abs(vol.h_axis - center[0])))
    ik = int(np.argmin(np.abs(vol.k_axis - center[1])))
    il = int(np.argmin(np.abs(vol.l_axis - center[2])))
    vol.data[ih, ik, il] = 100.0
    return vol


def _q_remover(**kw):
    """Integer-mode remover that punches the planted peak only (no origin)."""
    base = dict(
        mode="integer", min_intensity=10.0, min_prominence=0.5,
        punch_incident_beam=False, intensity_scale=False, margin=0.0,
        phi_tail_hkl=0.0, integer_optimize_position=False,
        integer_optimize_shape=False,
    )
    base.update(kw)
    return BraggRemover(**base)


# --------------------------------------------------------------------------- #
# shape-matrix construction
# --------------------------------------------------------------------------- #

def test_default_frame_is_hkl_no_q_shape():
    vol = _single_peak_vol(UB_DIAG)
    assert BraggRemover().punch_frame == "hkl"
    assert BraggRemover()._q_shape_matrix(vol) is None
    # a stray Q radius with the default hkl frame stays inert
    assert BraggRemover(punch_q_radius=0.1)._q_shape_matrix(vol) is None


def test_q_shape_matrix_isotropic_is_metric_over_rho2():
    vol = _single_peak_vol(UB_22K)
    rho = 0.1
    a = _q_remover(punch_frame="q", punch_q_radius=rho)._q_shape_matrix(vol)
    np.testing.assert_allclose(a, (UB_22K.T @ UB_22K) / rho**2, rtol=1e-12)


def test_q_shape_matrix_per_axis_on_diagonal_metric():
    vol = _single_peak_vol(UB_DIAG)
    radii = (0.08, 0.05, 0.2)
    a = _q_remover(punch_frame="q", punch_q_radii=radii)._q_shape_matrix(vol)
    expected = np.diag([(BSTAR[i] / radii[i]) ** 2 for i in range(3)])
    np.testing.assert_allclose(a, expected, atol=1e-12)


def test_q_frame_requires_a_radius():
    vol = _single_peak_vol(UB_DIAG)
    with pytest.raises(ValueError):
        _q_remover(punch_frame="q")._q_shape_matrix(vol)
    with pytest.raises(ValueError):
        _q_remover(punch_frame="q", punch_q_radius=-0.1)._q_shape_matrix(vol)
    with pytest.raises(ValueError):
        _q_remover(punch_frame="q", punch_q_radii=(0.1, 0.0, 0.1))._q_shape_matrix(vol)


# --------------------------------------------------------------------------- #
# punch behaviour
# --------------------------------------------------------------------------- #

def test_q_isotropic_punch_is_a_metric_sphere():
    """Every punched voxel is within ρ (Å^-1) of the peak; every kept voxel near
    it is outside ρ — i.e. the hole is the true Q-sphere |δQ| ≤ ρ."""
    rho = 0.12
    vol = _single_peak_vol(UB_22K, center=(1.0, 0.0, 0.0))
    keep = _q_remover(punch_frame="q", punch_q_radius=rho).build_mask(vol)
    punched = ~keep

    H, K, L = vol.hkl_grid()
    dhkl = np.stack([H - 1.0, K - 0.0, L - 0.0], axis=-1)
    qmag = np.linalg.norm(dhkl @ vol.ub_matrix.T, axis=-1)  # |δQ| Å^-1

    assert punched.any()
    # disagreements between the punch and the ideal sphere sit only on the rim
    disagree = punched ^ (qmag <= rho)
    if disagree.any():
        assert np.abs(qmag[disagree] - rho).max() < 0.02
    # interior is punched, well-outside is kept
    assert punched[qmag < rho - 0.02].all()
    assert (~punched[qmag > rho + 0.02]).all()


def test_q_isotropic_equals_equivalent_hkl_on_diagonal_metric():
    """On a diagonal metric, a Q-sphere of radius ρ punches the same voxels as an
    HKL ellipsoid with radii (ρ/|a*|, ρ/|b*|, ρ/|c*|) — the Phase 0 equivalence,
    now through the real punch path.  Masks agree except at boundary ties."""
    rho = 0.12
    vol_q = _single_peak_vol(UB_DIAG)
    vol_h = _single_peak_vol(UB_DIAG)
    keep_q = _q_remover(punch_frame="q", punch_q_radius=rho).build_mask(vol_q)
    keep_h = _q_remover(punch_radii=tuple(rho / BSTAR)).build_mask(vol_h)

    disagree = keep_q ^ keep_h
    # only a thin rim may differ; bound it well below the punched volume
    assert int(disagree.sum()) <= 0.05 * int((~keep_q).sum()) + 5


def test_q_punch_honors_margin():
    """The margin guard band inflates the Q-space hole too (it was silently
    ignored before — a Q-mode/HKL-mode inconsistency)."""
    vol = _single_peak_vol(UB_22K, center=(1.0, 0.0, 0.0))
    keep0 = _q_remover(punch_frame="q", punch_q_radius=0.10,
                       margin=0.0).build_mask(vol)
    keepm = _q_remover(punch_frame="q", punch_q_radius=0.10,
                       margin=0.05).build_mask(vol)
    assert int((~keepm).sum()) > int((~keep0).sum())


def test_q_per_axis_radii_punch_further_along_softer_axis():
    """Per-axis Q radii give independent control: a larger c*-axis radius punches
    further along L than the a*-axis radius does along H."""
    vol = _single_peak_vol(UB_DIAG, center=(1.0, 0.0, 0.0))
    keep = _q_remover(punch_frame="q",
                      punch_q_radii=(0.05, 0.05, 0.20)).build_mask(vol)
    punched = ~keep
    ih = int(np.argmin(np.abs(vol.h_axis - 1.0)))
    i0k = int(np.argmin(np.abs(vol.k_axis)))
    i0l = int(np.argmin(np.abs(vol.l_axis)))
    n_h = int(punched[:, i0k, i0l].sum())   # extent along H (tight a* radius)
    n_l = int(punched[ih, i0k, :].sum())    # extent along L (wide c* radius)
    assert n_l > n_h
