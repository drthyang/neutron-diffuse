"""Phase 1 — the quadratic-form punch kernel ``_ellipsoid_inside``.

Phase 1 introduces a single punch-shape kernel ``δᵀ A δ ≤ 1`` and routes the
existing axis-aligned ellipsoids through it.  Production keeps using the
*diagonal fast path* (``radii=``), which reproduces the historical ``(d/r)²``
arithmetic exactly — so the Phase 0 golden masters pass untouched (verified in
``test_bragg_qspace_phase0.py``).

These tests prove the *general matrix* path (``shape_matrix=``) — the one the
Q-space work will use — agrees with the legacy radii path when ``A = diag(1/r²)``,
and behaves correctly for a non-diagonal ``A`` (the Q-isotropic case ``A = g/ρ²``).

Per the Phase 0 finding, the general matrix path uses different floating-point
arithmetic than ``(d/r)²``, so equivalence is asserted on the *continuous*
quadratic values (tolerance), with the thresholded masks allowed to differ only
at voxels sitting on the ``quad == 1`` boundary.
"""

import numpy as np
import pytest

from ndiff.analysis.bragg import _ellipsoid_inside

# Real 22 K UB (metric g = UBᵀUB diagonal to ~0.5%); see Phase 0.
UB_22K = np.array([
    [-0.73475, -0.43626,  0.03571],
    [-0.75725,  0.41688,  0.03877],
    [-0.22948, -0.00539, -0.24912],
])


def _offset_grid(n: int = 41, span: float = 1.0):
    """A cube of HKL offsets δ = (dh, dk, dl) about the origin."""
    ax = np.linspace(-span, span, n)
    return np.meshgrid(ax, ax, ax, indexing="ij")


def test_radii_branch_is_exact_legacy_arithmetic():
    """The diagonal fast path reproduces the historical (d/r)² mask byte-for-byte."""
    dh, dk, dl = _offset_grid()
    for radii in [(0.3, 0.5, 0.7), (0.09, 0.12, 0.45), (0.2, 0.2, 0.2)]:
        rh, rk, rl = radii
        manual = (dh / rh) ** 2 + (dk / rk) ** 2 + (dl / rl) ** 2 <= 1.0
        got = _ellipsoid_inside(dh, dk, dl, radii=radii)
        assert np.array_equal(manual, got)


def test_diagonal_matrix_matches_radii_path():
    """A = diag(1/r²) (general path) equals the radii path: the continuous
    quadratic values are identical to tolerance, and the thresholded masks
    differ only at boundary ties (|quad - 1| ~ machine epsilon)."""
    dh, dk, dl = _offset_grid()
    for radii in [(0.3, 0.5, 0.7), (0.09, 0.12, 0.45), (0.2, 0.2, 0.2)]:
        rh, rk, rl = radii
        A = np.diag([1.0 / rh ** 2, 1.0 / rk ** 2, 1.0 / rl ** 2])
        m_radii = _ellipsoid_inside(dh, dk, dl, radii=radii)
        m_matrix = _ellipsoid_inside(dh, dk, dl, shape_matrix=A)

        q_radii = (dh / rh) ** 2 + (dk / rk) ** 2 + (dl / rl) ** 2
        q_matrix = A[0, 0] * dh * dh + A[1, 1] * dk * dk + A[2, 2] * dl * dl
        np.testing.assert_allclose(q_radii, q_matrix, rtol=1e-12, atol=1e-12)

        disagree = m_radii ^ m_matrix
        if disagree.any():
            assert np.abs(q_radii[disagree] - 1.0).max() < 1e-6


def test_ellipsoid_inside_requires_exactly_one_shape_spec():
    """Supplying neither or both of radii / shape_matrix is a usage error."""
    dh, dk, dl = _offset_grid(11)
    with pytest.raises(ValueError):
        _ellipsoid_inside(dh, dk, dl)
    with pytest.raises(ValueError):
        _ellipsoid_inside(dh, dk, dl, radii=(1.0, 1.0, 1.0),
                          shape_matrix=np.eye(3))


def test_q_isotropic_shape_matrix_equals_metric_sphere():
    """The Q-space isotropic punch A = g/ρ² is exactly the metric sphere
    |δQ| ≤ ρ — the general path's headline use case.  δᵀ(g/ρ²)δ ≤ 1 ⟺ δᵀgδ ≤ ρ²."""
    dh, dk, dl = _offset_grid(41, span=0.5)
    g = UB_22K.T @ UB_22K
    rho = 0.1
    A = g / rho ** 2
    m_matrix = _ellipsoid_inside(dh, dk, dl, shape_matrix=A)

    dhkl = np.stack([dh, dk, dl], axis=-1)
    qsq = np.einsum("...i,ij,...j->...", dhkl, g, dhkl)  # |δQ|²
    m_sphere = qsq <= rho ** 2

    assert int(m_matrix.sum()) > 0  # punches a non-trivial region
    disagree = m_matrix ^ m_sphere
    if disagree.any():
        # disagreements only at the spherical boundary
        assert np.abs(qsq[disagree] / rho ** 2 - 1.0).max() < 1e-6


def test_non_diagonal_shape_matrix_tilts_the_ellipsoid():
    """A genuinely off-diagonal A produces a tilted ellipsoid — the capability
    an HKL-axis radii triple cannot express (Phase 2/3 will drive this from a
    Q-space resolution metric)."""
    dh, dk, dl = _offset_grid(61, span=1.0)
    # Ellipsoid elongated along the H=K diagonal: rotate diag(1/a², 1/b²) by 45°.
    a, b = 0.6, 0.15
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    A = R @ np.diag([1.0 / a ** 2, 1.0 / b ** 2, 1.0 / 25.0]) @ R.T
    inside = _ellipsoid_inside(dh, dk, dl, shape_matrix=A)

    # The long axis points along (+1,+1,0): a point near (a/√2, a/√2, 0) is in,
    # while the same distance along (+1,-1,0) (the short axis) is out.
    def _val(h, k, l):
        d = np.array([h, k, l])
        return float(d @ A @ d)

    assert _val(a * c, a * s, 0.0) <= 1.0 + 1e-9     # along long axis → inside
    assert _val(a * c, -a * s, 0.0) > 1.0            # along short axis → outside
    assert inside.any()
