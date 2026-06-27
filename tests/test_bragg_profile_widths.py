# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tsung-han Yang

"""Regression tests for Bragg-profile axis-width extraction.

A degenerate peak shape (one extremely large / unbounded width → a near-singular
shape matrix) used to crash the Bragg-punch profile step with
``LinAlgError: Singular matrix`` when inverting the Q-space shape matrix.  The
widths are now derived via ``pinv`` + a direct UB congruence, which is identical
for well-conditioned shapes and finite for degenerate ones.
"""

import numpy as np

from nebula3d.core import HKLVolume
from nebula3d.pipeline import _axis_widths_from_shape

UB = 2 * np.pi * np.eye(3) / 4.0


def _vol():
    return HKLVolume.from_arrays(
        np.zeros((4, 4, 4)), (-3, 3), (-3, 3), (-3, 3), ub_matrix=UB
    )


def test_axis_widths_match_inverse_for_well_conditioned_shape():
    """pinv path is bit-equivalent to the old inv path for a normal shape."""
    vol = _vol()
    radii = (0.1, 0.15, 0.2)
    shape = np.diag([1.0 / r**2 for r in radii])

    widths_hkl, widths_q = _axis_widths_from_shape(vol, shape)
    assert np.allclose(widths_hkl, radii)

    inv_ub = np.linalg.inv(UB)
    cov_q_old = np.linalg.inv(inv_ub.T @ shape @ inv_ub)
    widths_q_old = [float(np.sqrt(max(cov_q_old[i, i], 0.0))) for i in range(3)]
    assert np.allclose(widths_q, widths_q_old)


def test_axis_widths_finite_for_singular_shape():
    """A singular / degenerate shape yields finite widths instead of raising."""
    vol = _vol()
    singular = np.diag([1.0 / 0.1**2, 1.0 / 0.15**2, 0.0])  # null direction
    widths_hkl, widths_q = _axis_widths_from_shape(vol, singular)
    assert all(np.isfinite(widths_hkl))
    assert all(np.isfinite(widths_q))


def test_axis_widths_finite_for_huge_width_shape():
    """A peak with one extremely large width (the real 22 K failure mode)."""
    vol = _vol()
    shape = np.diag([1.0 / 0.1**2, 1.0 / 0.15**2, 1.0 / 1e6**2])
    widths_hkl, widths_q = _axis_widths_from_shape(vol, shape)
    assert all(np.isfinite(widths_hkl))
    assert all(np.isfinite(widths_q))
