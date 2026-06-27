# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tsung-han Yang

"""Reciprocal-space utility functions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ub_from_lattice(
    a: float, b: float, c: float,
    alpha: float = 90.0, beta: float = 90.0, gamma: float = 90.0,
) -> NDArray[np.float64]:
    """Construct a UB matrix (in Å^-1) for an orthorhombic-or-lower lattice.

    The B matrix is the standard crystallographic B matrix; U is assumed to be
    the identity (crystal mounted with a* along x, b* in xy plane).

    Parameters
    ----------
    a, b, c:
        Direct-lattice parameters in Å.
    alpha, beta, gamma:
        Direct-lattice angles in degrees.

    Returns
    -------
    UB:
        (3, 3) matrix in Å^-1. Column j is the j-th reciprocal basis vector.
    """
    alpha_r, beta_r, gamma_r = np.radians([alpha, beta, gamma])
    ca, cb, cg = np.cos([alpha_r, beta_r, gamma_r])
    sg = np.sin(gamma_r)
    V = (a * b * c * np.sqrt(
        1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg
    ))
    # reciprocal lattice parameters
    a_star = b * c * np.sin(alpha_r) / V
    b_star = a * c * np.sin(beta_r) / V
    c_star = a * b * sg / V
    cos_beta_star = (ca * cg - cb) / (np.sin(alpha_r) * sg)
    sin_beta_star = np.sqrt(max(1.0 - cos_beta_star**2, 0.0))

    _cos_gamma_star = min(max((cg - ca * cb) / (np.sin(alpha_r) * np.sin(beta_r)), -1), 1)
    _sin_gamma_star = np.sqrt(max(1 - _cos_gamma_star**2, 0))
    _c_perp = np.sqrt(max(1 - cos_beta_star**2 - sin_beta_star**2, 0))
    B = np.array([
        [a_star, b_star * np.cos(np.arccos(_cos_gamma_star)), c_star * cos_beta_star],
        [0,      b_star * _sin_gamma_star,                    c_star * sin_beta_star],
        [0,      0,                                           c_star * _c_perp],
    ], dtype=np.float64)
    # Multiply by 2π (physics convention Q = 2π/d)
    return 2 * np.pi * B


def d_spacing(h: int, k: int, l: int, a: float, b: float, c: float) -> float:
    """Return d-spacing (Å) for (hkl) in an orthorhombic lattice."""
    return 1.0 / np.sqrt((h / a) ** 2 + (k / b) ** 2 + (l / c) ** 2)


def q_to_hkl(
    q_cart: NDArray[np.float64],
    ub_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert Cartesian Q-vector(s) in Å^-1 to fractional HKL.

    Parameters
    ----------
    q_cart:
        Array of shape (..., 3) in Å^-1.
    ub_matrix:
        (3, 3) UB matrix (columns = reciprocal basis vectors × 2π).

    Returns
    -------
    hkl:
        Array of shape (..., 3).
    """
    return q_cart @ np.linalg.inv(ub_matrix).T
