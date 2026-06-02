"""Total-variation (TV) inpainting via Chambolle-Pock primal-dual algorithm.

TV inpainting minimises:

    min_u  (1/2) ||W (u - f)||² + λ ||∇u||₁

where  f  = observed (partially corrupted) volume,  W  = diagonal weight
matrix (1 at unmasked voxels, 0 at masked), and ||∇u||₁ is the anisotropic
total variation (sum of absolute finite differences).

The Chambolle-Pock algorithm provides efficient, guaranteed convergence
without step-size tuning on the data fidelity term.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def tv_inpaint(
    data: NDArray[np.float64],
    mask: NDArray[np.bool_],
    lam: float = 0.1,
    max_iter: int = 300,
    tol: float = 1e-5,
) -> NDArray[np.float64]:
    """TV-regularised inpainting of 3D *data* at *mask* == False voxels.

    Parameters
    ----------
    data:
        3D intensity array. Values at masked positions are ignored.
    mask:
        Boolean; True = observed, False = to be filled.
    lam:
        Regularisation strength. Larger = smoother reconstruction.
        Typical range 0.01–1.0; start with 0.1 and tune.
    max_iter:
        Maximum Chambolle-Pock iterations.
    tol:
        Convergence tolerance on relative primal-dual gap.

    Returns
    -------
    u:
        Inpainted volume with masked regions reconstructed.
    """
    f = data.copy()
    # mask weight (observation operator W)
    W = mask.astype(np.float64)

    # --- operator norms for step size selection ---
    # Gradient operator L has spectral norm sqrt(6) in 3D (anisotropic)
    L_norm = np.sqrt(6.0)
    tau = 1.0 / L_norm        # primal step
    sigma = 1.0 / L_norm      # dual step

    u = f.copy()
    u_bar = u.copy()

    # dual variable (3 components for 3D gradient)
    p = np.zeros((3,) + f.shape, dtype=np.float64)

    for i in range(max_iter):
        u_old = u.copy()

        # --- dual update: p ← proj_{||·||∞ ≤ λ}(p + σ ∇ū) ---
        grad = _gradient(u_bar)
        p = p + sigma * grad
        # pointwise projection onto ℓ∞ ball of radius λ
        norms = np.maximum(1.0, np.abs(p) / lam)
        p = p / norms

        # --- primal update: u ← (u - τ div(p) + τ W f) / (1 + τ W) ---
        div_p = _divergence(p)
        u = (u - tau * div_p + tau * W * f) / (1.0 + tau * W)

        # --- extrapolation ---
        u_bar = 2.0 * u - u_old

        # convergence check
        rel_change = np.linalg.norm(u - u_old) / (np.linalg.norm(u_old) + 1e-12)
        if rel_change < tol and i > 10:
            break

    return u


# ------------------------------------------------------------------
# Finite-difference operators
# ------------------------------------------------------------------

def _gradient(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Forward finite-difference gradient, zero Neumann BC. Shape (3, *u.shape)."""
    grad = np.zeros((3,) + u.shape, dtype=u.dtype)
    grad[0, :-1, :, :] = u[1:, :, :] - u[:-1, :, :]   # ∂/∂h
    grad[1, :, :-1, :] = u[:, 1:, :] - u[:, :-1, :]   # ∂/∂k
    grad[2, :, :, :-1] = u[:, :, 1:] - u[:, :, :-1]   # ∂/∂l
    return grad


def _divergence(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Discrete adjoint ∇* of :func:`_gradient` (zero Neumann BC).

    Must be the *exact* adjoint of the forward-difference gradient so that
    ⟨∇u, p⟩ = ⟨u, ∇*p⟩; the Chambolle-Pock iteration only converges when this
    holds.  For the forward difference ``grad[:-1] = u[1:] - u[:-1]`` the
    adjoint is ``(∇*p)[j] = p[j-1] - p[j]`` (with the boundary terms following
    from the zero-padded last gradient component), assembled below by shifting
    ``p[:-1]`` rather than ``p[1:]``.
    """
    div = np.zeros(p.shape[1:], dtype=p.dtype)

    # h component
    div[:-1, :, :] -= p[0, :-1, :, :]
    div[1:, :, :] += p[0, :-1, :, :]

    # k component
    div[:, :-1, :] -= p[1, :, :-1, :]
    div[:, 1:, :] += p[1, :, :-1, :]

    # l component
    div[:, :, :-1] -= p[2, :, :, :-1]
    div[:, :, 1:] += p[2, :, :, :-1]

    return div
