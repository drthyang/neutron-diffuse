"""Tests for inpainting methods."""

import numpy as np
import pytest

from ndiff.inpainting.tv_inpainting import tv_inpaint
from ndiff.inpainting.interpolation import rbf_fill, biharmonic_fill
from ndiff.inpainting.symmetry import symmetry_fill, ORTHORHOMBIC_MMM
from ndiff.core import HKLVolume


def _smooth_volume(shape=(15, 15, 15)):
    """A smooth sinusoidal volume as ground truth."""
    x = np.linspace(0, 2 * np.pi, shape[0])
    y = np.linspace(0, 2 * np.pi, shape[1])
    z = np.linspace(0, 2 * np.pi, shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return np.sin(X) * np.cos(Y) + 0.5 * np.cos(Z)


def _center_mask(shape):
    """Mask that hides the central 3x3x3 cube."""
    mask = np.ones(shape, dtype=bool)
    c = [s // 2 for s in shape]
    mask[c[0]-1:c[0]+2, c[1]-1:c[1]+2, c[2]-1:c[2]+2] = False
    return mask


def test_tv_inpaint_recovers_smooth():
    truth = _smooth_volume()
    mask = _center_mask(truth.shape)
    corrupted = truth.copy()
    corrupted[~mask] = 0.0

    recovered = tv_inpaint(corrupted, mask, lam=0.05, max_iter=500)

    rms = np.sqrt(np.mean((recovered[~mask] - truth[~mask]) ** 2))
    scale = np.std(truth)
    # TV minimises an L1 gradient penalty, so it favours piecewise-constant
    # fields and staircases smooth curvature: even fully converged with weak
    # regularisation it bottoms out near 0.22 RMS/scale on this smooth
    # sinusoid (cf. the RBF sibling test, which tolerates < 0.3).  This bound
    # still catches a broken gradient/divergence adjoint, which drives the
    # error to ~0.93.
    assert rms / scale < 0.3, f"TV inpainting RMS error too large: {rms/scale:.3f}"


def test_biharmonic_fill_converges():
    truth = _smooth_volume(shape=(12, 12, 12))
    mask = _center_mask(truth.shape)
    corrupted = truth.copy()

    filled = biharmonic_fill(corrupted, mask, max_iter=300)
    # should not be NaN or inf
    assert np.isfinite(filled).all()
    # filled region should be closer to truth than zero
    rms_fill = np.sqrt(np.mean((filled[~mask] - truth[~mask]) ** 2))
    rms_zero = np.sqrt(np.mean(truth[~mask] ** 2))
    assert rms_fill < rms_zero


def test_rbf_fill_recovers_smooth():
    truth = _smooth_volume(shape=(10, 10, 10))
    mask = _center_mask(truth.shape)

    filled = rbf_fill(truth, mask, neighbors=20)
    rms = np.sqrt(np.mean((filled[~mask] - truth[~mask]) ** 2))
    scale = np.std(truth)
    assert rms / scale < 0.3, f"RBF fill RMS error too large: {rms/scale:.3f}"


def test_symmetry_fill_centrosymmetric():
    """With inversion symmetry, (-h,-k,-l) fills (h,k,l)."""
    shape = (11, 11, 11)
    data = _smooth_volume(shape)
    ub = 2 * np.pi * np.eye(3)
    h = np.linspace(-2, 2, shape[0])
    k = np.linspace(-2, 2, shape[1])
    l = np.linspace(-2, 2, shape[2])
    sigma = np.ones(shape) * 0.01
    mask = np.ones(shape, dtype=bool)

    # mask a voxel at index (8, 8, 8) → hkl ~ (1.2, 1.2, 1.2)
    mask[8, 8, 8] = False
    vol = HKLVolume(data=data, sigma=sigma, mask=mask,
                    h_axis=h, k_axis=k, l_axis=l, ub_matrix=ub)

    data_f, _, flag = symmetry_fill(vol, symmetry_ops=ORTHORHOMBIC_MMM)
    assert flag[8, 8, 8], "Centrosymmetric equivalent should fill the voxel"
