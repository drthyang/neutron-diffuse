"""Tests for Al peak detection and masking."""

import numpy as np
import pytest

from ndiff.background.aluminum import AluminumRemover, _al_q_values, AL_LATTICE_A
from ndiff.core import HKLVolume


def test_al_q_values_known_peaks():
    qs = _al_q_values(a=AL_LATTICE_A, q_max=8.0)
    # 111 peak: |Q| = 2π*sqrt(3)/a
    q_111 = 2 * np.pi * np.sqrt(3) / AL_LATTICE_A
    assert any(abs(q - q_111) < 1e-4 for q in qs)

    # 200 peak: |Q| = 2π*sqrt(4)/a = 4π/a
    q_200 = 4 * np.pi / AL_LATTICE_A
    assert any(abs(q - q_200) < 1e-4 for q in qs)


def test_al_q_values_fcc_selection_rule():
    qs = _al_q_values(a=AL_LATTICE_A, q_max=10.0)
    # 100 is forbidden in FCC; |Q| = 2π/a ≈ 1.552
    q_100 = 2 * np.pi / AL_LATTICE_A
    assert not any(abs(q - q_100) < 1e-4 for q in qs), "FCC-forbidden 100 peak found"


def _make_simple_volume(shape=(20, 20, 20), hkl_range=(-3, 3)):
    nh, nk, nl = shape
    data = np.ones(shape, dtype=float)
    h = np.linspace(hkl_range[0], hkl_range[1], nh)
    k = np.linspace(hkl_range[0], hkl_range[1], nk)
    l = np.linspace(hkl_range[0], hkl_range[1], nl)
    ub = 2 * np.pi * np.eye(3) / 4.0  # cubic, a=4 Å
    return HKLVolume.from_arrays(data, (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]),
                                  ub_matrix=ub)


def test_mask_removes_voxels():
    vol = _make_simple_volume()
    remover = AluminumRemover(al_lattice=AL_LATTICE_A, width_angstrom=0.3, width_sigma=None)
    mask = remover.build_mask(vol)
    # some voxels should be masked
    assert not mask.all(), "Expected some masked voxels near Al peaks"


def test_mask_all_true_when_no_peaks_in_range():
    vol = _make_simple_volume(hkl_range=(-0.1, 0.1))  # very small Q-range
    remover = AluminumRemover(al_lattice=AL_LATTICE_A, width_angstrom=0.01, width_sigma=None)
    mask = remover.build_mask(vol)
    # near Q=0 there are no Al peaks, so mask should be all True
    assert mask.all(), "Expected no masked voxels far from Al peaks"


def test_soft_weights_range():
    vol = _make_simple_volume()
    remover = AluminumRemover(al_lattice=AL_LATTICE_A, width_angstrom=0.2, taper=0.05)
    weights = remover.soft_weights(vol)
    assert weights.min() >= 0.0
    assert weights.max() <= 1.0
