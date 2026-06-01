"""Aluminum powder-ring detection and masking in 3D HKL space.

Aluminum (FCC, Fm-3m, a ≈ 4.046 Å) produces powder rings at specific |Q| values.
Given the sample UB matrix we compute |Q| per voxel and mask voxels whose |Q|
falls within a tunable width of any Al reflection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ndiff.core import HKLVolume

# Al lattice parameter in Angstrom (room temperature; adjust for cryo)
AL_LATTICE_A = 4.0494


def _al_q_values(a: float = AL_LATTICE_A, q_max: float = 10.0) -> list[float]:
    """Return |Q| in Å^-1 for all allowed Al powder reflections up to q_max.

    Al is FCC: allowed when h,k,l are all-odd or all-even (including 0).
    |Q| = 2π/d = 2π * sqrt(h²+k²+l²) / a
    """
    q_vals: list[float] = []
    hmax = int(np.ceil(q_max * a / (2 * np.pi))) + 1
    seen: set[float] = set()
    for h in range(0, hmax + 1):
        for k in range(0, hmax + 1):
            for l in range(0, hmax + 1):
                if h == k == l == 0:
                    continue
                # FCC selection rule: all even or all odd
                parity = {h % 2, k % 2, l % 2}
                if len(parity) > 1:
                    continue
                q = 2 * np.pi * np.sqrt(h**2 + k**2 + l**2) / a
                if q > q_max:
                    continue
                # round to avoid float duplicates from equivalent reflections
                qr = round(q, 6)
                if qr not in seen:
                    seen.add(qr)
                    q_vals.append(q)
    return sorted(q_vals)


@dataclass
class AluminumRemover:
    """Detect and mask Al powder-ring contamination in an HKLVolume.

    Parameters
    ----------
    al_lattice:
        Al lattice parameter in Å. Default 4.0494 Å (room temperature).
    q_max:
        Maximum |Q| to consider in Å^-1.
    width_sigma:
        Half-width of mask in |Q| space for sigma-clipping mode.
        If None, ``width_angstrom`` is used directly.
    width_angstrom:
        Fixed half-width of mask in Å^-1. Overridden if ``width_sigma`` set.
    taper:
        Sigmoid-taper width (Å^-1). Values > 0 produce soft mask edges,
        reducing Gibbs-like truncation artifacts.
    """

    al_lattice: float = AL_LATTICE_A
    q_max: float = 10.0
    width_sigma: Optional[float] = 5.0
    width_angstrom: float = 0.05
    taper: float = 0.01

    _q_al: list[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._q_al = _al_q_values(self.al_lattice, self.q_max)

    @property
    def q_al(self) -> list[float]:
        """Sorted list of Al powder-ring |Q| positions in Å^-1."""
        return list(self._q_al)

    # ------------------------------------------------------------------

    def build_mask(self, vol: HKLVolume) -> NDArray[np.bool_]:
        """Return a boolean mask (True = keep, False = Al-contaminated).

        Parameters
        ----------
        vol:
            Input HKLVolume. The UB matrix is used to convert HKL to Q.
        """
        q_mag = vol.q_magnitude()
        width = self._adaptive_width(vol, q_mag) if self.width_sigma is not None else self.width_angstrom
        keep = np.ones(vol.shape, dtype=bool)
        for q0 in self._q_al:
            dq = np.abs(q_mag - q0)
            if self.taper > 0:
                # sigmoid taper: smooth transition from 1 (keep) to 0 (mask)
                weight = _sigmoid_taper(dq, width, self.taper)
                # treat as mask: voxels with weight < 0.5 are masked
                keep &= weight > 0.5
            else:
                keep &= dq > width
        return keep

    def _adaptive_width(self, vol: HKLVolume, q_mag: NDArray) -> float:
        """Estimate mask half-width via sigma-clipping on radial profile."""
        # use a coarse radial binning to find background level; then set width
        # so that we capture sigma-clipping outliers near each Al peak
        assert self.width_sigma is not None
        # Rough estimate: use the minimum inter-peak spacing as context,
        # and pick a width that corresponds to width_sigma * median(dq step)
        dq_step = np.median(np.diff(sorted(set(q_mag.ravel()[::100]))))
        return float(self.width_sigma * dq_step)

    def soft_weights(self, vol: HKLVolume) -> NDArray[np.float64]:
        """Return per-voxel weight in [0, 1]; 0 = fully Al, 1 = clean.

        Useful for weighted averaging rather than hard masking.
        """
        q_mag = vol.q_magnitude()
        width = self.width_angstrom
        weights = np.ones(vol.shape, dtype=np.float64)
        for q0 in self._q_al:
            dq = np.abs(q_mag - q0)
            weights = np.minimum(weights, _sigmoid_taper(dq, width, max(self.taper, 1e-4)))
        return weights


def aluminum_mask(
    vol: HKLVolume,
    al_lattice: float = AL_LATTICE_A,
    width_sigma: Optional[float] = 5.0,
    width_angstrom: float = 0.05,
    taper: float = 0.01,
) -> NDArray[np.bool_]:
    """Convenience wrapper. Returns True-where-valid boolean mask."""
    remover = AluminumRemover(
        al_lattice=al_lattice,
        width_sigma=width_sigma,
        width_angstrom=width_angstrom,
        taper=taper,
    )
    return remover.build_mask(vol)


def _sigmoid_taper(dq: NDArray, width: float, taper: float) -> NDArray[np.float64]:
    """Smooth step from 0 (inside peak) to 1 (outside peak).

    dq < width-taper  → ~0  (masked)
    dq > width+taper  → ~1  (kept)
    """
    x = (dq - width) / (taper + 1e-12)
    return 1.0 / (1.0 + np.exp(-x))
