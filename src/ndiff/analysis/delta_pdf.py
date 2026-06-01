"""3D-ΔPDF computation via Fourier transform of diffuse scattering.

The three-dimensional difference pair distribution function (3D-ΔPDF) is:

    Δρ(r) = FT[ I_diffuse(Q) ]
           = FT[ I_total(Q) − I_Bragg(Q) ]

where I_diffuse is the background-corrected, Bragg-punched, backfilled
volume. The result reveals real-space pair correlations from local disorder.

References
----------
Weber & Simonov, Z. Kristallogr. 227, 238–247 (2012)
Simonov, Weber & Steurer, J. Appl. Cryst. 47, 2011–2018 (2014)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fftn, fftshift, fftfreq


Window = Literal["hann", "gaussian", "none"]


@dataclass
class DeltaPDF:
    """Real-space 3D-ΔPDF result.

    Attributes
    ----------
    data:
        Shape (na, nb, nc) real-valued ΔPDF in Å^-3 (or arbitrary units).
    x_axis, y_axis, z_axis:
        Real-space coordinate arrays in Å (or fractional units if
        ``real_space_angstrom=False`` was passed to :func:`compute_delta_pdf`).
    q_max:
        |Q|_max used in the transform (Å^-1), for reference.
    apodization:
        Window function applied before FFT.
    """

    data: NDArray[np.float64]
    x_axis: NDArray[np.float64]
    y_axis: NDArray[np.float64]
    z_axis: NDArray[np.float64]
    q_max: float
    apodization: str

    def slice_hk0(self) -> NDArray[np.float64]:
        """Return the l=0 (z=0) slice."""
        mid = self.data.shape[2] // 2
        return self.data[:, :, mid]

    def slice_h0l(self) -> NDArray[np.float64]:
        """Return the k=0 (y=0) slice."""
        mid = self.data.shape[1] // 2
        return self.data[:, mid, :]

    def slice_0kl(self) -> NDArray[np.float64]:
        """Return the h=0 (x=0) slice."""
        mid = self.data.shape[0] // 2
        return self.data[mid, :, :]


def compute_delta_pdf(
    vol: "HKLVolume",  # noqa: F821
    apodization: Window = "hann",
    gaussian_sigma: float = 0.5,
    zero_pad: bool = True,
    subtract_mean: bool = True,
    real_space_angstrom: bool = True,
) -> DeltaPDF:
    """Compute the 3D-ΔPDF from a diffuse scattering volume.

    The input *vol* should be the fully cleaned, Bragg-punched, and
    backfilled diffuse scattering volume.

    Parameters
    ----------
    vol:
        Cleaned HKLVolume (output of backfill_bragg).
    apodization:
        Window function applied in Q-space before FFT to suppress
        termination ripples:
        - ``"hann"``: cosine-squared taper (recommended for most cases)
        - ``"gaussian"``: Gaussian with σ = *gaussian_sigma* × Q_max
        - ``"none"``: no window (hard truncation)
    gaussian_sigma:
        Width parameter for Gaussian window (fraction of Q_max).
    zero_pad:
        Pad to next power-of-2 grid size for efficient FFT.
    subtract_mean:
        Subtract the mean intensity before FFT to suppress the r=0 peak.
    real_space_angstrom:
        If True, compute real-space axes in Å using the UB matrix.
        If False, axes are in fractional units (1/HKL step).

    Returns
    -------
    DeltaPDF
    """
    from ndiff.core import HKLVolume  # local import to avoid circular

    data = vol.masked_data()  # NaN at masked voxels
    # Replace NaN with zero (filled volume should have no NaN)
    data = np.where(np.isfinite(data), data, 0.0)

    if subtract_mean:
        data -= data.mean()

    # Apply apodization window
    win = _build_window(data.shape, apodization, gaussian_sigma)
    data = data * win

    # Zero-pad to next power-of-2 for efficiency
    if zero_pad:
        padded_shape = tuple(_next_power_of_2(s) for s in data.shape)
        pad_width = [(0, ps - s) for s, ps in zip(data.shape, padded_shape)]
        data = np.pad(data, pad_width, mode="constant")
    else:
        padded_shape = data.shape

    # 3D FFT → shift origin to centre
    ft = fftshift(fftn(data))
    delta_pdf = np.real(ft)  # take real part (valid for centrosymmetric I(Q))

    # Build real-space axes
    nh, nk, nl = padded_shape
    dh = (vol.h_axis[-1] - vol.h_axis[0]) / max(len(vol.h_axis) - 1, 1)
    dk = (vol.k_axis[-1] - vol.k_axis[0]) / max(len(vol.k_axis) - 1, 1)
    dl = (vol.l_axis[-1] - vol.l_axis[0]) / max(len(vol.l_axis) - 1, 1)

    # FFT frequency grid (in reciprocal of HKL step → direct lattice units)
    x_frac = fftshift(fftfreq(nh, d=dh))
    y_frac = fftshift(fftfreq(nk, d=dk))
    z_frac = fftshift(fftfreq(nl, d=dl))

    if real_space_angstrom:
        # Convert fractional direct-lattice coordinates to Å
        # Real-space basis vectors = columns of (UB/2π)^{-T}  times 2π
        # i.e., direct lattice = 2π * inv(UB)^T
        try:
            direct = 2 * np.pi * np.linalg.inv(vol.ub_matrix).T
            a_vec = direct[:, 0]
            b_vec = direct[:, 1]
            c_vec = direct[:, 2]
            x_axis = x_frac * np.linalg.norm(a_vec)
            y_axis = y_frac * np.linalg.norm(b_vec)
            z_axis = z_frac * np.linalg.norm(c_vec)
        except np.linalg.LinAlgError:
            x_axis, y_axis, z_axis = x_frac, y_frac, z_frac
    else:
        x_axis, y_axis, z_axis = x_frac, y_frac, z_frac

    q_max = float(np.max(vol.q_magnitude()))

    return DeltaPDF(
        data=delta_pdf,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        q_max=q_max,
        apodization=apodization,
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_window(shape: tuple[int, ...], kind: Window, sigma: float) -> NDArray:
    """Build a 3D separable apodization window."""
    def _1d(n: int) -> NDArray:
        if kind == "hann":
            return np.hanning(n)
        if kind == "gaussian":
            x = np.linspace(-1, 1, n)
            return np.exp(-0.5 * (x / sigma) ** 2)
        return np.ones(n)

    wh = _1d(shape[0])[:, None, None]
    wk = _1d(shape[1])[None, :, None]
    wl = _1d(shape[2])[None, None, :]
    return wh * wk * wl


def _next_power_of_2(n: int) -> int:
    return 1 if n == 0 else 2 ** int(np.ceil(np.log2(n)))
