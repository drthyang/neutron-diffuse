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
from scipy.fft import fftn, fftshift, ifftshift, fftfreq
from scipy.ndimage import gaussian_filter


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
    crop_hkl: tuple[float, float, float] | None = None,
    subtract_smooth_bg: float | tuple[float, float, float] | None = None,
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
    crop_hkl:
        Optional ``(h_max, k_max, l_max)`` in r.l.u.  When given, the
        volume is symmetrically cropped to ``|H| ≤ h_max``,
        ``|K| ≤ k_max``, ``|L| ≤ l_max`` before the FFT.  Cropping to a
        smaller, more uniform region of Q-space suppresses edge artifacts
        that arise from incomplete coverage or detector gaps at high Q.
    subtract_smooth_bg:
        Optional Gaussian-blur sigma in r.l.u.  When set, a smooth
        Gaussian-blurred background is subtracted from the (filled) volume
        *before* windowing so that only the oscillatory diffuse modulation
        transforms.  The broad diffuse envelope that survives ring removal /
        Bragg punch / backfill is approximately separable; its FT would
        otherwise concentrate on the principal axes as a bright cross through
        the origin (see ``docs/algorithms/delta_pdf.md``).  Typical value
        ``~1.5``.  Trade-off: also removes genuine very-long-period / low-``r``
        correlations, which live at the same scale as the background.

        Pass a scalar for an isotropic-in-r.l.u. 3D blur, or a tuple
        ``(sigma_h, sigma_k, sigma_l)`` for per-axis control.  Use
        ``sigma_h = 0`` (e.g. ``(0, 1.5, 1.5)``) to estimate the background
        **slice-wise** — independently on each H plane — which is the right
        choice for H-layered/modulated data (an isotropic H-blur would smear
        the H=0/±1/3/±2/3 layers into each other's background).  This is
        mathematically identical to doing the 2D per-plane background
        subtraction and then a single 3D FFT (subtraction is linear and
        commutes with the transform).

    Returns
    -------
    DeltaPDF
    """
    from ndiff.core import HKLVolume  # local import to avoid circular

    data = vol.masked_data()  # NaN at masked voxels

    # Crop Q-space symmetrically to ±(h_max, k_max, l_max) in r.l.u.
    h_axis = vol.h_axis.copy()
    k_axis = vol.k_axis.copy()
    l_axis = vol.l_axis.copy()
    if crop_hkl is not None:
        h_max, k_max, l_max = crop_hkl
        ih = np.where(np.abs(h_axis) <= h_max)[0]
        ik = np.where(np.abs(k_axis) <= k_max)[0]
        il = np.where(np.abs(l_axis) <= l_max)[0]
        data    = data[ih[0]:ih[-1]+1, ik[0]:ik[-1]+1, il[0]:il[-1]+1]
        h_axis  = h_axis[ih[0]:ih[-1]+1]
        k_axis  = k_axis[ik[0]:ik[-1]+1]
        l_axis  = l_axis[il[0]:il[-1]+1]

    # Replace NaN with zero (filled volume should have no NaN)
    data = np.where(np.isfinite(data), data, 0.0)

    # Subtract a smooth (Gaussian-blurred) background BEFORE windowing so that
    # only the oscillatory diffuse modulation transforms.  Without this, the
    # broad ~separable diffuse envelope that survives ring removal / punch /
    # backfill FTs into a bright cross on the y_K=0 / z_L=0 axes (the scalar
    # subtract_mean below only removes the DC term, not the envelope shape).
    # sigma is in r.l.u.  See docs/algorithms/delta_pdf.md.
    #
    # A scalar sigma blurs all three axes equally.  A per-axis (sigma_h, sigma_k,
    # sigma_l) lets you set sigma_h=0 to estimate the background INDEPENDENTLY on
    # each H plane (slice-wise) — identical to running the 2D per-plane bg, then
    # one 3D FFT.  This is the right model for H-layered/modulated data, where an
    # isotropic H-blur (e.g. 1.5 r.l.u. ≈ 45 px on a 0.033-step H axis) would
    # smear the H=0/±1/3/±2/3 layers into each other's background.
    if subtract_smooth_bg:
        if np.isscalar(subtract_smooth_bg):
            sig_h = sig_k = sig_l = float(subtract_smooth_bg)
        else:
            sig_h, sig_k, sig_l = (float(s) for s in subtract_smooth_bg)
        dh0 = (h_axis[-1] - h_axis[0]) / max(len(h_axis) - 1, 1)
        dk0 = (k_axis[-1] - k_axis[0]) / max(len(k_axis) - 1, 1)
        dl0 = (l_axis[-1] - l_axis[0]) / max(len(l_axis) - 1, 1)
        sigma_px = (sig_h / dh0, sig_k / dk0, sig_l / dl0)
        data = data - gaussian_filter(data, sigma=sigma_px, mode="nearest")

    # Apply apodization window first, then subtract mean of the windowed
    # data so that the DC component (sum) is exactly zero and the r=0
    # DeltaPDF peak is suppressed. Subtracting before windowing leaves a
    # nonzero sum = ∫(data−mean)·w dQ, producing a spurious 10^5-amplitude
    # spike at r=0 that overwhelms near-origin structure.
    win = _build_window(data.shape, apodization, gaussian_sigma)
    data = data * win

    if subtract_mean:
        data -= data.mean()

    # Zero-pad to next power-of-2 for efficiency.  Pad SYMMETRICALLY so the
    # Q=0 origin (at index s//2 of each axis) stays at the centre of the
    # padded array — one-sided padding would shift the origin and reintroduce
    # the phase ramp that ifftshift (below) removes.
    if zero_pad:
        padded_shape = tuple(_next_power_of_2(s) for s in data.shape)
    else:
        padded_shape = data.shape
    pad_width = []
    for s, ps in zip(data.shape, padded_shape):
        lo = ps // 2 - s // 2          # land the origin on the new centre ps//2
        pad_width.append((lo, ps - s - lo))
    data = np.pad(data, pad_width, mode="constant")

    # The input has its Q=0 origin at the array centre, but fftn treats index
    # [0,0,0] as the origin.  Without ifftshift the transform picks up a linear
    # phase ramp e^{-iπk} → (-1)^k, which flips the sign of real-space features
    # by pixel parity and splits each correlation peak into mixed +/- lobes.
    # The correct centred transform is fftshift(fftn(ifftshift(·))).
    ft = fftshift(fftn(ifftshift(data)))
    delta_pdf = np.real(ft)  # take real part (valid for centrosymmetric I(Q))

    # Build real-space axes (use possibly-cropped local axes)
    nh, nk, nl = padded_shape
    dh = (h_axis[-1] - h_axis[0]) / max(len(h_axis) - 1, 1)
    dk = (k_axis[-1] - k_axis[0]) / max(len(k_axis) - 1, 1)
    dl = (l_axis[-1] - l_axis[0]) / max(len(l_axis) - 1, 1)

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
