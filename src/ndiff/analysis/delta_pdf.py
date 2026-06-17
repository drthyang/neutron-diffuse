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
from scipy.fft import fftfreq, fftn, fftshift, ifftn, ifftshift
from scipy.ndimage import gaussian_filter

from ndiff.core import HKLVolume

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

    # --- inverse-transform metadata (populated by compute_delta_pdf) ----------
    # Everything needed to map the ΔPDF back to the reciprocal-space volume it
    # came from (see invert_delta_pdf).  Optional / default-None so older callers
    # and serialised results stay valid.
    pad_width: tuple[tuple[int, int], ...] | None = None
    cropped_shape: tuple[int, int, int] | None = None
    window_axes: tuple[NDArray[np.float64], ...] | None = None
    subtracted_mean: float = 0.0
    smooth_bg: NDArray[np.float64] | None = None
    h_axis_c: NDArray[np.float64] | None = None
    k_axis_c: NDArray[np.float64] | None = None
    l_axis_c: NDArray[np.float64] | None = None
    ub_matrix: NDArray[np.float64] | None = None

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
    vol: HKLVolume,
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
    smooth_bg: NDArray[np.float64] | None = None
    if subtract_smooth_bg:
        if isinstance(subtract_smooth_bg, tuple):
            sig_h, sig_k, sig_l = (float(s) for s in subtract_smooth_bg)
        else:
            sig_h = sig_k = sig_l = float(subtract_smooth_bg)
        dh0 = (h_axis[-1] - h_axis[0]) / max(len(h_axis) - 1, 1)
        dk0 = (k_axis[-1] - k_axis[0]) / max(len(k_axis) - 1, 1)
        dl0 = (l_axis[-1] - l_axis[0]) / max(len(l_axis) - 1, 1)
        sigma_px = (sig_h / dh0, sig_k / dk0, sig_l / dl0)
        smooth_bg = gaussian_filter(data, sigma=sigma_px, mode="nearest")
        data = data - smooth_bg

    # Shape of the (cropped, bg-subtracted) volume that actually enters the
    # transform — recorded so invert_delta_pdf can un-pad back to it.
    cropped_shape = data.shape

    # Apply apodization window first, then subtract mean of the windowed
    # data so that the DC component (sum) is exactly zero and the r=0
    # DeltaPDF peak is suppressed. Subtracting before windowing leaves a
    # nonzero sum = ∫(data−mean)·w dQ, producing a spurious 10^5-amplitude
    # spike at r=0 that overwhelms near-origin structure.
    window_axes = _window_axes(data.shape, apodization, gaussian_sigma)
    win = window_axes[0][:, None, None] * window_axes[1][None, :, None] \
        * window_axes[2][None, None, :]
    data = data * win

    subtracted_mean = 0.0
    if subtract_mean:
        subtracted_mean = float(data.mean())
        data -= subtracted_mean

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
        pad_width=tuple(tuple(p) for p in pad_width),  # type: ignore[misc]
        cropped_shape=cropped_shape,  # type: ignore[arg-type]
        window_axes=window_axes,
        subtracted_mean=subtracted_mean,
        smooth_bg=smooth_bg,
        h_axis_c=h_axis,
        k_axis_c=k_axis,
        l_axis_c=l_axis,
        ub_matrix=vol.ub_matrix.copy(),
    )


def invert_delta_pdf(
    dpdf: DeltaPDF,
    *,
    deapodize: bool = True,
    add_back_smooth_bg: bool = True,
    window_floor: float = 1e-3,
) -> HKLVolume:
    """Inverse-transform a 3D-ΔPDF back to its reciprocal-space diffuse volume.

    This is the exact mathematical inverse of :func:`compute_delta_pdf`: undo the
    centred FFT, strip the symmetric zero-padding, then (optionally) divide out
    the apodization window and restore the subtracted mean / smooth background —
    recovering the cleaned diffuse intensity ``I(Q)`` that produced the ΔPDF.

    Round-tripping ``compute_delta_pdf → invert_delta_pdf`` is the consistency
    check: the reconstruction should reproduce the transformed volume to
    numerical precision (the input is centrosymmetric — ``mmm`` Laue — so the
    real-part projection in the forward transform loses nothing).  Where it
    *doesn't* match, the discrepancy localises what the transform settings
    discard: high-|Q| detail removed by ``crop_hkl`` and, for ``hann``, the
    edge planes where the window vanishes.

    Parameters
    ----------
    deapodize:
        Divide out the apodization window to recover the un-tapered intensity.
        Stable for ``gaussian`` (never zero); for ``hann`` the division is
        clamped by *window_floor* and the edge planes are unreliable.  ``False``
        returns the windowed reconstruction — what a naive inverse FFT yields.
    add_back_smooth_bg:
        Re-add the smooth background removed by ``subtract_smooth_bg`` (only if
        it was used).  ``False`` keeps only the oscillatory modulation.
    window_floor:
        Smallest window value (relative to its peak) the deapodization divides
        by; also defines the reliable-region ``mask``.

    Returns
    -------
    HKLVolume
        Reciprocal-space reconstruction on the (possibly cropped) HKL grid of
        the transform input.  ``mask`` marks where the window exceeds
        *window_floor* (the reliably recoverable region).
    """
    if (dpdf.pad_width is None or dpdf.cropped_shape is None
            or dpdf.window_axes is None or dpdf.h_axis_c is None
            or dpdf.k_axis_c is None or dpdf.l_axis_c is None):
        raise ValueError(
            "DeltaPDF is missing the inverse metadata (pad_width / cropped_shape "
            "/ window_axes / cropped axes); recompute it with compute_delta_pdf "
            "from this build before inverting.")

    # Exact inverse of fftshift(fftn(ifftshift(·))).  The stored ΔPDF is real
    # (FT of centrosymmetric I(Q)); ifftn of it is real up to round-off.
    prep_pad = np.real(fftshift(ifftn(ifftshift(dpdf.data))))

    # Strip the symmetric zero-padding → the windowed, mean-subtracted volume.
    sl = tuple(slice(lo, lo + n)
               for (lo, _hi), n in zip(dpdf.pad_width, dpdf.cropped_shape))
    prep = prep_pad[sl] + dpdf.subtracted_mean   # restore mean → win·(I − bg)

    win = (dpdf.window_axes[0][:, None, None]
           * dpdf.window_axes[1][None, :, None]
           * dpdf.window_axes[2][None, None, :])
    reliable = win >= window_floor * float(win.max())

    if deapodize:
        recon = np.divide(prep, win, out=np.zeros_like(prep), where=reliable)
    else:
        recon = prep
        reliable = np.ones(recon.shape, dtype=bool)

    if add_back_smooth_bg and dpdf.smooth_bg is not None:
        recon = recon + dpdf.smooth_bg

    ub = (np.asarray(dpdf.ub_matrix, dtype=np.float64)
          if dpdf.ub_matrix is not None else np.eye(3, dtype=np.float64))
    return HKLVolume(
        data=recon.astype(np.float64),
        sigma=np.zeros(recon.shape, dtype=np.float64),
        mask=reliable,
        h_axis=dpdf.h_axis_c.copy(),
        k_axis=dpdf.k_axis_c.copy(),
        l_axis=dpdf.l_axis_c.copy(),
        ub_matrix=ub.copy(),
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _window_axes(
    shape: tuple[int, ...], kind: Window, sigma: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """The three separable 1-D apodization factors (kept for exact inversion)."""
    def _1d(n: int) -> NDArray[np.float64]:
        if kind == "hann":
            return np.hanning(n).astype(np.float64)
        if kind == "gaussian":
            x = np.linspace(-1, 1, n)
            return np.exp(-0.5 * (x / sigma) ** 2).astype(np.float64)
        return np.ones(n, dtype=np.float64)

    return _1d(shape[0]), _1d(shape[1]), _1d(shape[2])


def _build_window(shape: tuple[int, ...], kind: Window, sigma: float) -> NDArray:
    """Build a 3D separable apodization window."""
    wh, wk, wl = _window_axes(shape, kind, sigma)
    return wh[:, None, None] * wk[None, :, None] * wl[None, None, :]


def _next_power_of_2(n: int) -> int:
    return 1 if n == 0 else 2 ** int(np.ceil(np.log2(n)))
