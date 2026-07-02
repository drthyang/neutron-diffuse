# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tsung-han Yang

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
from scipy.fft import fftfreq, fftn, fftshift, ifftn, ifftshift, next_fast_len
from scipy.ndimage import gaussian_filter

from nebula3d.core import HKLVolume, q_magnitude_from_axes

Window = Literal["hann", "gaussian", "none"]

#: Threads for the 3D FFTs.  ``-1`` = all cores (scipy.fft / pocketfft).  The
#: transform is the dominant cost of the Q–R band round trip; multithreading it
#: is bit-for-bit identical to the single-threaded result (same pocketfft plan,
#: just split over independent 1-D transforms).
_FFT_WORKERS = -1


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
    q_band: tuple[float, float] | None = None,
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
        Pad to the next fast FFT length (5-smooth) for an efficient FFT.
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
    q_band:
        Optional spherical ``(q_min, q_max)`` shell in Å⁻¹.  Voxels outside
        the shell are set to zero before windowing/FFT.  ``None`` keeps the
        full cropped reciprocal-space range.
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

    q_mag = None
    if q_band is not None:
        qmin, qmax = q_band
        if qmax <= qmin:
            raise ValueError("q_band must satisfy q_max > q_min")
        q_mag = q_magnitude_from_axes(h_axis, k_axis, l_axis, vol.ub_matrix)
        in_band = (q_mag >= qmin) & (q_mag <= qmax)
        if not np.any(in_band):
            raise ValueError(f"q_band {q_band} selects no voxels")
        data = np.where(in_band, data, 0.0)

    # Shape of the (cropped, bg-subtracted) volume that actually enters the
    # transform — recorded so invert_delta_pdf can un-pad back to it.
    cropped_shape = data.shape

    # Apply apodization window first, then subtract mean of the windowed
    # data so that the DC component (sum) is exactly zero and the r=0
    # DeltaPDF peak is suppressed. Subtracting before windowing leaves a
    # nonzero sum = ∫(data−mean)·w dQ, producing a spurious 10^5-amplitude
    # spike at r=0 that overwhelms near-origin structure.
    window_axes = _window_axes(data.shape, apodization, gaussian_sigma)
    # Apply the separable window as three broadcast in-place multiplies —
    # never materialising the full 3-D window array (a volume-sized float64).
    data *= window_axes[0][:, None, None]
    data *= window_axes[1][None, :, None]
    data *= window_axes[2][None, None, :]

    subtracted_mean = 0.0
    if subtract_mean:
        subtracted_mean = float(data.mean())
        data -= subtracted_mean

    # Zero-pad to the next fast FFT length (5-smooth; scipy.fft.next_fast_len).
    # pocketfft transforms these just as fast as powers of two, but the pad is
    # far smaller (e.g. 360→375 instead of 360→512 — ~2.6× less memory for the
    # padded and complex arrays), which is what keeps full-resolution volumes
    # inside the browser's WASM heap.  Pad SYMMETRICALLY so the Q=0 origin (at
    # index s//2 of each axis) stays at the centre of the padded array —
    # one-sided padding would shift the origin and reintroduce the phase ramp
    # that ifftshift (below) removes.
    if zero_pad:
        padded_shape = tuple(next_fast_len(s) for s in data.shape)
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
    # The correct centred transform is fftshift(fftn(ifftshift(·))) — computed
    # in explicit steps that free each intermediate before the next allocates
    # (the complex spectrum alone is 2 padded volumes), and taking the real
    # part BEFORE fftshift (they commute: fftshift only permutes elements) so
    # the shift copies a float64 array, not a complex128 one.
    data = ifftshift(data)
    ft = fftn(data, workers=_FFT_WORKERS)
    del data  # free the padded input before the real-part copy below
    # Materialise the real part (valid for centrosymmetric I(Q)): np.real()
    # returns a VIEW that would otherwise pin the complex128 buffer (2 padded
    # volumes) for the DeltaPDF's whole lifetime.
    delta_pdf = np.ascontiguousarray(ft.real)
    del ft
    delta_pdf = fftshift(delta_pdf)

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

    if q_mag is None:
        # max|Q| over the (cropped) grid.  |Q| = ‖UB·hkl‖ is convex, so its
        # maximum over the axis-aligned hkl box is attained at a corner — and the
        # grid endpoints ARE those corners.  Evaluating 8 corners is exact and
        # avoids materialising a full (nh,nk,nl,3) meshgrid just for one scalar.
        q_max = _q_max_from_axes(h_axis, k_axis, l_axis, vol.ub_matrix)
    else:
        if q_band is not None:
            qmin, qmax_in = q_band
            retained = q_mag[(q_mag >= qmin) & (q_mag <= qmax_in)]
            q_max = float(np.max(retained))
        else:
            q_max = float(np.max(q_mag))

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
    consume: bool = False,
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
    consume:
        Release ``dpdf.data`` (replaced by an empty array) as soon as it has
        been copied for the transform.  Opt-in for memory-critical callers
        (the in-browser pipeline's consistency check) that never slice the
        ΔPDF afterwards: the padded volume is 1 volume-equivalent that would
        otherwise sit under the inverse FFT's complex transient.

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
    # Stepwise with prompt frees, real part before fftshift (they commute) —
    # same memory discipline as the forward transform in compute_delta_pdf.
    work = ifftshift(dpdf.data)
    if consume:
        dpdf.data = np.empty((0, 0, 0), dtype=np.float64)
    ft = ifftn(work, workers=_FFT_WORKERS)
    del work
    prep_pad = np.ascontiguousarray(ft.real)
    del ft
    prep_pad = fftshift(prep_pad)

    # Strip the symmetric zero-padding → the windowed, mean-subtracted volume.
    sl = tuple(slice(lo, lo + n)
               for (lo, _hi), n in zip(dpdf.pad_width, dpdf.cropped_shape))
    prep = prep_pad[sl] + dpdf.subtracted_mean   # restore mean → win·(I − bg)
    del prep_pad  # the padded inverse is no longer needed

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
    # Zero-stride broadcast view instead of a materialised zeros volume: the
    # reconstruction has no error estimate and nothing writes to it.
    zero_sigma = np.broadcast_to(np.float64(0.0), recon.shape)
    return HKLVolume(
        data=recon.astype(np.float64),
        sigma=zero_sigma,
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


def _q_max_from_axes(
    h_axis: NDArray[np.float64],
    k_axis: NDArray[np.float64],
    l_axis: NDArray[np.float64],
    ub_matrix: NDArray[np.float64],
) -> float:
    """Exact max |Q| (Å⁻¹) over the hkl box spanned by the axes, from 8 corners.

    ``|Q| = ‖UB·hkl‖`` is convex in ``hkl``; its maximum over the axis-aligned
    box ``[h_axis bounds]×[k]×[l]`` is therefore at a vertex, and the axis
    endpoints supply those vertices.  Bit-identical to ``q_magnitude().max()``
    over the regular grid, without the full meshgrid + matmul.
    """
    corners = np.array(
        [(hh, kk, ll)
         for hh in (h_axis[0], h_axis[-1])
         for kk in (k_axis[0], k_axis[-1])
         for ll in (l_axis[0], l_axis[-1])],
        dtype=np.float64,
    )
    return float(np.linalg.norm(corners @ np.asarray(ub_matrix).T, axis=1).max())
