"""Radial and azimuthal intensity profile plots for an HKLVolume."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from ndiff.core import HKLVolume

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_radial_profile(
    vol: HKLVolume,
    n_bins: int = 500,
    stat: str = "mean",
    q_range: Optional[tuple[float, float]] = None,
    mark_q: Optional[list[float]] = None,
    ax: Optional["Axes"] = None,
    title: Optional[str] = None,
    **kwargs: object,
) -> "Axes":
    """Plot the 1D radial intensity profile |Q| vs I.

    Wraps :func:`ndiff.preprocessing.powder_rings.radial_profile` for display.

    Parameters
    ----------
    vol : HKLVolume
    n_bins : int
        Number of |Q| bins.
    stat : {'mean', 'median'}
        Per-bin statistic.
    q_range : (q_lo, q_hi), optional
        Restrict the x-axis to this |Q| range (Å⁻¹).
    mark_q : list of float, optional
        Draw vertical dashed lines at these |Q| positions (e.g. ring centres).
    ax : Axes, optional
        Existing axes; a new figure is created if *None*.
    title : str, optional
    **kwargs
        Passed to ``ax.plot`` (e.g. ``color``, ``lw``).

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from ndiff.preprocessing.powder_rings import radial_profile

    q_centers, profile, counts = radial_profile(vol, n_bins=n_bins, stat=stat)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    valid = np.isfinite(profile) & (counts > 0)
    plot_kw: dict[str, object] = {"lw": 1.0, "color": "C0"}
    plot_kw.update(kwargs)  # type: ignore[arg-type]
    ax.plot(q_centers[valid], profile[valid], **plot_kw)

    if mark_q:
        for q in mark_q:
            ax.axvline(q, color="r", lw=0.8, ls="--", alpha=0.7)

    ax.set_xlabel("|Q| (Å⁻¹)")
    ax.set_ylabel(f"Intensity ({stat}, arb.)")
    if q_range is not None:
        ax.set_xlim(q_range)
    if title:
        ax.set_title(title)

    return ax


def plot_azimuthal_map(
    vol: HKLVolume,
    q_center: float,
    q_width: float = 0.1,
    n_phi_bins: int = 72,
    ax: Optional["Axes"] = None,
    title: Optional[str] = None,
    **kwargs: object,
) -> "Axes":
    """Plot intensity vs azimuthal angle φ for a thin |Q| shell.

    Useful for visualising the azimuthal texture T(φ) of a powder ring.

    Parameters
    ----------
    vol : HKLVolume
    q_center : float
        Centre of the |Q| shell (Å⁻¹).
    q_width : float
        Half-width of the shell (Å⁻¹).  Voxels with
        ||Q| − q_center| < q_width are included.
    n_phi_bins : int
        Number of azimuthal bins over [−π, π).
    ax : Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    q_mag = vol.q_magnitude()
    H, K, L = vol.hkl_grid()
    q_cart = np.stack([H, K, L], axis=-1) @ vol.ub_matrix.T
    phi = np.arctan2(q_cart[..., 1], q_cart[..., 0])

    shell = vol.mask & (np.abs(q_mag - q_center) < q_width)
    if not shell.any():
        raise ValueError(
            f"No valid voxels in shell |Q| = {q_center:.3f} ± {q_width:.3f} Å⁻¹"
        )

    phi_flat = phi[shell]
    I_flat = vol.data[shell]

    edges = np.linspace(-np.pi, np.pi, n_phi_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(phi_flat, edges) - 1, 0, n_phi_bins - 1)

    az_profile = np.full(n_phi_bins, np.nan)
    for b in range(n_phi_bins):
        mask_b = idx == b
        if mask_b.sum() >= 3:
            az_profile[b] = float(np.mean(I_flat[mask_b]))

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))

    plot_kw: dict[str, object] = {"lw": 1.0, "color": "C1"}
    plot_kw.update(kwargs)  # type: ignore[arg-type]
    valid = np.isfinite(az_profile)
    ax.plot(np.degrees(centers[valid]), az_profile[valid], **plot_kw)

    ax.set_xlabel("φ (°)")
    ax.set_ylabel("Intensity (arb.)")
    ax.set_xlim(-180, 180)
    ax.set_xticks(range(-180, 181, 45))
    default_title = f"|Q| = {q_center:.3f} ± {q_width:.3f} Å⁻¹"
    ax.set_title(title or default_title)

    return ax
