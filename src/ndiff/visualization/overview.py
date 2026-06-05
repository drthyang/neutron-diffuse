"""Multi-panel diagnostic overview of an HKLVolume."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ndiff.core import HKLVolume

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_overview(
    vol: HKLVolume,
    title: str | None = None,
    log_scale: bool = False,
    cmap: str = "hot",
    percentile: float = 99.5,
    vmin: float | None = None,
    vmax: float | None = None,
    mark_q: list[float] | None = None,
) -> Figure:
    """Four-panel diagnostic figure: three principal slices + radial profile.

    Layout
    ------
    ┌─────────────┬─────────────┐
    │  K-L (H=0)  │  H-L (K=0) │
    ├─────────────┼─────────────┤
    │  H-K (L=0)  │  |Q| profile│
    └─────────────┴─────────────┘

    Parameters
    ----------
    vol : HKLVolume
    title : str, optional
        Figure suptitle; defaults to ``vol.instrument``.
    log_scale : bool
        Apply log₁₀ to slice intensities.
    cmap : str
        Matplotlib colormap for the slices.
    percentile : float
        Upper percentile for colour-scale clipping (default 99.5).
    vmin, vmax : float, optional
        Shared colour limits applied to all three slice panels, overriding the
        per-panel percentile clip (on the log₁₀ scale when ``log_scale``).
    mark_q : list of float, optional
        Mark these |Q| positions with dashed lines on the radial profile.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    from ndiff.visualization.profiles import plot_radial_profile
    from ndiff.visualization.slices import plot_slice

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    slice_cfg = [
        ("kl", 0.0, axes[0, 0]),
        ("hl", 0.0, axes[0, 1]),
        ("hk", 0.0, axes[1, 0]),
    ]
    for plane, val, ax in slice_cfg:
        plot_slice(
            vol, plane=plane, value=val, ax=ax,
            cmap=cmap, percentile=percentile, log_scale=log_scale,
            vmin=vmin, vmax=vmax,
        )

    plot_radial_profile(vol, ax=axes[1, 1], mark_q=mark_q)

    main_title = title or vol.instrument or "HKLVolume diagnostic overview"
    fig.suptitle(main_title, fontsize=12)
    fig.tight_layout()
    return fig
