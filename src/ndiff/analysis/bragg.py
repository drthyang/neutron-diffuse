"""Bragg peak removal (punch step) for 3D-ΔPDF preparation.

Bragg peaks sit at (near-)integer (h, k, l) positions and are orders of magnitude
stronger than the diffuse signal. They must be excised ("punched") before
Fourier transforming to the 3D-ΔPDF.

Strategy
--------
1. Enumerate the integer (h,k,l) nodes within the HKL grid extent.
2. **Data-driven detection** (``min_intensity`` set): keep only nodes that carry
   a real peak (local max above ``min_intensity`` and above the local background
   by ``min_prominence``).  This crystal has many systematic absences — punching
   every node would gouge diffuse signal at the ~3/4 of nodes that are extinct.
   Each surviving peak is re-centred on its local argmax (peaks drift off the
   exact integer by thermal contraction etc.).
3. Punch a 3D ellipsoidal hole at each detected peak.  Radii are **anisotropic**
   (Bragg peaks are several-fold broader along the coarse axis — here L) and
   optionally **scale with intensity** (bright peaks have longer tails).
4. The mask is built on **local windows** around each peak, never a full-volume
   array per peak, so it scales to thousands of peaks on a 50M-voxel volume.

Punch → backfill (``ndiff.analysis.backfill_bragg``) → ΔPDF.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ndiff.core import HKLVolume


@dataclass
class BraggRemover:
    """Detect and punch Bragg reflections in an HKLVolume.

    Parameters
    ----------
    mode:
        ``"integer"`` (default) punches at integer (h,k,l) nodes (symmetry-based).
        ``"search"`` finds *any* sharp peak as a high-tail outlier above the
        robust per-|Q|-shell diffuse level — catches off-integer satellites
        (small-domain / superlattice reflections) the integer mode misses, at the
        cost of also removing any sharp *structural* diffuse (acceptable when only
        magnetic diffuse is wanted).  ``"both"`` takes the union.
    punch_radius_hkl:
        Isotropic punch half-radius in fractional HKL units.  Used only when
        ``punch_radii`` is not given.
    punch_radii:
        ``(rh, rk, rl)`` per-axis half-radii (Å⁻¹-free, fractional HKL).  Bragg
        peaks are anisotropic — broad along a coarse axis — so prefer this over
        the isotropic radius (e.g. ``(0.12, 0.12, 0.45)`` here, L being broad).
    min_intensity:
        Detection threshold.  ``None`` (default) punches **every** integer node
        (legacy behaviour).  When set, only nodes whose local peak intensity
        exceeds this value (and the local background by ``min_prominence``) are
        punched — the data-driven path that skips systematic absences.
    min_prominence:
        A detected peak must exceed its local-window median by at least this.
    detect_window_hkl:
        Half-width (HKL) of the window used to locate/centre a peak and estimate
        its local background.
    intensity_scale:
        If True, multiply the punch radii by ``clip((I/intensity_ref)**(1/3), 1,
        max_radius_scale)`` so bright peaks (longer tails) get larger holes.
    intensity_ref:
        Reference intensity for the scaling.  ``None`` → the median detected-peak
        intensity (computed once).
    max_radius_scale:
        Upper clamp on the intensity radius multiplier.
    margin:
        Extra half-width (HKL) added to every punch radius — a guard band so the
        peak's faint wings are removed too.
    subtract_profile:
        Reserved (profile-subtraction path not implemented in this pass).
    """

    mode: str = "integer"
    punch_radius_hkl: float = 0.3
    punch_radii: Optional[tuple[float, float, float]] = None
    min_intensity: Optional[float] = None
    min_prominence: float = 1.0
    detect_window_hkl: float = 0.2
    intensity_scale: bool = False
    intensity_ref: Optional[float] = None
    max_radius_scale: float = 3.0
    margin: float = 0.0
    # --- search mode (|Q|-shell outlier detection) ---
    search_q_step: float = 0.05
    search_n_mad: float = 8.0
    search_min_intensity: float = 2.0
    subtract_profile: bool = False

    def _radii(self) -> tuple[float, float, float]:
        if self.punch_radii is not None:
            return tuple(float(r) for r in self.punch_radii)  # type: ignore[return-value]
        r = float(self.punch_radius_hkl)
        return r, r, r

    @staticmethod
    def _steps(vol: HKLVolume) -> tuple[float, float, float]:
        return (
            float(vol.h_axis[1] - vol.h_axis[0]),
            float(vol.k_axis[1] - vol.k_axis[0]),
            float(vol.l_axis[1] - vol.l_axis[0]),
        )

    def enumerate_bragg(self, vol: HKLVolume) -> list[tuple[int, int, int]]:
        """Integer (h,k,l) nodes within the grid extent."""
        hs = range(int(np.ceil(vol.h_axis.min())), int(np.floor(vol.h_axis.max())) + 1)
        ks = range(int(np.ceil(vol.k_axis.min())), int(np.floor(vol.k_axis.max())) + 1)
        ls = range(int(np.ceil(vol.l_axis.min())), int(np.floor(vol.l_axis.max())) + 1)
        return [(h, k, l) for h in hs for k in ks for l in ls]

    def detect_peaks(self, vol: HKLVolume) -> list[tuple[int, int, int, float]]:
        """Return ``(ih, ik, il, intensity)`` peak-centre voxels to punch.

        Dispatches on ``mode``:

        - ``"integer"`` — peaks at integer (h,k,l) nodes (symmetry-based; skips
          systematic absences when ``min_intensity`` is set).
        - ``"search"`` — any sharp peak, found as a high-tail outlier above the
          robust per-|Q|-shell diffuse level.  Catches off-integer satellites
          (e.g. small-domain / superlattice reflections) the integer mode misses.
        - ``"both"`` — the union of the two (integer centres take precedence).
        """
        if self.mode == "integer":
            return self._detect_integer(vol)
        if self.mode == "search":
            return self._detect_search(vol)
        if self.mode == "both":
            # Sequential: punch the integer Bragg first, then search on the
            # residual.  With the strong integer peaks already masked out, the
            # per-|Q|-shell statistics are no longer inflated by them, so the
            # off-integer satellites stand out as clean outliers.
            integer = self._detect_integer(vol)
            keep = self._punch_centers(vol, np.ones(vol.shape, dtype=bool), integer)
            residual = dataclasses.replace(vol, mask=vol.mask & keep)
            return integer + self._detect_search(residual)
        raise ValueError(f"Unknown mode: {self.mode!r}")

    def _detect_integer(self, vol: HKLVolume) -> list[tuple[int, int, int, float]]:
        """Peaks at integer (h,k,l) nodes.

        With ``min_intensity`` unset every node is returned at its nearest voxel
        (legacy punch-all).  When set, each node is examined in a local window:
        the peak is re-centred on the window argmax and kept only if it clears the
        intensity and prominence thresholds — extinct nodes are dropped.
        """
        dh, dk, dl = self._steps(vol)
        nh, nk, nl = vol.shape
        data, valid = vol.data, (vol.mask & np.isfinite(vol.data))

        def nearest(axis: NDArray, val: int) -> int:
            return int(np.argmin(np.abs(axis - val)))

        out: list[tuple[int, int, int, float]] = []
        if self.min_intensity is None:
            for h, k, l in self.enumerate_bragg(vol):
                out.append((nearest(vol.h_axis, h), nearest(vol.k_axis, k),
                            nearest(vol.l_axis, l), float("nan")))
            return out

        wph = max(1, int(round(self.detect_window_hkl / abs(dh))))
        wpk = max(1, int(round(self.detect_window_hkl / abs(dk))))
        wpl = max(1, int(round(self.detect_window_hkl / abs(dl))))
        for h, k, l in self.enumerate_bragg(vol):
            ih, ik, il = (nearest(vol.h_axis, h), nearest(vol.k_axis, k),
                          nearest(vol.l_axis, l))
            hs, he = max(0, ih - wph), min(nh, ih + wph + 1)
            ks, ke = max(0, ik - wpk), min(nk, ik + wpk + 1)
            ls, le = max(0, il - wpl), min(nl, il + wpl + 1)
            win = data[hs:he, ks:ke, ls:le]
            wval = valid[hs:he, ks:ke, ls:le]
            if wval.sum() < 3:
                continue
            wv = np.where(wval, win, np.nan)
            peak = float(np.nanmax(wv))
            if not np.isfinite(peak) or peak < self.min_intensity:
                continue
            if peak - float(np.nanmedian(wv)) < self.min_prominence:
                continue
            # re-centre on the true peak (thermal/lattice drift off the integer)
            off = np.unravel_index(int(np.nanargmax(wv)), wv.shape)
            out.append((hs + off[0], ks + off[1], ls + off[2], peak))
        return out

    def _detect_search(self, vol: HKLVolume) -> list[tuple[int, int, int, float]]:
        """Peaks found as sharp |Q|-shell outliers (mode-agnostic to hkl).

        Reuses the ring-removal insight: at a given |Q| the diffuse is the bulk
        and any Bragg / satellite reflection is a sharp high-tail outlier.  For
        each |Q| shell (width ``search_q_step``) the robust level (median) and
        scale (MAD) are measured over the valid voxels; a voxel is a peak
        candidate when it exceeds ``median + search_n_mad · 1.4826·MAD`` (and an
        absolute floor).  Candidates are grouped into connected components and
        each component's brightest voxel is returned as a peak centre — so the
        shared ellipsoid punch removes the whole peak, not just its hottest voxel.
        """
        from scipy import ndimage

        q = vol.q_magnitude()
        valid = vol.mask & np.isfinite(vol.data)
        if not valid.any():
            return []
        qs = float(self.search_q_step)
        qv = q[valid]
        edges = np.arange(qv.min(), qv.max() + qs, qs)
        nb = max(len(edges) - 1, 1)
        bin_idx = np.clip(np.digitize(q, edges) - 1, 0, nb - 1)

        # Per-shell robust threshold (median + n·MAD), computed once over the
        # sorted valid voxels so it is O(N log N), not O(N · n_bins).
        flat_b = bin_idx[valid]
        flat_I = vol.data[valid]
        order = np.argsort(flat_b, kind="stable")
        sb, sI = flat_b[order], flat_I[order]
        bounds = np.searchsorted(sb, np.arange(nb + 1))
        thr = np.full(nb, np.inf)
        for b in range(nb):
            seg = sI[bounds[b]:bounds[b + 1]]
            if seg.size < 20:
                continue
            med = float(np.median(seg))
            mad = float(np.median(np.abs(seg - med)))
            scale = 1.4826 * mad if mad > 0 else (float(np.std(seg)) or 1.0)
            thr[b] = med + self.search_n_mad * scale
        thr = np.maximum(thr, self.search_min_intensity)

        cand = valid & (vol.data > thr[bin_idx])
        if not cand.any():
            return []
        # One centre per peak *summit*: a candidate voxel that is a local maximum
        # (≥ its 3×3×3 neighbours).  This catches every peak even when several are
        # joined into one above-threshold blob (taking a single max per connected
        # component would miss all but the brightest — e.g. satellites at the
        # measured-volume edge that touch a residual arc).
        scored = np.where(valid, vol.data, -np.inf)
        local_max = ndimage.maximum_filter(scored, size=3, mode="nearest")
        peaks = np.argwhere(cand & (scored >= local_max))
        return [(int(ih), int(ik), int(il), float(vol.data[ih, ik, il]))
                for ih, ik, il in peaks]

    def _scale_factor(self, peak: float, ref: float) -> float:
        if not self.intensity_scale or not np.isfinite(peak) or ref <= 0:
            return 1.0
        return float(np.clip((peak / ref) ** (1.0 / 3.0), 1.0, self.max_radius_scale))

    def build_mask(self, vol: HKLVolume) -> NDArray[np.bool_]:
        """Return a keep-mask (True = valid, False = punched Bragg voxel).

        Built on local windows around each detected peak, so the cost is
        ``n_peaks × small_window`` rather than ``n_peaks × whole_volume``.
        """
        return self._punch_centers(
            vol, np.ones(vol.shape, dtype=bool), self.detect_peaks(vol))

    def _punch_centers(
        self,
        vol: HKLVolume,
        keep: NDArray[np.bool_],
        peaks: list[tuple[int, int, int, float]],
    ) -> NDArray[np.bool_]:
        """Punch an anisotropic, intensity-scaled ellipsoid at each peak centre,
        in place on *keep* (local windows only)."""
        dh, dk, dl = self._steps(vol)
        nh, nk, nl = vol.shape
        rh0, rk0, rl0 = self._radii()

        ref = self.intensity_ref
        if self.intensity_scale and ref is None:
            ints = np.array([p[3] for p in peaks if np.isfinite(p[3])])
            ref = float(np.median(ints)) if ints.size else 1.0

        for ih, ik, il, peak in peaks:
            s = self._scale_factor(peak, ref if ref is not None else 1.0)
            rh, rk, rl = rh0 * s + self.margin, rk0 * s + self.margin, rl0 * s + self.margin
            ch, ck, cl = vol.h_axis[ih], vol.k_axis[ik], vol.l_axis[il]
            wh, wk, wl = (int(np.ceil(rh / abs(dh))), int(np.ceil(rk / abs(dk))),
                          int(np.ceil(rl / abs(dl))))
            hs, he = max(0, ih - wh), min(nh, ih + wh + 1)
            ks, ke = max(0, ik - wk), min(nk, ik + wk + 1)
            ls, le = max(0, il - wl), min(nl, il + wl + 1)
            HH, KK, LL = np.meshgrid(vol.h_axis[hs:he], vol.k_axis[ks:ke],
                                     vol.l_axis[ls:le], indexing="ij")
            ell = ((HH - ch) / rh) ** 2 + ((KK - ck) / rk) ** 2 + ((LL - cl) / rl) ** 2
            keep[hs:he, ks:ke, ls:le] &= ell > 1.0
        return keep

    def apply(self, vol: HKLVolume) -> HKLVolume:
        """Return a new volume with detected Bragg peaks masked out."""
        keep = self.build_mask(vol)
        return dataclasses.replace(vol, mask=vol.mask & keep)


def bragg_mask(
    vol: HKLVolume,
    punch_radius_hkl: float = 0.3,
    punch_radii: Optional[tuple[float, float, float]] = None,
    min_intensity: Optional[float] = None,
    intensity_scale: bool = False,
    margin: float = 0.0,
) -> NDArray[np.bool_]:
    """Convenience wrapper.  Returns a keep-mask (True = valid)."""
    return BraggRemover(
        punch_radius_hkl=punch_radius_hkl,
        punch_radii=punch_radii,
        min_intensity=min_intensity,
        intensity_scale=intensity_scale,
        margin=margin,
    ).build_mask(vol)
