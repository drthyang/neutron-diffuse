// Bragg profile · width diagnostics — a single-screen QC dashboard for the punch
// peak widths of one dataset (see design_handoff_bragg_profile_Bragg/).  Reads the
// BraggProfile and presents a stat strip, a width-vs-|Q| scatter, per-axis width
// histograms, a sortable peak table, and a selected-peak detail with the real
// intensity around the peak in three orthogonal slices + fit ellipses.

import { useMemo, useState } from "react";

import { useBraggProfile, useDatasets, useMeta } from "../api/hooks";
import type { BraggPeakWidth } from "../api/types";
import { BraggPeakSlice, type Ellipse } from "../components/BraggPeakSlice";
import { ColormapBar, EmptyState, IconAlert, IconLattice, Slider, Switch } from "../components/ui";
import { COLORMAPS, SEQUENTIAL_NAMES } from "../colormaps/luts";
import { useDatasetStore, useInitializeDataset } from "../state/datasetStore";
import { useViewerStore } from "../state/viewerStore";

// Axis colour-coding shared across the whole page (matches Configure / Reciprocal).
// Labels are profile-driven: the spherical punch reports widths along (ρ, θ, φ),
// the legacy frames along the reciprocal axes (a*, b*, c*).
type Axis = { i: 0 | 1 | 2; label: string; color: string };
const AXIS_COLORS = ["#f1a73a", "#74a8ff", "#34c98e"] as const;
const DEFAULT_AXIS_LABELS = ["a*", "b*", "c*"] as const;
function buildAxes(labels?: string[]): Axis[] {
  return [0, 1, 2].map((i) => ({
    i: i as 0 | 1 | 2,
    label: labels?.[i] ?? DEFAULT_AXIS_LABELS[i],
    color: AXIS_COLORS[i],
  }));
}
const CMAPS = ["inferno", "magma", "viridis", "plasma", "turbo", "cividis"].filter(
  (c) => c in COLORMAPS || SEQUENTIAL_NAMES.includes(c),
);

type SortKey = "q" | "width" | "intensity";
const MAX_ROWS = 400; // cap rendered table rows; scatter + stats cover the full set

// ---- width accessors (measured = pad-free; fall back to fitted for legacy) ----
const fittedQ = (p: BraggPeakWidth, a: number) => p.width_q[a];
const measuredQ = (p: BraggPeakWidth, a: number): number | null => {
  const m = p.measured_width_q;
  if (m === undefined) return p.width_q[a];
  return m === null ? null : m[a];
};
// For display (scatter / table / medians) prefer the pad-free measured width, but
// fall back to the fitted width when measured is absent (older profiles store only
// the fitted widths) so the dataset still plots.
const displayQ = (p: BraggPeakWidth, a: number): number | null => {
  const m = measuredQ(p, a);
  if (m != null && Number.isFinite(m)) return m;
  return Number.isFinite(p.width_q[a]) ? p.width_q[a] : null;
};
const fittedHkl = (p: BraggPeakWidth, a: number) => p.width_hkl[a];
const measuredHkl = (p: BraggPeakWidth, a: number): number | null => {
  const m = p.measured_width_hkl;
  if (m === undefined || m === null) return null;
  return m[a];
};
const resLimited = (p: BraggPeakWidth, a: number) =>
  Boolean(p.resolution_limited?.[a]);
const measurable = (p: BraggPeakWidth) => p.measured_width_q !== null;

function median(xs: (number | null | undefined)[]): number | null {
  const v = xs.filter((x): x is number => x != null && Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return null;
  const m = Math.floor(v.length / 2);
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

// Approximate the half-voxel pad floor (not in the payload) from the boundary
// between resolution-limited and clean measured widths.  `null` when underivable.
function deriveFloor(peaks: BraggPeakWidth[], axis: number, hkl: boolean): number | null {
  let flaggedMax = -Infinity;
  let cleanMin = Infinity;
  for (const p of peaks) {
    const w = hkl ? measuredHkl(p, axis) : measuredQ(p, axis);
    if (w == null || !Number.isFinite(w)) continue;
    if (resLimited(p, axis)) flaggedMax = Math.max(flaggedMax, w);
    else cleanMin = Math.min(cleanMin, w);
  }
  if (flaggedMax > -Infinity && cleanMin < Infinity) return (flaggedMax + cleanMin) / 2;
  if (flaggedMax > -Infinity) return flaggedMax;
  return null;
}

function fmt(v: number | null, d = 3): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return v.toFixed(d);
}
function hklStr(c: [number, number, number]): string {
  const r = (x: number) => (Math.abs(x - Math.round(x)) < 0.05 ? String(Math.round(x)) : x.toFixed(2));
  return `${r(c[0])} ${r(c[1])} ${r(c[2])}`;
}

function ticks(min: number, max: number, n = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return [min, max];
  return Array.from({ length: n }, (_, i) => min + (i * (max - min)) / (n - 1));
}

// ---------------------------------------------------------------------------
// Per-axis Gaussian fit of the width histogram (coarse grid → local refine).
// ---------------------------------------------------------------------------
type FitStats = { mean: number; sigma: number; fwhm: number; amplitude: number; error: number };

function histCounts(values: number[], domain: [number, number], bins: number) {
  const xs = values.filter(Number.isFinite);
  const bw = (domain[1] - domain[0]) / bins;
  const counts = Array.from({ length: bins }, () => 0);
  for (const v of xs) {
    if (v < domain[0] || v > domain[1]) continue;
    const raw = Math.floor(((v - domain[0]) / (domain[1] - domain[0])) * bins);
    counts[Math.max(0, Math.min(bins - 1, raw))] += 1;
  }
  const centers = counts.map((_c, i) => domain[0] + (i + 0.5) * bw);
  return { counts, centers, bw };
}

function fitGaussian(values: number[], domain: [number, number], bins: number): FitStats | null {
  const xs = values.filter(Number.isFinite);
  if (xs.length < 3) return null;
  const { counts, centers, bw } = histCounts(xs, domain, bins);
  const maxCount = Math.max(...counts);
  const total = counts.reduce((a, c) => a + c, 0);
  if (maxCount <= 0 || total <= 0) return null;
  const wMean = centers.reduce((a, c, i) => a + c * counts[i], 0) / total;
  const wSigma = Math.sqrt(Math.max(centers.reduce((a, c, i) => a + counts[i] * (c - wMean) ** 2, 0) / total, 0));
  const peakMean = centers[counts.indexOf(maxCount)];
  const span = domain[1] - domain[0];
  const minSigma = Math.max(bw * 0.35, span * 0.002);
  const maxSigma = Math.max(minSigma * 1.1, span * 0.5);
  const score = (mean: number, sigma: number): FitStats | null => {
    if (!Number.isFinite(mean) || !Number.isFinite(sigma) || sigma <= 0) return null;
    if (mean < domain[0] || mean > domain[1]) return null;
    const g = centers.map((c) => Math.exp(-0.5 * ((c - mean) / sigma) ** 2));
    const gg = g.reduce((a, v) => a + v * v, 0);
    if (gg <= 0) return null;
    const amp = Math.max(0, g.reduce((a, v, i) => a + counts[i] * v, 0) / gg);
    const err = g.reduce((a, v, i) => a + (counts[i] - amp * v) ** 2, 0);
    return { mean, sigma, fwhm: 2.354820045 * sigma, amplitude: amp, error: err };
  };
  const means = [wMean, peakMean, peakMean - bw, peakMean + bw, peakMean - 2 * bw, peakMean + 2 * bw];
  const sigmas = [minSigma, bw * 0.5, bw, bw * 1.5, bw * 2.5, Math.max(minSigma, wSigma * 0.5), Math.max(minSigma, wSigma)]
    .map((s) => Math.min(Math.max(s, minSigma), maxSigma));
  let best: FitStats | null = null;
  for (const m of means) for (const s of sigmas) {
    const cand = score(m, s);
    if (cand && (!best || cand.error < best.error)) best = cand;
  }
  if (!best) return null;
  let stepM = Math.max(bw, best.sigma), stepS = Math.max(bw * 0.5, best.sigma * 0.5);
  for (let it = 0; it < 24; it++) {
    let improved = false;
    const cands: [number, number][] = [
      [best.mean - stepM, best.sigma], [best.mean + stepM, best.sigma],
      [best.mean, best.sigma - stepS], [best.mean, best.sigma + stepS],
      [best.mean - stepM, best.sigma - stepS], [best.mean - stepM, best.sigma + stepS],
      [best.mean + stepM, best.sigma - stepS], [best.mean + stepM, best.sigma + stepS],
    ];
    for (const [m, sr] of cands) {
      const cand = score(m, Math.min(Math.max(sr, minSigma), maxSigma));
      if (cand && cand.error < best.error) { best = cand; improved = true; }
    }
    if (!improved) { stepM *= 0.55; stepS *= 0.55; }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Accurate fitted ellipse on a slice plane from the principal-axis ellipsoid.
// The slice passes through the peak centre, so the cross-section is the 2×2
// block of the precision matrix M = Σ (1/rᵢ²) vᵢ vᵢᵀ restricted to the in-plane
// axes (ix, iy); its eigen-decomposition gives the (possibly tilted) ellipse.
// Returns r.l.u. semi-axes (rx along `angle`, ry perpendicular) + SVG-space angle.
// ---------------------------------------------------------------------------
function fittedEllipse(peak: BraggPeakWidth, ix: number, iy: number): Ellipse {
  const dirs = peak.principal_directions_hkl;
  const rs = peak.principal_width_hkl;
  if (dirs && dirs.length === 3 && rs && rs.length === 3) {
    let a = 0, b = 0, c = 0;
    for (let i = 0; i < 3; i++) {
      const w = Math.abs(rs[i]);
      if (!(w > 0) || !dirs[i]) continue;
      const inv = 1 / (w * w);
      const vx = dirs[i][ix], vy = dirs[i][iy];
      a += inv * vx * vx; b += inv * vx * vy; c += inv * vy * vy;
    }
    const half = (a + c) / 2;
    const d = Math.sqrt(((a - c) / 2) ** 2 + b * b);
    const l1 = half + d, l2 = half - d;
    if (l1 > 0 && l2 > 0) {
      // SVG y points down, so negate the y-component of the eigenvector.
      const angle = (Math.atan2(-(l1 - a), b) * 180) / Math.PI;
      return { rx: 1 / Math.sqrt(l1), ry: 1 / Math.sqrt(l2), angle };
    }
  }
  return { rx: Math.abs(fittedHkl(peak, ix)), ry: Math.abs(fittedHkl(peak, iy)) };
}

// ---------------------------------------------------------------------------
// Scatter — width vs |Q|, three axis series
// ---------------------------------------------------------------------------
function Scatter({
  peaks,
  axes,
  selected,
  onSelect,
  padFloorQ,
}: {
  peaks: BraggPeakWidth[];
  axes: Axis[];
  selected: number;
  onSelect: (i: number) => void;
  padFloorQ: number | null;
}) {
  const qs = peaks.map((p) => p.q_abs).filter(Number.isFinite);
  const ws: number[] = [];
  peaks.forEach((p) => axes.forEach((a) => { const w = displayQ(p, a.i); if (w != null) ws.push(w); }));
  const qMin = Math.min(...qs, 0), qMax = Math.max(...qs, 1);
  const wMin = Math.max(0, Math.min(...ws) * 0.92), wMax = Math.max(...ws) * 1.05 || 1;
  const X = (q: number) => ((q - qMin) / (qMax - qMin || 1)) * 100;
  const Y = (w: number) => (1 - (w - wMin) / (wMax - wMin || 1)) * 100;

  // Large profiles (thousands of peaks) would mint tens of thousands of DOM dots;
  // sample evenly down to a sane count, always keeping the selected peak.
  const MAX_DOTS = 1200;
  const stride = Math.max(1, Math.ceil(peaks.length / MAX_DOTS));
  const sample = peaks.map((_, i) => i).filter((i) => i % stride === 0 || i === selected);

  return (
    <div className="bragg-panel bragg-scatter">
      <div className="bragg-panel-head">
        <span className="bragg-eyebrow">Resolution function · width vs |Q|</span>
        <div className="bragg-legend">
          {axes.map((a) => (
            <span key={a.i} className="bragg-leg"><i style={{ background: a.color }} />{a.label}</span>
          ))}
          <span className="bragg-leg"><i className="bragg-leg-ring" />res-limited</span>
        </div>
      </div>
      <div className="bragg-plot">
        <div className="bragg-plot-grid" />
        {ticks(wMin, wMax).map((t, i) => (
          <span key={`y${i}`} className="bragg-ytick" style={{ top: `${Y(t)}%` }}>{t.toFixed(3)}</span>
        ))}
        {ticks(qMin, qMax).map((t, i) => (
          <span key={`x${i}`} className="bragg-xtick" style={{ left: `${X(t)}%` }}>{t.toFixed(1)}</span>
        ))}
        {padFloorQ != null && (
          <div className="bragg-padline" style={{ top: `${Y(padFloorQ)}%` }}>
            <span>half-voxel pad {padFloorQ.toFixed(3)}</span>
          </div>
        )}
        {sample.map((pi) =>
          axes.map((a) => {
            const p = peaks[pi];
            const w = displayQ(p, a.i);
            if (w == null || !Number.isFinite(p.q_abs)) return null;
            const flagged = resLimited(p, a.i);
            return (
              <button
                key={`${pi}-${a.i}`}
                type="button"
                className={`bragg-dot${flagged ? " flagged" : ""}${pi === selected ? " sel" : ""}`}
                style={{
                  left: `${X(p.q_abs)}%`,
                  top: `${Y(w)}%`,
                  // Semi-transparent fill so dense, overlapping points read as
                  // density (and the grid shows through) instead of a solid blob.
                  background: `${a.color}80`,
                  boxShadow: flagged ? `0 0 0 2px #0d1014, 0 0 0 4px ${a.color}` : undefined,
                }}
                title={`${hklStr(p.center_hkl)} · ${a.label} ${w.toFixed(3)} Å⁻¹`}
                onClick={() => onSelect(pi)}
              />
            );
          }),
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Histograms — per-axis width distribution
// ---------------------------------------------------------------------------
function Histograms({ peaks, axes, showMeasured }: { peaks: BraggPeakWidth[]; axes: Axis[]; showMeasured: boolean }) {
  const all: number[] = [];
  peaks.forEach((p) => axes.forEach((a) => { const w = displayQ(p, a.i); if (w != null) all.push(w); }));
  all.sort((x, y) => x - y);
  const q = (f: number) =>
    all.length ? all[Math.max(0, Math.min(all.length - 1, Math.round(f * (all.length - 1))))] : 0;

  // Per-axis values + a first Gaussian fit over a robust data range, used only to
  // locate where each bell sits so we can widen the displayed range to show the
  // *full* curve (both wings) — not just the right half where data happens to be.
  const perAxisVals = axes.map((a) =>
    peaks.map((p) => displayQ(p, a.i)).filter((v): v is number => v != null),
  );
  const fitDomain: [number, number] = [Math.max(0, q(0.005)), Math.max(q(0.98), q(0.005) + 1e-6)];
  const preFits = perAxisVals.map((v) => fitGaussian(v, fitDomain, 30));
  let lo = q(0.01);
  let hi = q(0.9);
  for (const f of preFits) {
    if (!f) continue;
    lo = Math.min(lo, f.mean - 4 * f.sigma);
    hi = Math.max(hi, f.mean + 4 * f.sigma);
  }
  lo = Math.max(0, lo);
  hi = Math.max(hi, lo + 1e-6);
  const domain: [number, number] = [lo, hi];
  const N = 40;
  return (
    <div className="bragg-panel bragg-hist">
      <div className="bragg-panel-head">
        <span className="bragg-eyebrow">Width distribution / axis</span>
        <div className="bragg-legend">
          <span className="bragg-leg"><i className="bragg-leg-curve" />Gaussian fit</span>
          {showMeasured && <span className="bragg-leg"><i className="bragg-tick-dash" />measured</span>}
        </div>
      </div>
      {axes.map((a, ai) => {
        const values = perAxisVals[ai];
        const { counts } = histCounts(values, domain, N);
        const fit = fitGaussian(values, domain, N);
        const measMed = median(peaks.map((p) => measuredQ(p, a.i)));
        const yMax = Math.max(1, ...counts, fit?.amplitude ?? 0) * 1.08;
        const pos = (v: number | null) => (v == null ? null : ((v - lo) / (hi - lo)) * 100);
        const curve = fit
          ? Array.from({ length: 64 }, (_, i) => {
              const x = lo + (i / 63) * (hi - lo);
              const yv = fit.amplitude * Math.exp(-0.5 * ((x - fit.mean) / fit.sigma) ** 2);
              return `${(i / 63) * 100},${Math.max(0, 100 - (yv / yMax) * 100)}`;
            }).join(" ")
          : "";
        return (
          <div key={a.i} className="bragg-hist-row">
            <div className="bragg-hist-label">
              <span style={{ color: a.color }}>{a.label}</span>
              <span className="bragg-hist-med">
                {fit ? `μ ${fmt(fit.mean)} · σ ${fmt(fit.sigma)} · FWHM ${fmt(fit.fwhm)} Å⁻¹` : "no Gaussian fit"}
              </span>
            </div>
            <div className="bragg-bars">
              {counts.map((c, i) => (
                <span key={i} className="bragg-bar" style={{ height: `${(c / yMax) * 100}%`, background: c ? `${a.color}cc` : `${a.color}22` }} />
              ))}
              {fit && (
                <svg className="bragg-hist-curve" viewBox="0 0 100 100" preserveAspectRatio="none">
                  <polyline points={curve} fill="none" stroke={a.color} strokeWidth={1.4} vectorEffect="non-scaling-stroke" />
                </svg>
              )}
              {pos(fit?.mean ?? null) != null && <i className="bragg-bar-fit" style={{ left: `${pos(fit!.mean)}%` }} />}
              {showMeasured && pos(measMed) != null && <i className="bragg-bar-meas" style={{ left: `${pos(measMed)}%` }} />}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Peak table
// ---------------------------------------------------------------------------
function PeakTable({
  peaks,
  axes,
  order,
  selected,
  sortKey,
  onSort,
  onSelect,
}: {
  peaks: BraggPeakWidth[];
  axes: Axis[];
  order: number[];
  selected: number;
  sortKey: SortKey;
  onSort: (k: SortKey) => void;
  onSelect: (i: number) => void;
}) {
  return (
    <div className="bragg-panel bragg-table">
      <div className="bragg-panel-head">
        <span className="bragg-eyebrow">Peak table · {peaks.length} peaks</span>
        <div className="bragg-sort">
          {(["q", "width", "intensity"] as SortKey[]).map((k) => (
            <button key={k} type="button" className={sortKey === k ? "on" : ""} onClick={() => onSort(k)}>
              {k === "q" ? "|Q|" : k === "width" ? "width" : "I"}
            </button>
          ))}
        </div>
      </div>
      <div className="bragg-thead">
        <span>hkl</span><span>|Q|</span><span>I</span>
        {axes.map((a) => <span key={a.i} style={{ color: a.color }}>w {a.label}</span>)}
        <span>fit</span>
      </div>
      <div className="bragg-tbody">
        {order.slice(0, MAX_ROWS).map((pi) => {
          const p = peaks[pi];
          const anyFlag = axes.some((a) => resLimited(p, a.i));
          return (
            <button
              key={pi}
              type="button"
              className={`bragg-trow${pi === selected ? " sel" : ""}`}
              onClick={() => onSelect(pi)}
            >
              <span className="bragg-hkl">{anyFlag && <i className="bragg-flag-dot" />}{hklStr(p.center_hkl)}</span>
              <span>{p.q_abs.toFixed(2)}</span>
              <span>{p.intensity != null ? Math.round(p.intensity).toLocaleString() : "—"}</span>
              {axes.map((a) => {
                const w = displayQ(p, a.i);
                return (
                  <span key={a.i} style={{ color: resLimited(p, a.i) ? "#e8b454" : "#cdd4df" }}>
                    {fmt(w)}
                  </span>
                );
              })}
              <span className="bragg-fit">{p.fit_kind?.slice(0, 3) || "—"}</span>
            </button>
          );
        })}
        {order.length > MAX_ROWS && (
          <div className="bragg-trow-more">
            showing {MAX_ROWS} of {order.length} — sort to surface peaks of interest
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Selected-peak detail — intensity slices + fit ellipses + readout
// ---------------------------------------------------------------------------
function SelectedPeak({
  peak,
  peaks,
  axes,
  volumeId,
  colormap,
  setColormap,
}: {
  peak: BraggPeakWidth;
  peaks: BraggPeakWidth[];
  axes: Axis[];
  volumeId: string | undefined;
  colormap: string;
  setColormap: (c: string) => void;
}) {
  const [contrast, setContrast] = useState(1);
  const [zoom, setZoom] = useState(1);
  const lut = COLORMAPS[colormap] ?? COLORMAPS.inferno;
  const [h, k, l] = peak.center_hkl;

  const maxW = Math.max(...peak.width_hkl.map(Math.abs), 0.02);
  const baseHalf = Math.min(0.6, Math.max(0.08, maxW * 3.5));
  const half = baseHalf * zoom; // higher zoom → larger window → zoom out (more context)

  const floorHkl = axes.map((a) => deriveFloor(peaks, a.i, true));
  const ell = (xa: number, ya: number, src: (p: BraggPeakWidth, a: number) => number | null): Ellipse | null => {
    const rx = src(peak, xa), ry = src(peak, ya);
    return rx != null && ry != null ? { rx: Math.abs(rx), ry: Math.abs(ry) } : null;
  };
  const floorEll = (xa: number, ya: number): Ellipse | null =>
    floorHkl[xa] != null && floorHkl[ya] != null ? { rx: floorHkl[xa]!, ry: floorHkl[ya]! } : null;

  const tiles = [
    { label: "a*·b*", color: "#f1a73a", plane: "hk0", value: l, cx: h, cy: k, xa: 0, ya: 1 },
    { label: "a*·c*", color: "#34c98e", plane: "h0l", value: k, cx: h, cy: l, xa: 0, ya: 2 },
    { label: "b*·c*", color: "#74a8ff", plane: "0kl", value: h, cx: k, cy: l, xa: 1, ya: 2 },
  ];

  return (
    <div className="bragg-panel bragg-detail">
      <div className="bragg-panel-head">
        <span className="bragg-eyebrow">Selected peak · intensity slices + fit</span>
        <span className="bragg-sel-hkl">{hklStr(peak.center_hkl)}</span>
      </div>
      <div className="bragg-sel-meta">
        |Q| {peak.q_abs.toFixed(3)} Å⁻¹ · I {peak.intensity != null ? Math.round(peak.intensity).toLocaleString() : "—"} · {peak.fit_kind || "fit"}
      </div>

      <div className="bragg-sliders">
        <Slider
          label="Contrast"
          readout={`× ${contrast.toFixed(1)}`}
          min={0.1}
          max={20}
          step={0.1}
          value={contrast}
          onChange={setContrast}
          grow
        />
        <Slider
          label="Zoom"
          readout={`× ${zoom.toFixed(1)}`}
          min={1}
          max={10}
          step={0.5}
          value={zoom}
          onChange={setZoom}
          grow
        />
      </div>

      <div className="bragg-tiles">
        {tiles.map((t) => (
          <BraggPeakSlice
            key={t.label}
            volumeId={volumeId}
            plane={t.plane}
            value={t.value}
            cx={t.cx}
            cy={t.cy}
            half={half}
            lut={lut}
            contrast={contrast}
            fitted={fittedEllipse(peak, t.xa, t.ya)}
            measured={ell(t.xa, t.ya, measuredHkl)}
            floor={floorEll(t.xa, t.ya)}
            axisLabel={t.label}
            axisColor={t.color}
          />
        ))}
      </div>

      <div className="bragg-leg-row">
        <span className="bragg-leg"><i className="bragg-ell-leg-fit" />fitted</span>
        <span className="bragg-leg"><i className="bragg-ell-leg-meas" />measured</span>
        <span className="bragg-leg"><i className="bragg-ell-leg-floor" />|Q|-floor</span>
      </div>

      <div className="bragg-cmap">
        <span className="bragg-ctl-label">Colormap</span>
        <select value={colormap} onChange={(e) => setColormap(e.target.value)}>
          {CMAPS.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <ColormapBar lut={lut} />
      </div>

      <div className="bragg-readout">
        <span />
        {axes.map((a) => <span key={a.i} style={{ color: a.color }}>{a.label}</span>)}
        <span className="bragg-readout-key">fitted</span>
        {axes.map((a) => <span key={a.i}>{fmt(fittedQ(peak, a.i))}</span>)}
        <span className="bragg-readout-key">meas.</span>
        {axes.map((a) => (
          <span key={a.i} style={{ color: resLimited(peak, a.i) ? "#e8b454" : "#cdd4df" }}>{fmt(measuredQ(peak, a.i))}</span>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------
export function BraggProfileViewer() {
  const datasetsQ = useDatasets();
  const datasets = useMemo(() => datasetsQ.data ?? [], [datasetsQ.data]);
  useInitializeDataset(datasets);
  const datasetId = useDatasetStore((s) => s.datasetId);
  const setDataset = useDatasetStore((s) => s.setDataset);
  const colormap = useViewerStore((s) => s.colormap);
  const setColormap = useViewerStore((s) => s.setColormap);

  const profileQ = useBraggProfile(datasetId ?? undefined);
  const dataset = datasets.find((d) => d.id === datasetId);
  const profile = profileQ.data;

  const [selected, setSelected] = useState(0);
  const [sortKey, setSortKey] = useState<SortKey>("q");
  const [integerHklOnly, setIntegerHklOnly] = useState(true);

  // Per-axis width frame: spherical (ρ, θ, φ) or reciprocal (a*, b*, c*).
  const axes = useMemo(() => buildAxes(profile?.width_labels), [profile?.width_labels]);

  // Optionally keep only near-integer-hkl peaks (the actual Bragg reflections),
  // hiding fractional satellite / diffuse entries.
  const rawPeaks = useMemo(() => profile?.peaks ?? [], [profile]);
  const peaks = useMemo(
    () =>
      integerHklOnly
        ? rawPeaks.filter((p) => p.center_hkl.every((x) => Math.abs(x - Math.round(x)) < 0.15))
        : rawPeaks,
    [rawPeaks, integerHklOnly],
  );

  // pre-punch volume so the peak is visible (ring-removed preferred, raw fallback)
  const sliceVolumeId =
    dataset?.stages.find((s) => s.name === "ringremoved" && s.exists)?.volume_id ??
    dataset?.stages.find((s) => s.name === "raw" && s.exists)?.volume_id;
  const meta = useMeta(sliceVolumeId);

  const order = useMemo(() => {
    const idx = peaks.map((_, i) => i);
    idx.sort((a, b) => {
      const pa = peaks[a], pb = peaks[b];
      if (sortKey === "q") return pa.q_abs - pb.q_abs;
      if (sortKey === "intensity") return (pb.intensity ?? -Infinity) - (pa.intensity ?? -Infinity);
      return (measuredQ(pb, 1) ?? -Infinity) - (measuredQ(pa, 1) ?? -Infinity); // width = b* desc
    });
    return idx;
  }, [peaks, sortKey]);

  // ---- derived stats ----
  const meas = peaks.filter(measurable);
  const flagged = meas.filter((p) => axes.some((a) => resLimited(p, a.i))).length;
  const medQ = axes.map((a) => median(peaks.map((p) => displayQ(p, a.i))));
  const floorQ = axes.map((a) => deriveFloor(peaks, a.i, false));
  const padQ = median(floorQ);
  const qs = peaks.map((p) => p.q_abs).filter(Number.isFinite);

  const lat = meta.data?.lattice;
  const grid = meta.data?.shape;

  // Keep selection in range when the dataset / profile changes.
  const selPeak = peaks[selected] ?? peaks[Math.floor(peaks.length / 2)];

  const empty = (() => {
    if (datasetsQ.isLoading) return <EmptyState title="Loading datasets…" />;
    if (datasetsQ.isError) return <EmptyState error icon={<IconAlert />} title="Backend unreachable" hint="Start the API server and reload." />;
    if (!datasetsQ.isLoading && datasets.length === 0) return <EmptyState icon={<IconLattice />} title="No datasets" hint="Load a volume and run the pipeline on the Configure page first." />;
    if (!datasetId) return <EmptyState icon={<IconLattice />} title="No dataset selected" hint="Pick a dataset to view its Bragg profile." />;
    if (profileQ.isLoading || !profile) return <EmptyState title="Loading Bragg profile…" />;
    if (profileQ.isError) return <EmptyState error icon={<IconAlert />} title="Could not load Bragg profile" hint={(profileQ.error as Error).message} />;
    if (profile && !profile.has_profile) return <EmptyState icon={<IconLattice />} title="No Bragg profile for this dataset" hint="Run the pipeline with Bragg-punch peak-shape fitting enabled to create the review profile." />;
    if (profile?.has_profile && rawPeaks.length === 0) return <EmptyState title="No peaks recorded" hint="The punch stage completed, but no Bragg peaks were detected." />;
    if (profile?.has_profile && peaks.length === 0) return <EmptyState title="All peaks filtered out" hint="Turn off “Integer HKL only” to show fractional-hkl peaks." />;
    return null;
  })();

  return (
    <div className="bragg-page">
      <div className="bragg-card">
        <div className="bragg-title-row">
          <div className="bragg-chips">
            {profile?.has_profile && (
              <span className="bragg-toggle">
                <Switch
                  label="Integer HKL only"
                  checked={integerHklOnly}
                  onChange={(v) => { setIntegerHklOnly(v); setSelected(0); }}
                />
              </span>
            )}
            <label className="bragg-chip bragg-chip-select">
              <span className="bragg-chip-key">Dataset</span>
              <select value={datasetId ?? ""} onChange={(e) => { setDataset(e.target.value); setSelected(0); }}>
                {datasets.map((d) => <option key={d.id} value={d.id} title={d.raw_name}>{d.temperature ?? d.stem}</option>)}
              </select>
            </label>
            {profile?.has_profile && (
              <span className="bragg-chip bragg-chip-fit">
                <i /> {profile.fit_covariance ? "covariance fit" : "moment fit"}
              </span>
            )}
          </div>
        </div>

        {empty ?? (
          <>
            <div className="bragg-stats">
              <div className="bragg-stat">
                <span className="bragg-stat-eyebrow">Peaks profiled</span>
                <span className="bragg-stat-row"><span className="bragg-stat-v">{peaks.length}</span><span className="bragg-stat-desc">{meas.length === peaks.length ? "all measurable" : `${meas.length} measurable`}</span></span>
              </div>
              <div className="bragg-stat">
                <span className="bragg-stat-eyebrow">Median width · Å⁻¹</span>
                <span className="bragg-stat-row bragg-stat-medrow">
                  {axes.map((a) => (
                    <span key={a.i} className="bragg-stat-med">
                      <span style={{ color: a.color }}>{a.label}</span> {fmt(medQ[a.i])}
                    </span>
                  ))}
                </span>
              </div>
              <div className="bragg-stat">
                <span className="bragg-stat-eyebrow">Resolution-limited</span>
                <span className="bragg-stat-row"><span className="bragg-stat-v">{meas.length ? Math.round((flagged / meas.length) * 100) : 0}%</span><span className="bragg-stat-desc">{flagged} of {meas.length} peaks</span></span>
              </div>
              <div className="bragg-stat">
                <span className="bragg-stat-eyebrow">|Q| range</span>
                <span className="bragg-stat-row"><span className="bragg-stat-v">{qs.length ? `${Math.min(...qs).toFixed(1)}–${Math.max(...qs).toFixed(1)}` : "—"}</span><span className="bragg-stat-desc">Å⁻¹{padQ != null ? ` · floor ${padQ.toFixed(3)}` : ""}</span></span>
              </div>
              <div className="bragg-stat">
                <span className="bragg-stat-eyebrow">Fit kind</span>
                <span className="bragg-stat-row"><span className="bragg-stat-v bragg-stat-fit">{profile?.fit_covariance ? "covariance" : "moment"}</span><span className="bragg-stat-desc">{padQ != null ? `pad ${padQ.toFixed(3)} Å⁻¹` : ""}</span></span>
              </div>
            </div>

            <div className="bragg-grid">
              <Scatter peaks={peaks} axes={axes} selected={selected} onSelect={setSelected} padFloorQ={padQ} />
              <Histograms peaks={peaks} axes={axes} showMeasured />
            </div>

            <div className="bragg-grid">
              <PeakTable peaks={peaks} axes={axes} order={order} selected={selected} sortKey={sortKey} onSort={setSortKey} onSelect={setSelected} />
              {selPeak && (
                <SelectedPeak peak={selPeak} peaks={peaks} axes={axes} volumeId={sliceVolumeId} colormap={colormap} setColormap={setColormap} />
              )}
            </div>

            <div className="bragg-meta">
              {([
                ["Source", dataset?.raw_name ?? "—"],
                ["Punch frame", profile?.punch_frame ?? "unknown"],
                ["Floor r", floorQ.every((f) => f != null) ? `(${floorQ.map((f) => f!.toFixed(3)).join(", ")})` : "—"],
                ...(lat && lat.a != null ? [["Lattice", `a=${lat.a}${lat.b != null ? `, b=${lat.b}` : ""}${lat.c != null ? `, c=${lat.c} Å` : ""}`] as [string, string]] : []),
                ...(grid ? [["Grid", grid.join(" × ")] as [string, string]] : []),
              ] as [string, string][]).map(([key, value]) => (
                <div key={key} className="bragg-meta-item"><span className="bragg-meta-key">{key}</span><span className="bragg-meta-val">{value}</span></div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
