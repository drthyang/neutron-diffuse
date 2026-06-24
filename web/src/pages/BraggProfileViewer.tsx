import {
  useEffect,
  useMemo,
  useState,
  type MouseEvent,
  type PointerEvent,
  type ReactNode,
} from "react";

import type { BraggPeakWidth } from "../api/types";
import { useBraggProfile, useDatasets } from "../api/hooks";
import {
  EmptyState,
  Field,
  IconAlert,
  IconLattice,
  MetaStrip,
} from "../components/ui";
import { useDatasetStore, useInitializeDataset } from "../state/datasetStore";

const AXES = [
  { key: 0, label: "Qx", color: "#e66a5c" },
  { key: 1, label: "Qy", color: "#4f9edc" },
  { key: 2, label: "Qz", color: "#58b06f" },
] as const;

const PAD = { left: 54, right: 18, top: 18, bottom: 42 };
const SIZE = { width: 760, height: 330 };

type FitStats = {
  mean: number;
  sigma: number;
  fwhm: number;
  amplitude: number;
  error: number;
};

function finiteValues(values: number[]): number[] {
  return values.filter((v) => Number.isFinite(v));
}

function extent(values: number[], pad = 0.06): [number, number] {
  const xs = finiteValues(values);
  if (xs.length === 0) return [0, 1];
  let min = Math.min(...xs);
  let max = Math.max(...xs);
  if (min === max) {
    const d = Math.abs(min) || 1;
    min -= 0.5 * d;
    max += 0.5 * d;
  }
  const d = max - min;
  return [Math.max(0, min - d * pad), max + d * pad];
}

function ticks(min: number, max: number, n = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return [0, 1];
  return Array.from({ length: n }, (_, i) => min + (i * (max - min)) / (n - 1));
}

function fmt(v: number): string {
  if (Math.abs(v) >= 10) return v.toFixed(1);
  if (Math.abs(v) >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

function widthOf(peak: BraggPeakWidth, axis: number): number {
  return peak.width_q[axis];
}

function histogramCounts(
  values: number[],
  domain: [number, number],
  binCount: number,
) {
  const xs = finiteValues(values);
  const binWidth = (domain[1] - domain[0]) / binCount;
  const counts = Array.from({ length: binCount }, () => 0);
  for (const v of xs) {
    if (v < domain[0] || v > domain[1]) continue;
    const raw = Math.floor(((v - domain[0]) / (domain[1] - domain[0])) * binCount);
    counts[Math.max(0, Math.min(binCount - 1, raw))] += 1;
  }
  const centers = counts.map((_count, i) => domain[0] + (i + 0.5) * binWidth);
  return { counts, centers, binWidth };
}

function histogramBinCount(nPeaks: number): number {
  return Math.max(8, Math.min(28, Math.ceil(Math.sqrt(nPeaks) * 1.8)));
}

function fitGaussianToHistogram(
  values: number[],
  domain: [number, number],
  binCount: number,
): FitStats | null {
  const xs = finiteValues(values);
  if (xs.length < 3) return null;
  const { counts, centers, binWidth } = histogramCounts(xs, domain, binCount);
  const maxCount = Math.max(...counts);
  if (maxCount <= 0) return null;
  const total = counts.reduce((acc, c) => acc + c, 0);
  if (total <= 0) return null;
  const weightedMean = centers.reduce((acc, center, i) => acc + center * counts[i], 0) / total;
  const weightedSigma = Math.sqrt(
    Math.max(
      centers.reduce((acc, center, i) => acc + counts[i] * (center - weightedMean) ** 2, 0) / total,
      0,
    ),
  );
  const peakMean = centers[counts.indexOf(maxCount)];
  const domainWidth = domain[1] - domain[0];
  const minSigma = Math.max(binWidth * 0.35, domainWidth * 0.002);
  const maxSigma = Math.max(minSigma * 1.1, domainWidth * 0.5);

  const score = (mean: number, sigma: number): FitStats | null => {
    if (!Number.isFinite(mean) || !Number.isFinite(sigma) || sigma <= 0) return null;
    if (mean < domain[0] || mean > domain[1]) return null;
    const g = centers.map((center) => Math.exp(-0.5 * ((center - mean) / sigma) ** 2));
    const gg = g.reduce((acc, v) => acc + v * v, 0);
    if (gg <= 0) return null;
    const yg = g.reduce((acc, v, i) => acc + counts[i] * v, 0);
    const amplitude = Math.max(0, yg / gg);
    const error = g.reduce((acc, v, i) => acc + (counts[i] - amplitude * v) ** 2, 0);
    return {
      mean,
      sigma,
      fwhm: 2.354820045 * sigma,
      amplitude,
      error,
    };
  };

  const means = [
    weightedMean,
    peakMean,
    peakMean - binWidth,
    peakMean + binWidth,
    peakMean - 2 * binWidth,
    peakMean + 2 * binWidth,
  ];
  const sigmas = [
    minSigma,
    binWidth * 0.5,
    binWidth,
    binWidth * 1.5,
    binWidth * 2.5,
    Math.max(minSigma, weightedSigma * 0.5),
    Math.max(minSigma, weightedSigma),
  ].map((s) => Math.min(Math.max(s, minSigma), maxSigma));

  let best: FitStats | null = null;
  for (const mean of means) {
    for (const sigma of sigmas) {
      const candidate = score(mean, sigma);
      if (candidate && (!best || candidate.error < best.error)) best = candidate;
    }
  }

  if (!best) return null;
  let stepMean = Math.max(binWidth, best.sigma);
  let stepSigma = Math.max(binWidth * 0.5, best.sigma * 0.5);
  for (let iter = 0; iter < 24; iter += 1) {
    let improved = false;
    const candidates = [
      [best.mean - stepMean, best.sigma],
      [best.mean + stepMean, best.sigma],
      [best.mean, best.sigma - stepSigma],
      [best.mean, best.sigma + stepSigma],
      [best.mean - stepMean, best.sigma - stepSigma],
      [best.mean - stepMean, best.sigma + stepSigma],
      [best.mean + stepMean, best.sigma - stepSigma],
      [best.mean + stepMean, best.sigma + stepSigma],
    ] as const;
    for (const [mean, sigmaRaw] of candidates) {
      const sigma = Math.min(Math.max(sigmaRaw, minSigma), maxSigma);
      const candidate = score(mean, sigma);
      if (candidate && candidate.error < best.error) {
        best = candidate;
        improved = true;
      }
    }
    if (!improved) {
      stepMean *= 0.55;
      stepSigma *= 0.55;
    }
  }
  return best;
}

function gaussianCurve(fit: FitStats) {
  return (x: number) => fit.amplitude * Math.exp(-0.5 * ((x - fit.mean) / fit.sigma) ** 2);
}

function clampDomain(domain: [number, number], bounds: [number, number]): [number, number] {
  const width = domain[1] - domain[0];
  const boundsWidth = bounds[1] - bounds[0];
  if (width >= boundsWidth) return bounds;
  let lo = domain[0];
  let hi = domain[1];
  if (lo < bounds[0]) {
    lo = bounds[0];
    hi = lo + width;
  }
  if (hi > bounds[1]) {
    hi = bounds[1];
    lo = hi - width;
  }
  return [lo, hi];
}

function initialHistogramDomain(
  allWidths: number[],
  stats: Array<FitStats | null>,
): { full: [number, number]; initial: [number, number] } {
  const full = extent(allWidths, 0.4);
  const fitRanges = stats
    .filter((s): s is FitStats => s !== null)
    .flatMap((s) => [s.mean - 3 * s.sigma, s.mean + 3 * s.sigma]);
  const fitDomain = fitRanges.length ? extent(fitRanges, 0.12) : extent(allWidths, 0.18);
  const fullWidth = full[1] - full[0];
  const fitWidth = fitDomain[1] - fitDomain[0];
  const minWindow = fullWidth * 0.34;
  const center = 0.5 * (fitDomain[0] + fitDomain[1]);
  const windowWidth = Math.min(fullWidth, Math.max(fitWidth, minWindow));
  const initial: [number, number] = [
    center - 0.5 * windowWidth,
    center + 0.5 * windowWidth,
  ];
  return { full, initial: clampDomain(initial, full) };
}

function ChartFrame({
  title,
  xLabel,
  yLabel,
  hint,
  yTickFormat = fmt,
  children,
  xDomain,
  yDomain,
}: {
  title: string;
  xLabel: string;
  yLabel: string;
  hint?: string;
  yTickFormat?: (v: number) => string;
  children: (scale: {
    x: (v: number) => number;
    y: (v: number) => number;
    innerW: number;
    innerH: number;
  }) => ReactNode;
  xDomain: [number, number];
  yDomain: [number, number];
}) {
  const innerW = SIZE.width - PAD.left - PAD.right;
  const innerH = SIZE.height - PAD.top - PAD.bottom;
  const x = (v: number) => PAD.left + ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * innerW;
  const y = (v: number) => PAD.top + innerH - ((v - yDomain[0]) / (yDomain[1] - yDomain[0])) * innerH;

  return (
    <div className="profile-chart">
      <div className="profile-chart-head">
        <div className="profile-chart-title">
          <h3>{title}</h3>
          {hint && <span>{hint}</span>}
        </div>
        <div className="profile-legend">
          {AXES.map((a) => (
            <span key={a.key}>
              <i style={{ background: a.color }} />
              {a.label}
            </span>
          ))}
        </div>
      </div>
      <svg viewBox={`0 0 ${SIZE.width} ${SIZE.height}`} role="img" aria-label={title}>
        <rect
          x={PAD.left}
          y={PAD.top}
          width={innerW}
          height={innerH}
          className="profile-plot-bg"
        />
        {ticks(...xDomain).map((t) => (
          <g key={`x-${t}`}>
            <line x1={x(t)} x2={x(t)} y1={PAD.top} y2={PAD.top + innerH} className="profile-grid" />
            <text x={x(t)} y={SIZE.height - 17} className="profile-tick" textAnchor="middle">
              {fmt(t)}
            </text>
          </g>
        ))}
        {ticks(...yDomain).map((t) => (
          <g key={`y-${t}`}>
            <line x1={PAD.left} x2={PAD.left + innerW} y1={y(t)} y2={y(t)} className="profile-grid" />
            <text x={PAD.left - 9} y={y(t) + 4} className="profile-tick" textAnchor="end">
              {yTickFormat(t)}
            </text>
          </g>
        ))}
        <line x1={PAD.left} x2={PAD.left + innerW} y1={PAD.top + innerH} y2={PAD.top + innerH} className="profile-axis" />
        <line x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={PAD.top + innerH} className="profile-axis" />
        {children({ x, y, innerW, innerH })}
        <text x={PAD.left + innerW / 2} y={SIZE.height - 3} className="profile-axis-label" textAnchor="middle">
          {xLabel}
        </text>
        <text x={15} y={PAD.top + innerH / 2} className="profile-axis-label" textAnchor="middle" transform={`rotate(-90 15 ${PAD.top + innerH / 2})`}>
          {yLabel}
        </text>
      </svg>
    </div>
  );
}

function WidthScatter({ peaks }: { peaks: BraggPeakWidth[] }) {
  const xDomain = extent(peaks.map((p) => p.q_abs));
  const yDomain = extent(peaks.flatMap((p) => AXES.map((a) => widthOf(p, a.key))));

  return (
    <ChartFrame
      title="Peak width versus |Q|"
      xLabel="|Q| (Å⁻¹)"
      yLabel="width (Å⁻¹)"
      xDomain={xDomain}
      yDomain={yDomain}
    >
      {({ x, y }) => (
        <>
          {AXES.map((axis) => (
            <g key={axis.key}>
              {peaks.map((p) => {
                const w = widthOf(p, axis.key);
                if (!Number.isFinite(p.q_abs) || !Number.isFinite(w)) return null;
                return (
                  <circle
                    key={`${p.index}-${axis.key}`}
                    cx={x(p.q_abs)}
                    cy={y(w)}
                    r={3}
                    fill={axis.color}
                    opacity={0.72}
                  />
                );
              })}
            </g>
          ))}
        </>
      )}
    </ChartFrame>
  );
}

function Histogram({ peaks }: { peaks: BraggPeakWidth[] }) {
  const [hoveredFit, setHoveredFit] = useState<{
    axis: string;
    color: string;
    mean: number;
    sigma: number;
    x: number;
    y: number;
  } | null>(null);
  const [drag, setDrag] = useState<{
    pointerId: number;
    startX: number;
    domain: [number, number];
  } | null>(null);
  const allWidths = peaks.flatMap((p) => AXES.map((a) => widthOf(p, a.key)));
  const binCount = histogramBinCount(peaks.length);
  const domains = useMemo(
    () => {
      const fullDomain = extent(allWidths, 0.4);
      const fullFits = AXES.map((axis) =>
        fitGaussianToHistogram(
          peaks.map((p) => widthOf(p, axis.key)),
          fullDomain,
          binCount,
        ),
      );
      return initialHistogramDomain(allWidths, fullFits);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [peaks],
  );
  const [xDomain, setXDomain] = useState<[number, number]>(domains.initial);
  useEffect(() => {
    setXDomain(domains.initial);
  }, [domains.initial]);
  const binWidth = (xDomain[1] - xDomain[0]) / binCount;
  const bins = AXES.map((axis) => {
    const values = peaks.map((p) => widthOf(p, axis.key));
    const { counts } = histogramCounts(values, xDomain, binCount);
    const fit = fitGaussianToHistogram(values, xDomain, binCount);
    return { axis, counts, fit };
  });
  const fitMax = Math.max(
    0,
    ...bins.flatMap((b) => {
      if (!b.fit) return [0];
      const curve = gaussianCurve(b.fit);
      return Array.from({ length: 80 }, (_, i) =>
        curve(xDomain[0] + (i * (xDomain[1] - xDomain[0])) / 79),
      );
    }),
  );
  const yMax = Math.max(1, ...bins.flatMap((b) => b.counts), fitMax) * 1.16;

  return (
    <ChartFrame
      title="Peak-width histogram"
      xLabel="width (Å⁻¹)"
      yLabel="number of peaks"
      hint="Drag horizontally to inspect tails"
      xDomain={xDomain}
      yDomain={[0, yMax]}
      yTickFormat={(v) => Math.round(v).toLocaleString()}
    >
      {({ x, y, innerW }) => {
        const pxBinW = Math.max(1, x(xDomain[0] + binWidth) - x(xDomain[0]));
        const startDrag = (event: PointerEvent<SVGRectElement>) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          setHoveredFit(null);
          setDrag({
            pointerId: event.pointerId,
            startX: event.clientX,
            domain: xDomain,
          });
        };
        const moveDrag = (event: PointerEvent<SVGRectElement>) => {
          if (!drag || drag.pointerId !== event.pointerId) return;
          const domainWidth = drag.domain[1] - drag.domain[0];
          const dx = event.clientX - drag.startX;
          const shift = -(dx / innerW) * domainWidth;
          setXDomain(clampDomain([
            drag.domain[0] + shift,
            drag.domain[1] + shift,
          ], domains.full));
        };
        const endDrag = (event: PointerEvent<SVGRectElement>) => {
          if (drag?.pointerId === event.pointerId) setDrag(null);
        };
        return (
          <>
            <rect
              x={PAD.left}
              y={PAD.top}
              width={innerW}
              height={SIZE.height - PAD.top - PAD.bottom}
              className={`profile-drag-plane${drag ? " dragging" : ""}`}
              onPointerDown={startDrag}
              onPointerMove={moveDrag}
              onPointerUp={endDrag}
              onPointerCancel={endDrag}
              onLostPointerCapture={() => setDrag(null)}
            />
            {bins.map(({ axis, counts }, seriesIndex) => (
              <g key={axis.key}>
                {counts.map((count, i) => {
                  const x0 = x(xDomain[0] + i * binWidth);
                  const h = y(0) - y(count);
                  return (
                    <rect
                      key={i}
                      x={x0 + seriesIndex * (pxBinW / 4)}
                      y={y(count)}
                      width={Math.max(1, pxBinW / 3)}
                      height={h}
                      fill={axis.color}
                      opacity={0.56}
                    />
                  );
                })}
              </g>
            ))}
            {bins.map(({ axis, fit }) => {
              if (!fit) return null;
              const fitCurve = gaussianCurve(fit);
              const points = Array.from({ length: 100 }, (_, i) => {
                const xv = xDomain[0] + (i * (xDomain[1] - xDomain[0])) / 99;
                return `${x(xv).toFixed(2)},${y(fitCurve(xv)).toFixed(2)}`;
              }).join(" ");
              const showTip = (event: MouseEvent<SVGPolylineElement>) => {
                const svg = event.currentTarget.ownerSVGElement;
                if (!svg) return;
                const screenCtm = svg.getScreenCTM();
                if (!screenCtm) return;
                const pt = svg.createSVGPoint();
                pt.x = event.clientX;
                pt.y = event.clientY;
                const loc = pt.matrixTransform(screenCtm.inverse());
                setHoveredFit({
                  axis: axis.label,
                  color: axis.color,
                  mean: fit.mean,
                  sigma: fit.sigma,
                  x: Math.min(Math.max(loc.x + 12, PAD.left + 6), SIZE.width - 162),
                  y: Math.min(Math.max(loc.y - 48, PAD.top + 6), SIZE.height - 74),
                });
              };
              return (
                <g key={`fit-${axis.key}`}>
                  <polyline
                    points={points}
                    className="profile-fit-line"
                    style={{ stroke: axis.color }}
                  />
                  <polyline
                    points={points}
                    className="profile-fit-hit"
                    onMouseMove={showTip}
                    onMouseEnter={showTip}
                    onMouseLeave={() => setHoveredFit(null)}
                  />
                </g>
              );
            })}
            {hoveredFit && (
              <g className="profile-tooltip" transform={`translate(${hoveredFit.x} ${hoveredFit.y})`}>
                <rect width={150} height={60} rx={6} />
                <circle cx={13} cy={15} r={4} fill={hoveredFit.color} />
                <text x={24} y={19}>{hoveredFit.axis} Gaussian fit</text>
                <text x={10} y={38}>mean {fmt(hoveredFit.mean)} Å⁻¹</text>
                <text x={10} y={53}>sigma {fmt(hoveredFit.sigma)} Å⁻¹</text>
              </g>
            )}
          </>
        );
      }}
    </ChartFrame>
  );
}

function SummaryRows({ peaks }: { peaks: BraggPeakWidth[] }) {
  const allWidths = peaks.flatMap((p) => AXES.map((a) => widthOf(p, a.key)));
  const domain = extent(allWidths, 0.4);
  const binCount = histogramBinCount(peaks.length);
  const rows = AXES.map((axis) => {
    const fit = fitGaussianToHistogram(peaks.map((p) => widthOf(p, axis.key)), domain, binCount);
    return { axis, fit };
  });
  return (
    <div className="profile-summary">
      {rows.map((row) => (
        <div className="profile-summary-row" key={row.axis.key}>
          <span><i style={{ background: row.axis.color }} />{row.axis.label}</span>
          {row.fit ? (
            <>
              <em>Gaussian mean <b>{fmt(row.fit.mean)}</b> Å⁻¹</em>
              <em>Sigma <b>{fmt(row.fit.sigma)}</b> Å⁻¹</em>
              <em>FWHM <b>{fmt(row.fit.fwhm)}</b> Å⁻¹</em>
            </>
          ) : (
            <em>Gaussian fit unavailable</em>
          )}
        </div>
      ))}
    </div>
  );
}

export function BraggProfileViewer() {
  const datasetsQ = useDatasets();
  const datasets = useMemo(() => datasetsQ.data ?? [], [datasetsQ.data]);
  useInitializeDataset(datasets);

  const datasetId = useDatasetStore((s) => s.datasetId);
  const setDataset = useDatasetStore((s) => s.setDataset);
  const profileQ = useBraggProfile(datasetId ?? undefined);
  const dataset = datasets.find((d) => d.id === datasetId);
  const profile = profileQ.data;
  const peaks = profile?.peaks ?? [];
  const tilted = peaks.filter((p) => p.fit_kind === "tilted").length;

  return (
    <div className="page-body">
      <div className="toolbar">
        <Field label="Dataset" grow>
          <select value={datasetId ?? ""} onChange={(e) => setDataset(e.target.value)}>
            {datasets.map((d) => (
              <option key={d.id} value={d.id} title={d.raw_name}>
                {d.temperature ?? d.stem}
              </option>
            ))}
          </select>
        </Field>

      </div>

      {datasetsQ.isLoading && <EmptyState title="Loading datasets..." />}
      {datasetsQ.isError && (
        <EmptyState
          error
          icon={<IconAlert />}
          title="Backend unreachable"
          hint="Start the API server and reload."
        />
      )}
      {profileQ.isLoading && <EmptyState title="Loading Bragg profile..." />}
      {profileQ.isError && (
        <EmptyState
          error
          icon={<IconAlert />}
          title="Could not load Bragg profile"
          hint={(profileQ.error as Error).message}
        />
      )}
      {profile && !profile.has_profile && (
        <EmptyState
          icon={<IconLattice />}
          title="No Bragg profile for this dataset"
          hint="Run the Bragg punch stage with peak-shape fitting enabled to create the review profile."
        />
      )}
      {profile?.has_profile && peaks.length === 0 && (
        <EmptyState
          title="No peaks recorded"
          hint="The punch stage completed, but no Bragg peaks were detected for this dataset."
        />
      )}

      {profile?.has_profile && peaks.length > 0 && (
        <>
          <SummaryRows peaks={peaks} />
          <div className="profile-chart-grid">
            <WidthScatter peaks={peaks} />
            <Histogram peaks={peaks} />
          </div>
          <MetaStrip
            items={[
              { key: "Source", value: dataset?.raw_name },
              { key: "Peaks", value: profile.n_peaks },
              { key: "Tilted fits", value: `${tilted}/${profile.n_peaks}` },
              { key: "Punch frame", value: profile.punch_frame ?? "unknown" },
              { key: "Profile", value: profile.profile_path ?? "" },
            ]}
          />
        </>
      )}
    </div>
  );
}
