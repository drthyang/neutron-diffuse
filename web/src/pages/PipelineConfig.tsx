// Pipeline configuration page — the form + live algorithm illustrations.  All
// values live in the pipeline store, so navigating away and back never resets
// them; pressing Run kicks off the job (also in the store) and jumps to the
// Execution page via `onStarted`.

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { useShallow } from "zustand/react/shallow";

import { browseDataRoot, fetchMeta, fetchSlice, setDataRoot } from "../api/client";
import { useDataRoot, useDatasets } from "../api/hooks";
import { COLORMAPS } from "../colormaps/luts";
import { SliceCanvas } from "../components/SliceCanvas";
import { Field, HelpTip, RangeSlider, Slider, Switch } from "../components/ui";
import { useDatasetStore, useInitializeDataset } from "../state/datasetStore";
import {
  STAGE_LABELS,
  STAGE_NO,
  usePipelineStore,
  type PunchPlane,
} from "../state/pipelineStore";

const DATASET_STAGE_BADGES = [
  { key: "raw", label: "Raw", group: "Input" },
  { key: "ringremoved", label: "Ring removed", group: "Cleanup" },
  { key: "braggpunched", label: "Bragg punched", group: "Cleanup" },
  { key: "backfilled", label: "Backfilled", group: "Cleanup" },
  { key: "flattened", label: "Background flattened", group: "Cleanup" },
  { key: "delta_pdf", label: "3D-ΔPDF", group: "Output" },
] as const;

const DEFAULT_PDF_CROP = { h: 4, k: 8, l: 15 };

// One pipeline stage rendered as its own card.  `step` shows the matching
// stepper number so the tiles read in pipeline order regardless of how the
// masonry packs them.
function StageCard({
  title,
  step,
  className = "",
  children,
}: {
  title: string;
  step?: number;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section className={`card stage-card ${className}`}>
      <div className="card-head">
        <h3>
          {step != null && <span className="config-step-no">{step}</span>}
          {title}
        </h3>
      </div>
      <div className="card-body">{children}</div>
    </section>
  );
}

// Parse a numeric form value, falling back to `dflt` for empty/invalid input and
// clamping into [lo, hi] so the illustrations stay well-formed while typing.
function clampInt(raw: string, dflt: number, lo: number, hi: number): number {
  const n = raw === "" ? dflt : Math.round(Number(raw));
  if (!Number.isFinite(n)) return dflt;
  return Math.max(lo, Math.min(hi, n));
}
function clampFloat(raw: string, dflt: number, lo: number, hi: number): number {
  const n = raw === "" ? dflt : Number(raw);
  if (!Number.isFinite(n)) return dflt;
  return Math.max(lo, Math.min(hi, n));
}

function qSpanFromMeta(meta: {
  h_range: [number, number];
  k_range: [number, number];
  l_range: [number, number];
  lattice: { a: number | null; b: number | null; c: number | null };
} | null | undefined): number {
  if (!meta) return 0;
  const h = Math.min(DEFAULT_PDF_CROP.h, Math.max(Math.abs(meta.h_range[0]), Math.abs(meta.h_range[1])));
  const k = Math.min(DEFAULT_PDF_CROP.k, Math.max(Math.abs(meta.k_range[0]), Math.abs(meta.k_range[1])));
  const l = Math.min(DEFAULT_PDF_CROP.l, Math.max(Math.abs(meta.l_range[0]), Math.abs(meta.l_range[1])));
  const a = meta.lattice.a ?? 1;
  const b = meta.lattice.b ?? 1;
  const c = meta.lattice.c ?? 1;
  return Math.sqrt(
    (h * 2 * Math.PI / a) ** 2 +
    (k * 2 * Math.PI / b) ** 2 +
    (l * 2 * Math.PI / c) ** 2,
  );
}

// ---------------------------------------------------------------------------
// Ring-removal illustration (pie = patches, ring = Fourier texture)
// ---------------------------------------------------------------------------

// Diverging tint for a texture value v ∈ [−1, 1]: neutral surface at 0, accent
// blue at the crests (+1), amber at the troughs (−1).
function textureColor(v: number): string {
  const mid = [35, 42, 51]; // --surface-3
  const hi = [79, 143, 247]; // --accent
  const lo = [232, 180, 84]; // --amber
  const end = v >= 0 ? hi : lo;
  const t = Math.min(1, Math.abs(v));
  const c = mid.map((m, i) => Math.round(m + (end[i] - m) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

// ring-model illustration geometry (viewBox 0 0 180 180)
const RC = 90; // centre
const RPIE = 58; // pie radius (patches)
const RIN = 65; // ring inner radius (texture)
const ROUT = 84; // ring outer radius
const RHUB = 20; // centre hub
const RTOP = -Math.PI / 2; // start angle at 12 o'clock
const rpt = (r: number, a: number) =>
  [(RC + r * Math.cos(a)).toFixed(2), (RC + r * Math.sin(a)).toFixed(2)] as const;

// ---------------------------------------------------------------------------
// Parametric ring-model illustration
//
// Depicts the separable model the parametric estimator fits: concentric ring
// shells at increasing |Q|, each tinted azimuthally by the texture harmonic
// T(φ) = cos(nFourier·φ).  The radial band thickness tracks `ringWidth`.  In
// `rolling` mode the shells are thick, soft-edged and overlapping (a continuous
// Ring(|Q|) swept Qmin→Qmax); in `peaks` mode they are thin, crisp pseudo-Voigt
// rings.  There are NO azimuthal patches — that is the patched model's scheme.
// ---------------------------------------------------------------------------
const PRING_RADII = [34, 55, 76]; // three concentric |Q| shells

function ParametricRingViz({
  nFourier,
  radialMode,
  ringWidth,
}: {
  nFourier: number;
  radialMode: string;
  ringWidth: number;
}) {
  const rolling = radialMode === "rolling";
  // band thickness in px: scales with ring width (0.05–0.6 Å⁻¹ → ~3–18 px),
  // rolling reads thicker/softer than the crisp peaks rings.
  const tw = Math.max(3, Math.min(18, ringWidth * 30)) * (rolling ? 1.35 : 0.8);

  const segs = useMemo(() => {
    const n = Math.min(220, Math.max(96, nFourier * 14));
    const out: { d: string; c: string; r: number }[] = [];
    for (const R of PRING_RADII) {
      for (let i = 0; i < n; i++) {
        const a0 = (i / n) * 2 * Math.PI + RTOP;
        const a1 = ((i + 1) / n) * 2 * Math.PI + RTOP;
        const phi = ((i + 0.5) / n) * 2 * Math.PI;
        const [x0, y0] = rpt(R, a0);
        const [x1, y1] = rpt(R, a1);
        out.push({
          d: `M${x0} ${y0} A${R} ${R} 0 0 1 ${x1} ${y1}`,
          c: textureColor(Math.cos(nFourier * phi)),
          r: R,
        });
      }
    }
    return out;
  }, [nFourier]);

  return (
    <svg
      className="ring-viz"
      viewBox="0 0 180 180"
      role="img"
      aria-label={`parametric ${radialMode} ring model, texture order ${nFourier}`}
    >
      {/* soft underlay (rolling = continuous Ring(|Q|)) */}
      {rolling &&
        PRING_RADII.map((R, i) => (
          <circle
            key={`u${i}`}
            cx={RC}
            cy={RC}
            r={R}
            fill="none"
            className="pring-underlay"
            strokeWidth={tw * 1.8}
          />
        ))}
      {segs.map((s, i) => (
        <path
          key={i}
          d={s.d}
          fill="none"
          stroke={s.c}
          strokeWidth={tw}
          strokeLinecap="butt"
          opacity={rolling ? 0.92 : 1}
        />
      ))}
      <circle cx={RC} cy={RC} r={RHUB - 4} className="ring-viz-hub" />
      <text
        x={RC}
        y={RC}
        textAnchor="middle"
        dominantBaseline="central"
        className="ring-viz-count"
      >
        {rolling ? "Q→" : nFourier}
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Radial line-shape Ring(|Q|) — the function the selected radial mode fits.
//
// `peaks`   → a discrete sum of pseudo-Voigt rings: smooth analytic peaks that
//             fall to zero between rings (the model can only represent
//             pseudo-Voigt-shaped lines).
// `rolling` → a continuous Ring(|Q|) sampled at every shell: a filled profile
//             with the swept window drawn over one ring and the roll-step
//             sampling ticks on the axis (no discrete-peak assumption).
// Schematic, not to scale — `ringWidth` sets the relative peak/window breadth.
// ---------------------------------------------------------------------------
const RPROF_QMIN = 1.5;
const RPROF_QMAX = 10.5;
const RPROF_CENTERS = [2.7, 5.2, 7.6];
const RPROF_AMPS = [1.0, 0.62, 0.42];

function pseudoVoigt(q: number, c: number, fwhm: number, eta: number): number {
  const w = Math.max(fwhm, 1e-3);
  const sg = w / 2.3548;
  const g = Math.exp(-0.5 * ((q - c) / sg) ** 2);
  const l = 1 / (1 + ((q - c) / (0.5 * w)) ** 2);
  return eta * l + (1 - eta) * g;
}

function RadialProfileViz({
  mode,
  ringWidth,
}: {
  mode: string;
  ringWidth: number;
}) {
  const rolling = mode === "rolling";
  const PX0 = 14;
  const PX1 = 166;
  const PY0 = 56; // baseline
  const PYT = 12; // top
  const xq = (q: number) => PX0 + ((q - RPROF_QMIN) / (RPROF_QMAX - RPROF_QMIN)) * (PX1 - PX0);
  const yv = (v: number) => PY0 - v * (PY0 - PYT);
  // peaks are crisp; rolling shows the same true profile a touch broader/softer
  const fwhm = Math.max(0.12, ringWidth) * (rolling ? 1.5 : 1.0);

  const { line, area } = useMemo(() => {
    const n = 200;
    const pts: [number, number][] = [];
    for (let i = 0; i <= n; i++) {
      const q = RPROF_QMIN + (i / n) * (RPROF_QMAX - RPROF_QMIN);
      let v = 0;
      for (let r = 0; r < RPROF_CENTERS.length; r++)
        v += RPROF_AMPS[r] * pseudoVoigt(q, RPROF_CENTERS[r], fwhm, 0.5);
      pts.push([xq(q), yv(Math.min(v, 1))]);
    }
    const l = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");
    const a = `M${PX0} ${PY0} ` + pts.map(([x, y]) => `L${x.toFixed(1)} ${y.toFixed(1)}`).join(" ") + ` L${PX1} ${PY0} Z`;
    return { line: l, area: a };
  }, [fwhm]);

  // rolling: the swept window over the middle ring + roll-step sampling ticks
  const winC = RPROF_CENTERS[1];
  const winX0 = xq(winC - ringWidth);
  const winX1 = xq(winC + ringWidth);
  const ticks = useMemo(() => {
    const out: number[] = [];
    for (let q = RPROF_QMIN; q <= RPROF_QMAX + 1e-6; q += 0.45) out.push(xq(q));
    return out;
  }, []);

  return (
    <svg className="rprof" viewBox="0 0 180 70" role="img"
         aria-label={`Ring(|Q|) radial profile, ${mode} mode`}>
      {rolling && <path d={area} className="rprof-fill" />}
      {rolling && (
        <rect x={winX0} y={PYT - 2} width={Math.max(2, winX1 - winX0)} height={PY0 - PYT + 2}
              className="rprof-win" />
      )}
      <line x1={PX0} y1={PY0} x2={PX1} y2={PY0} className="rprof-axis" />
      {rolling &&
        ticks.map((x, i) => (
          <line key={i} x1={x} y1={PY0} x2={x} y2={PY0 + 3} className="rprof-tick" />
        ))}
      {!rolling &&
        RPROF_CENTERS.map((c, i) => (
          <line key={i} x1={xq(c)} y1={PY0} x2={xq(c)} y2={PY0 + 3} className="rprof-tick" />
        ))}
      <path d={line} className="rprof-line" fill="none" />
      <text x={PX1} y={PY0 + 10} textAnchor="end" className="rprof-lbl">|Q|</text>
      <text x={PX0} y={PYT - 3} className="rprof-lbl">Ring(|Q|)</text>
    </svg>
  );
}

// The inner pie is split into `nPatches` equal sectors (the azimuthal bins);
// the surrounding ring is tinted by the highest texture harmonic
// T(φ) = cos(nFourier · φ), so the colour lobes count the Fourier order.
function RingTextureViz({
  nPatches,
  nFourier,
}: {
  nPatches: number;
  nFourier: number;
}) {
  const wedges = useMemo(() => {
    const out: string[] = [];
    for (let i = 0; i < nPatches; i++) {
      const a0 = (i / nPatches) * 2 * Math.PI + RTOP;
      const a1 = ((i + 1) / nPatches) * 2 * Math.PI + RTOP;
      const [x0, y0] = rpt(RPIE, a0);
      const [x1, y1] = rpt(RPIE, a1);
      out.push(`M${RC} ${RC} L${x0} ${y0} A${RPIE} ${RPIE} 0 0 1 ${x1} ${y1} Z`);
    }
    return out;
  }, [nPatches]);

  const segs = useMemo(() => {
    const n = Math.min(600, Math.max(180, nFourier * 12));
    const out: { d: string; c: string }[] = [];
    for (let i = 0; i < n; i++) {
      const a0 = (i / n) * 2 * Math.PI + RTOP;
      const a1 = ((i + 1) / n) * 2 * Math.PI + RTOP;
      const phi = ((i + 0.5) / n) * 2 * Math.PI;
      const [xo0, yo0] = rpt(ROUT, a0);
      const [xo1, yo1] = rpt(ROUT, a1);
      const [xi1, yi1] = rpt(RIN, a1);
      const [xi0, yi0] = rpt(RIN, a0);
      out.push({
        d: `M${xo0} ${yo0} A${ROUT} ${ROUT} 0 0 1 ${xo1} ${yo1} L${xi1} ${yi1} A${RIN} ${RIN} 0 0 0 ${xi0} ${yi0} Z`,
        c: textureColor(Math.cos(nFourier * phi)),
      });
    }
    return out;
  }, [nFourier]);

  return (
    <svg
      className="ring-viz"
      viewBox="0 0 180 180"
      role="img"
      aria-label={`${nPatches} azimuthal patches, texture order ${nFourier}`}
    >
      <g className="ring-viz-tex">
        {segs.map((s, i) => (
          <path key={i} d={s.d} fill={s.c} stroke={s.c} strokeWidth={0.8} />
        ))}
      </g>
      <g className="ring-viz-pie">
        {wedges.map((d, i) => (
          <path key={i} d={d} className={i % 2 ? "odd" : "even"} />
        ))}
      </g>
      <circle cx={RC} cy={RC} r={RHUB} className="ring-viz-hub" />
      <text
        x={RC}
        y={RC}
        textAnchor="middle"
        dominantBaseline="central"
        className="ring-viz-count"
      >
        {nPatches}
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Bragg-punch footprint illustration
//
// `PunchShapeViz` is a method dispatcher: the punch currently has a single
// algorithm (the anisotropic ellipsoid), but the registry seam below means a
// future method is just a new `case` + its own renderer + parameter group —
// the surrounding stage (lattice grid, axes, plane toggle, legend) is shared.
// ---------------------------------------------------------------------------

type HklAxis = "H" | "K" | "L";

const PUNCH_PLANES: {
  id: PunchPlane;
  title: string;
  xl: string;
  yl: string;
  cutAxis: HklAxis;
  cutLabel: string;
}[] = [
  { id: "hk", title: "a*–b*", xl: "a*", yl: "b*", cutAxis: "L", cutLabel: "c*" },
  { id: "hl", title: "a*–c*", xl: "a*", yl: "c*", cutAxis: "K", cutLabel: "b*" },
  { id: "kl", title: "b*–c*", xl: "b*", yl: "c*", cutAxis: "H", cutLabel: "a*" },
];

interface PunchGeom {
  // r0/r1/r2 are the Q-space footprint half-radii along a*, b*, c* in Å⁻¹.
  r0: number;
  r1: number;
  r2: number;
  margin: number;
  phiTail: number;
  mode: string;
  isQ: boolean;
  unit: string;
  ax: [string, string, string]; // axis labels, e.g. [H,K,L] or [a*,b*,c*]
  directBeamRadiiQ: [number, number, number];
  directBeamMargin: number;
  fitCovariance: boolean;
}

function planeHklAxes(p: PunchPlane): [HklAxis, HklAxis] {
  if (p === "hk") return ["H", "K"];
  if (p === "hl") return ["H", "L"];
  return ["K", "L"];
}

type LatticeLike = { a: number | null; b: number | null; c: number | null };
type Matrix3 = [[number, number, number], [number, number, number], [number, number, number]];

function axisLattice(axis: HklAxis, lattice?: LatticeLike): number | null | undefined {
  if (axis === "H") return lattice?.a;
  if (axis === "K") return lattice?.b;
  return lattice?.c;
}

function axisIndex(axis: HklAxis): number {
  if (axis === "H") return 0;
  if (axis === "K") return 1;
  return 2;
}

function axisRadiusQ(axis: HklAxis, geom: PunchGeom, lattice?: LatticeLike): number {
  const idx = axisIndex(axis);
  const r = idx === 0 ? geom.r0 : idx === 1 ? geom.r1 : geom.r2;
  if (geom.isQ) return r;
  const lat = axisLattice(axis, lattice);
  return lat ? r * (2 * Math.PI / lat) : r;
}

function asMatrix3(raw?: number[][]): Matrix3 | null {
  if (!raw || raw.length !== 3 || raw.some((row) => row.length !== 3)) return null;
  if (raw.some((row) => row.some((v) => !Number.isFinite(v)))) return null;
  return raw as Matrix3;
}

function qShapeMatrix(geom: PunchGeom, ubRaw?: number[][]): Matrix3 | null {
  const ub = asMatrix3(ubRaw);
  if (!ub) return null;
  const radii = [geom.r0, geom.r1, geom.r2];
  if (radii.some((r) => r <= 0 || !Number.isFinite(r))) return null;
  const unit: Matrix3 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let col = 0; col < 3; col++) {
    const norm = Math.hypot(ub[0][col], ub[1][col], ub[2][col]);
    if (norm <= 0 || !Number.isFinite(norm)) return null;
    for (let row = 0; row < 3; row++) unit[row][col] = ub[row][col] / norm;
  }

  const p: Matrix3 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      p[i][j] = unit[0][i] * ub[0][j] + unit[1][i] * ub[1][j] + unit[2][i] * ub[2][j];
    }
  }

  const a: Matrix3 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let j = 0; j < 3; j++) {
    for (let k = 0; k < 3; k++) {
      let value = 0;
      for (let i = 0; i < 3; i++) value += p[i][j] * (1 / (radii[i] * radii[i])) * p[i][k];
      a[j][k] = value;
    }
  }
  return a;
}

function ellipseFromQuadratic(
  a00: number,
  a01: number,
  a11: number,
  fallback: { rx: number; ry: number; angle: number },
): { rx: number; ry: number; angle: number } {
  if (![a00, a01, a11].every(Number.isFinite)) return fallback;
  const trace = a00 + a11;
  const root = Math.hypot(a00 - a11, 2 * a01);
  const lambdaMin = (trace - root) / 2;
  const lambdaMax = (trace + root) / 2;
  if (lambdaMin <= 0 || lambdaMax <= 0) return fallback;

  let vx: number;
  let vy: number;
  if (Math.abs(a01) > 1e-12) {
    vx = a01;
    vy = lambdaMin - a00;
  } else if (a00 <= a11) {
    vx = 1;
    vy = 0;
  } else {
    vx = 0;
    vy = 1;
  }
  const norm = Math.hypot(vx, vy) || 1;
  const angle = Math.atan2(vy / norm, vx / norm) * 180 / Math.PI;
  return { rx: 1 / Math.sqrt(lambdaMin), ry: 1 / Math.sqrt(lambdaMax), angle };
}

function qEllipseForPlane(
  plane: PunchPlane,
  geom: PunchGeom,
  lattice: LatticeLike,
  ubMatrix?: number[][],
): { rx: number; ry: number; angle: number } {
  const [axisX, axisY] = planeHklAxes(plane);
  const latX = axisLattice(axisX, lattice) ?? 1;
  const latY = axisLattice(axisY, lattice) ?? 1;
  const qScaleX = 2 * Math.PI / latX;
  const qScaleY = 2 * Math.PI / latY;
  const fallback = {
    rx: axisRadiusQ(axisX, geom, lattice),
    ry: axisRadiusQ(axisY, geom, lattice),
    angle: 0,
  };
  const shape = qShapeMatrix(geom, ubMatrix);
  if (!shape) return fallback;
  const ix = axisIndex(axisX);
  const iy = axisIndex(axisY);
  return ellipseFromQuadratic(
    shape[ix][ix] / (qScaleX * qScaleX),
    shape[ix][iy] / (qScaleX * qScaleY),
    shape[iy][iy] / (qScaleY * qScaleY),
    fallback,
  );
}

function axisRange(
  meta: {
    h_range: [number, number];
    k_range: [number, number];
    l_range: [number, number];
    shape: number[];
  } | null | undefined,
  axis: HklAxis,
): { min: number; max: number; step: number } {
  if (!meta) return { min: -1, max: 1, step: 0.01 };
  const idx = axis === "H" ? 0 : axis === "K" ? 1 : 2;
  const [min, max] = axis === "H" ? meta.h_range : axis === "K" ? meta.k_range : meta.l_range;
  const n = meta.shape[idx] ?? 1;
  return { min, max, step: n > 1 ? Math.abs(max - min) / (n - 1) : 0.01 };
}

function PunchDataOverlay({
  slice,
  geom,
  spec,
  lattice,
  ubMatrix,
  sourceLabel,
  zoom,
  vmax,
}: {
  slice: import("../api/types").Slice;
  geom: PunchGeom;
  spec: (typeof PUNCH_PLANES)[number];
  lattice: LatticeLike;
  ubMatrix?: number[][];
  sourceLabel?: string;
  zoom: number;
  vmax: number;
}) {
  const plane = spec.id;
  const [axisX, axisY] = planeHklAxes(plane);
  const braggGeom = geom.isQ
    ? {
        ...geom,
        r0: geom.r0 + geom.margin,
        r1: geom.r1 + geom.margin,
        r2: geom.r2 + geom.margin,
      }
    : geom;
  const braggEllipse = qEllipseForPlane(plane, braggGeom, lattice, ubMatrix);
  const latX = axisLattice(axisX, lattice) ?? 1;
  const latY = axisLattice(axisY, lattice) ?? 1;
  const qScaleX = 2 * Math.PI / latX;
  const qScaleY = 2 * Math.PI / latY;
  const qScaleH = 2 * Math.PI / (axisLattice("H", lattice) ?? 1);
  const qScaleK = 2 * Math.PI / (axisLattice("K", lattice) ?? 1);
  const qScaleL = 2 * Math.PI / (axisLattice("L", lattice) ?? 1);
  const baseHalfRlu = 4;
  const zoomSafe = Math.max(0.5, Math.min(4, zoom));
  const qHalf = (baseHalfRlu * Math.max(qScaleH, qScaleK, qScaleL)) / zoomSafe;
  const previewHalfX = qHalf / qScaleX;
  const previewHalfY = qHalf / qScaleY;
  const directBeamGeom = {
    ...geom,
    r0: geom.directBeamRadiiQ[0] + geom.directBeamMargin,
    r1: geom.directBeamRadiiQ[1] + geom.directBeamMargin,
    r2: geom.directBeamRadiiQ[2] + geom.directBeamMargin,
  };
  const ib = qEllipseForPlane(plane, directBeamGeom, lattice, ubMatrix);
  const showTail = !geom.isQ && plane === "kl" && geom.phiTail > 0;
  const showMargin = !geom.isQ && geom.margin > 0;
  const showSat = geom.mode !== "integer";
  const nodes: { x: number; y: number }[] = [];
  for (let x = Math.ceil(-previewHalfX); x <= Math.floor(previewHalfX); x++) {
    for (let y = Math.ceil(-previewHalfY); y <= Math.floor(previewHalfY); y++) {
      nodes.push({ x, y });
    }
  }

  return (
    <div className="punch-preview">
      <div className="punch-preview-frame">
        <SliceCanvas
          slice={slice}
          lut={COLORMAPS.inferno}
          vmax={vmax}
          log={false}
          windowX={previewHalfX}
          windowY={previewHalfY}
          size={260}
          reciprocalAxes
          latX={latX}
          latY={latY}
        />
        <svg
          className="punch-overlay"
          viewBox={`${-qHalf} ${-qHalf} ${2 * qHalf} ${2 * qHalf}`}
          preserveAspectRatio="none"
          role="img"
          aria-label={`Bragg punch profile over ${spec.title}`}
        >
          <g transform="scale(1, -1)">
            {showSat && (
              <g>
                <ellipse
                  cx={1.5 * qScaleX}
                  cy={-0.5 * qScaleY}
                  rx={braggEllipse.rx}
                  ry={braggEllipse.ry}
                  transform={`rotate(${braggEllipse.angle.toFixed(2)} ${1.5 * qScaleX} ${-0.5 * qScaleY})`}
                  className="punch-sat"
                  vectorEffect="non-scaling-stroke"
                />
              </g>
            )}
            {nodes.map(({ x, y }) => {
              const key = `${x},${y}`;
              const cx = x * qScaleX;
              const cy = y * qScaleY;
              if (x === 0 && y === 0) {
                return (
                  <g key={key}>
                    <ellipse
                      cx={cx}
                      cy={cy}
                      rx={ib.rx}
                      ry={ib.ry}
                      transform={`rotate(${ib.angle.toFixed(2)} ${cx} ${cy})`}
                      className="punch-ib"
                      vectorEffect="non-scaling-stroke"
                    />
                  </g>
                );
              }
              const ang = (Math.atan2(cy, cx) * 180) / Math.PI + 90;
              return (
                <g key={key}>
                  {showTail && (
                    <ellipse
                      cx={cx}
                      cy={cy}
                      rx={braggEllipse.rx}
                      ry={braggEllipse.ry}
                      transform={`rotate(${ang.toFixed(1)} ${cx} ${cy})`}
                      className="punch-tail"
                    />
                  )}
                  {showMargin && (
                    <ellipse
                      cx={cx}
                      cy={cy}
                      rx={braggEllipse.rx + geom.margin * qScaleX}
                      ry={braggEllipse.ry + geom.margin * qScaleY}
                      transform={`rotate(${braggEllipse.angle.toFixed(2)} ${cx} ${cy})`}
                      className="punch-margin"
                      vectorEffect="non-scaling-stroke"
                    />
                  )}
                  <ellipse
                    cx={cx}
                    cy={cy}
                    rx={braggEllipse.rx}
                    ry={braggEllipse.ry}
                    transform={`rotate(${braggEllipse.angle.toFixed(2)} ${cx} ${cy})`}
                    className="punch-fp"
                    vectorEffect="non-scaling-stroke"
                  />
                </g>
              );
            })}
          </g>
        </svg>
      </div>
      <div className="punch-preview-meta">
        {sourceLabel ?? "source"} · {slice.header.cut_label}
      </div>
    </div>
  );
}

function PunchPreviewGrid({
  geom,
  slices,
  loading,
  lattice,
  sourceLabel,
  zoom,
  contrast,
  meta,
  cuts,
  onCut,
}: {
  geom: PunchGeom;
  slices: Partial<Record<PunchPlane, import("../api/types").Slice>>;
  loading: Partial<Record<PunchPlane, boolean>>;
  lattice?: LatticeLike;
  sourceLabel?: string;
  zoom: number;
  contrast: number;
  meta?: {
    h_range: [number, number];
    k_range: [number, number];
    l_range: [number, number];
    shape: number[];
    ub_matrix?: number[][];
  };
  cuts: Record<HklAxis, number>;
  onCut: (axis: HklAxis, value: number) => void;
}) {
  const showSat = geom.mode !== "integer";
  const contrastSafe = Math.max(0.2, Math.min(6, contrast));
  const sharedVmax = Math.max(
    1,
    ...Object.values(slices).map((slice) => slice?.header.robust_max ?? 0),
  ) * contrastSafe;
  return (
    <div className="punch-preview-grid">
      {PUNCH_PLANES.map((spec) => {
        const range = axisRange(meta, spec.cutAxis);
        const cutValue = Math.max(range.min, Math.min(range.max, cuts[spec.cutAxis] ?? 0));
        const slice = slices[spec.id];
        return (
          <div className="punch-preview-panel" key={spec.id}>
            <div className="punch-preview-panel-head">
              <span>{spec.title}</span>
              <span>
                cut {spec.cutLabel}
              </span>
            </div>
            <Slider
              label={`${spec.cutLabel} cut`}
              readout={`${cutValue.toFixed(3)} r.l.u.`}
              min={range.min}
              max={range.max}
              step={range.step}
              value={cutValue}
              disabled={!meta}
              onChange={(v) => onCut(spec.cutAxis, Number(v.toFixed(4)))}
            />
            {slice && lattice ? (
              <PunchDataOverlay
                slice={slice}
                geom={geom}
                spec={spec}
                lattice={lattice}
                ubMatrix={meta?.ub_matrix}
                sourceLabel={sourceLabel}
                zoom={zoom}
                vmax={sharedVmax}
              />
            ) : (
              <div className="punch-preview-empty">
                {loading[spec.id] ? "Loading slice..." : "Slice unavailable"}
              </div>
            )}
            <div className="punch-legend">
              <span>
                <span className="sw fp" />Bragg
              </span>
              <span>
                <span className="sw ib" />beam
              </span>
              {showSat && (
                <span>
                  <span className="sw sat" />satellite
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// 3D-ΔPDF preview grid
//
// The three orthogonal raw-volume planes through the origin (a*–b*, a*–c*,
// b*–c*), each with the spherical |Q| transform band drawn as a shell — the
// real-space layout twin of the Bragg-punch preview grid.  The shared `|Q|`
// band slider lives above this grid (in the preview pane), not per panel,
// because the band is one radial window applied to the whole volume.
// ---------------------------------------------------------------------------
function PdfPreviewGrid({
  slices,
  loading,
  lattice,
  sourceReady,
  sourceLabel,
  bands,
  contrast,
}: {
  slices: Partial<Record<PunchPlane, import("../api/types").Slice>>;
  loading: Partial<Record<PunchPlane, boolean>>;
  lattice?: LatticeLike;
  sourceReady: boolean;
  sourceLabel?: string;
  bands: [number, number];
  contrast: number;
}) {
  const contrastSafe = Math.max(0.2, Math.min(6, contrast));
  const sharedVmax = Math.max(
    1,
    ...Object.values(slices).map((slice) => slice?.header.robust_max ?? 0),
  ) * contrastSafe;
  return (
    <div className="punch-preview-grid">
      {PUNCH_PLANES.map((spec) => {
        const slice = slices[spec.id];
        const [axisX, axisY] = planeHklAxes(spec.id);
        const latX = (lattice && axisLattice(axisX, lattice)) || 1;
        const latY = (lattice && axisLattice(axisY, lattice)) || 1;
        const latCut = (lattice && axisLattice(spec.cutAxis, lattice)) || 1;
        const qScaleX = (2 * Math.PI) / latX;
        const qScaleY = (2 * Math.PI) / latY;
        // Crop each tile to an origin-centred square |Q| window so it fills the
        // frame with the shell circular (equal Å⁻¹ per axis) — the Bragg-punch
        // preview does the same.  The window is the largest square that fits the
        // plane's data, capped so the |Q| band's outer radius sits just inside.
        let windowX: number | undefined;
        let windowY: number | undefined;
        if (slice) {
          const xs = slice.header.x_axis;
          const ys = slice.header.y_axis;
          const qFit = Math.min(
            Math.max(Math.abs(xs[0]), Math.abs(xs[xs.length - 1])) * qScaleX,
            Math.max(Math.abs(ys[0]), Math.abs(ys[ys.length - 1])) * qScaleY,
          );
          const qOuter = bands[1] > 0 ? bands[1] : qFit;
          const qHalf = Math.max(0.1, Math.min(qOuter * 1.12, qFit));
          windowX = qHalf / qScaleX;
          windowY = qHalf / qScaleY;
        }
        return (
          <div className="punch-preview-panel" key={spec.id}>
            <div className="punch-preview-panel-head">
              <span>{spec.title}</span>
              <span>cut {spec.cutLabel} = 0</span>
            </div>
            {slice && lattice ? (
              <div className="punch-preview">
                <div className="pdf-slice-frame">
                  <SliceCanvas
                    slice={slice}
                    lut={COLORMAPS.inferno}
                    vmax={sharedVmax}
                    log={false}
                    windowX={windowX}
                    windowY={windowY}
                    bands={bands}
                    cutDistance={0}
                    reciprocalAxes
                    latX={latX}
                    latY={latY}
                    latCut={latCut}
                  />
                </div>
                <div className="punch-preview-meta">
                  {sourceLabel ?? "source"} · {slice.header.cut_label}
                </div>
              </div>
            ) : (
              <div className="punch-preview-empty">
                {loading[spec.id]
                  ? "Loading slice…"
                  : sourceReady
                    ? "Slice unavailable"
                    : "Raw volume needed"}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------

export function PipelineConfig({ onStarted }: { onStarted: () => void }) {
  const datasetsQ = useDatasets();
  const dataRootQ = useDataRoot();
  const queryClient = useQueryClient();
  const datasets = useMemo(() => datasetsQ.data ?? [], [datasetsQ.data]);
  useInitializeDataset(datasets);
  const datasetId = useDatasetStore((st) => st.datasetId);
  const setDataset = useDatasetStore((st) => st.setDataset);
  const resetDataset = useDatasetStore((st) => st.resetDataset);
  const selectedDataset = datasets.find((d) => d.id === datasetId);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const [rootDraft, setRootDraft] = useState("");
  const [rootError, setRootError] = useState<string | null>(null);
  const [rootNote, setRootNote] = useState<string | null>(null);

  useEffect(() => {
    if (dataRootQ.data?.data_root) setRootDraft(dataRootQ.data.data_root);
  }, [dataRootQ.data?.data_root]);

  const s = usePipelineStore(
    useShallow((st) => ({
      flatten: st.flatten,
      force: st.force,
      ringModel: st.ringModel,
      ringRadialMode: st.ringRadialMode,
      ringNPatches: st.ringNPatches,
      ringNFourier: st.ringNFourier,
      ringSliceAxis: st.ringSliceAxis,
      ringWidth: st.ringWidth,
      punchMinI: st.punchMinI,
      punchMethod: st.punchMethod,
      punchMode: st.punchMode,
      punchQA: st.punchQA,
      punchQB: st.punchQB,
      punchQC: st.punchQC,
      punchFitCovariance: st.punchFitCovariance,
      punchMargin: st.punchMargin,
      punchPhiTail: st.punchPhiTail,
      incidentBeamQA: st.incidentBeamQA,
      incidentBeamQB: st.incidentBeamQB,
      incidentBeamQC: st.incidentBeamQC,
      incidentBeamMargin: st.incidentBeamMargin,
      punchSliceZoom: st.punchSliceZoom,
      punchSliceContrast: st.punchSliceContrast,
      punchCutH: st.punchCutH,
      punchCutK: st.punchCutK,
      punchCutL: st.punchCutL,
      backfillMethod: st.backfillMethod,
      flattenEstimator: st.flattenEstimator,
      pdfApod: st.pdfApod,
      pdfQMin: st.pdfQMin,
      pdfQMax: st.pdfQMax,
      running: st.running,
    })),
  );
  const patch = usePipelineStore((st) => st.patch);
  const run = usePipelineStore((st) => st.run);

  const vizPatches = clampInt(s.ringNPatches, 36, 4, 96);
  const vizFourier = clampInt(s.ringNFourier, 6, 0, 40);
  const vizRingWidth = clampFloat(s.ringWidth, 0.24, 0.02, 1.0);
  const punchGeom: PunchGeom = {
    r0: clampFloat(s.punchQA, 0.097, 0.005, 0.6),
    r1: clampFloat(s.punchQB, 0.072, 0.005, 0.6),
    r2: clampFloat(s.punchQC, 0.115, 0.005, 0.6),
    margin: clampFloat(s.punchMargin, 0.02, 0, 0.5),
    phiTail: clampFloat(s.punchPhiTail, 0.12, 0, 1.0),
    mode: s.punchMode || "both",
    isQ: true,
    unit: "Å⁻¹",
    ax: ["a*", "b*", "c*"],
    directBeamRadiiQ: [
      clampFloat(s.incidentBeamQA, 0.16, 0.005, 2.0),
      clampFloat(s.incidentBeamQB, 0.30, 0.005, 2.0),
      clampFloat(s.incidentBeamQC, 0.25, 0.005, 2.0),
    ],
    directBeamMargin: clampFloat(s.incidentBeamMargin, 0.0, 0, 1.0),
    fitCovariance: s.punchFitCovariance,
  };
  const pdfPreviewStage = useMemo(() => {
    if (!selectedDataset) return undefined;
    return selectedDataset.stages.find((stage) => stage.name === "raw" && stage.exists);
  }, [selectedDataset]);
  const punchPreviewStage = useMemo(() => {
    if (!selectedDataset) return undefined;
    return (
      selectedDataset.stages.find((stage) => stage.name === "ringremoved" && stage.exists) ??
      selectedDataset.stages.find((stage) => stage.name === "raw" && stage.exists)
    );
  }, [selectedDataset]);
  const punchInputId = punchPreviewStage?.volume_id;
  const punchMetaQ = useQuery({
    queryKey: ["meta", punchInputId, "config-punch"],
    queryFn: () => fetchMeta(punchInputId as string),
    enabled: Boolean(punchInputId),
  });
  const punchCuts: Record<HklAxis, number> = {
    H: s.punchCutH,
    K: s.punchCutK,
    L: s.punchCutL,
  };
  const punchSliceQueries = useQueries({
    queries: PUNCH_PLANES.map((spec) => {
      const cut = punchCuts[spec.cutAxis] ?? 0;
      return {
        queryKey: ["slice", punchInputId, spec.id, cut, false, "config-punch"],
        queryFn: () => fetchSlice(punchInputId as string, spec.id, cut, false),
        enabled: Boolean(punchInputId),
      };
    }),
  });
  const punchSlices = Object.fromEntries(
    PUNCH_PLANES.map((spec, i) => [spec.id, punchSliceQueries[i]?.data]),
  ) as Partial<Record<PunchPlane, import("../api/types").Slice>>;
  const punchLoading = Object.fromEntries(
    PUNCH_PLANES.map((spec, i) => [spec.id, Boolean(punchInputId) && punchSliceQueries[i]?.isFetching]),
  ) as Partial<Record<PunchPlane, boolean>>;
  const pdfInputId = pdfPreviewStage?.volume_id;
  const pdfMetaQ = useQuery({
    queryKey: ["meta", pdfInputId, "config-pdf"],
    queryFn: () => fetchMeta(pdfInputId as string),
    enabled: Boolean(pdfInputId),
  });
  // Three orthogonal raw-volume planes (a*–b*, a*–c*, b*–c*) through the
  // origin, mirroring the Bragg-punch preview grid.
  const pdfSliceQueries = useQueries({
    queries: PUNCH_PLANES.map((spec) => ({
      queryKey: ["slice", pdfInputId, spec.id, 0, false, "config-pdf"],
      queryFn: () => fetchSlice(pdfInputId as string, spec.id, 0, false),
      enabled: Boolean(pdfInputId),
    })),
  });
  const pdfSlices = Object.fromEntries(
    PUNCH_PLANES.map((spec, i) => [spec.id, pdfSliceQueries[i]?.data]),
  ) as Partial<Record<PunchPlane, import("../api/types").Slice>>;
  const pdfLoading = Object.fromEntries(
    PUNCH_PLANES.map((spec, i) => [spec.id, Boolean(pdfInputId) && pdfSliceQueries[i]?.isFetching]),
  ) as Partial<Record<PunchPlane, boolean>>;
  const pdfQSpanMax = Math.ceil(qSpanFromMeta(pdfMetaQ.data) * 20) / 20;
  const pdfQMin = s.pdfQMin ? Number(s.pdfQMin) : 0;
  const pdfQMax = s.pdfQMax ? Number(s.pdfQMax) : pdfQSpanMax;
  const pdfQBandIsFull = !s.pdfQMin && !s.pdfQMax;

  const updatePdfQBand = (lo: number, hi: number) => {
    if (!pdfQSpanMax) return;
    const nextLo = Math.max(0, Math.min(lo, pdfQSpanMax));
    const nextHi = Math.max(nextLo, Math.min(hi, pdfQSpanMax));
    const isFull = nextLo <= 0 && Math.abs(nextHi - pdfQSpanMax) < 0.025;
    patch({
      pdfQMin: isFull ? "" : nextLo.toFixed(2),
      pdfQMax: isFull ? "" : nextHi.toFixed(2),
    });
  };

  const onRun = async () => {
    await run();
    onStarted();
  };

  const applyDataRoot = async () => {
    const nextRoot = rootDraft.trim();
    if (!nextRoot) return;
    setRootError(null);
    setRootNote(null);
    try {
      const next = await setDataRoot(nextRoot);
      setRootDraft(next.data_root);
      resetDataset();
      await queryClient.invalidateQueries();
      setRootNote(`${next.n_datasets} datasets`);
    } catch (e) {
      setRootError((e as Error).message);
    }
  };

  const onBrowseFolder = async () => {
    setRootError(null);
    setRootNote("Opening folder picker…");
    try {
      const next = await browseDataRoot();
      setRootDraft(next.data_root);
      resetDataset();
      await queryClient.invalidateQueries();
      setRootNote(`${next.n_datasets} datasets`);
    } catch (e) {
      const message = (e as Error).message;
      if (message.startsWith("409 ")) {
        setRootNote(null);
        return;
      }
      if (
        message.startsWith("404 ") ||
        message.startsWith("405 ") ||
        message.includes("Failed to fetch")
      ) {
        setRootNote(null);
        folderInputRef.current?.click();
        return;
      }
      setRootNote(null);
      setRootError(message);
    }
  };

  const onFolderPicked = (files: FileList | null) => {
    const first = files?.[0] as (File & { path?: string }) | undefined;
    if (!first) return;
    const rel = first.webkitRelativePath;
    const folderName = rel.split("/")[0] || first.name;
    if (first.path && rel && first.path.endsWith(rel)) {
      const root = first.path.slice(0, first.path.length - rel.length).replace(/[\\/]$/, "");
      setRootDraft(root);
      setRootNote(`Selected ${folderName}`);
      setRootError(null);
    } else {
      setRootNote(`Selected ${folderName}`);
      setRootError("This browser cannot share the full folder path. Paste it, then apply.");
    }
    if (folderInputRef.current) folderInputRef.current.value = "";
  };

  const stageStatus = DATASET_STAGE_BADGES.map((stage) => ({
    ...stage,
    exists: selectedDataset?.stages.some((s) => s.name === stage.key && s.exists) ?? false,
  }));
  const availableStageCount = stageStatus.filter((stage) => stage.exists).length;
  const missingStageCount = stageStatus.length - availableStageCount;
  const rootStatus = rootError
    ? rootError
    : rootNote ??
      (dataRootQ.data
        ? `${dataRootQ.data.raw_exists ? "raw" : "no raw"} · ${dataRootQ.data.processed_exists ? "processed" : "no processed"} · ${dataRootQ.data.n_datasets} datasets`
        : " ");
  const readinessLabel = selectedDataset
    ? missingStageCount === 0
      ? "All expected files available"
      : `${availableStageCount}/${stageStatus.length} files available`
    : "No dataset selected";

  return (
    <div className="config-page">
      {/* ----------------------------------------------------------- data card */}
      <section className="card data-card">
        <div className="data-card-head">
          <div className="data-card-title">
            <h3>DATA</h3>
            <span className="data-card-subtitle">{readinessLabel}</span>
          </div>
          <div className="dataset-stage-board" aria-label="Available dataset files">
            {stageStatus.map((stage) => (
              <div
                key={stage.key}
                className={`dataset-stage-item ${stage.exists ? "ok" : "missing"}`}
                title={`${stage.group} · ${stage.label}: ${stage.exists ? "available" : "missing"}`}
              >
                <span className="dataset-stage-dot" />
                <span className="dataset-stage-label">{stage.label}</span>
                <span className="dataset-stage-state">
                  {stage.exists ? "available" : "missing"}
                </span>
              </div>
            ))}
          </div>
          <div className="data-actions">
            <Switch
              label="Force"
              checked={s.force}
              onChange={(v) => patch({ force: v })}
            />
            <button
              type="button"
              className="btn btn-primary"
              onClick={onRun}
              disabled={s.running || !datasetId}
            >
              {s.running && <span className="spin" />}
              {s.running ? "Running…" : "Run pipeline"}
            </button>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={onStarted}
              disabled={!s.running}
            >
              View execution →
            </button>
          </div>
        </div>

        <div className="data-card-grid">
          <div className="data-source-panel">
            <Field label="Data folder" grow>
              <div className="data-root-row">
                <input
                  type="text"
                  value={rootDraft}
                  placeholder="/path/to/data"
                  onChange={(e) => {
                    setRootDraft(e.target.value);
                    setRootError(null);
                    setRootNote(null);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") void applyDataRoot();
                  }}
                />
                <input
                  ref={folderInputRef}
                  type="file"
                  className="visually-hidden"
                  // @ts-expect-error Chromium directory picker attribute.
                  webkitdirectory=""
                  directory=""
                  multiple
                  onChange={(e) => onFolderPicked(e.target.files)}
                />
                <button
                  type="button"
                  className="btn btn-ghost"
                  onClick={() => void onBrowseFolder()}
                >
                  Browse
                </button>
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={!rootDraft.trim() || dataRootQ.isFetching}
                  onClick={() => void applyDataRoot()}
                >
                  Apply
                </button>
              </div>
              <span className={`data-root-status${rootError ? " error" : ""}`}>
                {rootStatus}
              </span>
            </Field>
          </div>

          <div className="dataset-panel">
            <Field label="Dataset" grow>
              <select
                value={datasetId ?? ""}
                onChange={(e) => setDataset(e.target.value)}
              >
                {datasets.map((d) => (
                  <option key={d.id} value={d.id} title={d.raw_name}>
                    {d.temperature ?? d.stem}
                  </option>
                ))}
              </select>
            </Field>
            <div className="dataset-meta">
              <span>{selectedDataset?.raw_name ?? "No raw file selected"}</span>
              <span>{datasets.length} datasets in folder</span>
            </div>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------- per-stage cards */}
      <div className="config-cards">
        <StageCard title={STAGE_LABELS.rings} step={STAGE_NO.rings} className="stage-card-wide">
          <div className="config-grid">
            <Field label="Model">
              <select
                value={s.ringModel}
                title="Patched: non-parametric per-azimuthal-patch radial subtraction. Parametric: separable pseudo-Voigt(|Q|) × per-ring Fourier texture, fit from thin radial shells + binning-free azimuthal LS (statistics don't vary with |Q|)."
                onChange={(e) => patch({ ringModel: e.target.value })}
              >
                <option value="patched">Patched (per-patch)</option>
                <option value="parametric">Parametric (pseudo-Voigt)</option>
              </select>
            </Field>
            <Field label="Slice axis">
              <select
                value={s.ringSliceAxis}
                title="Axis sliced over when fitting the powder rings plane-by-plane"
                onChange={(e) => patch({ ringSliceAxis: e.target.value })}
              >
                <option value="H">H · fit 0kl planes</option>
                <option value="K">K · fit h0l planes</option>
                <option value="L">L · fit hk0 planes</option>
              </select>
            </Field>
          </div>
          {s.ringModel === "parametric" && (
            <div className="config-grid">
              <Field label="Radial mode">
                <select
                  value={s.ringRadialMode}
                  title="Rolling: a thick window swept Qmin→Qmax fits a continuous Ring(|Q|) × per-shell texture (no discrete peaks; thicker = smoother). Peaks: detect discrete rings and fit a pseudo-Voigt each."
                  onChange={(e) => patch({ ringRadialMode: e.target.value })}
                >
                  <option value="rolling">Rolling (continuous)</option>
                  <option value="peaks">Peaks (pseudo-Voigt)</option>
                </select>
              </Field>
              <Field
                label={s.ringRadialMode === "rolling" ? "Window (Å⁻¹)" : "Ring width (Å⁻¹)"}
              >
                <input
                  type="number"
                  min="0.02"
                  step="0.02"
                  placeholder="0.24"
                  value={s.ringWidth}
                  title={
                    s.ringRadialMode === "rolling"
                      ? "Rolling-window half-width in |Q| (Å⁻¹). Thicker = more azimuthal voxels per shell = smoother texture."
                      : "Max powder-ring full width / SNIP baseline window in |Q| (Å⁻¹). Broader features are kept as diffuse."
                  }
                  onChange={(e) => patch({ ringWidth: e.target.value })}
                />
              </Field>
            </div>
          )}
          <div className="config-grid">
            {s.ringModel === "patched" && (
              <Field label="Patches (n)">
                <input
                  type="number"
                  min="4"
                  step="1"
                  placeholder="36"
                  value={s.ringNPatches}
                  title="Number of azimuthal patches the powder rings are divided into"
                  onChange={(e) => patch({ ringNPatches: e.target.value })}
                />
              </Field>
            )}
            <Field label="Fourier order">
              <input
                type="number"
                min="0"
                step="1"
                placeholder="6"
                value={s.ringNFourier}
                title="Fourier order of the azimuthal texture T(φ) modelling the powder rings"
                onChange={(e) => patch({ ringNFourier: e.target.value })}
              />
            </Field>
          </div>
          <div className="stage-visual">
            {s.ringModel === "parametric" ? (
              <>
                <RadialProfileViz mode={s.ringRadialMode} ringWidth={vizRingWidth} />
                <ParametricRingViz
                  nFourier={vizFourier}
                  radialMode={s.ringRadialMode}
                  ringWidth={vizRingWidth}
                />
              </>
            ) : (
              <RingTextureViz nPatches={vizPatches} nFourier={vizFourier} />
            )}
            <div className="ring-viz-cap">
              {s.ringModel === "parametric" ? (
                s.ringRadialMode === "rolling" ? (
                  <>
                    continuous Ring(|Q|) swept Qmin→Qmax · texture order{" "}
                    <b>{vizFourier}</b>
                  </>
                ) : (
                  <>
                    pseudo-Voigt(|Q|) × per-ring texture order <b>{vizFourier}</b>
                  </>
                )
              ) : (
                <>
                  pie = <b>{vizPatches}</b> azimuthal patches · ring = texture order{" "}
                  <b>{vizFourier}</b>
                </>
              )}
            </div>
          </div>
        </StageCard>

        <StageCard title={STAGE_LABELS.punch} step={STAGE_NO.punch} className="stage-card-wide punch-stage-card">
          <div className="punch-workspace">
            <div className="punch-controls">
              <div className="config-grid-3 punch-basis">
                <Field label="Method">
                  <select
                    value={s.punchMethod}
                    title="Bragg-punch algorithm (more shapes coming)"
                    onChange={(e) => patch({ punchMethod: e.target.value })}
                  >
                    <option value="ellipsoid">Ellipsoid</option>
                  </select>
                </Field>
                <Field label="Min I">
                  <input
                    type="number"
                    step="0.1"
                    placeholder="0.8"
                    value={s.punchMinI}
                    title="Minimum intensity above background for a voxel to be punched as Bragg"
                    onChange={(e) => patch({ punchMinI: e.target.value })}
                  />
                </Field>
                <Field label="Mode">
                  <select
                    value={s.punchMode}
                    onChange={(e) => patch({ punchMode: e.target.value })}
                  >
                    <option value="">both (default)</option>
                    <option value="integer">integer</option>
                    <option value="search">search</option>
                    <option value="both">both</option>
                  </select>
                </Field>
              </div>

              <div className="punch-group">
                <div className="punch-group-head">
                  <span className="punch-group-title">Bragg footprint</span>
                  <span className="punch-group-unit">Å⁻¹</span>
                  <HelpTip>
                    Punch half-radii along a*, b*, c* in Q-space. The run request
                    always uses Q-space; blank fields use the validated defaults
                    (0.097, 0.072, 0.115).
                  </HelpTip>
                </div>
                <div className="config-grid-3">
                  <Field label={<>r<sub>a*</sub></>}>
                    <input
                      type="number"
                      step="0.005"
                      min="0"
                      placeholder="0.097"
                      value={s.punchQA}
                      title="Punch half-radius along a* (Å⁻¹)"
                      onChange={(e) => patch({ punchQA: e.target.value })}
                    />
                  </Field>
                  <Field label={<>r<sub>b*</sub></>}>
                    <input
                      type="number"
                      step="0.005"
                      min="0"
                      placeholder="0.072"
                      value={s.punchQB}
                      title="Punch half-radius along b* (Å⁻¹)"
                      onChange={(e) => patch({ punchQB: e.target.value })}
                    />
                  </Field>
                  <Field label={<>r<sub>c*</sub></>}>
                    <input
                      type="number"
                      step="0.005"
                      min="0"
                      placeholder="0.115"
                      value={s.punchQC}
                      title="Punch half-radius along c* (Å⁻¹)"
                      onChange={(e) => patch({ punchQC: e.target.value })}
                    />
                  </Field>
                </div>
                <div className="config-grid">
                  <Field label="Margin">
                    <input
                      type="number"
                      step="0.005"
                      min="0"
                      placeholder="0.02"
                      value={s.punchMargin}
                      title="Q-space guard band added to every punch half-radius (Å⁻¹)"
                      onChange={(e) => patch({ punchMargin: e.target.value })}
                    />
                  </Field>
                  <Field label="φ-tail (K–L)">
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      placeholder="0.12"
                      value={s.punchPhiTail}
                      title="Extra K–L tangential half-width along the powder-ring φ direction"
                      onChange={(e) => patch({ punchPhiTail: e.target.value })}
                    />
                  </Field>
                </div>
                <div className="switch-row">
                  <Switch
                    label="Fit tilted ellipsoid (covariance)"
                    checked={s.punchFitCovariance}
                    onChange={(v) => patch({ punchFitCovariance: v })}
                  />
                  <HelpTip>
                    Fit a tilted 3×3 ellipsoid to each Bragg peak during punching
                    and fold the φ-tail into it. The preview uses the exact UB-derived
                    Q-space floor; the per-peak covariance tilt is fitted from data
                    during the run and is best checked in the punched slices.
                  </HelpTip>
                </div>
              </div>

              <div className="punch-group">
                <div className="punch-group-head">
                  <span className="punch-group-title">Direct beam</span>
                  <span className="punch-group-unit">Å⁻¹</span>
                  <HelpTip>
                    Origin-centered incident/direct-beam ellipsoid in Q-space,
                    using half-radii along a*, b*, c*. The backend converts these
                    through UB, matching the Bragg footprint geometry.
                  </HelpTip>
                </div>
                <div className="config-grid-3">
                  <Field label={<>r<sub>a*</sub></>}>
                    <input
                      type="number"
                      step="0.005"
                      min="0"
                      placeholder="0.16"
                      value={s.incidentBeamQA}
                      title="Direct-beam half-radius along a* (Å⁻¹)"
                      onChange={(e) => patch({ incidentBeamQA: e.target.value })}
                    />
                  </Field>
                  <Field label={<>r<sub>b*</sub></>}>
                    <input
                      type="number"
                      step="0.005"
                      min="0"
                      placeholder="0.30"
                      value={s.incidentBeamQB}
                      title="Direct-beam half-radius along b* (Å⁻¹)"
                      onChange={(e) => patch({ incidentBeamQB: e.target.value })}
                    />
                  </Field>
                  <Field label={<>r<sub>c*</sub></>}>
                    <input
                      type="number"
                      step="0.005"
                      min="0"
                      placeholder="0.25"
                      value={s.incidentBeamQC}
                      title="Direct-beam half-radius along c* (Å⁻¹)"
                      onChange={(e) => patch({ incidentBeamQC: e.target.value })}
                    />
                  </Field>
                </div>
                <div className="config-grid">
                  <Field label="Margin">
                    <input
                      type="number"
                      step="0.005"
                      min="0"
                      placeholder="0.00"
                      value={s.incidentBeamMargin}
                      title="Q-space guard band added to every direct-beam half-radius (Å⁻¹)"
                      onChange={(e) => patch({ incidentBeamMargin: e.target.value })}
                    />
                  </Field>
                </div>
              </div>
            </div>
            <div className="stage-visual punch-preview-pane">
              <div className="punch-preview-controls">
                <Slider
                  label="Preview zoom"
                  readout={`${s.punchSliceZoom.toFixed(1)}×`}
                  min={0.5}
                  max={4}
                  step={0.1}
                  value={s.punchSliceZoom}
                  onChange={(v) => patch({ punchSliceZoom: Number(v.toFixed(1)) })}
                />
                <Slider
                  label="Contrast"
                  readout={`${s.punchSliceContrast.toFixed(1)}×`}
                  min={0.2}
                  max={6}
                  step={0.1}
                  value={s.punchSliceContrast}
                  onChange={(v) => patch({ punchSliceContrast: Number(v.toFixed(1)) })}
                />
              </div>
              <PunchPreviewGrid
                geom={punchGeom}
                slices={punchSlices}
                loading={punchLoading}
                lattice={punchMetaQ.data?.lattice}
                sourceLabel={punchPreviewStage?.name}
                zoom={s.punchSliceZoom}
                contrast={s.punchSliceContrast}
                meta={punchMetaQ.data}
                cuts={punchCuts}
                onCut={(axis, value) => {
                  if (axis === "H") patch({ punchCutH: value });
                  else if (axis === "K") patch({ punchCutK: value });
                  else patch({ punchCutL: value });
                }}
              />
              <div className="ring-viz-cap">
                r = (<b>{punchGeom.r0}</b>, <b>{punchGeom.r1}</b>, <b>{punchGeom.r2}</b>){" "}
                {punchGeom.unit} along {punchGeom.ax.join("/")} · margin{" "}
                <b>{punchGeom.margin}</b> · φ-tail <b>{punchGeom.phiTail}</b> · mode{" "}
                <b>{punchGeom.mode}</b>
              </div>
            </div>
          </div>
        </StageCard>

        <StageCard
          title={STAGE_LABELS.backfill}
          step={STAGE_NO.backfill}
          className="stage-card-compact"
        >
          <Field label="Method">
            <select
              value={s.backfillMethod}
              onChange={(e) => patch({ backfillMethod: e.target.value })}
            >
              <option value="">q_shell (default)</option>
              <option value="local">local</option>
              <option value="tv">tv</option>
              <option value="symmetry+tv">symmetry+tv</option>
            </select>
          </Field>
        </StageCard>

        <StageCard
          title={STAGE_LABELS.flatten}
          step={STAGE_NO.flatten}
          className="stage-card-compact"
        >
          <Switch
            label="Enable stage"
            checked={s.flatten}
            onChange={(v) => patch({ flatten: v })}
          />
          <Field label="Estimator">
            <select
              value={s.flattenEstimator}
              disabled={!s.flatten}
              onChange={(e) => patch({ flattenEstimator: e.target.value })}
            >
              <option value="">floor (default)</option>
              <option value="median">median</option>
              <option value="mode">mode</option>
              <option value="snip">snip</option>
            </select>
          </Field>
        </StageCard>

        <StageCard
          title={STAGE_LABELS.pdf}
          step={STAGE_NO.pdf}
          className="stage-card-wide pdf-stage-card"
        >
          <div className="punch-workspace">
            <div className="punch-controls">
              <div className="punch-group">
                <div className="punch-group-head">
                  <span className="punch-group-title">Transform</span>
                  <HelpTip>
                    3D-FFT of the cleaned, backfilled diffuse volume. The
                    apodization window tapers Q-space before the transform to
                    suppress real-space termination ripples.
                  </HelpTip>
                </div>
                <div className="config-grid">
                  <Field label="Apodization">
                    <select
                      value={s.pdfApod}
                      title="Q-space window applied before the FFT to suppress termination ripples"
                      onChange={(e) => patch({ pdfApod: e.target.value })}
                    >
                      <option value="">gaussian (default)</option>
                      <option value="hann">hann</option>
                      <option value="none">none</option>
                    </select>
                  </Field>
                </div>
              </div>
              <p className="pdf-q-note">
                Previews show the raw reciprocal-space volume; the ΔPDF runs on
                the cleaned, backfilled data. The outlined shell marks the |Q|
                band the transform keeps.
              </p>
            </div>
            <div className="stage-visual punch-preview-pane">
              <div className="punch-preview-controls pdf-preview-controls">
                <RangeSlider
                  grow
                  label="|Q| band"
                  readout={
                    pdfQBandIsFull
                      ? `full 0.00 … ${(pdfQSpanMax || 0).toFixed(2)} Å⁻¹`
                      : `${pdfQMin.toFixed(2)} … ${pdfQMax.toFixed(2)} Å⁻¹`
                  }
                  min={0}
                  max={pdfQSpanMax || 1}
                  step={0.05}
                  valueMin={pdfQMin}
                  valueMax={pdfQMax || 1}
                  disabled={!pdfQSpanMax}
                  onChange={updatePdfQBand}
                />
              </div>
              <PdfPreviewGrid
                slices={pdfSlices}
                loading={pdfLoading}
                lattice={pdfMetaQ.data?.lattice}
                sourceReady={Boolean(pdfInputId)}
                sourceLabel={pdfPreviewStage?.name}
                bands={[pdfQMin, pdfQMax]}
                contrast={1.5}
              />
              <div className="ring-viz-cap">
                source <b>{pdfPreviewStage?.name ?? "none"}</b> · transform band{" "}
                <b>
                  {pdfQBandIsFull
                    ? "full"
                    : `${pdfQMin.toFixed(2)}–${pdfQMax.toFixed(2)} Å⁻¹`}
                </b>
              </div>
            </div>
          </div>
        </StageCard>
      </div>
    </div>
  );
}
