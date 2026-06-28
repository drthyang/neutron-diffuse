// Pipeline configuration page — the form + live algorithm illustrations.  All
// values live in the pipeline store, so navigating away and back never resets
// them; pressing Run kicks off the job (also in the store) and jumps to the
// Execution page via `onStarted`.

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { useShallow } from "zustand/react/shallow";

import { browseDataRoot, fetchMeta, fetchSlice, setDataRoot } from "../api/client";
import {
  engine,
  getBootStatus,
  PYODIDE_MODE,
  subscribeBoot,
  type BootStatus,
} from "../api/pyodideEngine";
import { useDataRoot, useDatasets } from "../api/hooks";
import { COLORMAPS } from "../colormaps/luts";
import { SliceCanvas } from "../components/SliceCanvas";
import { Field, HelpTip, RangeSlider, Slider, Switch } from "../components/ui";
import { useDatasetStore, useInitializeDataset } from "../state/datasetStore";
import {
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

// Half-extent (in r.l.u. along the coarsest reciprocal axis) of the orthoslice
// preview window at zoom = 1.  Shared by the Bragg-punch and 3D-ΔPDF preview
// grids so both render the same isotropic Q window per plane — at equal zoom the
// a*–b* / a*–c* / b*–c* tiles line up peak-for-peak between the two cards.
const PREVIEW_BASE_HALF_RLU = 4;

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
  step?: number | string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section className={`card stage-card ${className}`}>
      <div className="card-title-row">
        {step != null && <span className="config-step-no">{step}</span>}
        <h3>{title}</h3>
        <span className="card-title-rule" />
      </div>
      <div className="card-body">{children}</div>
    </section>
  );
}

// Floating header for one workflow stage: number chip + title, an optional help
// bubble, and the per-stage "Enable stage" toggle on the right.  Disabling a
// stage skips it — its input passes through to the next enabled stage.
function StageHead({
  no,
  title,
  enabled,
  onToggle,
  children,
}: {
  no: number | string;
  title: string;
  enabled: boolean;
  onToggle: (b: boolean) => void;
  children?: ReactNode;
}) {
  return (
    <div className="cfg-stage-head">
      <span className="punch-group-no">{no}</span>
      <span className="punch-group-title">{title}</span>
      {children}
      <div className="cfg-stage-toggle">
        <Switch label="Enable stage" checked={enabled} onChange={onToggle} />
      </div>
    </div>
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

// Closed-form of the fitted ring model, mirroring the actual computation:
//   parametric → nebula3d.preprocessing.parametric_ring (Iᵣ = Σᵢ PVᵢ(|Q|)·Tᵢ(φ))
//   patched    → nebula3d.preprocessing.radial_background (per-patch radial subtraction)
function RingEquation({
  model,
  nFourier,
  nPatches,
}: {
  model: string;
  nFourier: number;
  nPatches: number;
}) {
  if (model === "parametric") {
    return (
      <div className="ring-eqn">
        <div className="eq">
          <i>I</i>
          <sub>ring</sub>(|<i>Q</i>|,&nbsp;φ) = <span className="op">Σ</span>
          <sub>i</sub> <span className="op">PV</span>
          <sub>i</sub>(|<i>Q</i>|)&nbsp;·&nbsp;<i>T</i>
          <sub>i</sub>(φ)
        </div>
        <div className="eq sub">
          <span className="op">PV</span>
          <sub>i</sub> = η&thinsp;<i>L</i> + (1&nbsp;−&nbsp;η)&thinsp;<i>G</i>
        </div>
        <div className="eq sub">
          <i>T</i>
          <sub>i</sub>(φ) = <i>a</i>
          <sub>0</sub> + <span className="op">Σ</span>
          <sub>n=1</sub>
          <sup>{nFourier}</sup> (<i>a</i>
          <sub>n</sub>&thinsp;<span className="op">cos</span>&thinsp;<i>n</i>φ + <i>b</i>
          <sub>n</sub>&thinsp;<span className="op">sin</span>&thinsp;<i>n</i>φ) ≥ 0
        </div>
      </div>
    );
  }
  return (
    <div className="ring-eqn">
      <div className="eq">
        <i>I</i>
        <sub>ring</sub>(|<i>Q</i>|,&nbsp;φ) = <span className="op">max</span>(0,&nbsp;
        <span className="op">prof</span>
        <sub>k</sub>(|<i>Q</i>|) − <span className="op">base</span>
        <sub>k</sub>(|<i>Q</i>|))
      </div>
      <div className="eq sub">
        <i>k</i> = patch(φ),&nbsp;&nbsp;<i>k</i> = 1…{nPatches}
      </div>
    </div>
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
  // Punch ellipsoid frame: "spherical" (rρ,rθ,rφ, default) | "q" (a*,b*,c*).
  frame: string;
  // Spherical-frame radii (Å⁻¹): rρ radial, rθ polar, rφ azimuth.
  sphRadii: [number, number, number];
  margin: number;
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

// --- spherical frame (rρ, rθ, rφ) -----------------------------------------
// The punch ellipsoid axes follow the *local* spherical frame at each peak, so
// the shape matrix is rebuilt per Bragg node from its Q direction (unlike the
// global a*/b*/c* q-frame).  Mirrors `BraggRemover._spherical_shape_matrix`.

function nodeFullHkl(plane: PunchPlane, x: number, y: number, cut: number): [number, number, number] {
  if (plane === "hk") return [x, y, cut];
  if (plane === "hl") return [x, cut, y];
  return [cut, x, y]; // kl
}

const cross = (a: number[], b: number[]): number[] => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const norm3 = (v: number[]) => Math.hypot(v[0], v[1], v[2]);

// HKL shape matrix A (δhklᵀAδhkl ≤ 1) for the spherical punch at one node.
function sphShapeMatrixHkl(
  radii: [number, number, number],
  hkl: [number, number, number],
  ub: Matrix3,
): Matrix3 | null {
  if (radii.some((r) => !(r > 0))) return null;
  const col = (j: number) => [ub[0][j], ub[1][j], ub[2][j]];
  const q = [
    ub[0][0] * hkl[0] + ub[0][1] * hkl[1] + ub[0][2] * hkl[2],
    ub[1][0] * hkl[0] + ub[1][1] * hkl[1] + ub[1][2] * hkl[2],
    ub[2][0] * hkl[0] + ub[2][1] * hkl[1] + ub[2][2] * hkl[2],
  ];
  const nq = norm3(q);
  if (!(nq > 0)) return null;
  const rho = q.map((v) => v / nq);
  const zc = col(2);
  const zn = norm3(zc);
  const zhat = zn > 0 ? zc.map((v) => v / zn) : [0, 0, 1];
  let phi = cross(zhat, rho);
  if (norm3(phi) < 1e-8) {
    const a = col(0);
    const an = norm3(a) || 1;
    phi = cross(a.map((v) => v / an), rho);
  }
  const pn = norm3(phi) || 1;
  const phiHat = phi.map((v) => v / pn);
  const theta = cross(phiHat, rho);
  // A_Q = R diag(1/r²) Rᵀ, R columns ρ̂, θ̂, φ̂
  const R = [rho, theta, phiHat]; // R[axis] = column vector
  const inv = radii.map((r) => 1 / (r * r));
  const aQ: Matrix3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      aQ[i][j] = inv[0] * R[0][i] * R[0][j] + inv[1] * R[1][i] * R[1][j] + inv[2] * R[2][i] * R[2][j];
  // A_hkl = UBᵀ A_Q UB
  const tmp: Matrix3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      tmp[i][j] = aQ[i][0] * ub[0][j] + aQ[i][1] * ub[1][j] + aQ[i][2] * ub[2][j];
  const aHkl: Matrix3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      aHkl[i][j] = ub[0][i] * tmp[0][j] + ub[1][i] * tmp[1][j] + ub[2][i] * tmp[2][j];
  return aHkl;
}

// Per-node punch ellipse: spherical → rebuilt from the node's Q; otherwise the
// shared global q-frame ellipse.
function ellipseForNode(
  plane: PunchPlane,
  geom: PunchGeom,
  globalEllipse: { rx: number; ry: number; angle: number },
  lattice: LatticeLike,
  ubRaw: number[][] | undefined,
  x: number,
  y: number,
  cut: number,
): { rx: number; ry: number; angle: number } {
  if (geom.frame !== "spherical") return globalEllipse;
  const ub = asMatrix3(ubRaw);
  if (!ub) return globalEllipse;
  const m = Math.max(0, geom.margin);
  const radii: [number, number, number] = [
    geom.sphRadii[0] + m,
    geom.sphRadii[1] + m,
    geom.sphRadii[2] + m,
  ];
  const a = sphShapeMatrixHkl(radii, nodeFullHkl(plane, x, y, cut), ub);
  if (!a) return globalEllipse;
  const [axisX, axisY] = planeHklAxes(plane);
  const ix = axisIndex(axisX);
  const iy = axisIndex(axisY);
  const qScaleX = (2 * Math.PI) / (axisLattice(axisX, lattice) ?? 1);
  const qScaleY = (2 * Math.PI) / (axisLattice(axisY, lattice) ?? 1);
  return ellipseFromQuadratic(
    a[ix][ix] / (qScaleX * qScaleX),
    a[ix][iy] / (qScaleX * qScaleY),
    a[iy][iy] / (qScaleY * qScaleY),
    globalEllipse,
  );
}

// |Q| = R boundary on a plane, in the same physical-Å⁻¹ overlay coordinates as
// the Bragg footprint.  The locus is the reciprocal-metric quadratic form
// Xᵀ·M·X = R² (M built from the UB columns), so it is a correctly *tilted*
// ellipse for oblique lattices and a true circle for orthogonal ones.  For an
// off-origin cut the in-plane radius shrinks by the perpendicular distance
// (exact for orthogonal axes — meta carries only a, b, c, not lattice angles).
function qShellEllipseForPlane(
  plane: PunchPlane,
  radiusQ: number,
  lattice: LatticeLike,
  ubMatrix?: number[][],
  cutQ = 0,
): { rx: number; ry: number; angle: number } | null {
  if (!(radiusQ > 0)) return null;
  const reff2 = radiusQ * radiusQ - cutQ * cutQ;
  if (reff2 <= 0) return null;
  const reff = Math.sqrt(reff2);
  const [axisX, axisY] = planeHklAxes(plane);
  const latX = axisLattice(axisX, lattice) ?? 1;
  const latY = axisLattice(axisY, lattice) ?? 1;
  const qScaleX = (2 * Math.PI) / latX;
  const qScaleY = (2 * Math.PI) / latY;
  if (!ubMatrix) return { rx: reff, ry: reff, angle: 0 };
  const ix = axisIndex(axisX);
  const iy = axisIndex(axisY);
  const ax = [ubMatrix[0][ix], ubMatrix[1][ix], ubMatrix[2][ix]];
  const ay = [ubMatrix[0][iy], ubMatrix[1][iy], ubMatrix[2][iy]];
  const dot = (p: number[], q: number[]) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
  const e = ellipseFromQuadratic(
    dot(ax, ax) / (qScaleX * qScaleX),
    dot(ax, ay) / (qScaleX * qScaleY),
    dot(ay, ay) / (qScaleY * qScaleY),
    { rx: 1, ry: 1, angle: 0 },
  );
  return { rx: e.rx * reff, ry: e.ry * reff, angle: e.angle };
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
  bands,
  cutValue,
}: {
  slice: import("../api/types").Slice;
  geom: PunchGeom;
  spec: (typeof PUNCH_PLANES)[number];
  lattice: LatticeLike;
  ubMatrix?: number[][];
  sourceLabel?: string;
  zoom: number;
  vmax: number;
  bands?: [number, number];
  cutValue?: number;
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
  const baseHalfRlu = PREVIEW_BASE_HALF_RLU;
  const zoomSafe = Math.max(0.5, Math.min(4, zoom));
  const qHalfReq = (baseHalfRlu * Math.max(qScaleH, qScaleK, qScaleL)) / zoomSafe;
  // Cap the (isotropic, square) window at the in-plane data extent.  Without the
  // cap, zooming out past the data makes SliceCanvas clamp the crop to the data
  // edge and stretch it into the square tile, while this overlay keeps the full
  // qHalf coordinates — which drifts the Bragg nodes/circles off the real peaks.
  const xs = slice.header.x_axis;
  const ys = slice.header.y_axis;
  const halfXQ = Math.max(Math.abs(xs[0]), Math.abs(xs[xs.length - 1])) * qScaleX;
  const halfYQ = Math.max(Math.abs(ys[0]), Math.abs(ys[ys.length - 1])) * qScaleY;
  const qHalf = Math.min(qHalfReq, halfXQ, halfYQ);
  const previewHalfX = qHalf / qScaleX;
  const previewHalfY = qHalf / qScaleY;
  const directBeamGeom = {
    ...geom,
    r0: geom.directBeamRadiiQ[0] + geom.directBeamMargin,
    r1: geom.directBeamRadiiQ[1] + geom.directBeamMargin,
    r2: geom.directBeamRadiiQ[2] + geom.directBeamMargin,
  };
  const ib = qEllipseForPlane(plane, directBeamGeom, lattice, ubMatrix);
  const showMargin = !geom.isQ && geom.margin > 0;
  const showSat = geom.mode !== "integer";
  // |Q| transform band as origin-centred inner/outer boundaries on this plane,
  // shrunk for the slice's off-origin cut.
  const qScaleCut = 2 * Math.PI / (axisLattice(spec.cutAxis, lattice) ?? 1);
  const cutQ = (cutValue ?? 0) * qScaleCut;
  const qBandInner = bands && bands[0] > 0
    ? qShellEllipseForPlane(plane, bands[0], lattice, ubMatrix, cutQ)
    : null;
  const qBandOuter = bands && bands[1] > 0
    ? qShellEllipseForPlane(plane, bands[1], lattice, ubMatrix, cutQ)
    : null;
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
            {qBandOuter && (
              <ellipse
                cx={0}
                cy={0}
                rx={qBandOuter.rx}
                ry={qBandOuter.ry}
                transform={`rotate(${qBandOuter.angle.toFixed(2)} 0 0)`}
                className="punch-qband"
                vectorEffect="non-scaling-stroke"
              />
            )}
            {qBandInner && (
              <ellipse
                cx={0}
                cy={0}
                rx={qBandInner.rx}
                ry={qBandInner.ry}
                transform={`rotate(${qBandInner.angle.toFixed(2)} 0 0)`}
                className="punch-qband"
                vectorEffect="non-scaling-stroke"
              />
            )}
            {showSat && (() => {
              const sat = ellipseForNode(
                plane, geom, braggEllipse, lattice, ubMatrix, 1.5, -0.5, cutValue ?? 0);
              return (
                <g>
                  <ellipse
                    cx={1.5 * qScaleX}
                    cy={-0.5 * qScaleY}
                    rx={sat.rx}
                    ry={sat.ry}
                    transform={`rotate(${sat.angle.toFixed(2)} ${1.5 * qScaleX} ${-0.5 * qScaleY})`}
                    className="punch-sat"
                    vectorEffect="non-scaling-stroke"
                  />
                </g>
              );
            })()}
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
              // Spherical frame: the ellipse is rebuilt from this node's Q so it
              // tilts to follow Q̂; q-frame uses the shared global ellipse.
              const e = ellipseForNode(
                plane, geom, braggEllipse, lattice, ubMatrix, x, y, cutValue ?? 0);
              return (
                <g key={key}>
                  {showMargin && (
                    <ellipse
                      cx={cx}
                      cy={cy}
                      rx={e.rx + geom.margin * qScaleX}
                      ry={e.ry + geom.margin * qScaleY}
                      transform={`rotate(${e.angle.toFixed(2)} ${cx} ${cy})`}
                      className="punch-margin"
                      vectorEffect="non-scaling-stroke"
                    />
                  )}
                  <ellipse
                    cx={cx}
                    cy={cy}
                    rx={e.rx}
                    ry={e.ry}
                    transform={`rotate(${e.angle.toFixed(2)} ${cx} ${cy})`}
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
  bands,
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
  bands?: [number, number];
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
                bands={bands}
                cutValue={cutValue}
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
              {bands && (bands[0] > 0 || bands[1] > 0) && (
                <span>
                  <span className="sw qband" />|Q| band
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
// In-browser boot progress panel (shown during the one-time WASM download)
// ---------------------------------------------------------------------------

function useBootStatus(): BootStatus {
  const [status, setStatus] = useState<BootStatus>(getBootStatus);
  useEffect(() => subscribeBoot(setStatus), []);
  return status;
}

const BOOT_PHASE_LABELS: Record<string, string> = {
  runtime: "Downloading Python runtime (~10 MB)…",
  packages: "Loading numpy, scipy, h5py…",
  wheel: "Installing nebula3d package…",
  ready: "Compute engine ready",
  error: "Boot failed",
};

function BootProgressPanel({ status }: { status: BootStatus }) {
  const phases = ["runtime", "packages", "wheel"] as const;
  const idx = phases.indexOf(status.phase as (typeof phases)[number]);
  const pct = idx >= 0 ? Math.round(((idx + 1) / phases.length) * 100) : 100;
  const isError = status.phase === "error";
  const isReady = status.phase === "ready";

  return (
    <div className={`boot-panel${isError ? " boot-panel--error" : isReady ? " boot-panel--ready" : ""}`}>
      <div className="boot-panel-icon">
        {isError ? (
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.5" />
            <path d="M5 5l4 4M9 5l-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        ) : isReady ? (
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.5" />
            <path d="M4.5 7l2 2 3-3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        ) : (
          <span className="spin" />
        )}
      </div>
      <div className="boot-panel-body">
        <span className="boot-panel-label">
          {BOOT_PHASE_LABELS[status.phase] ?? status.message}
        </span>
        {!isReady && !isError && (
          <div className="boot-panel-bar" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
            <div className="boot-panel-fill" style={{ width: `${pct}%` }} />
          </div>
        )}
        {isError && <span className="boot-panel-error">{status.error}</span>}
      </div>
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

  // In-browser (Pyodide) mode: the user supplies a volume file that is loaded
  // into the local runtime instead of pointing the server at a data folder.
  const bootStatus = useBootStatus();
  const bootBusy = PYODIDE_MODE && !bootStatus.ready && bootStatus.phase !== "idle";
  const volumeInputRef = useRef<HTMLInputElement>(null);
  const [loadBusy, setLoadBusy] = useState(false);
  const [loadNote, setLoadNote] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const loadLocalVolume = async (run: () => Promise<string>, label: string) => {
    setLoadBusy(true);
    setLoadError(null);
    setLoadNote(`Loading ${label}… (first load boots the in-browser engine)`);
    try {
      const id = await run();
      await queryClient.invalidateQueries({ queryKey: ["datasets"] });
      setDataset(id);
      setLoadNote(`Loaded ${label}`);
    } catch (e) {
      setLoadNote(null);
      setLoadError((e as Error).message);
    } finally {
      setLoadBusy(false);
    }
  };

  useEffect(() => {
    if (dataRootQ.data?.data_root) setRootDraft(dataRootQ.data.data_root);
  }, [dataRootQ.data?.data_root]);

  const s = usePipelineStore(
    useShallow((st) => ({
      ringsEnabled: st.ringsEnabled,
      punchEnabled: st.punchEnabled,
      backfillEnabled: st.backfillEnabled,
      flatten: st.flatten,
      pdfEnabled: st.pdfEnabled,
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
      punchFrame: st.punchFrame,
      punchRho: st.punchRho,
      punchTheta: st.punchTheta,
      punchPhi: st.punchPhi,
      punchQA: st.punchQA,
      punchQB: st.punchQB,
      punchQC: st.punchQC,
      punchFitCovariance: st.punchFitCovariance,
      punchFitUnconstrained: st.punchFitUnconstrained,
      punchMargin: st.punchMargin,
      incidentBeamQA: st.incidentBeamQA,
      incidentBeamQB: st.incidentBeamQB,
      incidentBeamQC: st.incidentBeamQC,
      incidentBeamMargin: st.incidentBeamMargin,
      incidentBeamFitCovariance: st.incidentBeamFitCovariance,
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
  const vizFourier = clampInt(s.ringNFourier, 8, 0, 40);
  const vizRingWidth = clampFloat(s.ringWidth, 0.24, 0.02, 1.0);
  const punchFrame = s.punchFrame === "q" ? "q" : "spherical";
  const sphRadii: [number, number, number] = [
    clampFloat(s.punchRho, 0.097, 0.005, 0.6),
    clampFloat(s.punchTheta, 0.072, 0.005, 0.6),
    clampFloat(s.punchPhi, 0.115, 0.005, 0.6),
  ];
  const punchGeom: PunchGeom = {
    r0: clampFloat(s.punchQA, 0.097, 0.005, 0.6),
    r1: clampFloat(s.punchQB, 0.072, 0.005, 0.6),
    r2: clampFloat(s.punchQC, 0.115, 0.005, 0.6),
    frame: punchFrame,
    sphRadii,
    margin: clampFloat(s.punchMargin, 0.02, 0, 0.5),
    mode: s.punchMode || "both",
    isQ: true,
    unit: "Å⁻¹",
    ax: punchFrame === "spherical" ? ["ρ", "θ", "φ"] : ["a*", "b*", "c*"],
    directBeamRadiiQ: [
      clampFloat(s.incidentBeamQA, 0.16, 0.005, 2.0),
      clampFloat(s.incidentBeamQB, 0.30, 0.005, 2.0),
      clampFloat(s.incidentBeamQC, 0.25, 0.005, 2.0),
    ],
    directBeamMargin: clampFloat(s.incidentBeamMargin, 0.0, 0, 1.0),
    fitCovariance: s.punchFitCovariance,
  };
  // Single reciprocal-space preview for the whole punch → ΔPDF card, sourced
  // from the raw volume (falls back to ringremoved if raw is absent).
  const punchPreviewStage = useMemo(() => {
    if (!selectedDataset) return undefined;
    return (
      selectedDataset.stages.find((stage) => stage.name === "raw" && stage.exists) ??
      selectedDataset.stages.find((stage) => stage.name === "ringremoved" && stage.exists)
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
  const pdfQSpanMax = Math.ceil(qSpanFromMeta(punchMetaQ.data) * 20) / 20;
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
  const rootStatus = rootError
    ? rootError
    : rootNote ??
      (dataRootQ.data
        ? `${dataRootQ.data.raw_exists ? "raw" : "no raw"} · ${dataRootQ.data.processed_exists ? "processed" : "no processed"} · ${dataRootQ.data.n_datasets} datasets`
        : " ");

  return (
    <div className="config-page">
      {/* ----------------------------------------------------------- data card */}
      <section className="card data-card">
        <div className="card-title-row">
          <h3>Data</h3>
          <span className="card-title-rule" />
        </div>
        <div className="data-card-head">
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
          {PYODIDE_MODE ? (
            <div className="data-source-panel">
              <Field label="Volume file" grow>
                <div className="data-root-row">
                  <input
                    ref={volumeInputRef}
                    type="file"
                    accept=".nxs,.h5,.hdf5"
                    className="visually-hidden"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) void loadLocalVolume(() => engine.loadFile(f), f.name);
                      if (volumeInputRef.current) volumeInputRef.current.value = "";
                    }}
                  />
                  <button
                    type="button"
                    className="btn btn-primary"
                    disabled={loadBusy || bootBusy}
                    onClick={() => volumeInputRef.current?.click()}
                  >
                    {loadBusy && <span className="spin" />}
                    Load volume…
                  </button>
                  <button
                    type="button"
                    className="btn btn-ghost"
                    disabled={loadBusy || bootBusy}
                    onClick={() => void loadLocalVolume(() => engine.loadDemo(), "demo volume")}
                  >
                    Use demo
                  </button>
                </div>
                {bootBusy ? (
                  <BootProgressPanel status={bootStatus} />
                ) : (
                  <span className={`data-root-status${loadError ? " error" : ""}`}>
                    {loadError ??
                      loadNote ??
                      "Your .nxs / .h5 is processed locally in your browser — nothing is uploaded."}
                  </span>
                )}
              </Field>
            </div>
          ) : (
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
          )}

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
        <StageCard
          title="3D-ΔPDF Workflow"
          className="stage-card-wide punch-stage-card"
        >
          <div className="punch-workspace">
            <div className="punch-controls">
              <div className={`cfg-stage${s.ringsEnabled ? "" : " cfg-stage--off"}`}>
                <StageHead
                  no={STAGE_NO.rings}
                  title="Ring removal"
                  enabled={s.ringsEnabled}
                  onToggle={(v) => patch({ ringsEnabled: v })}
                >
                  <HelpTip>
                    Powder-ring subtraction before punching. Patched removes a
                    per-azimuthal-patch radial background; parametric fits a
                    separable Ring(|Q|) × Fourier-texture model.
                  </HelpTip>
                </StageHead>
                <div className="cfg-stage-grid">
                  <div className="cfg-box">
                  <div className="ring-removal-controls">
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
                    {s.ringModel === "parametric" && (
                      <>
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
                      </>
                    )}
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
                        placeholder="8"
                        value={s.ringNFourier}
                        title="Fourier order of the azimuthal texture T(φ) modelling the powder rings"
                        onChange={(e) => patch({ ringNFourier: e.target.value })}
                      />
                    </Field>
                  </div>
                  </div>
                  <div className="cfg-box cfg-box--plot" aria-hidden="true">
                    <span className="ring-removal-spacer-hint">
                      radial PV fit · ring-width estimate
                    </span>
                  </div>
                  <div className="cfg-box ring-removal-figure">
                    <div className="ring-removal-viz">
                      {s.ringModel === "parametric" ? (
                        <ParametricRingViz
                          nFourier={vizFourier}
                          radialMode={s.ringRadialMode}
                          ringWidth={vizRingWidth}
                        />
                      ) : (
                        <RingTextureViz nPatches={vizPatches} nFourier={vizFourier} />
                      )}
                      <div className="ring-viz-cap">
                        {s.ringModel === "parametric" ? (
                          <>
                            azimuthal Fourier texture · order <b>{vizFourier}</b>
                          </>
                        ) : (
                          <>
                            pie = <b>{vizPatches}</b> azimuthal patches · ring = texture order{" "}
                            <b>{vizFourier}</b>
                          </>
                        )}
                      </div>
                    </div>
                    <RingEquation
                      model={s.ringModel}
                      nFourier={vizFourier}
                      nPatches={vizPatches}
                    />
                  </div>
                </div>
              </div>

              <div className={`cfg-stage${s.punchEnabled ? "" : " cfg-stage--off"}`}>
                <StageHead
                  no={STAGE_NO.punch}
                  title="Punch"
                  enabled={s.punchEnabled}
                  onToggle={(v) => patch({ punchEnabled: v })}
                >
                  <HelpTip>
                    Detection method, the intensity floor above background, and
                    which peaks to punch — integer nodes, |Q|-shell search, or both.
                  </HelpTip>
                </StageHead>
                <div className="cfg-stage-grid">
                  <div className="cfg-box">
                    <span className="cfg-box-eyebrow">Detection</span>
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
                  </div>
                  <div className="cfg-box">
                    <div className="cfg-box-head">
                      <span className="cfg-box-eyebrow cfg-box-eyebrow--sub">Bragg footprint</span>
                      <span className="punch-group-unit">Å⁻¹</span>
                      <HelpTip>
                        Punch ellipsoid radii in Q-space. <b>Spherical</b> (default)
                        sets axes in the local spherical frame at each peak —
                        r<sub>ρ</sub> radial (along Q̂), r<sub>θ</sub> polar,
                        r<sub>φ</sub> azimuth (c* pole) — so every peak is oriented
                        correctly with no tilt angle. <b>Reciprocal</b> uses fixed
                        a*, b*, c* axes. Blank fields use the validated defaults
                        (0.097, 0.072, 0.115).
                      </HelpTip>
                    </div>
                <div className="config-grid">
                  <Field label="Frame">
                    <select
                      value={s.punchFrame || "spherical"}
                      title="Punch ellipsoid axis frame"
                      onChange={(e) => patch({ punchFrame: e.target.value })}
                    >
                      <option value="spherical">spherical (ρ, θ, φ)</option>
                      <option value="q">reciprocal (a*, b*, c*)</option>
                    </select>
                  </Field>
                </div>
                {(s.punchFrame || "spherical") === "spherical" ? (
                <div className="config-grid-3">
                  <Field label={<>r<sub>ρ</sub></>}>
                    <input
                      type="number"
                      step="0.005"
                      min="0"
                      placeholder="0.097"
                      value={s.punchRho}
                      title="Punch half-radius along the radial direction Q̂ (Å⁻¹)"
                      onChange={(e) => patch({ punchRho: e.target.value })}
                    />
                  </Field>
                  <Field label={<>r<sub>θ</sub></>}>
                    <input
                      type="number"
                      step="0.005"
                      min="0"
                      placeholder="0.072"
                      value={s.punchTheta}
                      title="Punch half-radius along the polar tangent (Å⁻¹)"
                      onChange={(e) => patch({ punchTheta: e.target.value })}
                    />
                  </Field>
                  <Field label={<>r<sub>φ</sub></>}>
                    <input
                      type="number"
                      step="0.005"
                      min="0"
                      placeholder="0.115"
                      value={s.punchPhi}
                      title="Punch half-radius along the azimuthal (a*–b* ring) tangent (Å⁻¹)"
                      onChange={(e) => patch({ punchPhi: e.target.value })}
                    />
                  </Field>
                </div>
                ) : (
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
                )}
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
                </div>
                <div className="switch-row">
                  <Switch
                    label="Fit tilted ellipsoid (covariance)"
                    checked={s.punchFitCovariance}
                    onChange={(v) =>
                      patch({
                        punchFitCovariance: v,
                        punchFitUnconstrained: v ? s.punchFitUnconstrained : false,
                      })
                    }
                  />
                  <HelpTip>
                    Fit a tilted 3×3 ellipsoid to each Bragg peak during punching.
                    The preview uses the exact UB-derived Q-space floor; the
                    per-peak covariance tilt is fitted from data during the run and
                    is best checked in the punched slices.
                  </HelpTip>
                </div>
                <div className="switch-row">
                  <Switch
                    label="Drop fit constraints"
                    checked={s.punchFitUnconstrained}
                    disabled={!s.punchFitCovariance}
                    onChange={(v) => patch({ punchFitUnconstrained: v })}
                  />
                  <HelpTip>
                    Let Bragg covariance-fit radii go below the Q-space floor or
                    above the max-radius cap. This is useful for profile diagnostics,
                    but can create unstable punch masks on weak or noisy peaks.
                  </HelpTip>
                </div>
                  </div>
                  <div className="cfg-box">
                    <div className="cfg-box-head">
                      <span className="cfg-box-eyebrow cfg-box-eyebrow--sub">Direct beam</span>
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
                <div className="switch-row">
                  <Switch
                    label="Fit ellipsoid (covariance)"
                    checked={s.incidentBeamFitCovariance}
                    onChange={(v) => patch({ incidentBeamFitCovariance: v })}
                  />
                  <HelpTip>
                    Fit a tilted ellipsoid to the direct-beam remnant at the origin
                    during punching (analogue of the Bragg footprint fit), floored at
                    the radii above so it only follows/expands the real beam shape.
                    The preview shows the UB-derived floor; the fitted tilt is best
                    checked in the punched slices.
                  </HelpTip>
                </div>
                  </div>
                </div>
              </div>

              <div className="cfg-stage-row">
                <div className={`cfg-stage${s.backfillEnabled ? "" : " cfg-stage--off"}`}>
                  <StageHead
                    no={STAGE_NO.backfill}
                    title="Backfill"
                    enabled={s.backfillEnabled}
                    onToggle={(v) => patch({ backfillEnabled: v })}
                  >
                    <HelpTip>
                      How punched Bragg / direct-beam holes are filled before the
                      transform. q_shell (default) interpolates each voxel from its
                      |Q| shell.
                    </HelpTip>
                  </StageHead>
                  <div className="cfg-box">
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
                  </div>
                </div>

                <div className={`cfg-stage${s.flatten ? "" : " cfg-stage--off"}`}>
                  <StageHead
                    no={STAGE_NO.flatten}
                    title="Flatten"
                    enabled={s.flatten}
                    onToggle={(v) => patch({ flatten: v })}
                  >
                    <HelpTip>
                      Optional background flattening of the backfilled volume before
                      the transform, using the selected baseline estimator.
                    </HelpTip>
                  </StageHead>
                  <div className="cfg-box">
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
                  </div>
                </div>

                <div className={`cfg-stage${s.pdfEnabled ? "" : " cfg-stage--off"}`}>
                  <StageHead
                    no={STAGE_NO.pdf}
                    title="Transform (3D-ΔPDF)"
                    enabled={s.pdfEnabled}
                    onToggle={(v) => patch({ pdfEnabled: v })}
                  >
                    <HelpTip>
                      3D-FFT of the cleaned, backfilled diffuse volume. The
                      apodization window tapers Q-space before the transform to
                      suppress real-space termination ripples; the |Q| band (shown
                      on the slices) sets the radial window the transform keeps.
                    </HelpTip>
                  </StageHead>
                  <div className="cfg-box">
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
              <PunchPreviewGrid
                geom={punchGeom}
                slices={punchSlices}
                loading={punchLoading}
                lattice={punchMetaQ.data?.lattice}
                sourceLabel={punchPreviewStage?.name}
                zoom={s.punchSliceZoom}
                contrast={s.punchSliceContrast}
                bands={[pdfQMin, s.pdfQMax ? pdfQMax : 0]}
                meta={punchMetaQ.data}
                cuts={punchCuts}
                onCut={(axis, value) => {
                  if (axis === "H") patch({ punchCutH: value });
                  else if (axis === "K") patch({ punchCutK: value });
                  else patch({ punchCutL: value });
                }}
              />
              <div className="ring-viz-cap">
                source <b>{punchPreviewStage?.name ?? "none"}</b> · r = (
                <b>{punchGeom.r0}</b>, <b>{punchGeom.r1}</b>, <b>{punchGeom.r2}</b>){" "}
                {punchGeom.unit} · |Q| band{" "}
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
