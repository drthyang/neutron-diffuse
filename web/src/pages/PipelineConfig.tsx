// Pipeline configuration page — the form + live algorithm illustrations.  All
// values live in the pipeline store, so navigating away and back never resets
// them; pressing Run kicks off the job (also in the store) and jumps to the
// Execution page via `onStarted`.

import { useEffect, useMemo, type ReactNode } from "react";
import { useShallow } from "zustand/react/shallow";

import { useDatasets } from "../api/hooks";
import { Field, HelpTip, Switch } from "../components/ui";
import {
  STAGE_LABELS,
  STAGE_NO,
  usePipelineStore,
  type PunchPlane,
} from "../state/pipelineStore";

// One pipeline stage rendered as its own card.  `step` shows the matching
// stepper number so the tiles read in pipeline order regardless of how the
// masonry packs them.
function StageCard({
  title,
  step,
  children,
}: {
  title: string;
  step?: number;
  children: ReactNode;
}) {
  return (
    <section className="card stage-card">
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

const PUNCH_PLANES: { id: PunchPlane; xl: string; yl: string }[] = [
  { id: "hk", xl: "H", yl: "K" },
  { id: "hl", xl: "H", yl: "L" },
  { id: "kl", xl: "K", yl: "L" },
];

// incident-beam ellipsoid default (PunchParams.incident_beam_ellipsoid_radii_hkl)
const PUNCH_IB = { h: 0.15, k: 0.5, l: 1.0 };

interface PunchGeom {
  // r0/r1/r2 are the footprint half-radii along the three axes, in the active
  // frame's units (r.l.u. for HKL, Å⁻¹ for Q).
  r0: number;
  r1: number;
  r2: number;
  margin: number;
  phiTail: number;
  mode: string;
  isQ: boolean;
  unit: string;
  ax: [string, string, string]; // axis labels, e.g. [H,K,L] or [a*,b*,c*]
}

function planeRadii(p: PunchPlane, g: PunchGeom): [number, number] {
  if (p === "hk") return [g.r0, g.r1];
  if (p === "hl") return [g.r0, g.r2];
  return [g.r1, g.r2];
}
function planeAxes(p: PunchPlane, ax: [string, string, string]): [string, string] {
  if (p === "hk") return [ax[0], ax[1]];
  if (p === "hl") return [ax[0], ax[2]];
  return [ax[1], ax[2]];
}

function EllipsoidPunchViz({
  geom,
  plane,
  onPlane,
}: {
  geom: PunchGeom;
  plane: PunchPlane;
  onPlane: (p: PunchPlane) => void;
}) {
  const PV = 200; // viewBox
  const C = 100; // centre
  const SP = 34; // px per reciprocal-lattice step
  const N = 2; // nodes span −N..N
  const [axX, axY] = planeAxes(plane, geom.ax);
  const [rx, ry] = planeRadii(plane, geom);
  // The footprint is anisotropic and its absolute size is unit-dependent, so we
  // auto-scale: the larger semi-axis fills ~42% of a lattice cell.  The drawing
  // shows the punch *shape* relative to the reciprocal lattice, not true scale.
  const fp = (0.42 * SP) / Math.max(rx, ry, 1e-6);
  // incident beam: HKL has its own ellipsoid radii; Q shows it ≈2× the floor.
  const ib: [number, number] = geom.isQ
    ? [rx * 2, ry * 2]
    : plane === "hk"
      ? [PUNCH_IB.h, PUNCH_IB.k]
      : plane === "hl"
        ? [PUNCH_IB.h, PUNCH_IB.l]
        : [PUNCH_IB.k, PUNCH_IB.l];
  // margin (guard band) and φ-tail are r.l.u. concepts — drawn only in HKL so the
  // Q view stays a clean resolution ellipsoid (they still apply numerically).
  const showTail = !geom.isQ && plane === "kl" && geom.phiTail > 0;
  const showMargin = !geom.isQ && geom.margin > 0;
  const showSat = geom.mode !== "integer";

  const grid: ReactNode[] = [];
  for (let i = -N; i <= N; i++) {
    const q = C + i * SP;
    grid.push(<line key={`v${i}`} x1={q} y1={16} x2={q} y2={PV - 16} className="punch-grid" />);
    grid.push(<line key={`h${i}`} x1={16} y1={q} x2={PV - 16} y2={q} className="punch-grid" />);
  }

  const marks: ReactNode[] = [];
  for (let i = -N; i <= N; i++) {
    for (let j = -N; j <= N; j++) {
      const cx = C + i * SP;
      const cy = C - j * SP;
      const k = `${i},${j}`;
      if (i === 0 && j === 0) {
        marks.push(
          <ellipse key={`ib-${k}`} cx={cx} cy={cy} rx={ib[0] * fp} ry={ib[1] * fp} className="punch-ib" />,
        );
      } else {
        if (showTail) {
          const ang = (Math.atan2(j, i) * 180) / Math.PI + 90; // local ring tangent
          marks.push(
            <ellipse
              key={`tl-${k}`}
              cx={cx}
              cy={cy}
              rx={(geom.r1 + geom.phiTail) * fp}
              ry={geom.r1 * fp}
              transform={`rotate(${ang.toFixed(1)} ${cx} ${cy})`}
              className="punch-tail"
            />,
          );
        }
        if (showMargin) {
          marks.push(
            <ellipse
              key={`mg-${k}`}
              cx={cx}
              cy={cy}
              rx={(rx + geom.margin) * fp}
              ry={(ry + geom.margin) * fp}
              className="punch-margin"
            />,
          );
        }
        marks.push(
          <ellipse key={`fp-${k}`} cx={cx} cy={cy} rx={rx * fp} ry={ry * fp} className="punch-fp" />,
        );
      }
      marks.push(<circle key={`nd-${k}`} cx={cx} cy={cy} r={1.6} className="punch-node" />);
    }
  }

  // off-integer satellite caught by search / both modes
  const satX = C + 1.5 * SP;
  const satY = C - 0.5 * SP;

  return (
    <div className="punch-viz">
      <div className="punch-tabs" role="tablist">
        {PUNCH_PLANES.map((p) => {
          const [x, y] = planeAxes(p.id, geom.ax);
          return (
            <button
              key={p.id}
              type="button"
              role="tab"
              aria-selected={p.id === plane}
              className={p.id === plane ? "on" : ""}
              onClick={() => onPlane(p.id)}
            >
              {x}–{y}
            </button>
          );
        })}
      </div>
      <svg
        viewBox={`0 0 ${PV} ${PV}`}
        role="img"
        aria-label={`Bragg punch ellipsoid cross-section in the ${axX}-${axY} plane (${geom.unit})`}
      >
        <g>{grid}</g>
        {showSat && (
          <g>
            <ellipse cx={satX} cy={satY} rx={rx * fp} ry={ry * fp} className="punch-sat" />
            <circle cx={satX} cy={satY} r={1.6} className="punch-sat-node" />
          </g>
        )}
        <g>{marks}</g>
        <text x={PV - 12} y={C - 5} textAnchor="end" className="punch-axis">
          {axX}
        </text>
        <text x={C + 5} y={20} className="punch-axis">
          {axY}
        </text>
      </svg>
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
        <span className="punch-legend-unit">{geom.unit}</span>
      </div>
    </div>
  );
}

function PunchShapeViz({
  method,
  geom,
  plane,
  onPlane,
}: {
  method: string;
  geom: PunchGeom;
  plane: PunchPlane;
  onPlane: (p: PunchPlane) => void;
}) {
  // method registry seam — add a case per future punch algorithm
  switch (method) {
    case "ellipsoid":
    default:
      return <EllipsoidPunchViz geom={geom} plane={plane} onPlane={onPlane} />;
  }
}

// ---------------------------------------------------------------------------

export function PipelineConfig({ onStarted }: { onStarted: () => void }) {
  const datasetsQ = useDatasets();
  const datasets = useMemo(() => datasetsQ.data ?? [], [datasetsQ.data]);

  const s = usePipelineStore(
    useShallow((st) => ({
      datasetId: st.datasetId,
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
      punchFrame: st.punchFrame,
      punchQA: st.punchQA,
      punchQB: st.punchQB,
      punchQC: st.punchQC,
      punchFitCovariance: st.punchFitCovariance,
      punchRH: st.punchRH,
      punchRK: st.punchRK,
      punchRL: st.punchRL,
      punchMargin: st.punchMargin,
      punchPhiTail: st.punchPhiTail,
      punchPlane: st.punchPlane,
      backfillMethod: st.backfillMethod,
      flattenEstimator: st.flattenEstimator,
      pdfApod: st.pdfApod,
      running: st.running,
    })),
  );
  const patch = usePipelineStore((st) => st.patch);
  const run = usePipelineStore((st) => st.run);

  // default to the first dataset once the list loads
  useEffect(() => {
    if (!s.datasetId && datasets.length) patch({ datasetId: datasets[0].id });
  }, [s.datasetId, datasets, patch]);

  const vizPatches = clampInt(s.ringNPatches, 36, 4, 96);
  const vizFourier = clampInt(s.ringNFourier, 6, 0, 40);
  const vizRingWidth = clampFloat(s.ringWidth, 0.24, 0.02, 1.0);
  const isQ = s.punchFrame === "q";
  const punchGeom: PunchGeom = {
    r0: isQ ? clampFloat(s.punchQA, 0.097, 0.005, 0.6)
      : clampFloat(s.punchRH, 0.09, 0.01, 1.2),
    r1: isQ ? clampFloat(s.punchQB, 0.072, 0.005, 0.6)
      : clampFloat(s.punchRK, 0.12, 0.01, 1.2),
    r2: isQ ? clampFloat(s.punchQC, 0.115, 0.005, 0.6)
      : clampFloat(s.punchRL, 0.45, 0.01, 1.5),
    margin: clampFloat(s.punchMargin, 0.02, 0, 0.5),
    phiTail: clampFloat(s.punchPhiTail, 0.12, 0, 1.0),
    mode: s.punchMode || "both",
    isQ,
    unit: isQ ? "Å⁻¹" : "r.l.u.",
    ax: isQ ? ["a*", "b*", "c*"] : ["H", "K", "L"],
  };

  const onRun = async () => {
    await run();
    onStarted();
  };

  return (
    <div className="config-page">
      {/* ----------------------------------------------------- run toolbar */}
      <div className="card config-toolbar">
        <Field label="Dataset" grow>
          <select
            value={s.datasetId}
            onChange={(e) => patch({ datasetId: e.target.value })}
          >
            {datasets.map((d) => (
              <option key={d.id} value={d.id} title={d.raw_name}>
                {d.temperature ?? d.stem}
              </option>
            ))}
          </select>
        </Field>
        <div className="toolbar-actions">
          <Switch
            label="Force"
            checked={s.force}
            onChange={(v) => patch({ force: v })}
          />
          <button
            type="button"
            className="btn btn-primary"
            onClick={onRun}
            disabled={s.running || !s.datasetId}
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

      {/* ------------------------------------------------- per-stage cards */}
      <div className="config-cards">
        <StageCard title={STAGE_LABELS.rings} step={STAGE_NO.rings}>
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
        </StageCard>

        <StageCard title={STAGE_LABELS.punch} step={STAGE_NO.punch}>
          <div className="config-grid">
            <Field label="Method">
              <select
                value={s.punchMethod}
                title="Bragg-punch algorithm (more shapes coming)"
                onChange={(e) => patch({ punchMethod: e.target.value })}
              >
                <option value="ellipsoid">Ellipsoid</option>
              </select>
            </Field>
            <Field label="Frame">
              <select
                value={s.punchFrame}
                title="Describe the punch in fractional HKL or in reciprocal Å⁻¹ (Q-space)"
                onChange={(e) => patch({ punchFrame: e.target.value })}
              >
                <option value="hkl">HKL (r.l.u.)</option>
                <option value="q">Q-space (Å⁻¹)</option>
              </select>
            </Field>
          </div>
          <div className="config-grid">
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
          {s.punchFrame === "q" ? (
            <>
              <div className="config-sub-row">
                <span className="config-sub-title">
                  Resolution floor (Å⁻¹)
                </span>
                <HelpTip>
                  Punch half-radii along the reciprocal axes a*, b*, c*, in Å⁻¹ —
                  a lattice- and temperature-independent resolution floor, still
                  modulated by the per-peak fit. Leave blank to use the validated
                  default (0.097, 0.072, 0.115).
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
            </>
          ) : (
            <>
              <div className="config-sub-row">
                <span className="config-sub-title">Ellipsoid half-radii (r.l.u.)</span>
                <HelpTip>
                  Punch half-radii along H, K, L in reciprocal-lattice units —
                  the legacy fractional-coordinate footprint.
                </HelpTip>
              </div>
              <div className="config-grid-3">
                <Field label={<>r<sub>H</sub></>}>
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    placeholder="0.09"
                    value={s.punchRH}
                    title="Punch half-radius along H (r.l.u.)"
                    onChange={(e) => patch({ punchRH: e.target.value })}
                  />
                </Field>
                <Field label={<>r<sub>K</sub></>}>
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    placeholder="0.12"
                    value={s.punchRK}
                    title="Punch half-radius along K (r.l.u.)"
                    onChange={(e) => patch({ punchRK: e.target.value })}
                  />
                </Field>
                <Field label={<>r<sub>L</sub></>}>
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    placeholder="0.45"
                    value={s.punchRL}
                    title="Punch half-radius along L (r.l.u.)"
                    onChange={(e) => patch({ punchRL: e.target.value })}
                  />
                </Field>
              </div>
            </>
          )}
          <div className="config-grid">
            <Field label="Margin">
              <input
                type="number"
                step="0.01"
                min="0"
                placeholder="0.02"
                value={s.punchMargin}
                title="Guard band added to every punch radius"
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
              label="Fit resolution ellipsoid (tilted, covariance)"
              checked={s.punchFitCovariance}
              onChange={(v) => patch({ punchFitCovariance: v })}
            />
            <HelpTip>
              Fit a tilted 3×3 ellipsoid to each Bragg peak (following its real
              orientation) and fold the φ-tail into it, instead of three
              axis-aligned half-radii plus a separate tail.
            </HelpTip>
          </div>
          <PunchShapeViz
            method={s.punchMethod}
            geom={punchGeom}
            plane={s.punchPlane}
            onPlane={(p) => patch({ punchPlane: p })}
          />
          <div className="ring-viz-cap">
            r = (<b>{punchGeom.r0}</b>, <b>{punchGeom.r1}</b>, <b>{punchGeom.r2}</b>){" "}
            {punchGeom.unit} along {punchGeom.ax.join("/")} · margin{" "}
            <b>{punchGeom.margin}</b> · φ-tail <b>{punchGeom.phiTail}</b> · mode{" "}
            <b>{punchGeom.mode}</b>
          </div>
        </StageCard>

        <StageCard title={STAGE_LABELS.backfill} step={STAGE_NO.backfill}>
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

        <StageCard title={STAGE_LABELS.flatten} step={STAGE_NO.flatten}>
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

        <StageCard title={STAGE_LABELS.pdf} step={STAGE_NO.pdf}>
          <Field label="Apodization">
            <select
              value={s.pdfApod}
              onChange={(e) => patch({ pdfApod: e.target.value })}
            >
              <option value="">gaussian (default)</option>
              <option value="hann">hann</option>
              <option value="none">none</option>
            </select>
          </Field>
        </StageCard>
      </div>
    </div>
  );
}
