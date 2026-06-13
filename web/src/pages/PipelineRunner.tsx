// Pipeline runner — drives ndiff.pipeline.run_pipeline as a background job and
// streams progress over SSE.  Configuration form on the left; stage stepper and
// live log on the right.

import type { ReactNode } from "react";
import { useEffect, useMemo, useRef, useState } from "react";

import { cancelJob, runPipeline } from "../api/client";
import { useDatasets } from "../api/hooks";
import type { JobEvent, StageParamsIn } from "../api/types";
import { Field, Switch } from "../components/ui";

const STAGES = ["rings", "punch", "backfill", "flatten", "pdf"];
const STAGE_LABELS: Record<string, string> = {
  rings: "Ring removal",
  punch: "Bragg punch",
  backfill: "Backfill",
  flatten: "Flatten",
  pdf: "3D-ΔPDF",
};

// step number (1-based) of each stage, to tie config groups to the stepper
const STAGE_NO: Record<string, number> = Object.fromEntries(
  STAGES.map((s, i) => [s, i + 1]),
);

// One grouped block in the configuration form.  `step` renders the matching
// stepper number so each algorithm's parameters line up with its stage.
function ConfigSection({
  title,
  step,
  children,
}: {
  title: string;
  step?: number;
  children: ReactNode;
}) {
  return (
    <div className="config-section">
      <span className="config-section-title">
        {step != null && <span className="config-step-no">{step}</span>}
        {title}
      </span>
      {children}
    </div>
  );
}

// Parse a numeric form value, falling back to `dflt` for empty/invalid input and
// clamping into [lo, hi] so the illustration stays well-formed while typing.
function clampInt(raw: string, dflt: number, lo: number, hi: number): number {
  const n = raw === "" ? dflt : Math.round(Number(raw));
  if (!Number.isFinite(n)) return dflt;
  return Math.max(lo, Math.min(hi, n));
}

// Float variant of clampInt for the continuous punch geometry parameters.
function clampFloat(raw: string, dflt: number, lo: number, hi: number): number {
  const n = raw === "" ? dflt : Number(raw);
  if (!Number.isFinite(n)) return dflt;
  return Math.max(lo, Math.min(hi, n));
}

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
const C = 90; // centre
const RPIE = 58; // pie radius (patches)
const RIN = 65; // ring inner radius (texture)
const ROUT = 84; // ring outer radius
const RHUB = 20; // centre hub
const TOP = -Math.PI / 2; // start angle at 12 o'clock
const pt = (r: number, a: number) =>
  [(C + r * Math.cos(a)).toFixed(2), (C + r * Math.sin(a)).toFixed(2)] as const;

// Live illustration of the powder-ring azimuthal model.  The inner pie is split
// into `nPatches` equal sectors (the azimuthal bins the ring is divided into);
// the surrounding ring is tinted by the highest texture harmonic
// T(φ) = cos(nFourier · φ), so the number of colour lobes around the ring equals
// the Fourier order (order 0 → uniform).
function RingTextureViz({
  nPatches,
  nFourier,
}: {
  nPatches: number;
  nFourier: number;
}) {
  // pie sectors — one wedge per azimuthal patch
  const wedges = useMemo(() => {
    const out: string[] = [];
    for (let i = 0; i < nPatches; i++) {
      const a0 = (i / nPatches) * 2 * Math.PI + TOP;
      const a1 = ((i + 1) / nPatches) * 2 * Math.PI + TOP;
      const [x0, y0] = pt(RPIE, a0);
      const [x1, y1] = pt(RPIE, a1);
      out.push(`M${C} ${C} L${x0} ${y0} A${RPIE} ${RPIE} 0 0 1 ${x1} ${y1} Z`);
    }
    return out;
  }, [nPatches]);

  // texture ring — fine segments tinted by cos(nFourier · φ)
  const segs = useMemo(() => {
    const n = Math.min(600, Math.max(180, nFourier * 12));
    const out: { d: string; c: string }[] = [];
    for (let i = 0; i < n; i++) {
      const a0 = (i / n) * 2 * Math.PI + TOP;
      const a1 = ((i + 1) / n) * 2 * Math.PI + TOP;
      const phi = ((i + 0.5) / n) * 2 * Math.PI; // measured from the top
      const [xo0, yo0] = pt(ROUT, a0);
      const [xo1, yo1] = pt(ROUT, a1);
      const [xi1, yi1] = pt(RIN, a1);
      const [xi0, yi0] = pt(RIN, a0);
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
      <circle cx={C} cy={C} r={RHUB} className="ring-viz-hub" />
      <text
        x={C}
        y={C}
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

type PunchPlane = "hk" | "hl" | "kl";

const PUNCH_PLANES: { id: PunchPlane; xl: string; yl: string }[] = [
  { id: "hk", xl: "H", yl: "K" },
  { id: "hl", xl: "H", yl: "L" },
  { id: "kl", xl: "K", yl: "L" },
];

// incident-beam ellipsoid default (PunchParams.incident_beam_ellipsoid_radii_hkl)
const PUNCH_IB = { h: 0.15, k: 0.5, l: 1.0 };

interface PunchGeom {
  rh: number;
  rk: number;
  rl: number;
  margin: number;
  phiTail: number;
  mode: string;
}

// the two in-plane half-radii for a cross-section
function planeRadii(p: PunchPlane, g: PunchGeom): [number, number] {
  if (p === "hk") return [g.rh, g.rk];
  if (p === "hl") return [g.rh, g.rl];
  return [g.rk, g.rl];
}
function planeIB(p: PunchPlane): [number, number] {
  if (p === "hk") return [PUNCH_IB.h, PUNCH_IB.k];
  if (p === "hl") return [PUNCH_IB.h, PUNCH_IB.l];
  return [PUNCH_IB.k, PUNCH_IB.l];
}

function EllipsoidPunchViz({ geom }: { geom: PunchGeom }) {
  const [plane, setPlane] = useState<PunchPlane>("hl");

  const PV = 200; // viewBox
  const C = 100; // centre
  const SP = 34; // px per HKL unit
  const N = 2; // nodes span −N..N
  const pl = PUNCH_PLANES.find((p) => p.id === plane)!;
  const [rx, ry] = planeRadii(plane, geom);
  const [ibx, iby] = planeIB(plane);
  const showTail = plane === "kl" && geom.phiTail > 0;
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
          <ellipse key={`ib-${k}`} cx={cx} cy={cy} rx={ibx * SP} ry={iby * SP} className="punch-ib" />,
        );
      } else {
        if (showTail) {
          const ang = (Math.atan2(j, i) * 180) / Math.PI + 90; // local ring tangent
          marks.push(
            <ellipse
              key={`tl-${k}`}
              cx={cx}
              cy={cy}
              rx={(geom.rk + geom.phiTail) * SP}
              ry={geom.rk * SP}
              transform={`rotate(${ang.toFixed(1)} ${cx} ${cy})`}
              className="punch-tail"
            />,
          );
        }
        if (geom.margin > 0) {
          marks.push(
            <ellipse
              key={`mg-${k}`}
              cx={cx}
              cy={cy}
              rx={(rx + geom.margin) * SP}
              ry={(ry + geom.margin) * SP}
              className="punch-margin"
            />,
          );
        }
        marks.push(
          <ellipse key={`fp-${k}`} cx={cx} cy={cy} rx={rx * SP} ry={ry * SP} className="punch-fp" />,
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
        {PUNCH_PLANES.map((p) => (
          <button
            key={p.id}
            type="button"
            role="tab"
            aria-selected={p.id === plane}
            className={p.id === plane ? "on" : ""}
            onClick={() => setPlane(p.id)}
          >
            {p.xl}–{p.yl}
          </button>
        ))}
      </div>
      <svg
        viewBox={`0 0 ${PV} ${PV}`}
        role="img"
        aria-label={`Bragg punch ellipsoid cross-section in the ${pl.xl}-${pl.yl} plane`}
      >
        <g>{grid}</g>
        {showSat && (
          <g>
            <ellipse cx={satX} cy={satY} rx={rx * SP} ry={ry * SP} className="punch-sat" />
            <circle cx={satX} cy={satY} r={1.6} className="punch-sat-node" />
          </g>
        )}
        <g>{marks}</g>
        <text x={PV - 12} y={C - 5} textAnchor="end" className="punch-axis">
          {pl.xl}
        </text>
        <text x={C + 5} y={20} className="punch-axis">
          {pl.yl}
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
      </div>
    </div>
  );
}

function PunchShapeViz({ method, geom }: { method: string; geom: PunchGeom }) {
  // method registry seam — add a case per future punch algorithm
  switch (method) {
    case "ellipsoid":
    default:
      return <EllipsoidPunchViz geom={geom} />;
  }
}

interface StageInfo {
  status: string;
  fraction: number | null;
}

function stepClass(status: string | undefined, off: boolean): string {
  if (off) return "off";
  if (!status) return "";
  if (status === "start" || status === "progress") return "active";
  return status; // done | skip | error
}

function stepSub(info: StageInfo | undefined, off: boolean): string {
  if (off) return "disabled";
  if (!info) return "pending";
  switch (info.status) {
    case "start":
      return "starting…";
    case "progress":
      return info.fraction != null
        ? `running · ${Math.round(info.fraction * 100)}%`
        : "running…";
    case "done":
      return "done";
    case "skip":
      return "skipped — output exists";
    case "error":
      return "failed";
    default:
      return info.status;
  }
}

export function PipelineRunner() {
  const datasetsQ = useDatasets();
  const datasets = useMemo(() => datasetsQ.data ?? [], [datasetsQ.data]);

  const [datasetId, setDatasetId] = useState("");
  const [flatten, setFlatten] = useState(true);
  const [force, setForce] = useState(false);
  const [ringNPatches, setRingNPatches] = useState("");
  const [ringNFourier, setRingNFourier] = useState("");
  const [punchMinI, setPunchMinI] = useState("");
  const [punchMethod, setPunchMethod] = useState("ellipsoid");
  const [punchMode, setPunchMode] = useState("");
  const [punchRH, setPunchRH] = useState("");
  const [punchRK, setPunchRK] = useState("");
  const [punchRL, setPunchRL] = useState("");
  const [punchMargin, setPunchMargin] = useState("");
  const [punchPhiTail, setPunchPhiTail] = useState("");
  const [backfillMethod, setBackfillMethod] = useState("");
  const [flattenEstimator, setFlattenEstimator] = useState("");
  const [pdfApod, setPdfApod] = useState("");

  const [jobId, setJobId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [terminal, setTerminal] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!datasetId && datasets.length) setDatasetId(datasets[0].id);
  }, [datasetId, datasets]);

  // stream job progress over SSE
  useEffect(() => {
    if (!jobId) return;
    const es = new EventSource(`/api/pipeline/jobs/${jobId}/events`);
    es.onmessage = (e) => {
      const ev = JSON.parse(e.data) as JobEvent;
      if (["done", "error", "cancelled"].includes(ev.type)) {
        setTerminal(ev.type);
        setRunning(false);
        es.close();
      } else {
        setEvents((prev) => [...prev, ev]);
      }
    };
    es.onerror = () => {
      es.close();
      setRunning(false);
    };
    return () => es.close();
  }, [jobId]);

  useEffect(() => {
    logRef.current?.scrollTo(0, logRef.current.scrollHeight);
  }, [events]);

  const onRun = async () => {
    setEvents([]);
    setTerminal(null);
    setRunning(true);
    const params: StageParamsIn = {};
    if (ringNPatches) params.rings_n_patches = Number(ringNPatches);
    if (ringNFourier) params.rings_n_fourier = Number(ringNFourier);
    if (punchMinI) params.punch_min_intensity = Number(punchMinI);
    if (punchMode) params.punch_mode = punchMode;
    if (punchRH) params.punch_radius_h = Number(punchRH);
    if (punchRK) params.punch_radius_k = Number(punchRK);
    if (punchRL) params.punch_radius_l = Number(punchRL);
    if (punchMargin) params.punch_margin = Number(punchMargin);
    if (punchPhiTail) params.punch_phi_tail_hkl = Number(punchPhiTail);
    if (backfillMethod) params.backfill_method = backfillMethod;
    if (flattenEstimator) params.flatten_estimator = flattenEstimator;
    if (pdfApod) params.pdf_apodization = pdfApod;
    try {
      const job = await runPipeline({
        dataset_id: datasetId,
        flatten_enabled: flatten,
        force,
        params,
      });
      setJobId(job.id);
    } catch (e) {
      setTerminal("error");
      setRunning(false);
      setEvents([{ type: "progress", status: "error", message: (e as Error).message }]);
    }
  };

  const onCancel = async () => {
    if (jobId) await cancelJob(jobId).catch(() => undefined);
  };

  // latest status (+ fraction) per stage from the event stream
  const stageInfo: Record<string, StageInfo> = {};
  for (const ev of events) {
    if (ev.stage && ev.status) {
      stageInfo[ev.stage] = { status: ev.status, fraction: ev.fraction ?? null };
    }
  }

  // clamped values driving the ring-model illustration (fall back to the
  // placeholder defaults while the inputs are empty)
  const vizPatches = clampInt(ringNPatches, 36, 4, 96);
  const vizFourier = clampInt(ringNFourier, 6, 0, 40);

  // clamped punch geometry driving the Bragg-punch illustration
  const punchGeom: PunchGeom = {
    rh: clampFloat(punchRH, 0.09, 0.01, 1.2),
    rk: clampFloat(punchRK, 0.12, 0.01, 1.2),
    rl: clampFloat(punchRL, 0.45, 0.01, 1.5),
    margin: clampFloat(punchMargin, 0.02, 0, 0.5),
    phiTail: clampFloat(punchPhiTail, 0.12, 0, 1.0),
    mode: punchMode || "both",
  };

  return (
    <div className="pipeline-layout">
      {/* ------------------------------------------------ configuration */}
      <div className="card">
        <div className="card-head">
          <h3>Configuration</h3>
        </div>
        <div className="card-body">
          <ConfigSection title="Input">
            <Field label="Dataset">
              <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
                {datasets.map((d) => (
                  <option key={d.id} value={d.id} title={d.raw_name}>
                    {d.temperature ?? d.stem}
                  </option>
                ))}
              </select>
            </Field>
          </ConfigSection>

          <ConfigSection title={STAGE_LABELS.rings} step={STAGE_NO.rings}>
            <div className="config-grid">
              <Field label="Patches (n)">
                <input
                  type="number"
                  min="4"
                  step="1"
                  placeholder="36"
                  value={ringNPatches}
                  title="Number of azimuthal patches the powder rings are divided into"
                  onChange={(e) => setRingNPatches(e.target.value)}
                />
              </Field>
              <Field label="Fourier order">
                <input
                  type="number"
                  min="0"
                  step="1"
                  placeholder="6"
                  value={ringNFourier}
                  title="Fourier order of the azimuthal texture T(φ) modelling the Al powder rings"
                  onChange={(e) => setRingNFourier(e.target.value)}
                />
              </Field>
            </div>
            <RingTextureViz nPatches={vizPatches} nFourier={vizFourier} />
            <div className="ring-viz-cap">
              pie = <b>{vizPatches}</b> azimuthal patches · ring = texture order{" "}
              <b>{vizFourier}</b>
            </div>
          </ConfigSection>

          <ConfigSection title={STAGE_LABELS.punch} step={STAGE_NO.punch}>
            <Field label="Method">
              <select
                value={punchMethod}
                title="Bragg-punch algorithm (more shapes coming)"
                onChange={(e) => setPunchMethod(e.target.value)}
              >
                <option value="ellipsoid">Ellipsoid</option>
              </select>
            </Field>
            <div className="config-grid">
              <Field label="Min I">
                <input
                  type="number"
                  step="0.1"
                  placeholder="0.8"
                  value={punchMinI}
                  title="Minimum intensity above background for a voxel to be punched as Bragg"
                  onChange={(e) => setPunchMinI(e.target.value)}
                />
              </Field>
              <Field label="Mode">
                <select value={punchMode} onChange={(e) => setPunchMode(e.target.value)}>
                  <option value="">both (default)</option>
                  <option value="integer">integer</option>
                  <option value="search">search</option>
                  <option value="both">both</option>
                </select>
              </Field>
            </div>
            <span className="config-sub-title">Ellipsoid half-radii (HKL)</span>
            <div className="config-grid-3">
              <Field label="r·H">
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  placeholder="0.09"
                  value={punchRH}
                  onChange={(e) => setPunchRH(e.target.value)}
                />
              </Field>
              <Field label="r·K">
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  placeholder="0.12"
                  value={punchRK}
                  onChange={(e) => setPunchRK(e.target.value)}
                />
              </Field>
              <Field label="r·L">
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  placeholder="0.45"
                  value={punchRL}
                  onChange={(e) => setPunchRL(e.target.value)}
                />
              </Field>
            </div>
            <div className="config-grid">
              <Field label="Margin">
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  placeholder="0.02"
                  value={punchMargin}
                  title="Guard band added to every punch radius"
                  onChange={(e) => setPunchMargin(e.target.value)}
                />
              </Field>
              <Field label="φ-tail (K–L)">
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  placeholder="0.12"
                  value={punchPhiTail}
                  title="Extra K–L tangential half-width along the powder-ring φ direction"
                  onChange={(e) => setPunchPhiTail(e.target.value)}
                />
              </Field>
            </div>
            <PunchShapeViz method={punchMethod} geom={punchGeom} />
            <div className="ring-viz-cap">
              r = (<b>{punchGeom.rh}</b>, <b>{punchGeom.rk}</b>, <b>{punchGeom.rl}</b>) ·
              margin <b>{punchGeom.margin}</b> · φ-tail <b>{punchGeom.phiTail}</b> · mode{" "}
              <b>{punchGeom.mode}</b>
            </div>
          </ConfigSection>

          <ConfigSection title={STAGE_LABELS.backfill} step={STAGE_NO.backfill}>
            <Field label="Method">
              <select
                value={backfillMethod}
                onChange={(e) => setBackfillMethod(e.target.value)}
              >
                <option value="">q_shell (default)</option>
                <option value="local">local</option>
                <option value="tv">tv</option>
                <option value="symmetry+tv">symmetry+tv</option>
              </select>
            </Field>
          </ConfigSection>

          <ConfigSection title={STAGE_LABELS.flatten} step={STAGE_NO.flatten}>
            <Switch label="Enable stage" checked={flatten} onChange={setFlatten} />
            <Field label="Estimator">
              <select
                value={flattenEstimator}
                disabled={!flatten}
                onChange={(e) => setFlattenEstimator(e.target.value)}
              >
                <option value="">floor (default)</option>
                <option value="median">median</option>
                <option value="mode">mode</option>
                <option value="snip">snip</option>
              </select>
            </Field>
          </ConfigSection>

          <ConfigSection title={STAGE_LABELS.pdf} step={STAGE_NO.pdf}>
            <Field label="Apodization">
              <select value={pdfApod} onChange={(e) => setPdfApod(e.target.value)}>
                <option value="">gaussian (default)</option>
                <option value="hann">hann</option>
                <option value="none">none</option>
              </select>
            </Field>
          </ConfigSection>

          <ConfigSection title="Run options">
            <Switch
              label="Force — recompute existing outputs"
              checked={force}
              onChange={setForce}
            />
          </ConfigSection>

          <div className="config-actions">
            <button
              type="button"
              className="btn btn-primary"
              onClick={onRun}
              disabled={running || !datasetId}
            >
              {running && <span className="spin" />}
              {running ? "Running…" : "Run pipeline"}
            </button>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={onCancel}
              disabled={!running}
            >
              Cancel
            </button>
          </div>
        </div>
      </div>

      {/* --------------------------------------------------- execution */}
      <div className="exec-col">
        <div className="card">
          <div className="card-head">
            <h3>Execution</h3>
            {running && <span className="spin" />}
          </div>
          <div className="card-body">
            {terminal === "done" && (
              <div className="banner ok">
                Job finished — open the 3D-ΔPDF tab to view the result.
              </div>
            )}
            {terminal === "cancelled" && (
              <div className="banner warn">Job cancelled.</div>
            )}
            {terminal === "error" && (
              <div className="banner err">Job failed — see the log below.</div>
            )}

            <ol className="stepper">
              {STAGES.map((s) => {
                const off = !flatten && s === "flatten";
                const info = stageInfo[s];
                const cls = stepClass(info?.status, off);
                return (
                  <li key={s} className={cls}>
                    <span className="step-rail">
                      <span className="step-dot" />
                    </span>
                    <span className="step-main">
                      <span className="step-name">{STAGE_LABELS[s]}</span>
                      <span className="step-sub">{stepSub(info, off)}</span>
                      {cls === "active" && info?.fraction != null && (
                        <span className="step-bar">
                          <span style={{ width: `${info.fraction * 100}%` }} />
                        </span>
                      )}
                    </span>
                  </li>
                );
              })}
            </ol>
          </div>
        </div>

        <div className="card">
          <div className="card-head">
            <h3>Log</h3>
            <span className="log-count">
              {events.length > 0 ? `${events.length} events` : ""}
            </span>
          </div>
          <div className="log" ref={logRef}>
            {events.length === 0 && !running && (
              <div className="log-empty">
                Pick a dataset and press Run. Stages with existing outputs are
                skipped — enable “force” to recompute them.
              </div>
            )}
            {events.map((ev, i) => (
              <div key={i} className={`log-line ${ev.status ?? ""}`}>
                <span className="log-stage">
                  {ev.stage ? STAGE_LABELS[ev.stage] ?? ev.stage : ""}
                </span>
                <span className="log-msg">{ev.message}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
