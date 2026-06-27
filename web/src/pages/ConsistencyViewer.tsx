// Back-FFT consistency viewer — the end-of-workflow check.  Inverse-transforms
// the ΔPDF back to reciprocal space and compares it to the diffuse data, with an
// adjustable |Q| band so you can see which signals come from low- vs high-|Q|
// data.  data | back-FFT | residual share plane / cut / contrast.

import { useEffect, useMemo, useRef, useState } from "react";

import { keepPreviousData, useQueries, useQuery } from "@tanstack/react-query";

import { fetchConsistencyMeta, fetchConsistencySlice, saveConsistencyDpdf } from "../api/client";
import { PYODIDE_MODE } from "../api/pyodideEngine";
import { useDatasets } from "../api/hooks";
import { COLORMAPS, SEQUENTIAL_NAMES, DIVERGING_NAMES, DIVERGING_NAME } from "../colormaps/luts";
import { SliceCanvas } from "../components/SliceCanvas";
import { UnitCellGrid } from "../components/UnitCellGrid";
import {
  ColormapBar,
  EmptyState,
  Field,
  IconAlert,
  RangeSlider,
  Segmented,
  Slider,
  Switch,
} from "../components/ui";
import { useDatasetStore, useInitializeDataset } from "../state/datasetStore";
import { useDpdfStore } from "../state/dpdfStore";
import { AXIS_INDEX, AXIS_TO_PLANE, type FixedAxis, REAL_AXIS_INDEX, REAL_AXIS_TO_PLANE, type RealAxis, useViewerStore } from "../state/viewerStore";

const AXES: FixedAxis[] = ["H", "K", "L"];
const REAL_AXES: RealAxis[] = ["X", "Y", "Z"];
const PANELS = [
  { key: "data", title: "Data (band-limited)" },
  { key: "recon", title: "Back-FFT  IFFT[ΔPDF]" },
  { key: "residual", title: "Residual (data − recon)" },
] as const;

export function ConsistencyViewer() {
  const datasetsQ = useDatasets();
  const datasets = useMemo(() => datasetsQ.data ?? [], [datasetsQ.data]);
  useInitializeDataset(datasets);
  const datasetId = useDatasetStore((s) => s.datasetId);
  const setDatasetId = useDatasetStore((s) => s.setDataset);
  // Only datasets whose ΔPDF input (flattened/backfilled) exists can be inverted.
  const usable = useMemo(
    () =>
      datasets.filter((d) =>
        d.stages.some(
          (s) =>
            s.kind === "hkl" &&
            s.exists &&
            (s.name === "flattened" || s.name === "backfilled"),
        ),
    ),
    [datasets],
  );
  const dataset = datasets.find((d) => d.id === datasetId);
  const selectedUsable = Boolean(dataset && usable.some((d) => d.id === dataset.id));

  const fixedAxis = useViewerStore((s) => s.fixedAxis);
  const cutIndex = useViewerStore((s) => s.cutIndex);
  const contrast = useViewerStore((s) => s.contrast);
  const zoom = useViewerStore((s) => s.zoom);
  const log = useViewerStore((s) => s.log);
  const colormap = useViewerStore((s) => s.colormap);
  const setFixedAxis = useViewerStore((s) => s.setFixedAxis);
  const setCutIndex = useViewerStore((s) => s.setCutIndex);
  const setContrast = useViewerStore((s) => s.setContrast);
  const setZoom = useViewerStore((s) => s.setZoom);
  const setLog = useViewerStore((s) => s.setLog);
  const setColormap = useViewerStore((s) => s.setColormap);

  // Display controls (contrast / window / colormap / gridlines) are shared with the
  // 3D-ΔPDF viewer via the store.  The cut *indices* are NOT shared: the
  // consistency round-trip can use a different real-space grid than the saved ΔPDF
  // volume, so sharing indices would push this viewer's centre cut onto the other
  // viewer's slider (out of range → it snaps to max).  Keep cuts local here.
  const dpdfContrast = useDpdfStore((s) => s.contrast);
  const windowFull = useDpdfStore((s) => s.windowFull);
  const dpdfColormap = useDpdfStore((s) => s.colormap);
  const dpdfGridlines = useDpdfStore((s) => s.gridlines);

  const setDpdfContrast = useDpdfStore((s) => s.setContrast);
  const setWindowFull = useDpdfStore((s) => s.setWindowFull);
  const setDpdfColormap = useDpdfStore((s) => s.setColormap);
  const setDpdfGridlines = useDpdfStore((s) => s.setGridlines);

  const [dpdfCuts, setDpdfCuts] = useState<{ X: number; Y: number; Z: number }>({
    X: 0,
    Y: 0,
    Z: 0,
  });

  const [band, setBand] = useState<{ min: number; max: number } | null>(null);
  const [draftMin, setDraftMin] = useState(0);
  const [draftMax, setDraftMax] = useState(0);

  const [rBand, setRBand] = useState<{ min: number; max: number } | null>(null);
  const [draftRMin, setDraftRMin] = useState(0);
  const [draftRMax, setDraftRMax] = useState(0);

  // Save the final band-limited 3D-ΔPDF to disk (end of the workflow).
  const [saveState, setSaveState] = useState<
    { status: "idle" | "saving" } | { status: "saved"; filename: string } | { status: "error"; message: string }
  >({ status: "idle" });

  const [dpdfFixedAxis, setDpdfFixedAxis] = useState<RealAxis>("Z");

  // Back-FFT panel swaps between the reconstruction and the residual (data − recon).
  const [backView, setBackView] = useState<"recon" | "residual">("recon");

  const dpdfCutIndex = dpdfCuts[dpdfFixedAxis];
  const setDpdfCutIndex = (i: number) =>
    setDpdfCuts((c) => ({ ...c, [dpdfFixedAxis]: i }));

  // Meta drives the metrics + grid ranges + |Q| span; it recomputes the (heavy)
  // round trip whenever the committed band changes.
  const metaQ = useQuery({
    queryKey: ["consMeta", datasetId, band?.min, band?.max, rBand?.min, rBand?.max],
    queryFn: () =>
      fetchConsistencyMeta(datasetId as string, band?.min, band?.max, rBand?.min, rBand?.max),
    enabled: Boolean(datasetId && selectedUsable),
    placeholderData: keepPreviousData,
  });
  const meta = metaQ.data;
  const spanMax = meta ? Math.ceil(meta.q_data_max) : 0;
  const rSpanMax = meta ? Math.ceil(meta.r_data_max) : 0;

  // Initialise the draft |Q| / |R| bands to the full span once it is known, and
  // re-initialise (clearing any applied band) when the dataset changes.  A single
  // effect keyed on (dataset, spans) — rather than separate "init to full" and
  // "reset to 0" effects — avoids a race where the two clobber each other and
  // leave both range-slider knobs stuck at 0: with meta cached the span is known
  // on mount, so both effects ran in one commit (worsened by StrictMode's
  // double-invocation) and the reset won.  The ref makes this idempotent, and it
  // does not depend on the drafts, so the user's own band edits are preserved.
  const bandInitKey = useRef("");
  useEffect(() => {
    if (spanMax <= 0 || rSpanMax <= 0) return;
    const key = `${datasetId}|${spanMax}|${rSpanMax}`;
    if (bandInitKey.current === key) return;
    bandInitKey.current = key;
    setBand(null);
    setDraftMin(0);
    setDraftMax(spanMax);
    setRBand(null);
    setDraftRMin(0);
    setDraftRMax(rSpanMax);
  }, [datasetId, spanMax, rSpanMax]);

  const axisInfo = useMemo(() => {
    if (!meta) return null;
    const i = AXIS_INDEX[fixedAxis];
    const [min, max] = [meta.h_range, meta.k_range, meta.l_range][i];
    const n = meta.shape[i];
    return { min, max, n, step: n > 1 ? (max - min) / (n - 1) : 0 };
  }, [meta, fixedAxis]);

  useEffect(() => {
    if (axisInfo) setCutIndex(Math.floor(axisInfo.n / 2));
  }, [axisInfo, setCutIndex]);

  const a = meta?.lattice.a ?? 1;
  const b = meta?.lattice.b ?? 1;
  const c = meta?.lattice.c ?? 1;

  let latX = 1, latY = 1, latCut = 1;
  if (fixedAxis === "H") { latCut = a; latX = b; latY = c; }
  else if (fixedAxis === "K") { latCut = b; latX = a; latY = c; }
  else if (fixedAxis === "L") { latCut = c; latX = a; latY = b; }

  let dpdfLatX = a, dpdfLatY = b;
  if (dpdfFixedAxis === "X") { dpdfLatX = b; dpdfLatY = c; }
  else if (dpdfFixedAxis === "Y") { dpdfLatX = c; dpdfLatY = a; }
  else if (dpdfFixedAxis === "Z") { dpdfLatX = a; dpdfLatY = b; }

  const dpdfAxisInfo = useMemo(() => {
    if (!meta || !meta.x_range || !meta.dpdf_shape) return null;
    const i = REAL_AXIS_INDEX[dpdfFixedAxis];
    const [min, max] = [meta.x_range, meta.y_range, meta.z_range][i];
    const n = meta.dpdf_shape[i];
    return { min, max, n, step: n > 1 ? (max - min) / (n - 1) : 0 };
  }, [meta, dpdfFixedAxis]);

  // Centre the selected real-space cut when the grid or axis changes (functional
  // setState keeps the dep list stable — setDpdfCutIndex is a fresh closure each
  // render).
  useEffect(() => {
    if (dpdfAxisInfo) {
      setDpdfCuts((c) => ({ ...c, [dpdfFixedAxis]: Math.floor(dpdfAxisInfo.n / 2) }));
    }
  }, [dpdfAxisInfo, dpdfFixedAxis]);

  const idx = axisInfo ? Math.min(cutIndex, axisInfo.n - 1) : 0;
  const value = axisInfo ? axisInfo.min + idx * axisInfo.step : 0;
  const plane = AXIS_TO_PLANE[fixedAxis];

  const dpdfIdx = dpdfAxisInfo ? Math.min(dpdfCutIndex, dpdfAxisInfo.n - 1) : 0;
  const dpdfValue = dpdfAxisInfo ? dpdfAxisInfo.min + dpdfIdx * dpdfAxisInfo.step : 0;
  const dpdfPlane = REAL_AXIS_TO_PLANE[dpdfFixedAxis];
  const seqLut = COLORMAPS[colormap] ?? COLORMAPS.inferno;
  const divLut = COLORMAPS[dpdfColormap] ?? COLORMAPS[DIVERGING_NAME];

  const commitCut = (v: number) => {
    if (!axisInfo || axisInfo.step === 0) return;
    const i = Math.round((v - axisInfo.min) / axisInfo.step);
    setCutIndex(Math.max(0, Math.min(axisInfo.n - 1, i)));
  };

  const sliceResults = useQueries({
    queries: PANELS.map((p) => ({
      queryKey: ["consSlice", datasetId, p.key, plane, value, band?.min, band?.max, rBand?.min, rBand?.max],
      queryFn: () =>
        fetchConsistencySlice(
          datasetId as string, p.key, plane, value, band?.min, band?.max, rBand?.min, rBand?.max,
        ),
      enabled: Boolean(datasetId && selectedUsable && axisInfo),
      placeholderData: keepPreviousData,
    })),
  });

  const dpdfSliceResult = useQuery({
    queryKey: ["consSlice", datasetId, "dpdf", dpdfPlane, dpdfValue, band?.min, band?.max, rBand?.min, rBand?.max],
    queryFn: () =>
      fetchConsistencySlice(
        datasetId as string, "dpdf", dpdfPlane, dpdfValue, band?.min, band?.max, rBand?.min, rBand?.max,
      ),
    enabled: Boolean(datasetId && selectedUsable && dpdfAxisInfo),
    placeholderData: keepPreviousData,
  });

  const rmOf = (i: number) => {
    const rm = sliceResults[i]?.data?.header.robust_max;
    return rm && Number.isFinite(rm) ? rm : 0;
  };
  const seqVmax = (Math.max(rmOf(0), rmOf(1)) || 1) * contrast;
  const dpdfVmax = (dpdfSliceResult.data?.header.robust_max || 1) * dpdfContrast;

  const m = meta?.metrics;
  const bandApplied = band !== null;
  const canApply = draftMax > draftMin;
  const rBandApplied = rBand !== null;
  const canApplyR = draftRMax > draftRMin;

  // Saving reuses the cached reconstruction for the *applied* bands (band/rBand),
  // i.e. exactly what the panels show — pending draft edits are saved only after
  // "Apply bounds".
  const saveDpdf = async () => {
    if (!datasetId) return;
    setSaveState({ status: "saving" });
    try {
      const res = await saveConsistencyDpdf(
        datasetId, band?.min, band?.max, rBand?.min, rBand?.max);
      setSaveState({ status: "saved", filename: res.filename });
    } catch (e) {
      setSaveState({ status: "error", message: (e as Error).message });
    }
  };
  // A saved file reflects one band selection; clear the confirmation when the
  // applied bands (or dataset) change so the message can't go stale.
  useEffect(() => {
    setSaveState({ status: "idle" });
  }, [datasetId, band?.min, band?.max, rBand?.min, rBand?.max]);

  const backResult = backView === "recon" ? sliceResults[1] : sliceResults[2];
  const consistent = m && Number.isFinite(m.pearson_r) && m.pearson_r >= 0.95;

  // The image area shared by every flow panel: error → message, data → canvas,
  // otherwise a loading skeleton, with a fetch spinner overlaid.
  const flowImage = (
    result: { data?: import("../api/types").Slice; isError?: boolean; error?: unknown; isFetching?: boolean } | undefined,
    canvas: React.ReactNode,
    overlay?: React.ReactNode,
  ) => (
    <div className="qr-img">
      {result?.isError ? (
        <div className="panel-err">{(result.error as Error)?.message}</div>
      ) : result?.data ? (
        <>
          {canvas}
          {overlay}
        </>
      ) : (
        <div className="skeleton" style={{ width: "100%", height: "100%" }} />
      )}
      {(result?.isFetching || metaQ.isFetching) && result?.data && (
        <span className="spin qr-img-spin" />
      )}
    </div>
  );

  return (
    <div className="page-body qr-page">
      {/* ── Header: dataset · round-trip · description · apply/full ───────── */}
      <div className="qr-header">
        <div className="qr-header-dataset">
          <span className="qr-eyebrow">Dataset</span>
          <select
            value={datasetId ?? ""}
            onChange={(e) => setDatasetId(e.target.value)}
          >
            {datasets.map((d) => (
              <option key={d.id} value={d.id} title={d.raw_name}>
                {d.temperature ?? d.stem}
              </option>
            ))}
          </select>
        </div>

        <div className="qr-divider" />

        <div className="qr-roundtrip">
          <span className="qr-rt qr-rt--q">Q</span>
          <span className="qr-rt-arrow">→</span>
          <span className="qr-rt qr-rt--r">R</span>
          <span className="qr-rt-arrow">→</span>
          <span className="qr-rt qr-rt--qp">Q′</span>
        </div>

        <span className="qr-desc">
          Band-limited round-trip · reconstruction compared to the input
        </span>

        <div className="qr-header-actions">
          <button
            type="button"
            className="btn btn-primary"
            disabled={(!canApply && !canApplyR) || metaQ.isFetching}
            onClick={() => {
              setBand({ min: draftMin, max: draftMax });
              setRBand({ min: draftRMin, max: draftRMax });
            }}
          >
            {metaQ.isFetching ? "Computing…" : "Apply bounds"}
          </button>
          <button
            type="button"
            className="btn btn-ghost"
            disabled={
              !bandApplied && draftMin === 0 && draftMax === spanMax &&
              !rBandApplied && draftRMin === 0 && draftRMax === rSpanMax
            }
            onClick={() => {
              setBand(null);
              setDraftMin(0);
              setDraftMax(spanMax);
              setRBand(null);
              setDraftRMin(0);
              setDraftRMax(rSpanMax);
            }}
          >
            Full
          </button>
        </div>
      </div>

      {/* ── Display-control clusters: reciprocal | real-space ────────────── */}
      <div className="qr-clusters">
        <div className="qr-cluster">
          <div className="qr-cluster-head">
            <span className="qr-cluster-title">
              Reciprocal display · Data · Back-FFT · Residual
            </span>
            <div className="qr-cluster-toggle">
              <Switch label="Log scale" checked={log} onChange={setLog} />
            </div>
          </div>
          <div className="qr-cluster-controls">
            <Field label="Axis">
              <Segmented
                options={AXES}
                value={fixedAxis}
                onChange={(a) => setFixedAxis(a as FixedAxis)}
              />
            </Field>
            <div className="qr-cluster-slider">
              <Slider
                label="Contrast"
                readout={`× ${contrast.toFixed(1)}`}
                min={0.1}
                max={20}
                step={0.1}
                value={contrast}
                onChange={setContrast}
              />
            </div>
            <div className="qr-cluster-slider">
              <Slider
                label="Zoom"
                readout={`× ${zoom.toFixed(1)}`}
                min={1}
                max={10}
                step={0.5}
                value={zoom}
                onChange={setZoom}
              />
            </div>
          </div>
          <div className="qr-cluster-cmap">
            <span className="field-label">Colormap</span>
            <select value={colormap} onChange={(e) => setColormap(e.target.value)}>
              {SEQUENTIAL_NAMES.map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
            <ColormapBar lut={seqLut} />
          </div>
        </div>

        <div className="qr-cluster">
          <div className="qr-cluster-head">
            <span className="qr-cluster-title">Real-space display · 3D-ΔPDF</span>
            <div className="qr-cluster-toggle">
              <Switch label="Unit cells" checked={dpdfGridlines} onChange={setDpdfGridlines} />
            </div>
          </div>
          <div className="qr-cluster-controls">
            <Field label="Axis">
              <Segmented
                options={REAL_AXES}
                value={dpdfFixedAxis}
                onChange={(a) => setDpdfFixedAxis(a as RealAxis)}
              />
            </Field>
            <div className="qr-cluster-slider">
              <Slider
                label="Contrast"
                readout={`× ${dpdfContrast.toFixed(1)}`}
                min={0.1}
                max={20}
                step={0.1}
                value={dpdfContrast}
                onChange={setDpdfContrast}
              />
            </div>
            <div className="qr-cluster-slider">
              <Slider
                label="Window"
                readout={`${windowFull.toFixed(0)} Å`}
                min={10}
                max={160}
                step={2}
                value={windowFull}
                onChange={setWindowFull}
              />
            </div>
          </div>
          <div className="qr-cluster-cmap">
            <span className="field-label">Colormap</span>
            <select value={dpdfColormap} onChange={(e) => setDpdfColormap(e.target.value)}>
              {DIVERGING_NAMES.map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
            <ColormapBar lut={divLut} />
          </div>
        </div>
      </div>

      {datasetsQ.isError && (
        <EmptyState
          error
          icon={<IconAlert />}
          title="Backend unreachable"
          hint="Start the API server (nebula3d-web or uvicorn on port 8000) and reload."
        />
      )}
      {!datasetsQ.isError && usable.length === 0 && (
        <EmptyState
          title="No invertible volumes yet"
          hint="Run the pipeline first — the flattened/backfilled volume feeds this back-FFT check."
        />
      )}
      {!datasetsQ.isError && usable.length > 0 && dataset && !selectedUsable && (
        <EmptyState
          title="No invertible volume for this dataset"
          hint="Run this dataset through backfill or flatten first — the flattened/backfilled volume feeds this back-FFT check."
        />
      )}
      {metaQ.isError && (
        <EmptyState
          error
          icon={<IconAlert />}
          title="Consistency check failed"
          hint={(metaQ.error as Error)?.message}
        />
      )}

      {/* ── Slice flow row: Data → fwd FFT → ΔPDF → back FFT → Back-FFT ──── */}
      {selectedUsable && (
        <div className="qr-flow-row">
          {/* Stage 1 — Data D(Q) */}
          <div className="qr-panel">
            <div className="qr-panel-head">
              <span className="qr-panel-titlegroup">
                <span className="qr-rt-badge qr-rt--q">Q</span>
                <span className="qr-panel-title">Data</span>
              </span>
              <span className="qr-panel-tag">recip</span>
            </div>
            {flowImage(
              sliceResults[0],
              <SliceCanvas
                slice={sliceResults[0]!.data!}
                lut={seqLut}
                vmax={seqVmax}
                vmin={0}
                log={log}
                fit
                zoom={zoom}
                bands={[draftMin, draftMax]}
                cutDistance={value}
                reciprocalAxes
                latX={latX}
                latY={latY}
                latCut={latCut}
              />,
            )}
            <div className="qr-foot">
              <div className="qr-foot-cut">
                <Slider
                  label={`Cut ${fixedAxis}`}
                  readout={axisInfo ? undefined : "—"}
                  valueInput={
                    axisInfo
                      ? { value, prefix: `${fixedAxis} =`, suffix: "r.l.u.", onCommit: commitCut }
                      : undefined
                  }
                  min={0}
                  max={axisInfo ? axisInfo.n - 1 : 0}
                  value={idx}
                  disabled={!axisInfo}
                  onChange={setCutIndex}
                />
              </div>
              <div className="qr-foot-band">
                <RangeSlider
                  label="|Q| band"
                  readout={`${draftMin.toFixed(2)} … ${draftMax.toFixed(2)} Å⁻¹`}
                  min={0}
                  max={spanMax || 1}
                  step={0.05}
                  valueMin={draftMin}
                  valueMax={draftMax}
                  disabled={!spanMax}
                  onChange={(lo, hi) => {
                    setDraftMin(lo);
                    setDraftMax(hi);
                  }}
                />
              </div>
            </div>
          </div>

          {/* Stage 2 — ΔPDF g(R) */}
          <div className="qr-panel qr-panel--hi">
            <div className="qr-panel-head">
              <span className="qr-panel-titlegroup">
                <span className="qr-rt-badge qr-rt--r">R</span>
                <span className="qr-panel-title">
                  3D-ΔPDF <span className="qr-panel-tag">· real</span>
                </span>
              </span>
              <div className="qr-panel-head-actions">
                {saveState.status === "saved" && (
                  <span className="qr-saved" title={saveState.filename}>
                    <span className="qr-dot" /> saved
                  </span>
                )}
                {saveState.status === "error" && (
                  <span className="qr-saved qr-saved--err" title={saveState.message}>
                    save failed
                  </span>
                )}
                <button
                  type="button"
                  className="btn btn-primary qr-save-btn"
                  title={
                    PYODIDE_MODE
                      ? "Saving to disk requires the desktop / server app"
                      : "Save the band-limited 3D-ΔPDF (final processed file) to data/processed"
                  }
                  disabled={
                    !selectedUsable || PYODIDE_MODE ||
                    metaQ.isFetching || saveState.status === "saving"
                  }
                  onClick={saveDpdf}
                >
                  ↓ {saveState.status === "saving" ? "Saving…" : "Save ΔPDF"}
                </button>
              </div>
            </div>
            {flowImage(
              dpdfSliceResult,
              <SliceCanvas
                slice={dpdfSliceResult.data!}
                lut={divLut}
                vmax={dpdfVmax}
                vmin={0}
                log={false}
                diverging
                bands={[draftRMin, draftRMax]}
                cutDistance={dpdfValue}
                latX={dpdfLatX}
                latY={dpdfLatY}
                windowA={windowFull / 2}
              />,
              dpdfGridlines ? (
                <UnitCellGrid half={windowFull / 2} latX={dpdfLatX ?? null} latY={dpdfLatY ?? null} />
              ) : undefined,
            )}
            <div className="qr-foot">
              <div className="qr-foot-cut">
                <Slider
                  label={`Cut ${dpdfFixedAxis}`}
                  readout={dpdfAxisInfo ? undefined : "—"}
                  valueInput={
                    dpdfAxisInfo
                      ? {
                          value: dpdfValue,
                          prefix: `${dpdfFixedAxis} =`,
                          suffix: "Å",
                          onCommit: (v) => {
                            if (!dpdfAxisInfo || dpdfAxisInfo.step === 0) return;
                            const i = Math.round((v - dpdfAxisInfo.min) / dpdfAxisInfo.step);
                            setDpdfCutIndex(Math.max(0, Math.min(dpdfAxisInfo.n - 1, i)));
                          },
                        }
                      : undefined
                  }
                  min={0}
                  max={dpdfAxisInfo ? dpdfAxisInfo.n - 1 : 0}
                  value={dpdfIdx}
                  disabled={!dpdfAxisInfo}
                  onChange={setDpdfCutIndex}
                />
              </div>
              <div className="qr-foot-band">
                <RangeSlider
                  label="|R| band"
                  readout={`${draftRMin.toFixed(2)} … ${draftRMax.toFixed(2)} Å`}
                  min={0}
                  max={rSpanMax || 1}
                  step={1}
                  valueMin={draftRMin}
                  valueMax={draftRMax}
                  disabled={!rSpanMax}
                  onChange={(lo, hi) => {
                    setDraftRMin(lo);
                    setDraftRMax(hi);
                  }}
                />
              </div>
            </div>
          </div>

          {/* Stage 3 — Back-FFT reconstruction / residual */}
          <div className="qr-panel">
            <div className="qr-panel-head">
              <span className="qr-panel-titlegroup">
                <span className="qr-rt-badge qr-rt--qp">Q′</span>
                <span className="qr-panel-title">Back-FFT</span>
              </span>
              <Segmented
                options={["Recon", "Residual"]}
                value={backView === "recon" ? "Recon" : "Residual"}
                onChange={(v) => setBackView(v === "Recon" ? "recon" : "residual")}
              />
            </div>
            {flowImage(
              backResult,
              <SliceCanvas
                slice={backResult!.data!}
                lut={seqLut}
                vmax={seqVmax}
                vmin={0}
                log={backView === "recon" ? log : false}
                fit
                zoom={zoom}
                reciprocalAxes
                latX={latX}
                latY={latY}
                latCut={latCut}
              />,
            )}
            <div className="qr-foot qr-foot--metrics">
              {backView === "recon" ? (
                <>
                  <div className="qr-metric-row">
                    {consistent && (
                      <span className="qr-verdict">
                        <span className="qr-dot" /> CONSISTENT
                      </span>
                    )}
                    <span className="qr-metric-r">
                      r = {m && Number.isFinite(m.pearson_r) ? m.pearson_r.toFixed(5) : "—"}
                    </span>
                    <span className="qr-metric-rms">
                      RMS {m ? m.normalized_rms.toExponential(2) : "—"}
                    </span>
                  </div>
                  <span className="qr-foot-caption">
                    shares the {fixedAxis} cut &amp; display of the Data panel
                  </span>
                </>
              ) : (
                <>
                  <div className="qr-metric-row">
                    <span className="qr-metric-r">D − D′</span>
                    <span className="qr-metric-rms">log scale off</span>
                  </div>
                  <span className="qr-foot-caption">
                    data minus reconstruction · same {fixedAxis} cut
                  </span>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
