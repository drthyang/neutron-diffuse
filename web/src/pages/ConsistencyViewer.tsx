// Back-FFT consistency viewer — the end-of-workflow check.  Inverse-transforms
// the ΔPDF back to reciprocal space and compares it to the diffuse data, with an
// adjustable |Q| band so you can see which signals come from low- vs high-|Q|
// data.  data | back-FFT | residual share plane / cut / contrast.

import { useEffect, useMemo, useState } from "react";

import { keepPreviousData, useQueries, useQuery } from "@tanstack/react-query";

import { fetchConsistencyMeta, fetchConsistencySlice } from "../api/client";
import { useDatasets } from "../api/hooks";
import { COLORMAPS, SEQUENTIAL_NAMES, DIVERGING_NAMES, DIVERGING_NAME } from "../colormaps/luts";
import { SlicePanel } from "../components/SlicePanel";
import {
  ColormapBar,
  EmptyState,
  Field,
  IconAlert,
  MetaStrip,
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
  const log = useViewerStore((s) => s.log);
  const colormap = useViewerStore((s) => s.colormap);
  const setFixedAxis = useViewerStore((s) => s.setFixedAxis);
  const setCutIndex = useViewerStore((s) => s.setCutIndex);
  const setContrast = useViewerStore((s) => s.setContrast);
  const setLog = useViewerStore((s) => s.setLog);
  const setColormap = useViewerStore((s) => s.setColormap);

  const dpdfContrast = useDpdfStore((s) => s.contrast);
  const windowFull = useDpdfStore((s) => s.windowFull);
  const dpdfColormap = useDpdfStore((s) => s.colormap);
  const dpdfGridlines = useDpdfStore((s) => s.gridlines);
  const cutX = useDpdfStore((s) => s.cutX);
  const cutY = useDpdfStore((s) => s.cutY);
  const cutZ = useDpdfStore((s) => s.cutZ);

  const setDpdfContrast = useDpdfStore((s) => s.setContrast);
  const setWindowFull = useDpdfStore((s) => s.setWindowFull);
  const setDpdfColormap = useDpdfStore((s) => s.setColormap);
  const setDpdfGridlines = useDpdfStore((s) => s.setGridlines);
  const setCutX = useDpdfStore((s) => s.setCutX);
  const setCutY = useDpdfStore((s) => s.setCutY);
  const setCutZ = useDpdfStore((s) => s.setCutZ);

  const [band, setBand] = useState<{ min: number; max: number } | null>(null);
  const [draftMin, setDraftMin] = useState(0);
  const [draftMax, setDraftMax] = useState(0);

  const [rBand, setRBand] = useState<{ min: number; max: number } | null>(null);
  const [draftRMin, setDraftRMin] = useState(0);
  const [draftRMax, setDraftRMax] = useState(0);

  const [dpdfFixedAxis, setDpdfFixedAxis] = useState<RealAxis>("Z");
  const dpdfCutIndex = dpdfFixedAxis === "X" ? cutX : dpdfFixedAxis === "Y" ? cutY : cutZ;
  const setDpdfCutIndex = dpdfFixedAxis === "X" ? setCutX : dpdfFixedAxis === "Y" ? setCutY : setCutZ;

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

  // Initialise the band drafts to the full |Q| span once it is known.
  useEffect(() => {
    if (spanMax > 0 && draftMax === 0) setDraftMax(spanMax);
  }, [spanMax, draftMax]);

  useEffect(() => {
    if (rSpanMax > 0 && draftRMax === 0) setDraftRMax(rSpanMax);
  }, [rSpanMax, draftRMax]);

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

  useEffect(() => {
    if (dpdfAxisInfo) setDpdfCutIndex(Math.floor(dpdfAxisInfo.n / 2));
  }, [dpdfAxisInfo, setDpdfCutIndex]);

  const idx = axisInfo ? Math.min(cutIndex, axisInfo.n - 1) : 0;
  const value = axisInfo ? axisInfo.min + idx * axisInfo.step : 0;
  const plane = AXIS_TO_PLANE[fixedAxis];

  const dpdfIdx = dpdfAxisInfo ? Math.min(dpdfCutIndex, dpdfAxisInfo.n - 1) : 0;
  const dpdfValue = dpdfAxisInfo ? dpdfAxisInfo.min + dpdfIdx * dpdfAxisInfo.step : 0;
  const dpdfPlane = REAL_AXIS_TO_PLANE[dpdfFixedAxis];
  const seqLut = COLORMAPS[colormap] ?? COLORMAPS.inferno;
  const divLut = COLORMAPS[dpdfColormap] ?? COLORMAPS[DIVERGING_NAME];

  useEffect(() => {
    // When the global dataset changes (e.g. from Reciprocal Viewer),
    // we need to reset the bands so they re-calculate limits.
    setBand(null);
    setDraftMin(0);
    setDraftMax(0);
    setRBand(null);
    setDraftRMin(0);
    setDraftRMax(0);
  }, [datasetId]);

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
  const residScale = (rmOf(2) || 1) * contrast;
  const dpdfVmax = (dpdfSliceResult.data?.header.robust_max || 1) * dpdfContrast;

  const m = meta?.metrics;
  const bandApplied = band !== null;
  const canApply = draftMax > draftMin;
  const rBandApplied = rBand !== null;
  const canApplyR = draftRMax > draftRMin;

  return (
    <div className="page-body">
      <div className="toolbar">
        <Field label="Dataset">
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
        </Field>
      </div>

      <div className="toolbar">
        <Field label="Fixed axis">
          <Segmented
            options={AXES}
            value={fixedAxis}
            onChange={(a) => setFixedAxis(a as FixedAxis)}
          />
        </Field>

        <Slider
          grow
          label={`Cut along ${fixedAxis}`}
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

        <Slider
          label="Contrast"
          readout={`× ${contrast.toFixed(1)}`}
          min={0.1}
          max={20}
          step={0.1}
          value={contrast}
          onChange={setContrast}
        />

        <Switch label="Log scale" checked={log} onChange={setLog} />

        <Field label="Colormap">
          <select value={colormap} onChange={(e) => setColormap(e.target.value)}>
            {SEQUENTIAL_NAMES.map((name) => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
          <ColormapBar lut={seqLut} />
        </Field>
      </div>

      <div className="toolbar">
        <Field label="Fixed axis">
          <Segmented
            options={REAL_AXES}
            value={dpdfFixedAxis}
            onChange={(a) => setDpdfFixedAxis(a as RealAxis)}
          />
        </Field>
        <Slider
          grow
          label={`Cut along ${dpdfFixedAxis}`}
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
        <Slider
          label="Contrast"
          readout={`× ${dpdfContrast.toFixed(1)}`}
          min={0.1}
          max={20}
          step={0.1}
          value={dpdfContrast}
          onChange={setDpdfContrast}
        />
        <Switch label="Unit cells" checked={dpdfGridlines} onChange={setDpdfGridlines} />
        <Slider
          label="Window"
          readout={`${windowFull.toFixed(0)} Å`}
          min={10}
          max={160}
          step={2}
          value={windowFull}
          onChange={setWindowFull}
        />
        <Field label="Colormap">
          <select value={dpdfColormap} onChange={(e) => setDpdfColormap(e.target.value)}>
            {DIVERGING_NAMES.map((name) => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
          <ColormapBar lut={divLut} />
        </Field>
      </div>
      
      <div className="toolbar">
        <RangeSlider
          grow
          label="|Q| band (band-limit the data)"
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
        <RangeSlider
          grow
          label="|R| band (band-limit back-FFT)"
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
        <Field label="Recompute">
          <div style={{ display: "flex", gap: 8 }}>
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
              disabled={(!bandApplied && draftMin === 0 && draftMax === spanMax) && (!rBandApplied && draftRMin === 0 && draftRMax === rSpanMax)}
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
        </Field>
      </div>

      {datasetsQ.isError && (
        <EmptyState
          error
          icon={<IconAlert />}
          title="Backend unreachable"
          hint="Start the API server (ndiff-web or uvicorn on port 8000) and reload."
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

      {selectedUsable && (
      <div className="panel-grid">
        {PANELS.map((p, i) => {
          const isData = p.key === "data";
          return (
            <SlicePanel
              key={p.key}
              title={p.title}
              data={sliceResults[i]?.data}
              isFetching={sliceResults[i]?.isFetching || metaQ.isFetching}
              isError={sliceResults[i]?.isError}
              error={sliceResults[i]?.error as Error | null}
              lut={seqLut}
              vmax={p.key === "residual" ? residScale : seqVmax}
              vmin={0}
              log={p.key === "residual" ? false : log}
              bands={isData ? [draftMin, draftMax] : undefined}
              cutDistance={isData ? value : undefined}
              reciprocalAxes
              latX={latX}
              latY={latY}
              latCut={latCut}
            />
          );
        })}
        <SlicePanel
          title="3D-ΔPDF (Real Space)"
          data={dpdfSliceResult.data}
          isFetching={dpdfSliceResult.isFetching || metaQ.isFetching}
          isError={dpdfSliceResult.isError}
          error={dpdfSliceResult.error as Error | null}
          lut={divLut}
          vmax={dpdfVmax}
          vmin={0}
          log={false}
          diverging={true}
          bands={[draftRMin, draftRMax]}
          cutDistance={dpdfValue}
          latX={dpdfLatX}
          latY={dpdfLatY}
          windowA={windowFull / 2}
          gridlines={dpdfGridlines}
        />
      </div>
      )}

      {selectedUsable && m && (
        <MetaStrip
          items={[
            {
              key: "Pearson r",
              value: Number.isFinite(m.pearson_r) ? m.pearson_r.toFixed(5) : "—",
            },
            { key: "Normalised RMS", value: m.normalized_rms.toExponential(2) },
            {
              key: "|Q| band",
              value: m.q_band
                ? `${m.q_band[0].toFixed(2)} … ${m.q_band[1].toFixed(2)} Å⁻¹`
                : `full (0 … ${spanMax.toFixed(2)})`,
            },
            {
              key: "|R| band",
              value: m.r_band
                ? `${m.r_band[0].toFixed(2)} … ${m.r_band[1].toFixed(2)} Å`
                : `full (0 … ${rSpanMax.toFixed(2)})`,
            },
            {
              key: "per-plane r",
              value: Object.entries(m.per_plane_r)
                .map(([h, r]) => `H${h}: ${Number(r).toFixed(3)}`)
                .join("   "),
            },
            { key: "Voxels", value: m.n_voxels.toLocaleString() },
            { key: "Plane", value: plane },
          ]}
        />
      )}
    </div>
  );
}
