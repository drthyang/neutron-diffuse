// Reciprocal-space cleanup viewer — replaces examples/explore_slice.py.
// One panel per existing HKLVolume stage, all sharing plane / cut / contrast /
// log / colormap controls so the cleanup stages are directly comparable.

import { useEffect, useMemo } from "react";

import { keepPreviousData, useQueries } from "@tanstack/react-query";

import { fetchSlice } from "../api/client";
import { useDatasets, useMeta } from "../api/hooks";
import { COLORMAPS, SEQUENTIAL_NAMES } from "../colormaps/luts";
import { SlicePanel } from "../components/SlicePanel";
import {
  ColormapBar,
  EmptyState,
  Field,
  IconAlert,
  MetaStrip,
  Segmented,
  Slider,
  Switch,
} from "../components/ui";
import {
  AXIS_INDEX,
  AXIS_TO_PLANE,
  type FixedAxis,
  useViewerStore,
} from "../state/viewerStore";

const STAGE_ORDER = ["raw", "ringremoved", "braggpunched", "backfilled", "flattened"];
const STAGE_LABELS: Record<string, string> = {
  raw: "Raw",
  ringremoved: "Ring-removed",
  braggpunched: "Bragg-punched",
  backfilled: "Backfilled",
  flattened: "Flattened",
};
const AXES: FixedAxis[] = ["H", "K", "L"];

export function ReciprocalViewer() {
  const datasetsQ = useDatasets();
  const datasets = useMemo(() => datasetsQ.data ?? [], [datasetsQ.data]);

  const datasetId = useViewerStore((s) => s.datasetId);
  const fixedAxis = useViewerStore((s) => s.fixedAxis);
  const cutIndex = useViewerStore((s) => s.cutIndex);
  const contrast = useViewerStore((s) => s.contrast);
  const log = useViewerStore((s) => s.log);
  const colormap = useViewerStore((s) => s.colormap);
  const setDataset = useViewerStore((s) => s.setDataset);
  const setFixedAxis = useViewerStore((s) => s.setFixedAxis);
  const setCutIndex = useViewerStore((s) => s.setCutIndex);
  const setContrast = useViewerStore((s) => s.setContrast);
  const setLog = useViewerStore((s) => s.setLog);
  const setColormap = useViewerStore((s) => s.setColormap);

  useEffect(() => {
    if (!datasetId && datasets.length) setDataset(datasets[0].id);
  }, [datasetId, datasets, setDataset]);

  const dataset = datasets.find((d) => d.id === datasetId);
  const stages = (dataset?.stages ?? [])
    .filter((s) => s.kind === "hkl" && s.exists)
    .sort((a, b) => STAGE_ORDER.indexOf(a.name) - STAGE_ORDER.indexOf(b.name));

  const metaVolId = stages[0]?.volume_id;
  const meta = useMeta(metaVolId).data;

  const axisInfo = useMemo(() => {
    if (!meta) return null;
    const i = AXIS_INDEX[fixedAxis];
    const [min, max] = [meta.h_range, meta.k_range, meta.l_range][i];
    const n = meta.shape[i];
    return { min, max, n, step: n > 1 ? (max - min) / (n - 1) : 0 };
  }, [meta, fixedAxis]);

  // centre the cut when the axis or dataset changes (axisInfo is re-memoised
  // only when meta or the fixed axis changes, not while scrubbing the slider).
  useEffect(() => {
    if (axisInfo) setCutIndex(Math.floor(axisInfo.n / 2));
  }, [axisInfo, setCutIndex]);

  const idx = axisInfo ? Math.min(cutIndex, axisInfo.n - 1) : 0;
  const value = axisInfo ? axisInfo.min + idx * axisInfo.step : 0;
  const plane = AXIS_TO_PLANE[fixedAxis];
  const lut = COLORMAPS[colormap] ?? COLORMAPS.inferno;

  // Snap a typed cut value to the nearest available data point.
  const commitCut = (v: number) => {
    if (!axisInfo || axisInfo.step === 0) return;
    const i = Math.round((v - axisInfo.min) / axisInfo.step);
    setCutIndex(Math.max(0, Math.min(axisInfo.n - 1, i)));
  };

  // Displayed slices at the current cut (one per stage).
  const sliceResults = useQueries({
    queries: stages.map((s) => ({
      queryKey: ["slice", s.volume_id, plane, value, false],
      queryFn: () => fetchSlice(s.volume_id, plane, value),
      enabled: Boolean(axisInfo),
      placeholderData: keepPreviousData,
    })),
  });

  // One global colour scale, fixed per (dataset, axis): the pooled robust level
  // of every stage at the CENTRE cut.  Keying it on the centre cut (not the
  // displayed cut) means the scale stays put while the cut slider is dragged, so
  // intensities stay comparable both across stages and across cut positions.
  const centerValue = axisInfo
    ? axisInfo.min + Math.floor(axisInfo.n / 2) * axisInfo.step
    : 0;
  const scaleResults = useQueries({
    queries: stages.map((s) => ({
      queryKey: ["slice", s.volume_id, plane, centerValue, false],
      queryFn: () => fetchSlice(s.volume_id, plane, centerValue),
      enabled: Boolean(axisInfo),
      staleTime: Infinity,
    })),
  });

  let globalVmax = 0;
  for (const r of scaleResults) {
    const rm = r.data?.header.robust_max;
    if (rm && Number.isFinite(rm)) globalVmax = Math.max(globalVmax, rm);
  }
  globalVmax = globalVmax || 1;
  const vmax = contrast * globalVmax;

  return (
    <div className="page-body">
      <div className="toolbar">
        <Field label="Dataset">
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
              ? {
                  value,
                  prefix: `${fixedAxis} =`,
                  suffix: "r.l.u.",
                  onCommit: commitCut,
                }
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
            {SEQUENTIAL_NAMES.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <ColormapBar lut={lut} />
        </Field>
      </div>

      {datasetsQ.isLoading && (
        <EmptyState title="Loading datasets…" />
      )}
      {datasetsQ.isError && (
        <EmptyState
          error
          icon={<IconAlert />}
          title="Backend unreachable"
          hint="Start the API server (ndiff-web or uvicorn on port 8000) and reload."
        />
      )}
      {dataset && stages.length === 0 && (
        <EmptyState
          title="No processed stages for this dataset"
          hint="Run the pipeline first — the Run pipeline tab will produce the cleanup stages shown here."
        />
      )}

      <div className="panel-grid">
        {stages.map((s, i) => (
          <SlicePanel
            key={s.volume_id}
            title={STAGE_LABELS[s.name] ?? s.name}
            data={sliceResults[i]?.data}
            isFetching={sliceResults[i]?.isFetching}
            isError={sliceResults[i]?.isError}
            error={sliceResults[i]?.error as Error | null}
            lut={lut}
            vmax={vmax}
            log={log}
          />
        ))}
      </div>

      {meta && (
        <MetaStrip
          items={[
            { key: "Source", value: dataset?.raw_name },
            { key: "Plane", value: plane },
            {
              key: "Colour scale",
              value: `global · 0 … ${vmax.toPrecision(3)}${log ? " (log)" : ""}`,
            },
            {
              key: "Lattice",
              value: `a=${meta.lattice.a?.toFixed(2)}  b=${meta.lattice.b?.toFixed(2)}  c=${meta.lattice.c?.toFixed(2)} Å`,
            },
            { key: "Grid", value: meta.shape.join(" × ") },
          ]}
        />
      )}
    </div>
  );
}
