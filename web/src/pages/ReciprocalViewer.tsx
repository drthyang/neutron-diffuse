// Reciprocal-space cleanup viewer — replaces examples/explore_slice.py.
// One panel per existing HKLVolume stage, all sharing plane / cut / contrast /
// log / colormap controls so the cleanup stages are directly comparable.

import { useEffect, useMemo } from "react";

import { useDatasets, useMeta } from "../api/hooks";
import { COLORMAPS, SEQUENTIAL_NAMES } from "../colormaps/luts";
import { SlicePanel } from "../components/SlicePanel";
import {
  AXIS_INDEX,
  AXIS_TO_PLANE,
  type FixedAxis,
  useViewerStore,
} from "../state/viewerStore";

const STAGE_ORDER = ["raw", "ringremoved", "braggpunched", "backfilled", "flattened"];
const STAGE_LABELS: Record<string, string> = {
  raw: "raw",
  ringremoved: "ring-removed",
  braggpunched: "Bragg-punched",
  backfilled: "backfilled",
  flattened: "flattened",
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

  return (
    <div className="viewer">
      <div className="controls">
        <label>
          dataset
          <select value={datasetId ?? ""} onChange={(e) => setDataset(e.target.value)}>
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.temperature ?? d.stem}
              </option>
            ))}
          </select>
        </label>

        <div className="axis-group">
          <span>fixed axis</span>
          {AXES.map((a) => (
            <button
              key={a}
              className={a === fixedAxis ? "active" : ""}
              onClick={() => setFixedAxis(a)}
            >
              {a}
            </button>
          ))}
        </div>

        <label className="grow">
          {axisInfo ? `${fixedAxis} = ${value.toFixed(3)} r.l.u.` : "cut"}
          <input
            type="range"
            min={0}
            max={axisInfo ? axisInfo.n - 1 : 0}
            step={1}
            value={idx}
            disabled={!axisInfo}
            onChange={(e) => setCutIndex(Number(e.target.value))}
          />
        </label>

        <label>
          contrast ×{contrast.toFixed(1)}
          <input
            type="range"
            min={0.1}
            max={20}
            step={0.1}
            value={contrast}
            onChange={(e) => setContrast(Number(e.target.value))}
          />
        </label>

        <label className="check">
          <input type="checkbox" checked={log} onChange={(e) => setLog(e.target.checked)} />
          log
        </label>

        <label>
          colormap
          <select value={colormap} onChange={(e) => setColormap(e.target.value)}>
            {SEQUENTIAL_NAMES.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
      </div>

      {datasetsQ.isLoading && <div className="status">loading datasets…</div>}
      {datasetsQ.isError && <div className="status error">backend unreachable</div>}
      {dataset && stages.length === 0 && (
        <div className="status">
          no processed stages for this dataset — run the pipeline first.
        </div>
      )}

      <div className="panels">
        {stages.map((s) => (
          <SlicePanel
            key={s.volume_id}
            title={STAGE_LABELS[s.name] ?? s.name}
            volumeId={s.volume_id}
            plane={plane}
            value={value}
            lut={lut}
            contrast={contrast}
            log={log}
          />
        ))}
      </div>

      {meta && (
        <div className="footer">
          {dataset?.raw_name} · plane {plane} · lattice a=
          {meta.lattice.a?.toFixed(2)} b={meta.lattice.b?.toFixed(2)} c=
          {meta.lattice.c?.toFixed(2)} Å
        </div>
      )}
    </div>
  );
}
