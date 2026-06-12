// Single-temperature 3D-ΔPDF orthoslice viewer — replaces
// examples/explore_delta_pdf_ortho.py.  Three linked orthogonal real-space cuts
// with movable cut sliders, a contrast control, and a unit-cell gridline toggle.

import { useEffect, useMemo } from "react";

import { useDatasets, useDpdfMeta } from "../api/hooks";
import { COLORMAPS, DIVERGING_NAME } from "../colormaps/luts";
import { DpdfPanel } from "../components/DpdfPanel";
import { useDpdfStore } from "../state/dpdfStore";

function tempNum(t: string | null): number {
  const m = t?.match(/(\d+)/);
  return m ? Number(m[1]) : 1e9;
}

function axisValue(
  range: [number, number] | undefined,
  n: number | undefined,
  idx: number,
): number {
  if (!range || !n || n < 2) return 0;
  return range[0] + Math.min(idx, n - 1) * ((range[1] - range[0]) / (n - 1));
}

export function DeltaPdfViewer() {
  const datasetsQ = useDatasets();
  const dpdfDatasets = useMemo(
    () =>
      (datasetsQ.data ?? [])
        .filter((d) => d.stages.some((s) => s.name === "delta_pdf" && s.exists))
        .sort((a, b) => tempNum(a.temperature) - tempNum(b.temperature)),
    [datasetsQ.data],
  );

  const datasetId = useDpdfStore((s) => s.datasetId);
  const cutX = useDpdfStore((s) => s.cutX);
  const cutY = useDpdfStore((s) => s.cutY);
  const cutZ = useDpdfStore((s) => s.cutZ);
  const contrast = useDpdfStore((s) => s.contrast);
  const gridlines = useDpdfStore((s) => s.gridlines);
  const centered = useDpdfStore((s) => s.centered);
  const setDataset = useDpdfStore((s) => s.setDataset);
  const setCutX = useDpdfStore((s) => s.setCutX);
  const setCutY = useDpdfStore((s) => s.setCutY);
  const setCutZ = useDpdfStore((s) => s.setCutZ);
  const setContrast = useDpdfStore((s) => s.setContrast);
  const setGridlines = useDpdfStore((s) => s.setGridlines);
  const center = useDpdfStore((s) => s.center);

  useEffect(() => {
    if (!datasetId && dpdfDatasets.length) setDataset(dpdfDatasets[0].id);
  }, [datasetId, dpdfDatasets, setDataset]);

  const dataset = dpdfDatasets.find((d) => d.id === datasetId);
  const volumeId = dataset?.stages.find((s) => s.name === "delta_pdf")?.volume_id;
  const meta = useDpdfMeta(volumeId).data;

  useEffect(() => {
    if (meta && !centered) {
      center(
        Math.floor(meta.shape[0] / 2),
        Math.floor(meta.shape[1] / 2),
        Math.floor(meta.shape[2] / 2),
      );
    }
  }, [meta, centered, center]);

  const lut = COLORMAPS[DIVERGING_NAME];
  const a = meta?.lattice.a ?? null;
  const b = meta?.lattice.b ?? null;
  const c = meta?.lattice.c ?? null;

  const xVal = axisValue(meta?.x_range, meta?.shape[0], cutX);
  const yVal = axisValue(meta?.y_range, meta?.shape[1], cutY);
  const zVal = axisValue(meta?.z_range, meta?.shape[2], cutZ);

  return (
    <div className="viewer">
      <div className="controls">
        <label>
          dataset
          <select value={datasetId ?? ""} onChange={(e) => setDataset(e.target.value)}>
            {dpdfDatasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.temperature ?? d.stem}
              </option>
            ))}
          </select>
        </label>

        <label className="grow">
          x_H {meta ? `= ${xVal.toFixed(1)} Å` : ""}
          <input
            type="range"
            min={0}
            max={meta ? meta.shape[0] - 1 : 0}
            value={cutX}
            disabled={!meta}
            onChange={(e) => setCutX(Number(e.target.value))}
          />
        </label>
        <label className="grow">
          y_K {meta ? `= ${yVal.toFixed(1)} Å` : ""}
          <input
            type="range"
            min={0}
            max={meta ? meta.shape[1] - 1 : 0}
            value={cutY}
            disabled={!meta}
            onChange={(e) => setCutY(Number(e.target.value))}
          />
        </label>
        <label className="grow">
          z_L {meta ? `= ${zVal.toFixed(1)} Å` : ""}
          <input
            type="range"
            min={0}
            max={meta ? meta.shape[2] - 1 : 0}
            value={cutZ}
            disabled={!meta}
            onChange={(e) => setCutZ(Number(e.target.value))}
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
          <input
            type="checkbox"
            checked={gridlines}
            onChange={(e) => setGridlines(e.target.checked)}
          />
          unit cells
        </label>
      </div>

      {!volumeId && (
        <div className="status">
          no ΔPDF available — run the pipeline through the ΔPDF stage.
        </div>
      )}

      <div className="panels">
        {volumeId && (
          <>
            <DpdfPanel
              title="x_H – y_K  (fix z_L)"
              volumeId={volumeId}
              plane="xy"
              value={zVal}
              lut={lut}
              contrast={contrast}
              gridlines={gridlines}
              latX={a}
              latY={b}
            />
            <DpdfPanel
              title="x_H – z_L  (fix y_K)"
              volumeId={volumeId}
              plane="xz"
              value={yVal}
              lut={lut}
              contrast={contrast}
              gridlines={gridlines}
              latX={a}
              latY={c}
            />
            <DpdfPanel
              title="y_K – z_L  (fix x_H)"
              volumeId={volumeId}
              plane="yz"
              value={xVal}
              lut={lut}
              contrast={contrast}
              gridlines={gridlines}
              latX={b}
              latY={c}
            />
          </>
        )}
      </div>

      {meta && (
        <div className="footer">
          {dataset?.raw_name} · ΔPDF · lattice a={a?.toFixed(2)} b={b?.toFixed(2)} c=
          {c?.toFixed(2)} Å · |Q|max {meta.q_max?.toFixed(1)} Å⁻¹
        </div>
      )}
    </div>
  );
}
