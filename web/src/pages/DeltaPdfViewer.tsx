// Single-temperature 3D-ΔPDF orthoslice viewer — replaces
// examples/explore_delta_pdf_ortho.py.  Three linked orthogonal real-space cuts
// with movable cut sliders, a contrast control, and a unit-cell gridline toggle.

import { useEffect, useMemo } from "react";

import { useDatasets, useDpdfMeta } from "../api/hooks";
import { COLORMAPS, DIVERGING_NAMES, DIVERGING_NAME } from "../colormaps/luts";
import { DpdfPanel } from "../components/DpdfPanel";
import { EmptyState, Field, MetaStrip, Slider, Switch } from "../components/ui";
import { useDatasetStore, useInitializeDataset } from "../state/datasetStore";
import { useDpdfStore } from "../state/dpdfStore";

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
  const datasets = useMemo(() => datasetsQ.data ?? [], [datasetsQ.data]);
  useInitializeDataset(datasets);

  const datasetId = useDatasetStore((s) => s.datasetId);
  const setDataset = useDatasetStore((s) => s.setDataset);
  const cutX = useDpdfStore((s) => s.cutX);
  const cutY = useDpdfStore((s) => s.cutY);
  const cutZ = useDpdfStore((s) => s.cutZ);
  const contrast = useDpdfStore((s) => s.contrast);
  const gridlines = useDpdfStore((s) => s.gridlines);
  const colormap = useDpdfStore((s) => s.colormap);
  const setColormap = useDpdfStore((s) => s.setColormap);
  const centered = useDpdfStore((s) => s.centered);
  const setCutX = useDpdfStore((s) => s.setCutX);
  const setCutY = useDpdfStore((s) => s.setCutY);
  const setCutZ = useDpdfStore((s) => s.setCutZ);
  const setContrast = useDpdfStore((s) => s.setContrast);
  const setGridlines = useDpdfStore((s) => s.setGridlines);
  const center = useDpdfStore((s) => s.center);

  // Square real-space window (full width in Å) shown for every orthoslice;
  // shared with the multi-temperature viewer via the store.
  const windowFull = useDpdfStore((s) => s.windowFull);
  const setWindowFull = useDpdfStore((s) => s.setWindowFull);
  const halfWindow = windowFull / 2;

  const dataset = datasets.find((d) => d.id === datasetId);
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

  const lut = COLORMAPS[colormap] ?? COLORMAPS[DIVERGING_NAME];
  const a = meta?.lattice.a ?? null;
  const b = meta?.lattice.b ?? null;
  const c = meta?.lattice.c ?? null;

  const xVal = axisValue(meta?.x_range, meta?.shape[0], cutX);
  const yVal = axisValue(meta?.y_range, meta?.shape[1], cutY);
  const zVal = axisValue(meta?.z_range, meta?.shape[2], cutZ);

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

        <Slider
          label="Window"
          readout={`${windowFull.toFixed(0)} Å`}
          min={10}
          max={160}
          step={2}
          value={windowFull}
          onChange={setWindowFull}
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

        <Switch label="Unit cells" checked={gridlines} onChange={setGridlines} />
        <Field label="Colormap">
          <select value={colormap} onChange={(e) => setColormap(e.target.value)}>
            {DIVERGING_NAMES.map((name) => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
        </Field>
      </div>

      {datasetsQ.isSuccess && !volumeId && (
        <EmptyState
          title="No ΔPDF volume available for this dataset"
          hint="Run this dataset through the ΔPDF stage — the Run pipeline tab produces the real-space volume shown here."
        />
      )}

      <div className="panel-grid">
        {volumeId && (
          <>
            <div className="panel-col">
              <Slider
                label="z_L"
                readout={meta ? `${zVal.toFixed(1)} Å` : "—"}
                min={0}
                max={meta ? meta.shape[2] - 1 : 0}
                value={cutZ}
                disabled={!meta}
                onChange={setCutZ}
              />
              <DpdfPanel
                title="x_H – y_K  ·  fixed z_L"
                volumeId={volumeId}
                plane="xy"
                value={zVal}
                lut={lut}
                contrast={contrast}
                gridlines={gridlines}
                latX={a}
                latY={b}
                windowA={halfWindow}
              />
            </div>
            <div className="panel-col">
              <Slider
                label="y_K"
                readout={meta ? `${yVal.toFixed(1)} Å` : "—"}
                min={0}
                max={meta ? meta.shape[1] - 1 : 0}
                value={cutY}
                disabled={!meta}
                onChange={setCutY}
              />
              <DpdfPanel
                title="x_H – z_L  ·  fixed y_K"
                volumeId={volumeId}
                plane="xz"
                value={yVal}
                lut={lut}
                contrast={contrast}
                gridlines={gridlines}
                latX={a}
                latY={c}
                windowA={halfWindow}
              />
            </div>
            <div className="panel-col">
              <Slider
                label="x_H"
                readout={meta ? `${xVal.toFixed(1)} Å` : "—"}
                min={0}
                max={meta ? meta.shape[0] - 1 : 0}
                value={cutX}
                disabled={!meta}
                onChange={setCutX}
              />
              <DpdfPanel
                title="y_K – z_L  ·  fixed x_H"
                volumeId={volumeId}
                plane="yz"
                value={xVal}
                lut={lut}
                contrast={contrast}
                gridlines={gridlines}
                latX={b}
                latY={c}
                windowA={halfWindow}
              />
            </div>
          </>
        )}
      </div>

      {meta && (
        <MetaStrip
          items={[
            { key: "Source", value: dataset?.raw_name },
            {
              key: "Window",
              value: `${windowFull.toFixed(0)} × ${windowFull.toFixed(0)} Å`,
            },
            {
              key: "Lattice",
              value: `a=${a?.toFixed(2)}  b=${b?.toFixed(2)}  c=${c?.toFixed(2)} Å`,
            },
            { key: "|Q| max", value: `${meta.q_max?.toFixed(1)} Å⁻¹` },
          ]}
        />
      )}
    </div>
  );
}
