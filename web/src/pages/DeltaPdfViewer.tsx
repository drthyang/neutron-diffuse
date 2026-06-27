// Single-temperature 3D-ΔPDF orthoslice viewer — replaces
// examples/explore_delta_pdf_ortho.py.  Three linked orthogonal real-space cuts
// with movable cut sliders, a contrast control, and a unit-cell gridline toggle.

import { useCallback, useEffect, useMemo } from "react";

import { useDatasets, useDpdfMeta } from "../api/hooks";
import { COLORMAPS, DIVERGING_NAMES, DIVERGING_NAME } from "../colormaps/luts";
import { DpdfPanel } from "../components/DpdfPanel";
import {
  ColormapBar,
  EmptyState,
  MetaStrip,
  Slider,
  Switch,
  type ValueInputConfig,
} from "../components/ui";
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

// Snap an Å value back to the nearest grid index along an axis.
function commitAngstrom(
  range: [number, number],
  n: number,
  setIdx: (i: number) => void,
): (v: number) => void {
  return (v) => {
    if (n < 2) return;
    const step = (range[1] - range[0]) / (n - 1);
    const i = Math.round((v - range[0]) / step);
    setIdx(Math.max(0, Math.min(n - 1, i)));
  };
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
  // shared with the multi-volume viewer via the store.
  const windowFull = useDpdfStore((s) => s.windowFull);
  const setWindowFull = useDpdfStore((s) => s.setWindowFull);
  const halfWindow = windowFull / 2;

  const dataset = datasets.find((d) => d.id === datasetId);
  const volumeId = dataset?.stages.find((s) => s.name === "delta_pdf")?.volume_id;
  const meta = useDpdfMeta(volumeId).data;

  const recenter = useCallback(() => {
    if (!meta) return;
    center(
      Math.floor(meta.shape[0] / 2),
      Math.floor(meta.shape[1] / 2),
      Math.floor(meta.shape[2] / 2),
    );
  }, [meta, center]);

  useEffect(() => {
    if (meta && !centered) recenter();
  }, [meta, centered, recenter]);

  const lut = COLORMAPS[colormap] ?? COLORMAPS[DIVERGING_NAME];
  const a = meta?.lattice.a ?? null;
  const b = meta?.lattice.b ?? null;
  const c = meta?.lattice.c ?? null;

  const xVal = axisValue(meta?.x_range, meta?.shape[0], cutX);
  const yVal = axisValue(meta?.y_range, meta?.shape[1], cutY);
  const zVal = axisValue(meta?.z_range, meta?.shape[2], cutZ);

  // Two editable boxes per axis: the cut in Å, and the cut divided by the lattice
  // parameter along that direction (a/b/c).  Editing either snaps the cut.
  const axisInputs = (
    range: [number, number] | undefined,
    n: number | undefined,
    valAng: number,
    lat: number | null,
    latLetter: string,
    setIdx: (i: number) => void,
  ): ValueInputConfig[] | undefined => {
    if (!range || !n) return undefined;
    const commit = commitAngstrom(range, n, setIdx);
    const inputs: ValueInputConfig[] = [
      { value: valAng, suffix: "Å", onCommit: commit },
    ];
    if (lat != null && lat !== 0) {
      inputs.push({
        value: valAng / lat,
        prefix: `/${latLetter}`,
        onCommit: (u) => commit(u * lat),
      });
    }
    return inputs;
  };

  // Per-axis cut slider rendered into a panel footer; the wrapper hue links the
  // slider fill + label to that panel's badge (amber x · blue y · green z).
  const cutSlider = (
    hue: string,
    label: string,
    range: [number, number] | undefined,
    n: number | undefined,
    valAng: number,
    lat: number | null,
    latLetter: string,
    value: number,
    setIdx: (i: number) => void,
  ) => (
    <div className={`qr-foot-cut dpdf-cut ${hue}`}>
      <Slider
        label={label}
        readout={meta ? undefined : "—"}
        valueInputs={axisInputs(range, n, valAng, lat, latLetter, setIdx)}
        min={0}
        max={n ? n - 1 : 0}
        value={value}
        disabled={!meta}
        onChange={setIdx}
      />
    </div>
  );

  return (
    <div className="page-body qr-page">
      {/* ── Header: dataset · x·y·z orthoslice identity · recenter ───────── */}
      <div className="qr-header">
        <div className="qr-header-dataset">
          <span className="qr-eyebrow">Dataset</span>
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
        </div>

        <div className="qr-divider" />

        <div className="qr-roundtrip">
          <span className="qr-rt qr-rt--q">x</span>
          <span className="qr-rt-arrow">·</span>
          <span className="qr-rt qr-rt--r">y</span>
          <span className="qr-rt-arrow">·</span>
          <span className="qr-rt qr-rt--qp">z</span>
        </div>
        <span className="qr-eyebrow">Orthoslices</span>

        <span className="qr-desc">
          Three linked real-space cuts about the origin
        </span>

        <div className="qr-header-actions">
          <button
            type="button"
            className="btn btn-ghost"
            disabled={!meta}
            onClick={recenter}
          >
            Recenter cuts
          </button>
        </div>
      </div>

      {/* ── Display-control cluster: window · contrast · unit cells · cmap ── */}
      <div className="qr-clusters">
        <div className="qr-cluster">
          <div className="qr-cluster-head">
            <span className="qr-cluster-title">Real-space display · 3D-ΔPDF</span>
            <div className="qr-cluster-toggle">
              <Switch label="Unit cells" checked={gridlines} onChange={setGridlines} />
            </div>
          </div>
          <div className="qr-cluster-controls">
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
          </div>
          <div className="qr-cluster-cmap">
            <span className="field-label">Colormap</span>
            <select value={colormap} onChange={(e) => setColormap(e.target.value)}>
              {DIVERGING_NAMES.map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
            <ColormapBar lut={lut} />
          </div>
        </div>
      </div>

      {datasetsQ.isSuccess && !volumeId && (
        <EmptyState
          title="No ΔPDF volume available for this dataset"
          hint="Run this dataset through the ΔPDF stage — the Run pipeline tab produces the real-space volume shown here."
        />
      )}

      {/* ── Three orthoslice panels — equal siblings, no connectors ──────── */}
      {volumeId && (
        <div className="qr-flow-row">
          <DpdfPanel
            badge="z_L"
            badgeClass="qr-rt--qp"
            title="x_H – y_K"
            tag="fixed z · real"
            volumeId={volumeId}
            plane="xy"
            value={zVal}
            lut={lut}
            contrast={contrast}
            gridlines={gridlines}
            latX={a}
            latY={b}
            windowA={halfWindow}
            footer={cutSlider(
              "dpdf-cut--z", "Cut z_L", meta?.z_range, meta?.shape[2], zVal, c, "c", cutZ, setCutZ,
            )}
          />
          <DpdfPanel
            badge="y_K"
            badgeClass="qr-rt--r"
            title="x_H – z_L"
            tag="fixed y · real"
            volumeId={volumeId}
            plane="xz"
            value={yVal}
            lut={lut}
            contrast={contrast}
            gridlines={gridlines}
            latX={a}
            latY={c}
            windowA={halfWindow}
            footer={cutSlider(
              "dpdf-cut--y", "Cut y_K", meta?.y_range, meta?.shape[1], yVal, b, "b", cutY, setCutY,
            )}
          />
          <DpdfPanel
            badge="x_H"
            badgeClass="qr-rt--q"
            title="y_K – z_L"
            tag="fixed x · real"
            volumeId={volumeId}
            plane="yz"
            value={xVal}
            lut={lut}
            contrast={contrast}
            gridlines={gridlines}
            latX={b}
            latY={c}
            windowA={halfWindow}
            footer={cutSlider(
              "dpdf-cut--x", "Cut x_H", meta?.x_range, meta?.shape[0], xVal, a, "a", cutX, setCutX,
            )}
          />
        </div>
      )}

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
