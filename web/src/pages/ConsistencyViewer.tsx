// Back-FFT consistency viewer — the end-of-workflow check.  Inverse-transforms
// the ΔPDF back to reciprocal space and compares it to the diffuse data, with an
// adjustable |Q| band so you can see which signals come from low- vs high-|Q|
// data.  data | back-FFT | residual share plane / cut / contrast.

import { useEffect, useMemo, useState } from "react";

import { keepPreviousData, useQueries, useQuery } from "@tanstack/react-query";

import { fetchConsistencyMeta, fetchConsistencySlice } from "../api/client";
import { useDatasets } from "../api/hooks";
import { COLORMAPS, SEQUENTIAL_NAMES } from "../colormaps/luts";
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
import { AXIS_INDEX, AXIS_TO_PLANE, type FixedAxis } from "../state/viewerStore";

const AXES: FixedAxis[] = ["H", "K", "L"];
const PANELS = [
  { key: "data", title: "Data (band-limited)" },
  { key: "recon", title: "Back-FFT  IFFT[ΔPDF]" },
  { key: "residual", title: "Residual (data − recon)" },
] as const;

export function ConsistencyViewer() {
  const datasetsQ = useDatasets();
  const datasets = useMemo(() => datasetsQ.data ?? [], [datasetsQ.data]);
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

  const [datasetId, setDatasetId] = useState<string | undefined>();
  const [fixedAxis, setFixedAxis] = useState<FixedAxis>("H");
  const [cutIndex, setCutIndex] = useState(0);
  const [contrast, setContrast] = useState(1);
  const [log, setLog] = useState(false);
  const [colormap, setColormap] = useState("inferno");
  const [band, setBand] = useState<{ min: number; max: number } | null>(null);
  const [draftMin, setDraftMin] = useState(0);
  const [draftMax, setDraftMax] = useState(0);

  useEffect(() => {
    if (!datasetId && usable.length) setDatasetId(usable[0].id);
  }, [datasetId, usable]);

  // Meta drives the metrics + grid ranges + |Q| span; it recomputes the (heavy)
  // round trip whenever the committed band changes.
  const metaQ = useQuery({
    queryKey: ["consMeta", datasetId, band?.min, band?.max],
    queryFn: () => fetchConsistencyMeta(datasetId as string, band?.min, band?.max),
    enabled: Boolean(datasetId),
    placeholderData: keepPreviousData,
  });
  const meta = metaQ.data;
  const spanMax = meta?.q_data_max ?? 0;

  // Initialise the band drafts to the full |Q| span once it is known.
  useEffect(() => {
    if (spanMax > 0 && draftMax === 0) setDraftMax(spanMax);
  }, [spanMax, draftMax]);

  const axisInfo = useMemo(() => {
    if (!meta) return null;
    const i = AXIS_INDEX[fixedAxis];
    const [min, max] = [meta.h_range, meta.k_range, meta.l_range][i];
    const n = meta.shape[i];
    return { min, max, n, step: n > 1 ? (max - min) / (n - 1) : 0 };
  }, [meta, fixedAxis]);

  useEffect(() => {
    if (axisInfo) setCutIndex(Math.floor(axisInfo.n / 2));
  }, [axisInfo]);

  const idx = axisInfo ? Math.min(cutIndex, axisInfo.n - 1) : 0;
  const value = axisInfo ? axisInfo.min + idx * axisInfo.step : 0;
  const plane = AXIS_TO_PLANE[fixedAxis];
  const seqLut = COLORMAPS[colormap] ?? COLORMAPS.inferno;

  const commitCut = (v: number) => {
    if (!axisInfo || axisInfo.step === 0) return;
    const i = Math.round((v - axisInfo.min) / axisInfo.step);
    setCutIndex(Math.max(0, Math.min(axisInfo.n - 1, i)));
  };

  const sliceResults = useQueries({
    queries: PANELS.map((p) => ({
      queryKey: ["consSlice", datasetId, p.key, plane, value, band?.min, band?.max],
      queryFn: () =>
        fetchConsistencySlice(
          datasetId as string, p.key, plane, value, band?.min, band?.max,
        ),
      enabled: Boolean(datasetId && axisInfo),
      placeholderData: keepPreviousData,
    })),
  });

  const rmOf = (i: number) => {
    const rm = sliceResults[i]?.data?.header.robust_max;
    return rm && Number.isFinite(rm) ? rm : 0;
  };
  const seqVmax = (Math.max(rmOf(0), rmOf(1)) || 1) * contrast;
  const residScale = (rmOf(2) || 1) * contrast;

  const m = meta?.metrics;
  const bandApplied = band !== null;
  const canApply = draftMax > draftMin;

  return (
    <div className="page-body">
      <div className="toolbar">
        <Field label="Dataset">
          <select
            value={datasetId ?? ""}
            onChange={(e) => {
              setDatasetId(e.target.value);
              setBand(null);
              setDraftMin(0);
              setDraftMax(0);
            }}
          >
            {usable.map((d) => (
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
            {SEQUENTIAL_NAMES.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <ColormapBar lut={seqLut} />
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
        <Field label="Recompute">
          <div style={{ display: "flex", gap: 8 }}>
            <button
              type="button"
              className="btn btn-primary"
              disabled={!canApply || metaQ.isFetching}
              onClick={() => setBand({ min: draftMin, max: draftMax })}
            >
              {metaQ.isFetching ? "Computing…" : "Apply |Q| band"}
            </button>
            <button
              type="button"
              className="btn btn-ghost"
              disabled={!bandApplied && draftMin === 0 && draftMax === spanMax}
              onClick={() => {
                setBand(null);
                setDraftMin(0);
                setDraftMax(spanMax);
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
      {metaQ.isError && (
        <EmptyState
          error
          icon={<IconAlert />}
          title="Consistency check failed"
          hint={(metaQ.error as Error)?.message}
        />
      )}

      <div className="panel-grid">
        {PANELS.map((p, i) => (
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
          />
        ))}
      </div>

      {m && (
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
