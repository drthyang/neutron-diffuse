// Multi-temperature 3D-ΔPDF comparison — replaces
// examples/explore_delta_pdf_multi.py.  Rows = temperatures × columns = the
// three orthoslice planes, with shared cut sliders and a per-plane colour scale
// pooled across temperatures (so temperatures are comparable within a column).

import { Fragment, useEffect, useMemo } from "react";

import { keepPreviousData, useQueries } from "@tanstack/react-query";

import { fetchDpdfSlice } from "../api/client";
import { useDatasets, useDpdfMeta } from "../api/hooks";
import { COLORMAPS, DIVERGING_NAME } from "../colormaps/luts";
import { SliceCanvas } from "../components/SliceCanvas";
import { useDpdfStore } from "../state/dpdfStore";

const PLANES = [
  { key: "xy", label: "x_H – y_K" },
  { key: "xz", label: "x_H – z_L" },
  { key: "yz", label: "y_K – z_L" },
];
const CELL = 220;

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

export function MultiTempViewer() {
  const datasetsQ = useDatasets();
  const temps = useMemo(
    () =>
      (datasetsQ.data ?? [])
        .filter((d) => d.stages.some((s) => s.name === "delta_pdf" && s.exists))
        .sort((a, b) => tempNum(a.temperature) - tempNum(b.temperature)),
    [datasetsQ.data],
  );

  const cutX = useDpdfStore((s) => s.cutX);
  const cutY = useDpdfStore((s) => s.cutY);
  const cutZ = useDpdfStore((s) => s.cutZ);
  const contrast = useDpdfStore((s) => s.contrast);
  const centered = useDpdfStore((s) => s.centered);
  const setCutX = useDpdfStore((s) => s.setCutX);
  const setCutY = useDpdfStore((s) => s.setCutY);
  const setCutZ = useDpdfStore((s) => s.setCutZ);
  const setContrast = useDpdfStore((s) => s.setContrast);
  const center = useDpdfStore((s) => s.center);

  const firstVolId = temps[0]?.stages.find((s) => s.name === "delta_pdf")?.volume_id;
  const meta = useDpdfMeta(firstVolId).data;

  useEffect(() => {
    if (meta && !centered) {
      center(
        Math.floor(meta.shape[0] / 2),
        Math.floor(meta.shape[1] / 2),
        Math.floor(meta.shape[2] / 2),
      );
    }
  }, [meta, centered, center]);

  const xVal = axisValue(meta?.x_range, meta?.shape[0], cutX);
  const yVal = axisValue(meta?.y_range, meta?.shape[1], cutY);
  const zVal = axisValue(meta?.z_range, meta?.shape[2], cutZ);
  const planeValue: Record<string, number> = { xy: zVal, xz: yVal, yz: xVal };

  const combos = temps.flatMap((t) => {
    const vid = t.stages.find((s) => s.name === "delta_pdf")?.volume_id ?? "";
    return PLANES.map((p) => ({ vid, plane: p.key, value: planeValue[p.key] }));
  });

  const results = useQueries({
    queries: combos.map((c) => ({
      queryKey: ["dpdfSlice", c.vid, c.plane, c.value],
      queryFn: () => fetchDpdfSlice(c.vid, c.plane, c.value),
      enabled: Boolean(meta) && Boolean(c.vid),
      placeholderData: keepPreviousData,
    })),
  });

  // pooled robust scale per plane (column), across temperatures
  const pooled: Record<string, number> = {};
  PLANES.forEach((p, ci) => {
    let m = 0;
    temps.forEach((_, ri) => {
      const rm = results[ri * PLANES.length + ci]?.data?.header.robust_max;
      if (rm && Number.isFinite(rm)) m = Math.max(m, rm);
    });
    pooled[p.key] = m || 1;
  });

  const lut = COLORMAPS[DIVERGING_NAME];

  return (
    <div className="viewer">
      <div className="controls">
        <label className="grow">
          x_H {meta ? `= ${xVal.toFixed(1)} Å` : ""}
          <input type="range" min={0} max={meta ? meta.shape[0] - 1 : 0} value={cutX}
                 disabled={!meta} onChange={(e) => setCutX(Number(e.target.value))} />
        </label>
        <label className="grow">
          y_K {meta ? `= ${yVal.toFixed(1)} Å` : ""}
          <input type="range" min={0} max={meta ? meta.shape[1] - 1 : 0} value={cutY}
                 disabled={!meta} onChange={(e) => setCutY(Number(e.target.value))} />
        </label>
        <label className="grow">
          z_L {meta ? `= ${zVal.toFixed(1)} Å` : ""}
          <input type="range" min={0} max={meta ? meta.shape[2] - 1 : 0} value={cutZ}
                 disabled={!meta} onChange={(e) => setCutZ(Number(e.target.value))} />
        </label>
        <label>
          contrast ×{contrast.toFixed(1)}
          <input type="range" min={0.1} max={20} step={0.1} value={contrast}
                 onChange={(e) => setContrast(Number(e.target.value))} />
        </label>
      </div>

      {temps.length === 0 && (
        <div className="status">no ΔPDF outputs found for any temperature.</div>
      )}

      {temps.length > 0 && (
        <div
          className="multi-grid"
          style={{ gridTemplateColumns: `70px repeat(${PLANES.length}, ${CELL}px)` }}
        >
          <div className="corner" />
          {PLANES.map((p) => (
            <div key={p.key} className="col-head">
              {p.label}
            </div>
          ))}
          {temps.map((t, ri) => (
            <Fragment key={t.id}>
              <div className="row-head">{t.temperature ?? t.stem}</div>
              {PLANES.map((p, ci) => {
                const data = results[ri * PLANES.length + ci]?.data;
                return (
                  <div key={p.key} className="cell">
                    {data ? (
                      <SliceCanvas
                        slice={data}
                        lut={lut}
                        vmax={contrast * pooled[p.key]}
                        log={false}
                        diverging
                        width={CELL}
                      />
                    ) : (
                      <div className="placeholder" style={{ width: CELL, height: CELL }} />
                    )}
                  </div>
                );
              })}
            </Fragment>
          ))}
        </div>
      )}
    </div>
  );
}
