// One reciprocal-space stage panel (presentational).  The parent fetches every
// stage's slice together and pools one global colour scale, so each panel just
// renders the slice it is handed with the shared vmin/vmax.

import type { Slice } from "../api/types";
import { SliceCanvas } from "./SliceCanvas";
import { UnitCellGrid } from "./UnitCellGrid";

interface Props {
  title: string;
  data?: Slice;
  isFetching?: boolean;
  isError?: boolean;
  error?: Error | null;
  lut: Uint8ClampedArray;
  vmax: number;
  vmin?: number;
  log: boolean;
  width?: number;
  bands?: [number, number];
  cutDistance?: number;
  reciprocalAxes?: boolean;
  latX?: number;
  latY?: number;
  latCut?: number;
  windowA?: number;
  diverging?: boolean;
  gridlines?: boolean;
}

export function SlicePanel({
  title,
  data,
  isFetching,
  isError,
  error,
  lut,
  vmax,
  vmin = 0,
  log,
  width = 320,
  bands,
  cutDistance,
  reciprocalAxes = false,
  latX,
  latY,
  latCut,
  windowA,
  diverging = false,
  gridlines = false,
}: Props) {
  return (
    <div className="panel-card">
      <div className="panel-head">
        <span className="panel-title">{title}</span>
        {isFetching && <span className="spin" />}
      </div>
      <div className="panel-body">
        {isError ? (
          <div className="panel-err" style={{ width, height: width }}>
            {error?.message}
          </div>
        ) : data ? (
          <div style={{ position: "relative", width, height: windowA != null ? width : "auto" }}>
            <SliceCanvas
              slice={data}
              lut={lut}
              vmax={vmax}
              vmin={vmin}
              log={log}
              width={width}
              bands={bands}
              cutDistance={cutDistance}
              reciprocalAxes={reciprocalAxes}
              latX={latX}
              latY={latY}
              latCut={latCut}
              windowA={windowA}
              size={width}
              diverging={diverging}
            />
            {gridlines && windowA != null && (
              <UnitCellGrid half={windowA} latX={latX ?? null} latY={latY ?? null} />
            )}
          </div>
        ) : (
          <div className="skeleton" style={{ width, height: width }} />
        )}
      </div>
      <div className="panel-foot">
        {data ? `${data.header.x_label} × ${data.header.y_label}` : " "}
      </div>
    </div>
  );
}
