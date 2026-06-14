// One ΔPDF orthoslice panel: fetches its slice and renders a square real-space
// window (±windowA Å on both axes) diverging about 0, with an optional gray
// dashed unit-cell overlay.  Self-scales to its own robust far-field level ×
// contrast (the multi-temp viewer pools scale separately).

import { useDpdfSlice } from "../api/hooks";
import { SliceCanvas } from "./SliceCanvas";
import { UnitCellGrid } from "./UnitCellGrid";

interface Props {
  title: string;
  volumeId: string;
  plane: string;
  value: number;
  lut: Uint8ClampedArray;
  contrast: number;
  gridlines: boolean;
  latX: number | null;
  latY: number | null;
  windowA?: number; // half-window in Å (square)
  size?: number; // square display size in px
}

export function DpdfPanel({
  title,
  volumeId,
  plane,
  value,
  lut,
  contrast,
  gridlines,
  latX,
  latY,
  windowA = 10,
  size = 320,
}: Props) {
  const { data, isFetching, isError, error } = useDpdfSlice(volumeId, plane, value);

  return (
    <div className="panel-card">
      <div className="panel-head">
        <span className="panel-title">{title}</span>
        {isFetching && <span className="spin" />}
      </div>
      <div className="panel-body">
        {isError ? (
          <div className="panel-err" style={{ width: size, height: size }}>
            {(error as Error).message}
          </div>
        ) : data ? (
          <div style={{ position: "relative", width: size, height: size }}>
            <SliceCanvas
              slice={data}
              lut={lut}
              vmax={contrast * data.header.robust_max}
              log={false}
              diverging
              windowA={windowA}
              size={size}
            />
            {gridlines && (
              <UnitCellGrid half={windowA} latX={latX} latY={latY} />
            )}
          </div>
        ) : (
          <div className="skeleton" style={{ width: size, height: size }} />
        )}
      </div>
      <div className="panel-foot">{data ? data.header.cut_label : " "}</div>
    </div>
  );
}
