// One ΔPDF orthoslice panel: fetches its slice, renders it diverging about 0
// with an optional unit-cell gridline overlay.  Self-scales to its own robust
// far-field level × contrast (the multi-temp viewer pools scale separately).

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
  width?: number;
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
  width = 300,
}: Props) {
  const { data, isFetching, isError, error } = useDpdfSlice(volumeId, plane, value);
  const height = data ? Math.round((width * data.header.ny) / data.header.nx) : width;

  return (
    <div className="panel">
      <div className="panel-title">
        {title}
        {isFetching && <span className="spinner"> ·</span>}
      </div>
      {isError ? (
        <div className="error" style={{ width, height: width }}>
          {(error as Error).message}
        </div>
      ) : data ? (
        <>
          <div style={{ position: "relative", width, height }}>
            <SliceCanvas
              slice={data}
              lut={lut}
              vmax={contrast * data.header.robust_max}
              log={false}
              diverging
              width={width}
            />
            {gridlines && (
              <UnitCellGrid
                width={width}
                height={height}
                xAxis={data.header.x_axis}
                yAxis={data.header.y_axis}
                latX={latX}
                latY={latY}
              />
            )}
          </div>
          <div className="panel-meta">{data.header.cut_label}</div>
        </>
      ) : (
        <div className="placeholder" style={{ width, height: width }} />
      )}
    </div>
  );
}
