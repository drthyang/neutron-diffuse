// One stage panel: fetches its slice and renders it with the shared controls.

import { useSlice } from "../api/hooks";
import { SliceCanvas } from "./SliceCanvas";

interface Props {
  title: string;
  volumeId: string;
  plane: string;
  value: number;
  lut: Uint8ClampedArray;
  contrast: number;
  log: boolean;
  width?: number;
}

export function SlicePanel({
  title,
  volumeId,
  plane,
  value,
  lut,
  contrast,
  log,
  width = 320,
}: Props) {
  const { data, isFetching, isError, error } = useSlice(volumeId, plane, value);

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
          <SliceCanvas
            slice={data}
            lut={lut}
            vmax={contrast * data.header.robust_max}
            log={log}
            width={width}
          />
          <div className="panel-meta">
            {data.header.x_label} × {data.header.y_label}
          </div>
        </>
      ) : (
        <div className="placeholder" style={{ width, height: width }} />
      )}
    </div>
  );
}
