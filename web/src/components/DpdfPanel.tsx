// One ΔPDF orthoslice panel: fetches its slice and renders a square real-space
// window (±windowA Å on both axes) diverging about 0, with an optional gray
// dashed unit-cell overlay.  Self-scales to its own robust far-field level ×
// contrast (the multi-temp viewer pools scale separately).
//
// Chrome reuses the Q–R Band Transform panel system (`.qr-panel`): a header with
// an axis-colored badge + plane title + fixed-axis tag, the square slice image,
// and a footer that holds this axis's cut slider (passed in as `footer`).

import type { ReactNode } from "react";

import { useDpdfSlice } from "../api/hooks";
import { SliceCanvas } from "./SliceCanvas";
import { UnitCellGrid } from "./UnitCellGrid";

interface Props {
  badge: string; // axis label shown in the header chip, e.g. "z_L"
  badgeClass: string; // axis hue class, e.g. "qr-rt--qp"
  title: string; // plane title, e.g. "x_H – y_K"
  tag: string; // fixed-axis tag, e.g. "fixed z · real"
  volumeId: string;
  plane: string;
  value: number;
  lut: Uint8ClampedArray;
  contrast: number;
  gridlines: boolean;
  latX: number | null;
  latY: number | null;
  windowA?: number; // half-window in Å (square)
  footer?: ReactNode; // this axis's cut slider
}

export function DpdfPanel({
  badge,
  badgeClass,
  title,
  tag,
  volumeId,
  plane,
  value,
  lut,
  contrast,
  gridlines,
  latX,
  latY,
  windowA = 10,
  footer,
}: Props) {
  const { data, isFetching, isError, error } = useDpdfSlice(volumeId, plane, value);

  return (
    <div className="qr-panel">
      <div className="qr-panel-head">
        <span className="qr-panel-titlegroup">
          <span className={`qr-rt-badge ${badgeClass}`}>{badge}</span>
          <span className="qr-panel-title">{title}</span>
        </span>
        <span className="qr-panel-tag">{tag}</span>
      </div>
      <div className="qr-img">
        {isError ? (
          <div className="panel-err">{(error as Error).message}</div>
        ) : data ? (
          <>
            <SliceCanvas
              slice={data}
              lut={lut}
              vmax={contrast * data.header.robust_max}
              log={false}
              diverging
              windowA={windowA}
            />
            {gridlines && <UnitCellGrid half={windowA} latX={latX} latY={latY} />}
          </>
        ) : (
          <div className="skeleton" style={{ width: "100%", height: "100%" }} />
        )}
        {isFetching && data && <span className="spin qr-img-spin" />}
      </div>
      {footer && <div className="qr-foot">{footer}</div>}
    </div>
  );
}
