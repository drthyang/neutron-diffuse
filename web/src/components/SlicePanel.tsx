// One reciprocal-space stage panel (presentational).  The parent fetches every
// stage's slice together and pools one global colour scale, so each panel just
// renders the slice it is handed with the shared vmin/vmax.
//
// Chrome reuses the Q–R Band Transform panel system (`.qr-panel`): a header with
// a numbered stage badge + stage name (+ an `output` tag on the final stage) and
// a square slice image on the shared scale.  No caption — the stage name already
// says what the step does, and the slices are the only thing meant to differ.

import type { Slice } from "../api/types";
import { SliceCanvas } from "./SliceCanvas";

interface Props {
  index: number; // stage number shown in the badge
  title: string;
  output?: boolean; // final stage carries a small green "output" tag
  data?: Slice;
  isFetching?: boolean;
  isError?: boolean;
  error?: Error | null;
  lut: Uint8ClampedArray;
  vmax: number;
  vmin?: number;
  log: boolean;
  bands?: [number, number];
  cutDistance?: number;
  reciprocalAxes?: boolean;
  latX?: number;
  latY?: number;
  latCut?: number;
  zoom?: number;
}

export function SlicePanel({
  index,
  title,
  output = false,
  data,
  isFetching,
  isError,
  error,
  lut,
  vmax,
  vmin = 0,
  log,
  bands,
  cutDistance,
  reciprocalAxes = false,
  latX,
  latY,
  latCut,
  zoom,
}: Props) {
  return (
    <div className="qr-panel">
      <div className="qr-panel-head">
        <span className="qr-panel-titlegroup">
          <span className="qr-stage-badge">{index}</span>
          <span className="qr-panel-title">{title}</span>
        </span>
        {output && <span className="qr-stage-output">output</span>}
      </div>
      <div className="qr-img qr-img--free">
        {isError ? (
          <div className="panel-err">{error?.message}</div>
        ) : data ? (
          <SliceCanvas
            slice={data}
            lut={lut}
            vmax={vmax}
            vmin={vmin}
            log={log}
            fit
            zoom={zoom}
            bands={bands}
            cutDistance={cutDistance}
            reciprocalAxes={reciprocalAxes}
            latX={latX}
            latY={latY}
            latCut={latCut}
          />
        ) : (
          <div className="skeleton" style={{ width: "100%", height: "100%" }} />
        )}
        {isFetching && data && <span className="spin qr-img-spin" />}
      </div>
    </div>
  );
}
