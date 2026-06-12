// Light unit-cell gridline overlay (SVG) at integer multiples of the lattice
// spacing along each displayed axis.  Drawn over a SliceCanvas of the same size.

interface Props {
  width: number;
  height: number;
  xAxis: number[];
  yAxis: number[];
  latX: number | null;
  latY: number | null;
}

function multiplesPx(
  lat: number | null,
  axis: number[],
  toPx: (v: number) => number,
): number[] {
  if (!lat || lat <= 0 || axis.length < 2) return [];
  const lo = axis[0];
  const hi = axis[axis.length - 1];
  const out: number[] = [];
  for (let k = Math.ceil(lo / lat); k <= Math.floor(hi / lat); k++) {
    out.push(toPx(k * lat));
  }
  return out;
}

export function UnitCellGrid({ width, height, xAxis, yAxis, latX, latY }: Props) {
  const xMin = xAxis[0];
  const xMax = xAxis[xAxis.length - 1];
  const yMin = yAxis[0];
  const yMax = yAxis[yAxis.length - 1];
  const toPxX = (v: number) => ((v - xMin) / (xMax - xMin)) * width;
  // canvas y is flipped (smallest y at the bottom), so mirror here too.
  const toPxY = (v: number) => height - ((v - yMin) / (yMax - yMin)) * height;

  const vx = multiplesPx(latX, xAxis, toPxX);
  const hy = multiplesPx(latY, yAxis, toPxY);

  return (
    <svg
      width={width}
      height={height}
      style={{ position: "absolute", left: 0, top: 0, pointerEvents: "none" }}
    >
      <g stroke="rgba(255,255,255,0.22)" strokeWidth={0.5}>
        {vx.map((px, i) => (
          <line key={`v${i}`} x1={px} y1={0} x2={px} y2={height} />
        ))}
        {hy.map((py, i) => (
          <line key={`h${i}`} x1={0} y1={py} x2={width} y2={py} />
        ))}
      </g>
    </svg>
  );
}
