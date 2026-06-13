// Unit-cell gridline overlay (SVG) for the ΔPDF square-window viewer.  Draws
// gray dashed lines at integer multiples of the direct-lattice spacing along each
// displayed axis, over a square [-half, +half] Å window mapped to a `size` px box.

interface Props {
  size: number; // square box side in px
  half: number; // half-window in Å (box spans [-half, +half] on both axes)
  latX: number | null;
  latY: number | null;
}

// Lattice-multiple positions (in Å) that fall inside [-half, +half].
function multiples(lat: number | null, half: number): number[] {
  if (!lat || lat <= 0) return [];
  const out: number[] = [];
  for (let k = Math.ceil(-half / lat); k <= Math.floor(half / lat); k++) {
    out.push(k * lat);
  }
  return out;
}

export function UnitCellGrid({ size, half, latX, latY }: Props) {
  const toPxX = (v: number) => ((v + half) / (2 * half)) * size;
  // canvas y is flipped (smallest y at the bottom), so mirror here too.
  const toPxY = (v: number) => size - ((v + half) / (2 * half)) * size;

  const vx = multiples(latX, half).map(toPxX);
  const hy = multiples(latY, half).map(toPxY);

  return (
    <svg
      width={size}
      height={size}
      style={{ position: "absolute", left: 0, top: 0, pointerEvents: "none" }}
    >
      <g
        stroke="rgba(150, 158, 172, 0.6)"
        strokeWidth={1}
        strokeDasharray="4 3"
        shapeRendering="crispEdges"
      >
        {vx.map((px, i) => (
          <line key={`v${i}`} x1={px} y1={0} x2={px} y2={size} />
        ))}
        {hy.map((py, i) => (
          <line key={`h${i}`} x1={0} y1={py} x2={size} y2={py} />
        ))}
      </g>
    </svg>
  );
}
