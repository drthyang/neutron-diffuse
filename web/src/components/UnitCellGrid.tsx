// Unit-cell gridline overlay (SVG) for the ΔPDF square-window viewer.  Draws
// gray dashed lines at integer multiples of the direct-lattice spacing along each
// displayed axis, over a square [-half, +half] Å window.  Renders into a
// normalized viewBox with a non-scaling stroke, so it fills its square parent
// responsively — used both at a fixed px size (single ΔPDF panel) and in the
// fluid multi-volume grid cells.

interface Props {
  half: number; // half-window in Å (box spans [-half, +half] on both axes)
  latX: number | null;
  latY: number | null;
}

const VB = 1000; // normalized viewBox side; stroke stays 1px (non-scaling)

// Lattice-multiple positions (in Å) that fall inside [-half, +half].
function multiples(lat: number | null, half: number): number[] {
  if (!lat || lat <= 0) return [];
  const out: number[] = [];
  for (let k = Math.ceil(-half / lat); k <= Math.floor(half / lat); k++) {
    out.push(k * lat);
  }
  return out;
}

export function UnitCellGrid({ half, latX, latY }: Props) {
  const toX = (v: number) => ((v + half) / (2 * half)) * VB;
  // canvas y is flipped (smallest y at the bottom), so mirror here too.
  const toY = (v: number) => VB - ((v + half) / (2 * half)) * VB;

  const vx = multiples(latX, half).map(toX);
  const hy = multiples(latY, half).map(toY);

  return (
    <svg
      viewBox={`0 0 ${VB} ${VB}`}
      preserveAspectRatio="none"
      width="100%"
      height="100%"
      style={{ position: "absolute", left: 0, top: 0, pointerEvents: "none" }}
    >
      <g
        stroke="rgba(150, 158, 172, 0.6)"
        strokeWidth={1}
        strokeDasharray="4 3"
        shapeRendering="crispEdges"
      >
        {vx.map((px, i) => (
          <line
            key={`v${i}`}
            x1={px}
            y1={0}
            x2={px}
            y2={VB}
            vectorEffect="non-scaling-stroke"
          />
        ))}
        {hy.map((py, i) => (
          <line
            key={`h${i}`}
            x1={0}
            y1={py}
            x2={VB}
            y2={py}
            vectorEffect="non-scaling-stroke"
          />
        ))}
      </g>
    </svg>
  );
}
