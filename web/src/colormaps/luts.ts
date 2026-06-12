// 256-entry RGBA colormap lookup tables, interpolated from control stops.
// Sequential maps (inferno/viridis/hot) approximate matplotlib's; the diverging
// map (RdBu_r: blue→white→red) is used for the signed ΔPDF viewers.

type RGB = [number, number, number];
type Stop = [number, RGB];

function buildLut(stops: Stop[]): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256 * 4);
  let si = 0;
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    while (si < stops.length - 2 && t > stops[si + 1][0]) si++;
    const [t0, c0] = stops[si];
    const [t1, c1] = stops[si + 1];
    const f = t1 > t0 ? (t - t0) / (t1 - t0) : 0;
    lut[i * 4] = c0[0] + (c1[0] - c0[0]) * f;
    lut[i * 4 + 1] = c0[1] + (c1[1] - c0[1]) * f;
    lut[i * 4 + 2] = c0[2] + (c1[2] - c0[2]) * f;
    lut[i * 4 + 3] = 255;
  }
  return lut;
}

const INFERNO: Stop[] = [
  [0.0, [0, 0, 4]],
  [0.13, [31, 12, 72]],
  [0.25, [85, 15, 109]],
  [0.38, [136, 34, 106]],
  [0.5, [186, 54, 85]],
  [0.63, [227, 89, 51]],
  [0.75, [249, 140, 10]],
  [0.88, [249, 201, 50]],
  [1.0, [252, 255, 164]],
];

const VIRIDIS: Stop[] = [
  [0.0, [68, 1, 84]],
  [0.25, [59, 82, 139]],
  [0.5, [33, 145, 140]],
  [0.75, [94, 201, 98]],
  [1.0, [253, 231, 37]],
];

const HOT: Stop[] = [
  [0.0, [10, 0, 0]],
  [0.33, [255, 0, 0]],
  [0.66, [255, 255, 0]],
  [1.0, [255, 255, 255]],
];

const RDBU_R: Stop[] = [
  [0.0, [5, 48, 97]],
  [0.25, [67, 147, 195]],
  [0.5, [247, 247, 247]],
  [0.75, [214, 96, 77]],
  [1.0, [103, 0, 31]],
];

export const COLORMAPS: Record<string, Uint8ClampedArray> = {
  inferno: buildLut(INFERNO),
  viridis: buildLut(VIRIDIS),
  hot: buildLut(HOT),
  RdBu_r: buildLut(RDBU_R),
};

export const SEQUENTIAL_NAMES = ["inferno", "viridis", "hot"];
export const DIVERGING_NAME = "RdBu_r";
