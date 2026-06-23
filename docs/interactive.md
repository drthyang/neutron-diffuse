# Interactive Exploration

## Overview

`nebula3d` ships a small **visualization** layer (`nebula3d.visualization`) and
interactive example scripts for inspecting an `HKLVolume` by eye: slices, radial
profiles, azimuthal ring texture, and cleanup before/after panels.

This is deliberately kept as a thin, scriptable layer rather than a monolithic
GUI: each plot function takes an `HKLVolume`, draws into a Matplotlib `Axes`
(or `Figure`), and returns it, so the same calls work in a one-shot script, an
IPython session, or a Jupyter notebook. Future interactive front-ends (widget
panels, a dashboard, Mantid-side hooks) should build on these same primitives.

> This page documents the **matplotlib viewers and the `nebula3d.visualization`
> API**. For copy-paste launch commands across 22/45/100 K, see
> [commands.md](commands.md); for the browser console, see [web.md](web.md).

---

## 1. Cleanup QA Viewer

The current real-data QA entry point is `examples/explore_slice.py`. It processes
all H planes, then opens a four-panel viewer with an **H/K/L plane selector** and
a cut-position slider:

```bash
env PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl USE_BACKGROUND=0 \
PUNCH_PRESET=cc_on MODE=both MIN_I=0.8 MIN_PROM=0.8 \
INTEGER_FIT_POSITION=1 INTEGER_FIT_SHAPE=1 INTEGER_H_GUARD=0.12 \
SEARCH_EXCLUDE_H=-0.6667,-0.3333,0.3333,0.6667 SEARCH_EXCLUDE_H_WIDTH=0.08 \
BACKFILL_METHOD=q_shell H_VALUE=0.3333 \
python3 examples/explore_slice.py
```

Panels:

- `data`
- `Removed ring`
- `Punched`
- `Backfilled`

The H/K/L radio selector (bottom-left, above the linear/log toggle) retargets the
slider to the matching fixed axis and redraws all four panels as `0kl`, `h0l`, or
`hk0` slices. Set the initial orientation with `VIEW_AXIS=H|K|L` (default `H`)
and the initial cut with `{H,K,L}_VALUE` (default `0.3333` for H, else `0.0`).

> The selector only changes how the volumes are **displayed**. The in-script
> ring-removal compute path always works along H (`0kl` planes); to ring-remove
> along K or L, process the volume with `remove_rings_3d.py SLICE_AXIS=K|L` and
> load the result via `RING_FILE`.

Use this viewer to inspect integer-H Bragg cleanup and fractional-H diffuse
preservation before running the final 3D-DeltaPDF transform.

## 2. ΔPDF Real-Space Viewers (standard preview)

Three interactive viewers preview the **real-space 3D-ΔPDF** (the FFT of the
cleaned diffuse volume): two single-temperature views (orthoslice and
single-plane) that read the cached `examples/_delta_pdf.h5` written by
`examples/delta_pdf.py` (run that first if it is missing), and a
multi-temperature comparison view. **Use these as the standard preview for ΔPDF
results — do not build ad-hoc plotting scripts.**

### Orthoslice viewer (recommended) — `examples/explore_delta_pdf_ortho.py`

All three orthogonal real-space planes at once — `x_H–y_K` (a–b), `x_H–z_L`
(a–c), `y_K–z_L` (b–c) — with sliders to move each cut position, a contrast
control, and a **unit-cell gridline toggle**. Each panel auto-scales to its own
robust level, so the three very different magnitudes stay readable.

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl RMAX=50 \
python3 examples/explore_delta_pdf_ortho.py
```

Controls: `x_H / y_K / z_L cut` sliders · `contrast ×` (scales the per-panel
colour limits) · `unit cells` checkbox (light-gray gridlines at the a/b/c lattice
spacings).

- `RMAX` — display half-window (Å) for all axes (default 50).
- `PERCENTILE` — per-panel colour-scale percentile at r>3 Å (default 98).
- `CONTRAST_MIN` / `CONTRAST_MAX` — contrast-slider range (default 0.1 .. 20).
- `LAT_A` / `LAT_B` / `LAT_C` — direct-lattice constants (Å) for the gridlines
  (default: ΔPDF `.h5` attrs, else the source backfilled UB matrix).
- `PDF_FILE` — ΔPDF `.h5` (default `examples/_delta_pdf.h5`).
- `SMOKE=1` — render one frame to PNG headless (verification, no GUI).

### Single-plane viewer — `examples/explore_delta_pdf.py`

The `y_K–z_L` plane with an `x_H` slider, a `|scale|` slider, and the same
**unit-cell gridline toggle** (b along y_K, c along z_L). Good for focusing on the
b–c correlation plane carried by the diffuse layers.

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl RMAX=25 \
python3 examples/explore_delta_pdf.py
```

- `RMAX` — display half-window (Å) for K and L (default 25).
- `SCALE_MAX` — upper `|scale|` slider multiple of the p99 level (default 20).
- `LAT_A` / `LAT_B` / `LAT_C` — lattice constants (Å) for the gridlines.
- `X_VALUE` — initial x_H plane (Å); `PDF_FILE` / `PROC_FILE`; `SMOKE=1`.

Both viewers store the lattice constants (`lat_a/b/c`) directly in the ΔPDF
`.h5` when written by `examples/delta_pdf.py`, so the gridlines work without any
fallback on freshly generated transforms.

Both use the `macosx` backend and block on `plt.show()` — launch with
`run_in_background: true`; exit code 0 means the window was closed. For the
static three-panel contact sheet (central cuts), `examples/delta_pdf.py` also
writes `_delta_pdf_hk0/h0l/0kl.png`.

### Multi-temperature comparison — `examples/explore_delta_pdf_multi.py`

The 22 K / 45 K / 100 K comparison: three rows (one per temperature) × the three
orthoslice planes, with shared `x_H / y_K / z_L` cut sliders, a `contrast ×`
slider, and the unit-cell gridline toggle. Each plane (column) is scaled to its
own level pooled across the three temperatures, so the temperatures are directly
comparable *within* a plane. It **auto-detects** the pipeline's per-temperature
`data/processed/*{T}*_delta_pdf.h5` outputs — no paths needed once the pipeline
has run for all three temperatures.

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl RMAX=28 \
python3 examples/explore_delta_pdf_multi.py
```

- `PDF_22K` / `PDF_45K` / `PDF_100K` — override the auto-detected paths.
- `RMAX` — display half-window (Å) for all axes (default 50).
- `PERCENTILE` — per-plane colour-scale percentile at r>3 Å (default 98).
- `CONTRAST_MIN` / `CONTRAST_MAX` — `contrast ×` slider range (default 0.1 .. 20).
- `SMOKE=1` — render one frame to PNG headless (verification, no GUI).

## 3. Launching A General Plotting Session

The older exploration preamble (`examples/explore.py`) loads raw/data/background
volumes when those files are present and pulls the plot helpers into scope. Run
it with IPython and an **interactive** Matplotlib backend so figures open in
live, pan/zoom windows and you stay at a prompt:

```bash
cd <repo root>
PYTHONPATH=src ipython --matplotlib=macosx -i examples/explore.py
```

- `--matplotlib=macosx` — interactive backend on macOS. Use `--matplotlib=qt`
  on Linux/Windows (requires a Qt binding), or run the same imports inside
  Jupyter with `%matplotlib widget`.
- `-i` — drop into the IPython prompt after the preamble runs.
- The preamble calls `plt.ion()`, so figures appear as soon as you plot; no
  explicit `plt.show()` is needed (call it manually in a plain `python` REPL).

After it loads, these names are in scope:

| Name | What |
|------|------|
| `data`, `bkg`, `sub` | `HKLVolume` for data / background / (data − bkg) |
| `plot_slice`, `plot_radial_profile`, `plot_azimuthal_map`, `plot_overview` | plot functions |
| `extract_slice` | numeric slice extractor (no plotting) |
| `plt` | `matplotlib.pyplot` |

> The background scan has no UB matrix of its own, so the preamble loads it with
> `ub_matrix=data.ub_matrix` — all three volumes then share one physics-convention
> (2π) |Q| scale. See [`io/mantid_nxs.py`](../src/nebula3d/io/mantid_nxs.py).

`examples/explore.py` auto-discovers the long Mantid filenames in `data/raw/`,
so it keeps working if you swap in a different dataset.

---

## 4. The Visualization API

All functions live in `nebula3d.visualization` and are re-exported at that level:

```python
from nebula3d.visualization import (
    extract_slice, plot_slice,
    plot_radial_profile, plot_azimuthal_map,
    plot_overview, SliceData,
)
```

### `plot_slice(vol, plane="kl", value=0.0, ...) -> Axes`

2D intensity slice through the volume. `plane` is read as
**`(horizontal, vertical)`** and selects the two displayed axes; the two
orderings of a pair are **transposes** of each other. The **remaining (fixed)
axis** is cut at `value` — so any `'hk'`/`'kh'` fixes L, and `value=0.3333` is
the L = 0.3333 plane:

| plane | x-axis (horizontal) | y-axis (vertical) | fixed (cut by `value`) |
|-------|---------------------|-------------------|------------------------|
| `'kl'` / `'0kl'` | K | L | H |
| `'lk'` | L | K | H |
| `'hl'` / `'h0l'` | H | L | K |
| `'lh'` | L | H | K |
| `'hk'` / `'hk0'` | H | K | L |
| `'kh'` | K | H | L |

(Mantid-style aliases `'0kl'`, `'h0l'`, `'hk0'` map to the principal planes.)

Key options: `log_scale` (log₁₀ with a 1%-of-max floor), `percentile` (colour
clip, default 99.5; lower clip is the symmetric `100 - percentile`), `vmin`/`vmax`
(manual colour limits — either may be given alone; interpreted on the log scale
when `log_scale=True`), `interp` (see below), `cmap` (default `"hot"`), `ax`,
`title`. Masked/empty voxels are drawn in grey.

```python
plot_slice(bkg, "kl", value=0.0, log_scale=True)          # background rings
plot_slice(data, "hk", value=0.3333, interp=True)         # exact L = 1/3 plane
plot_slice(data, "hk", value=0.3333, interp=True,
           vmin=0.0, vmax=0.4)                            # manual colour limits
```

**Off-grid cuts (`interp`).** By default `value` snaps to the nearest grid
plane (e.g. with an L step of 0.12, asking for 0.3333 silently gives 0.36).
Pass `interp=True` to linearly interpolate between the two bracketing planes
so the exact value is honoured. The interpolation is NaN-aware: where only one
bracketing plane is valid, that value is used, so the cut does not erode the
masked boundary. Out-of-range values clamp to the first/last plane.

### `extract_slice(vol, plane="kl", value=0.0) -> SliceData`

Numeric version of the above — returns a `SliceData` NamedTuple
(`data, y_axis, x_axis, y_label, x_label, cut_label`) with masked voxels as NaN.
Use it when you want the array, not a plot.

### `plot_radial_profile(vol, n_bins=500, stat="mean", ...) -> Axes`

1D |Q| vs intensity profile (wraps
`nebula3d.preprocessing.powder_rings.radial_profile`). Options: `stat`
(`'mean'`/`'median'`), `q_range=(lo, hi)`, `mark_q=[...]` (vertical dashed lines
at given |Q|, e.g. ring centres), plus any `**kwargs` forwarded to `ax.plot`.

```python
plot_radial_profile(data, mark_q=[2.69])           # mark Al(111) at 2.69 Å⁻¹
```

> `mark_q` is a **list** of |Q| values, not a scalar.

### `plot_azimuthal_map(vol, q_center, q_width=0.1, n_phi_bins=72, ...) -> Axes`

Intensity vs azimuthal angle φ within a thin |Q| shell — i.e. the ring texture
T(φ). Voxels with `||Q| − q_center| < q_width` are binned over [−180°, 180°];
bins with fewer than 3 voxels are left blank. Raises `ValueError` if the shell
is empty.

```python
plot_azimuthal_map(data, q_center=2.69)            # texture of the Al(111) ring
```

### `plot_overview(vol, log_scale=False, ...) -> Figure`

2×2 diagnostic: K-L (H=0), H-L (K=0), H-K (L=0) slices + radial profile. Good
first look at any volume. Options mirror the per-panel functions (`log_scale`,
`cmap`, `percentile`, `vmin`/`vmax` — shared across all three slice panels —
`mark_q`, `title`; default title is `vol.instrument`).

```python
fig = plot_overview(data, log_scale=True)
```

---

## 5. Recipes

```python
# Compare data vs background on a shared colour treatment.
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plot_slice(data, "kl", 0.0, ax=ax[0], log_scale=True, title="data")
plot_slice(bkg,  "kl", 0.0, ax=ax[1], log_scale=True, title="background")

# Save instead of show (headless / batch).
fig.savefig("data/compare.png", dpi=110)   # data/*.png is gitignored
```

For non-interactive/batch use, set the Agg backend before importing pyplot:

```python
import matplotlib; matplotlib.use("Agg")
```

---

## 6. Design Notes & Future Directions

- **Primitive-first.** Every plot function accepts an `HKLVolume`, draws into a
  caller-supplied `Axes`/`Figure`, and returns it. Composite views
  (`plot_overview`) are built by composing the primitives — new dashboards or
  widget panels should do the same rather than re-implementing plotting.
- **Separation of compute and display.** `extract_slice` (numeric) is split from
  `plot_slice` (display); profile plots wrap the existing
  `preprocessing.powder_rings` math rather than duplicating it. Keep this split
  as more views are added.
- **Implemented interactive front-ends** (see sections 1–2):
  - cleanup before/after panel wired to the pipeline with an H/K/L plane
    selector and a cut slider (`examples/explore_slice.py`);
  - linked ΔPDF real-space views — single-plane `x_H`-slider viewer
    (`examples/explore_delta_pdf.py`) and the three-plane orthoslice viewer
    with movable cuts (`examples/explore_delta_pdf_ortho.py`);
  - temperature-comparison grid with shared cuts and a global colour scale
    (`examples/explore_delta_pdf_multi.py`).
- **Still planned:**
  - live `|Q|`-shell scrubbing for the azimuthal/ring texture views;
  - migrating the standalone viewers onto the `nebula3d.visualization` primitives
    so the core stays free of GUI dependencies.

---

## See also

- [`powder_rings.md`](algorithms/powder_rings.md) — the |Q| binning and ring
  detection that `plot_radial_profile` / `plot_azimuthal_map` visualise.
- [`bragg_cleanup.md`](algorithms/bragg_cleanup.md) — Bragg punch/backfill
  workflow and current guarded `MODE=both` settings.
- `examples/explore.py` — the live-session preamble described above.
