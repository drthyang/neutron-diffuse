# Interactive Exploration

## Overview

`ndiff` ships a small **visualization** layer (`ndiff.visualization`) and an
interactive **exploration preamble** (`examples/explore.py`) for inspecting an
`HKLVolume` by eye — slices, radial profiles, azimuthal ring texture, and a
multi-panel diagnostic overview.

This is deliberately kept as a thin, scriptable layer rather than a monolithic
GUI: each plot function takes an `HKLVolume`, draws into a Matplotlib `Axes`
(or `Figure`), and returns it, so the same calls work in a one-shot script, an
IPython session, or a Jupyter notebook. Future interactive front-ends (widget
panels, a dashboard, Mantid-side hooks) should build on these same primitives.

---

## 1. Launching a live plotting session

The exploration preamble loads the three real 28K volumes and pulls the plot
helpers into scope. Run it with IPython and an **interactive** Matplotlib
backend so figures open in live, pan/zoom windows and you stay at a prompt:

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
> (2π) |Q| scale. See [`io/mantid_nxs.py`](../src/ndiff/io/mantid_nxs.py).

`examples/explore.py` auto-discovers the long Mantid filenames in `data/raw/`,
so it keeps working if you swap in a different dataset.

---

## 2. The visualization API

All functions live in `ndiff.visualization` and are re-exported at that level:

```python
from ndiff.visualization import (
    extract_slice, plot_slice,
    plot_radial_profile, plot_azimuthal_map,
    plot_overview, SliceData,
)
```

### `plot_slice(vol, plane="kl", value=0.0, ...) -> Axes`

2D intensity slice through the volume. `plane` selects the two displayed axes
(`'kl'`, `'hl'`, `'hk'`; Mantid aliases `'0kl'`, `'h0l'`, `'hk0'` accepted) and
`value` is the cut coordinate on the third axis (nearest grid point used).

Key options: `log_scale` (log₁₀ with a 1%-of-max floor), `percentile` (colour
clip, default 99.5; lower clip is the symmetric `100 - percentile`), `vmin`/`vmax`
(override the clip), `cmap` (default `"hot"`), `ax`, `title`. Masked/empty voxels
are drawn in grey.

```python
plot_slice(bkg, "kl", value=0.0, log_scale=True)   # background powder rings
```

### `extract_slice(vol, plane="kl", value=0.0) -> SliceData`

Numeric version of the above — returns a `SliceData` NamedTuple
(`data, y_axis, x_axis, y_label, x_label, cut_label`) with masked voxels as NaN.
Use it when you want the array, not a plot.

### `plot_radial_profile(vol, n_bins=500, stat="mean", ...) -> Axes`

1D |Q| vs intensity profile (wraps
`ndiff.preprocessing.powder_rings.radial_profile`). Options: `stat`
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
`cmap`, `percentile`, `mark_q`, `title`; default title is `vol.instrument`).

```python
fig = plot_overview(data, log_scale=True)
```

---

## 3. Recipes

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

## 4. Design notes & future directions

- **Primitive-first.** Every plot function accepts an `HKLVolume`, draws into a
  caller-supplied `Axes`/`Figure`, and returns it. Composite views
  (`plot_overview`) are built by composing the primitives — new dashboards or
  widget panels should do the same rather than re-implementing plotting.
- **Separation of compute and display.** `extract_slice` (numeric) is split from
  `plot_slice` (display); profile plots wrap the existing
  `preprocessing.powder_rings` math rather than duplicating it. Keep this split
  as more views are added.
- **Planned interactive front-ends** (not yet implemented):
  - slider widgets to scrub the cut `value` and |Q| shell live
    (`ipywidgets` / Matplotlib `Slider`);
  - a before/after panel wired to the processing pipeline
    (`EmptySubtractor` → `PatchedRingModel` → backfill);
  - linked ΔPDF real-space views.

  Each should consume the `ndiff.visualization` primitives so the core stays
  free of GUI dependencies.

---

## See also

- [`powder_rings.md`](algorithms/powder_rings.md) — the |Q| binning and ring
  detection that `plot_radial_profile` / `plot_azimuthal_map` visualise.
- `examples/explore.py` — the live-session preamble described above.
