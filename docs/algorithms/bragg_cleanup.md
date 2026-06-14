# Bragg Cleanup

## Purpose

Before 3D-DeltaPDF, sharp Bragg and satellite peaks must be removed from the
diffuse volume and replaced with a plausible diffuse background. The current
workflow is:

```text
ring-removed volume
    -> BraggRemover.build_mask()
    -> backfill_bragg()
    -> cleaned diffuse volume
```

The direct beam at `(0,0,0)` is handled separately from ordinary Bragg peaks.

In practice, use the guarded `mode="both"` workflow for current real-data QA:
integer-node Bragg peaks are handled with lattice-aware punches, and the
hkl-agnostic search stage is constrained so it does not remove known
fractional-H diffuse planes.

## Detection Modes

`BraggRemover(mode=...)` supports three modes:

| Mode | Behavior |
|------|----------|
| `integer` | Enumerate integer `(h,k,l)` nodes and decide per node whether a peak is present. |
| `search` / `auto` | Search all valid voxels for sharp high-tail outliers in robust `|Q|` shells. |
| `both` | Run `integer` first, punch those peaks, then run `search` on the residual. |

The current visual preference is guarded `mode="both"`: integer-node Bragg peaks
are handled lattice-aware, while search catches off-integer satellites where it
is safe to do so.

## Integer-Node Path

The integer path is lattice-aware:

1. Enumerate integer `(h,k,l)` nodes in the volume.
2. Inspect a local HKL window around each node.
3. Keep the node only if a real nearby peak is present:
   - `min_intensity`
   - `min_prominence`
   - optional `integer_n_mad` against a robust per-`|Q|` shell level.
4. Recenter to the measured local peak.
5. Optionally fit peak position and anisotropic shape:
   - `integer_optimize_position=True`
   - `integer_optimize_shape=True`
   - `integer_fit_covariance=True` (Phase 3) fits a **tilted** 3×3 resolution
     ellipsoid following the peak's real orientation (a full weighted covariance,
     eigen-clipped to the base/max bounds) instead of three axis-aligned radii,
     and folds the φ-tail into it as a rank-1 tangential inflation rather than a
     separate unioned ellipsoid. Default off; reduces exactly to the diagonal
     radii fit for an axis-aligned peak.
6. Punch a continuous-HKL ellipsoid at the fitted centre.

Useful guards:

- `integer_h_guard_hkl`: clips integer-node punches to a slab around the source
  integer-H plane. This prevents strong integer-H Bragg holes from extending into
  fractional-H diffuse planes such as `H=±1/3` or `H=±2/3`.
- `integer_fit_max_radius_hkl`: caps fitted per-peak radii.

### Small but sharp weak Bragg (`integer_local_prominence_n_mad`)

Weak Bragg peaks at integer nodes can sit below the absolute `min_intensity` /
`min_prominence` floors yet still be sharp, local outliers. A purely
sharpness-based catch over the whole volume just finds noise (a small spike in a
flat region looks "sharp"); the reliable discriminator is **position** — Bragg
sits at integer nodes, which are 4–5× more likely to carry a residual sharp peak
than random control positions. So the catch is applied **only at integer nodes**:

- `integer_local_prominence_n_mad`: keep a node when its prominence
  `(peak − local_bg)` is at least this many **local** MADs (measured in the
  detection window), regardless of the absolute floors and the `|Q|`-shell
  threshold. `integer_local_min_prominence` adds an optional small absolute floor
  to reject pure noise in flat regions.

Because it is locked to integer nodes (never a fractional-H plane) and obeys
`integer_h_guard_hkl`, it cannot touch the q=1/3 diffuse. Default `cc_on` value
is `8` (~+0.4 % extra punched on test data, all at lattice nodes).

## Search Path

Search mode is hkl-agnostic. At each `|Q|`, it estimates a robust background
(`median + n*MAD`) and keeps local maxima above that level and the absolute
floor.

Because search does not know the lattice or magnetic diffuse planes, protect
known fractional-H diffuse planes. Either an explicit centre list or — preferred
for a modulation that repeats at every integer — a **periodic** fractional rule
that shields the whole family across the full H range:

```text
# explicit centres (fixed planes only):
SEARCH_EXCLUDE_H=-0.6667,-0.3333,0.3333,0.6667
# OR periodic: protect every integer±1/3 plane (q=1/3 family: ±1/3, ±2/3,
# ±4/3, ±5/3, ±7/3 …):
SEARCH_EXCLUDE_H_FRACTIONS=0.3333,0.6667
SEARCH_EXCLUDE_H_WIDTH=0.08
```

`search_exclude_h_fractions` matches H by its fractional part mod 1, so a single
setting covers the higher-order satellites (`±4/3`, `±5/3`, …) that a fixed
centre list misses. This allows `mode="both"` to keep useful off-integer
satellite detection without punching structured diffuse on any thirds plane.

## Direct Beam

The direct beam is not a Bragg reflection. It is punched after ordinary peak
detection using independent settings:

```text
INCIDENT_ELLIPSOID_R_HKL=0.15,0.50,1.00
INCIDENT_SPHERE_R_HKL=
```

The ellipsoid is sized from H/K/L linecuts through the origin. The direct-beam
backfill uses a special just-outside-`|Q|` shell so the fill does not sample the
negative over-subtraction halo adjacent to the beam.

## Punch Coordinate Space (HKL today; Q-space planned)

The punch ellipsoid is currently defined in **fractional HKL**: a peak at
`(h₀,k₀,l₀)` is removed where

```text
((H−h₀)/rh)² + ((K−k₀)/rk)² + ((L−l₀)/rl)² ≤ 1
```

with `punch_radii = (rh, rk, rl)` in r.l.u. This is convenient (it matches the
grid axes) but the *physical* peak profile is a function of **Q** — instrument
resolution plus size/strain/mosaic — and does not depend on the lattice
constants. HKL radii therefore bake in the `b*` scaling: on TbTi3Bi4 the default
`(0.09, 0.12, 0.45)` r.l.u. looks 5× anisotropic but is `≈(0.097, 0.072, 0.115)`
Å⁻¹ — **near-isotropic in Q**. The factor-5 "L is broad" is almost entirely the
small `c* = 0.25 Å⁻¹` and the coarse `ΔL = 0.15` r.l.u. sampling, not the peak.

The metric `g = UBᵀUB` for these crystals is diagonal to ~0.5% (orthorhombic),
so an axis-aligned ellipsoid in Q is — to that precision — the same set of voxels
as today's HKL punch. The motivation to move to Q is therefore not this dataset's
masks but capability: correct off-axis peak orientation (radial vs tangential),
oblique crystals (where an HKL-axis ellipsoid shears relative to the resolution
ellipsoid), and parameters in Å⁻¹ that transfer across temperatures and samples.

**A single quadratic-form kernel** `δhklᵀ A δhkl ≤ 1` (`_ellipsoid_inside`)
subsumes all current shapes and the Q-space upgrade:

| Shape spec | `A` |
|------------|-----|
| legacy HKL radii `(rh,rk,rl)` | `diag(1/rh², 1/rk², 1/rl²)` |
| Q isotropic radius `ρ` (Å⁻¹) | `g / ρ²` |
| Q per-axis radii `(ra,rb,rc)` (Å⁻¹) | `Pᵀ diag(1/r²) P`, `P = ê·UB` |
| fitted resolution ellipsoid (Phase 3) | per-peak 3×3 `A` from the covariance (φ-tail = rank-1 mod) |

**Using the Q-space punch (opt-in).** Set `punch_frame="q"` and either
`punch_q_radius` (isotropic, Å⁻¹) or `punch_q_radii` (along a*, b*, c*, Å⁻¹) on
`PunchParams` / `BraggRemover`, or pick **Frame → Q-space (Å⁻¹)** in the web
Run-pipeline panel. The Q radii are the **resolution floor** in Å⁻¹ (lattice- and
temperature-independent). `punch_frame="hkl"` (default) keeps the radii path
above, so existing configs and saved pipelines are unaffected.

In Q-mode the punch is **adaptive**, not fixed: the per-peak shape-fit is floored
to the Q resolution and then punches through the same mechanism as the HKL path
(axis-aligned ellipsoid + union φ-tail, intensity-scaled), so the Q frame only
relocates the floor from r.l.u. to Å⁻¹ — it does not discard the fit or the
φ-tail. Off-integer search peaks (no per-peak fit) use the fixed Q metric
ellipsoid. Validated against the production HKL punch at Jaccard ≈ 0.89–0.93 on
the 22/45/100 K series (see `examples/compare_punch_frames.py`); the matched
isotropic radius ρ ≈ 0.093 Å⁻¹ is temperature-invariant.

Phase 0/1/2 tests live in `tests/test_bragg_qspace_phase{0,1,2}.py`. Note that
HKL- and Q-axis punches are bit-identical only for an *exactly* diagonal metric —
on the real (~0.5%-sheared) UB they differ at a handful of boundary voxels.

## Backfill Modes

`backfill_bragg` supports:

| Method | Use |
|--------|-----|
| `local` | Fill each connected component from a local dilated shell median. |
| `q_shell` | Fill ordinary Bragg holes from the robust radial background at the same `|Q|`. |
| `tv`, `symmetry`, `symmetry+tv`, etc. | General inpainting fallbacks. |

Current real-data QA uses `METHOD=q_shell` for ordinary Bragg holes and keeps the
special direct-beam fill enabled.

## Recommended QA Settings

```bash
PUNCH_PRESET=cc_on MODE=both MIN_I=0.8 MIN_PROM=0.8 \
INTEGER_FIT_POSITION=1 INTEGER_FIT_SHAPE=1 INTEGER_H_GUARD=0.12 \
SEARCH_EXCLUDE_H=-0.6667,-0.3333,0.3333,0.6667 SEARCH_EXCLUDE_H_WIDTH=0.08 \
BACKFILL_METHOD=q_shell
```

Inspect `H=0` for residual Bragg peaks and `H=±1/3`, `±2/3` for diffuse
preservation.
