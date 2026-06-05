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
6. Punch a continuous-HKL ellipsoid at the fitted centre.

Useful guards:

- `integer_h_guard_hkl`: clips integer-node punches to a slab around the source
  integer-H plane. This prevents strong integer-H Bragg holes from extending into
  fractional-H diffuse planes such as `H=±1/3` or `H=±2/3`.
- `integer_fit_max_radius_hkl`: caps fitted per-peak radii.

## Search Path

Search mode is hkl-agnostic. At each `|Q|`, it estimates a robust background
(`median + n*MAD`) and keeps local maxima above that level and the absolute
floor.

Because search does not know the lattice or magnetic diffuse planes, protect
known fractional-H diffuse planes with:

```text
SEARCH_EXCLUDE_H=-0.6667,-0.3333,0.3333,0.6667
SEARCH_EXCLUDE_H_WIDTH=0.08
```

This allows `mode="both"` to keep useful off-integer satellite detection without
allowing search to punch structured diffuse features on those planes.

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

## Backfill Modes

`backfill_bragg` supports:

| Method | Use |
|--------|-----|
| `local` | Fill each connected component from a local dilated shell median. |
| `q_shell` | Fill ordinary Bragg holes from the robust radial background at the same `|Q|`. |
| `tv`, `symmetry`, `symmetry+tv`, etc. | General inpainting fallbacks. |

Current real-data QA uses `METHOD=q_shell` for ordinary Bragg holes and keeps the
special direct-beam fill enabled.

## Recommended Current QA Settings

```bash
PUNCH_PRESET=cc_on MODE=both MIN_I=0.8 MIN_PROM=0.8 \
INTEGER_FIT_POSITION=1 INTEGER_FIT_SHAPE=1 INTEGER_H_GUARD=0.12 \
SEARCH_EXCLUDE_H=-0.6667,-0.3333,0.3333,0.6667 SEARCH_EXCLUDE_H_WIDTH=0.08 \
BACKFILL_METHOD=q_shell
```

Inspect `H=0` for residual Bragg peaks and `H=±1/3`, `±2/3` for diffuse
preservation.
