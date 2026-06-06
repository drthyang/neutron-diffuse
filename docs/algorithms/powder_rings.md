# Powder Ring Removal

## Purpose

Polycrystalline material in the beam path can add powder rings to a single-crystal
diffuse scattering volume. Common sources include the sample environment,
cryostat, sample holder, and capsule walls.

The goal is to subtract the ring contribution while preserving real diffuse
structure. The current production path is subtractive: it estimates only the
azimuthally smooth ring intensity and subtracts that estimate. It does not mask
or replace voxels just because they have radial excess, because radial excess can
also be genuine diffuse scattering.

## Physical Basis

A powder ring is localized in `|Q|` (a thin spherical shell at the powder
d-spacing), but its amplitude is not uniform around the shell. Detector
solid-angle coverage, absorption path length, and normalization artifacts
modulate the ring intensity with azimuthal direction. In real data, rings are
therefore not isotropic.

A useful model for a voxel with `|Q|` and azimuthal angle `phi` is:

```
I_ring(Q, phi) = T(phi) x sum_i A_i G(|Q| - q_i, sigma_i)
```

- **G(|Q| - q_i, sigma_i)**: radial profile of ring *i*.
- **Aᵢ**: per-ring amplitude (structure factor × absorption).
- **T(phi)**: azimuthal texture from detector coverage, absorption, and
  normalization.

The measured signal is `I_measured(Q, φ) = I_diffuse(Q) + I_ring(Q, φ)`, where the
diffuse signal we want is direction-dependent and does **not** share the ring's radial
peak structure or azimuthal texture.

## Current Production Path

The original factored Gaussian/SVD model remains in the package as
`PatchedRingModel`, but the current real-data driver uses
`PatchedRadialRingModel` through `examples/remove_rings_3d.py`.

That path is non-parametric:

1. Fit each H plane independently in the displayed `0kl` frame.
2. Build robust radial profiles in azimuthal patches.
3. Estimate a smooth baseline with SNIP-like clipping.
4. Subtract only azimuthally smooth ring intensity.
5. Carry cross-H confirmed ring shells and amplitude ceilings into each plane so
   integer-H Bragg artifacts do not become fake powder rings.

The key design rule is: **ring removal is subtractive only**. Do not replace
masked/excess regions unless the mask is based on azimuthal smoothness, not
radial excess.

## Legacy Factored Ring Algorithm

The older algorithm is still useful background and remains available for
comparison. It assumes a shared azimuthal texture across all rings and then
optionally backfills masked shells.

### Step 1 — Empty-scan subtraction  (`EmptySubtractor`)

```
I_residual(Q) = I_sample(Q) − s × I_empty(Q)
```

The empty-environment scan removes the cryostat/furnace ring. The scale `s` is estimated
analytically by minimising the residual in a ring-dominated |Q| window
(`s = Σ I_sample·I_empty / Σ I_empty²`). A residual ring from the **sample holder**
remains, because the holder is present only during the sample scan.

### Step 2 — Factored ring model  (`PatchedRingModel`)

Detect ring |Q| positions, then fit the factored model:

1. **Detect rings** (`detect_ring_shells`): bin voxels into a 1D radial profile, estimate a
   baseline with a rolling median (robust to peaks wider than the ring), subtract it, and
   pick peaks above a noise threshold. Returns ring |Q| ranges `(q_lo, q_center, q_hi)`.
2. **Azimuthal patches**: divide φ ∈ [0, 2π) (in a reference plane, default hk0,
   φ = atan2(k_Q, h_Q)) into N overlapping Hann-weighted patches.
3. **Per-patch NNLS**: with ring positions qᵢ and widths σᵢ fixed from the global fit,
   solve a non-negative least-squares problem for the per-patch amplitudes →
   amplitude matrix `A[n_rings × n_patches]`.
4. **Rank-1 SVD** of `A`: `A[i, P] ≈ Aᵢ × T[P]` → per-ring amplitudes and per-patch
   texture values.
5. **Fourier series** fit to `(φ_P, T[P])`: `T(φ) = c₀ + Σₖ (aₖ cos kφ + bₖ sin kφ)`.
   Smooth, periodic, C∞ → C¹ continuity across patch boundaries is automatic.

Subtract the full model from **every voxel**. Voxels where the ring dominates
(`I_ring / σ_data > threshold`) are masked for backfill downstream.

> **Note on aluminium**: Al (FCC, Fm-3m, a ≈ 4.046 Å) is the most common source.
> Its peak positions can be pre-computed with `al_ring_q_positions()` and passed as
> `ring_hints` to seed the fit. But the algorithm is material-agnostic and works
> without this prior.

### Step 3 — Backfill the masked shell  (`backfill_ring_shells`)

The masked region forms a **thin spherical shell** in HKL space. For each masked voxel,
the nearest uncontaminated neighbours in 3D HKL space lie at nearly the same direction but
just inside or outside the shell. A distance- and inverse-variance-weighted interpolation
across the shell:

- imposes **no assumption** on the diffuse signal shape;
- is C¹ at the shell boundary by construction;
- is physically motivated, since the diffuse signal is smooth in |Q| and the shell is
  thin relative to that scale.

Voxels with too few clean neighbours fall back to TV inpainting (Chambolle-Pock).

## Diagnostics

The factored model assumes all rings share one T(φ). Check this after fitting:

- `rank1_variance` — fraction of amplitude-matrix variance explained by the rank-1
  (shared-texture) approximation. Values ≥ 0.90 confirm the assumption. Lower values
  indicate that higher-|Q| rings have a different azimuthal texture and may need per-ring
  T_i(φ) fits.
- `per_ring_texture_residual()` — per-ring RMS deviation from the shared T(φ); identifies
  which ring drives a rank-1 failure.

## Artifact considerations

| Artifact | Cause | Mitigation |
|----------|-------|-----------|
| Residual ring after subtraction | Gaussian width too narrow | Widen σᵢ; check detection |
| Texture mismatch | Shared T(φ) too restrictive | Inspect `rank1_variance` |
| Over-subtraction of diffuse | Ring model absorbs diffuse | Reduce detection sensitivity |
| Gibbs ringing in ΔPDF | Hard mask boundary | Sigmoid taper (`taper_width > 0`) |
| Biased fill values | Interpolation can't capture sharp diffuse | TV fallback; tune λ |

## References

- Weber & Simonov, *Z. Kristallogr.* 227, 238–247 (2012) — 3D-ΔPDF.
- Simonov, Weber & Steurer, *J. Appl. Cryst.* 47, 2011–2018 (2014) —
  3D-ΔPDF and punch-and-fill.
