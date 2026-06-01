# Powder Ring Removal

## Physical basis

Polycrystalline material in the beam path (sample environment, cryostat, sample holder,
capsule walls, etc.) produces powder rings — sharp peaks in the radial intensity profile.

A ring is **localised in |Q|** (a thin spherical shell at the powder d-spacing) but its
amplitude is **not uniform around the shell**. Detector solid-angle coverage, absorption
path length, and normalisation artefacts modulate the ring intensity with azimuthal
direction. In real data the rings are therefore *far from isotropic*.

We model the ring contribution at a voxel with |Q| and azimuthal angle φ as:

```
I_ring(Q, φ) = T(φ) × Σᵢ Aᵢ G(|Q| − qᵢ, σᵢ)
```

- **G(|Q| − qᵢ, σᵢ)**: Gaussian radial profile of ring *i* (shared across all φ).
- **Aᵢ**: per-ring amplitude (structure factor × absorption).
- **T(φ)**: one shared azimuthal texture function. All rings from the same
  polycrystalline material share the same T(φ) because they see the same detector
  geometry.

The measured signal is `I_measured(Q, φ) = I_diffuse(Q) + I_ring(Q, φ)`, where the
diffuse signal we want is direction-dependent and does **not** share the ring's radial
peak structure or azimuthal texture.

## Algorithm

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
| Per-ring texture mismatch | Shared T(φ) too restrictive | Inspect `rank1_variance`; per-ring T_i(φ) |
| Over-subtraction of diffuse | Ring model absorbs diffuse | Reduce detection sensitivity |
| Gibbs ringing in ΔPDF | Hard mask boundary | Sigmoid taper (`taper_width > 0`) |
| Biased fill values | Interpolation can't capture sharp diffuse | TV fallback; tune λ |

## References

- Weber & Simonov, *Z. Kristallogr.* 227, 238–247 (2012) — 3D-ΔPDF.
- Simonov, Weber & Steurer, *J. Appl. Cryst.* 47, 2011–2018 (2014) — 3D-ΔPDF and punch-and-fill.
