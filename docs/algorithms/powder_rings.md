# Powder Ring Removal

## Physical basis

Polycrystalline material in the beam path (sample environment, cryostat, capsule walls, etc.)
produces powder rings — sharp peaks in the radial intensity profile. Their intensity is
**isotropic in |Q|**: it depends only on the magnitude of the scattering vector, not its direction.

The measured signal is therefore:

```
I_measured(Q) = I_diffuse(Q) + I_ring(|Q|)
```

- **I_diffuse(Q)**: direction-dependent (anisotropic) — the signal we want.
- **I_ring(|Q|)**: spherically symmetric — depends only on |Q|.

This separation is the foundation of the removal algorithm.

## Algorithm

### Step 1 — Detect ring positions

Compute the radial intensity profile by binning all valid voxels by |Q|:

1. Sigma-clipped mean per shell → mean profile.
2. Fit a smooth spline to the mean profile as the "diffuse background" (slowly varying).
3. Subtract spline → residuals highlight powder ring peaks.
4. `scipy.signal.find_peaks` on residuals with height threshold (default: 5σ_rms).
5. Fit a Gaussian to each detected peak → (q₀, σ_ring, amplitude).

**This step is material-agnostic.** It detects any polycrystalline contaminant, not just Al.

> **Note on aluminum**: Al (FCC, Fm-3m, a ≈ 4.046 Å) is the most common source.
> Its peak positions can be pre-computed with `al_ring_q_positions()` and used to
> cross-check or seed the detection. But the algorithm works without this prior.

### Step 2 — Fit ring profiles

For each detected ring, refine the Gaussian profile by fitting to the data in a narrow
|Q| window around the peak. This accounts for instrument-resolution effects.

```
I_ring(|Q|) = Σ_rings  A_i · exp(−(|Q| − q₀ᵢ)² / 2σᵢ²)
```

### Step 3 — Subtract

Subtract the modelled ring contribution from **every voxel**:

```
I_diffuse_est(Q) = I_measured(Q) − I_ring(|Q|_voxel)
```

The subtracted signal is valid everywhere. Near the ring peak centre, however, the
residual noise is large (ring dominated → subtraction error ∼ I_ring). These voxels
are masked using a sigmoid-tapered threshold on I_ring / σ_data.

### Step 4 — Backfill the masked shell

The masked region forms a **thin spherical shell** in HKL space. The sources for filling are
the clean I_diffuse values at nearby (slightly different |Q|) unmasked voxels.

Key observations:
- The diffuse signal is smooth in 3D HKL space.
- The shell is thin (typically < 0.1 Å⁻¹ wide) relative to the scale of diffuse features.
- Interpolation across the shell is therefore physically well-posed.

Default method: **TV inpainting** (Chambolle-Pock) with λ = 0.08.
- Preserves anisotropic diffuse streaks and sharp features.
- Handles the thin-shell topology naturally.
- λ = 0.05–0.15 is appropriate for most diffuse scattering cases.

## Artifact considerations

| Artifact | Cause | Mitigation |
|----------|-------|-----------|
| Residual ring after subtraction | Profile model too narrow | Increase mask_halfwidth or σ_ring |
| Over-subtraction of diffuse | Profile too broad | Reduce detection sensitivity or tighten window |
| Gibbs ringing in ΔPDF | Hard mask boundary | Sigmoid taper (taper_width > 0) |
| Biased fill values | Interpolation can't capture sharp diffuse | Increase TV iterations; reduce λ |

## References

- Simonov, Weber & Steurer, J. Appl. Cryst. 47, 2011–2018 (2014) — 3D-ΔPDF and punch-fill
- Baity-Jesi et al. (2022) — background subtraction strategies for diffuse scattering
