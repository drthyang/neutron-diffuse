# Inpainting Methods

## Overview

After masking Al-contaminated voxels we need to reconstruct physically reasonable
intensities. Three complementary strategies are implemented, applied in a pipeline.

---

## 1. Symmetry-based averaging (primary)

**Principle:** In a single crystal the diffuse scattering respects the Laue symmetry of
the crystal. Symmetry-related voxels at **g** and **Rg** (R ∈ Laue group) should have
equal intensity. If a masked voxel has one or more unmasked symmetry equivalents, we
fill it by weighted averaging.

**Weight:** inverse-variance weighting (w = 1/σ²), so high-count equivalents dominate.

**Strengths:** exact, no smoothing, preserves all features present in the data.

**Limitations:** fails for high-multiplicity mask positions where all equivalents are also
masked (e.g., along a symmetry axis coinciding with a powder ring).

---

## 2. Total-Variation (TV) inpainting (secondary)

**Formulation:**

```
min_{u}  (1/2) ||W(u − f)||²  +  λ ||∇u||₁
```

- **f** = observed data (zero / arbitrary in masked region)
- **W** = diagonal mask (1 for valid, 0 for masked)
- **∇u** = 3D forward finite difference gradient
- **λ** = regularisation parameter (default 0.1)

**Algorithm:** Chambolle–Pock primal-dual (τσ = 1/6, projection onto ℓ∞ ball).

**Why TV?** TV allows the reconstruction to be piecewise smooth — it preserves sharp
features (streaks, sheets of diffuse intensity) while suppressing noise. This is
physically appropriate because diffuse scattering can be anisotropic and structured.

**Parameter tuning:**
- λ ≪ 1 → closer to data fidelity (noisy interpolation)
- λ ≫ 1 → over-smoothed
- Start with λ = 0.1; increase if filled region looks speckled.

---

## 3. RBF interpolation (fallback)

**Method:** `scipy.interpolate.RBFInterpolator` with thin-plate-spline kernel and
k-nearest-neighbour support.

**Use case:** small isolated masks in low-symmetry regions where TV may be slow.

---

## 4. Biharmonic relaxation

Iterative solution of ∇⁴u = 0 inside the mask. Produces very smooth fills, appropriate
for broad, diffuse backgrounds. Slower than TV for large masks.

---

## Default pipeline: `"symmetry+tv"`

1. Symmetry equivalents fill as many masked voxels as possible.
2. Remaining unfilled voxels are passed to TV inpainting.
3. Output includes a `filled_flag` channel marking reconstructed voxels.

---

## Uncertainty propagation

| Method | σ_filled |
|--------|----------|
| Symmetry | √(1/Σwᵢ) where wᵢ = 1/σᵢ² |
| TV | Not propagated (mark as "reconstructed") |
| RBF | Not propagated (mark as "reconstructed") |

For downstream RMC refinement, reconstructed voxels should be assigned lower weight
(e.g., σ_filled = 2× the local unmasked σ).

---

## References

- Chambolle & Pock, *J. Math. Imaging Vision* 2011 — primal-dual TV algorithm
- Bertalmio et al., SIGGRAPH 2000 — PDE inpainting (origin of diffusion-based approach)
- Bertero & Boccacci, *Introduction to Inverse Problems in Imaging* (1998) — general theory
