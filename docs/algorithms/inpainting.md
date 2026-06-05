# Inpainting Methods

## Overview

After masking contaminated voxels (Bragg punches, or ring shells when radial
interpolation cannot fill them) we need to reconstruct physically reasonable
intensities. Several complementary strategies are implemented, combined in a pipeline.

> **Scope.** This pipeline is the **general-purpose** inpainter. The current
> real-data Bragg workflow usually uses `backfill_bragg(method="q_shell")` or
> `method="local"` before falling back to TV/symmetry methods. Powder-ring shells
> are filled first by `backfill_ring_shells`, which interpolates radially across
> the thin shell from uncontaminated neighbours (see
> [powder_rings.md](powder_rings.md)). Symmetry averaging is **not** used for ring
> removal, because the Laue equivalents of a ring voxel lie on the same ring and
> are equally contaminated.

---

## 1. Symmetry-based averaging

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

## General fallback pipeline: `"symmetry+tv"`

1. Symmetry equivalents fill as many masked voxels as possible.
2. Remaining unfilled voxels are passed to TV inpainting.
3. Output includes a `filled_flag` channel marking reconstructed voxels.

## Real-data Bragg backfill

For Bragg-punched volumes, prefer the dedicated wrapper:

```python
from ndiff.analysis import backfill_bragg

filled = backfill_bragg(punched, method="q_shell")
```

`method="q_shell"` fills ordinary Bragg components from the robust radial
background level at the same `|Q|`, while the direct beam keeps its special
just-outside-`|Q|` fill. `method="local"` remains useful for fast visual checks
or sparse synthetic volumes.

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
