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

## Selectable Models: Patched vs Parametric

Two interchangeable removers expose the same `fit` / `subtract` interface and the
same cross-stack confirmed-shell guards; select with `RingParams.ring_model`
(`"patched"` — default | `"parametric"`).

- **`PatchedRadialRingModel`** (`"patched"`, default) — the non-parametric
  per-(azimuthal-patch × |Q|-bin) estimator described above.
- **`ParametricRingModel`** (`"parametric"`) — separable and binning-free:
  `I_ring(|Q|,φ) = Σᵢ Tᵢ(φ)·PVᵢ(|Q|)`, a unit-peak pseudo-Voigt radial line shape
  per ring × that ring's own non-negative Fourier azimuthal texture
  `Tᵢ(φ) = c₀ + Σₖ (cₖ cos kφ + sₖ sin kφ)`. Two radial modes (`ring_radial_mode`):
  **rolling** (default — a continuous `Ring(|Q|)·T(|Q|,φ)` swept over thick
  overlapping |Q| windows, no peak detection) and **peaks** (discrete
  pseudo-Voigt rings). Motivation: the patched grid's per-cell voxel count scales
  with arc length ∝ |Q|, starving the low-|Q| patches; the parametric fit pools
  all azimuths per radial shell for uniform statistics.

### A/B status (2026-06-16)

Compared with `examples/compare_ring_models.py` (representative H planes, same
confirmed shells). The two are **close** but fail in *opposite* directions:
**patched over-subtracts** (digs shallow negative troughs at the ring centres,
worst at the first ring ≈1.93 Å⁻¹) while **parametric rolling under-subtracts**
(leaves ring behind, most on the magnetic H=1/3 plane). Judged on the slice
figures below, **patched hugs the diffuse baseline better overall and is kept as
the default**; parametric rolling is a validated, selectable alternative.

### The dominant residual error is texture-contrast compression

The main arc-by-arc error in **both** models is that the fitted azimuthal texture
`T(φ)` is **flattened toward its φ-mean**. At |Q|≈2.69 Å⁻¹ (H=0) the data-truth
ring excess swings ≈0.04→0.16, but every model reaches only ≈0.078→0.135 —
roughly half the contrast. So `T(φ)` sits *below* truth at the bright arcs
(→ under-subtraction / leftover) and *above* it at the dim arcs
(→ over-subtraction / digs a hole). Cause: the harmonic ridge (`texture_ridge`,
penalty ∝ order², with the mean `c₀` left free) + Fourier truncation
(`n_fourier`) + the amplitude ceiling. A constant/background offset **cannot** fix
this — it shifts every azimuth equally, whereas the error is *differential*; the
lever is texture **contrast** (lower `texture_ridge`, higher `n_fourier`).

> **Metric caveat.** The mean per-shell "ring removed %" is *blind* to this,
> because the bright-under and dim-over errors cancel in the azimuthal average
> (parametric scores ≈98% at H=0 with a visibly wrong texture). Judge ring
> quality on the azimuthal **texture overlay** and the **per-φ / diverging
> residual** figures, not on the mean %.

### A/B tooling

- `examples/compare_ring_models.py` — per-plane metrics + three figures: (a) the
  magma `data | patched | parametric` residuals; (b) a **diverging
  deviation-from-baseline** map (red = ring leftover, blue = over-subtraction)
  that makes over-subtraction visible — the magma view hides it; (c) a 1-D
  azimuthally-averaged **ring-residual profile** vs |Q| measured against one
  common diffuse baseline.
- `examples/tune_parametric_ring.py` — the azimuthal **texture overlay**
  (data-truth `median_on(φ) − median_off(φ)` vs each model's `T(φ)` at a shell);
  the diagnostic that exposes the contrast compression.

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
