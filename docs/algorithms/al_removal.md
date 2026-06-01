# Aluminum Background Removal

## Physical basis

Aluminum (FCC, Fm-3m, a ≈ 4.046 Å) is ubiquitous in neutron sample environments (cryostats,
pressure cells, furnaces). Al nuclei scatter neutrons coherently, producing sharp powder rings
at fixed |Q| values superimposed on the diffuse scattering signal.

## Al reflection positions

Allowed reflections (FCC selection rule: h, k, l all even or all odd):

| hkl | |Q| (Å⁻¹) | d (Å) |
|-----|-----------|-------|
| 111 | 2.687 | 2.338 |
| 200 | 3.104 | 2.024 |
| 220 | 4.389 | 1.431 |
| 311 | 5.148 | 1.221 |
| 222 | 5.374 | 1.169 |
| 400 | 6.207 | 1.012 |

Values for a = 4.046 Å at room temperature. At low temperature, a decreases slightly.

## Mask strategy

### Step 1 — enumerate Al Q-positions
Given the experimental |Q|-range, compute all Al |Q| values up to Q_max.

### Step 2 — map to HKL space
Using the sample UB matrix, compute |Q| for every voxel in the 3D HKL grid:

```
Q_cart = UB · hkl
|Q| = ||Q_cart||
```

### Step 3 — adaptive mask width
For each Al peak at Q₀:

**Fixed-width mode:** mask all voxels with `|Q - Q₀| < Δ` (user-specified Δ).

**Sigma-clipping mode (default):** compute radial intensity profile in a shell around Q₀,
fit a smooth background, identify outliers (> N·σ above background), and set Δ to encompass
all outliers. This adapts to the actual instrument resolution.

### Step 4 — soft (tapered) mask boundary
A hard step mask creates a discontinuity in the 3D Fourier domain, producing
Gibbs-like ringing in real-space correlation functions (pair distribution functions, etc.).
We apply a sigmoid taper:

```
w(dQ) = 1 / (1 + exp(-(dQ - Δ) / τ))
```

where τ controls the transition width (default 0.01 Å⁻¹). Voxels with w < 0.5 are
treated as masked; the continuous weight is also available for weighted averaging.

## Artifact considerations

| Artifact | Cause | Mitigation |
|----------|-------|-----------|
| Gibbs ringing in PDF | Sharp mask edge | Sigmoid taper (τ > 0) |
| Over-masking diffuse | Mask too wide | Sigma-clip width tuning |
| Under-masking at wings | Mask too narrow | Dilate mask by 1–2 voxels |
| Absorption-dependent residual | Sample geometry | Q-dependent scale factor |

## References
- Welberry & Butler, *Chem. Rev.* 1995 — diffuse scattering background subtraction methods
- Neef et al., *J. Appl. Cryst.* 2020 — CORELLI Al subtraction
