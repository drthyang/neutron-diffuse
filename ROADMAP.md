# Roadmap

## Pipeline overview

Input: symmetrised 3D HKL volume from Mantid (or equivalent).

```
[ Symmetrised 3D HKL volume ]
        │
        ▼  DATA PROCESSING
        │
        │  (1) Empty-scan subtraction
        │        Removes environment ring (cryostat, furnace, etc.).
        │        Residual: sample-holder Al ring remains.
        │
        │  (2) Factored ring model — fit and subtract
        │        I_ring(Q, φ) = T(φ) × Σᵢ Aᵢ G(|Q|−qᵢ, σᵢ)
        │        See "Residual ring removal" section below.
        │
        │  (3) Backfill subtracted region
        │        Radial interpolation from nearest uncontaminated neighbours.
        │
        ▼  FURTHER ANALYSIS
        │
        │  (4) Bragg peak removal  (punch-and-fill)
        │  (5) 3D-ΔPDF via Fourier transform
        │
        ▼
  [ 3D-ΔPDF in real space ]
```

---

## Phase 1 — Foundation (months 1–2)

| Task | Details |
|------|---------|
| `HKLVolume` | 3D ndarray + UB matrix + validity mask + σ array |
| I/O — HDF5/NeXus | Read Mantid-output SXD, CORELLI, TOPAZ formats |
| I/O — ASCII | Legacy `.hkl`, tab-delimited grids |
| Slice viewer | h/k/l = const slices + \|Q\| shell projections |
| CI | pytest, GitHub Actions, ruff, mypy |

**Milestone:** load real Mantid output → display slices → save back.

---

## Phase 2 — Powder Ring Removal (months 2–6)

### 2-1. Empty-scan subtraction

```
I_residual(Q) = I_sample(Q) − s × I_empty(Q)
```

Scale factor `s` estimated analytically from ring-dominated |Q| shells.
Removes environment ring; sample-holder Al ring remains.

### 2-2. Residual ring removal: factored patch model

**Physical model:**

All rings from the same polycrystalline material share the same azimuthal
texture T(φ), because they all experience the same detector geometry.
Per-ring amplitudes Aᵢ scale with each ring's structure factor and absorption.

```
I_ring(Q, φ) = T(φ) × Σᵢ Aᵢ × G(|Q| − qᵢ, σᵢ)
```

**Coordinate system:**

φ = azimuthal angle in the reference plane (default: hk0).
  `φ = atan2(k_Q, h_Q)` where (h_Q, k_Q) are Cartesian Q-space coordinates.
Gaussians G are evaluated at the full 3D |Q| of each voxel.
This extends the 2D fit naturally to the full 3D dataset.

**Fitting procedure:**

```
(a) Divide φ ∈ [0, 2π) into N overlapping patches
    (Hann weighting within overlap; typical N = 24–72, overlap 30 %)

(b) Per patch: fit Gaussians to weighted 1D profile I(|Q|)
    Ring positions qᵢ and widths σᵢ shared across patches (global fit).
    Non-negative least-squares for per-patch amplitudes Aᵢ(P).
    → Amplitude matrix A[n_rings × n_patches]

(c) Rank-1 SVD factorisation of A:
    A[i, P] ≈ Aᵢ × T[P]
    → first singular vectors give per-ring amplitudes Aᵢ
      and per-patch texture T[P]

(d) Fit Fourier series to {φ_P, T[P]}:
    T(φ) = c₀ + Σₖ (aₖ cos kφ + bₖ sin kφ)
    → smooth, periodic, C∞ → C¹ continuity automatic
    (typical n_fourier = 4–8)

(e) Subtract full model from data:
    I_est(voxel) = I_measured(voxel) − T(φ_voxel) × Σᵢ Aᵢ G(|Q_voxel|−qᵢ,σᵢ)
    Mask voxels where I_ring / σ_data > threshold.
```

**Why no assumption on diffuse signal:**
- Gaussians in |Q| are fitted to the local radial profile within each patch.
  The diffuse signal averages differently from the ring (it is not Gaussian in |Q|
  and varies anisotropically), so the Gaussian captures mainly the ring.
- The shared texture T(φ) further constrains the fit: the diffuse signal
  does not share the same azimuthal pattern across all rings.

**Open design questions:**
- How many patches and what overlap for a typical dataset?
- What to do when a patch has very few voxels (detector gap)?
- How to handle overlapping Gaussian peaks (closely spaced rings)?
- Validation: what residual ring level is acceptable for 3D-ΔPDF quality?

### 2-3. Backfill

Masked voxels (where ring dominated) are filled by:
1. Finding nearest valid 3D-HKL neighbours outside the ring |Q| range.
2. Weighted interpolation in |Q| → C¹ at shell boundary by construction.
3. TV inpainting fallback for voxels with too few clean neighbours.

No assumption is imposed on the diffuse signal.

---

## Phase 3 — Bragg Peak Removal (months 6–8)

- Ellipsoidal punch at each integer (h,k,l).
- Adaptive punch radius (scale with intensity).
- Sigmoid-tapered boundary.
- TV inpainting backfill.

---

## Phase 4 — 3D-ΔPDF (months 8–10)

```
Δρ(r) = FT[ I_diffuse(Q) ]
```

1. Hann apodization in 3D.
2. Zero-pad to next power-of-2.
3. `scipy.fft.fftn` → fftshift → real part.
4. Real-space axes in Å via UB matrix.

---

## Phase 5 — Validation & Release (months 10–12)

| Task | Details |
|------|---------|
| Synthetic tests | Inject anisotropic ring + diffuse; evaluate ΔPDF quality |
| Real benchmarks | ≥ 2 datasets with Mantid-symmetrised input |
| Mantid integration | Export-compatible format |
| PyPI v1.0 | Changelog, DOI via Zenodo |

---

## Module layout

```
src/ndiff/
├── core.py
├── io/
├── preprocessing/
│   ├── empty_subtraction.py    # (1) subtract empty-environment scan
│   ├── ring_model.py           # (2) PatchedRingModel: fit + subtract
│   ├── powder_rings.py         # utilities: detect_ring_shells, radial_profile
│   └── backfill.py             # (3) radial interpolation backfill
├── analysis/
│   ├── bragg.py                # (4) Bragg punch
│   ├── bragg_fill.py           # (4b) Bragg backfill
│   └── delta_pdf.py            # (5) 3D-ΔPDF
├── inpainting/
│   ├── tv_inpainting.py
│   ├── interpolation.py
│   └── pipeline.py
└── utils/
    └── reciprocal_space.py
```
