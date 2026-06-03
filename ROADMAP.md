# Roadmap

## Pipeline overview

Input: symmetrised 3D HKL volume from Mantid (or equivalent).

```
[ Symmetrised 3D HKL volume ]
        │
        ▼  DATA PROCESSING
        │
        │  (1) Empty-scan subtraction                       [implemented]
        │        Removes environment ring (cryostat, furnace, etc.).
        │        Residual: sample-holder ring remains.
        │
        │  (2) Factored ring model — fit and subtract       [implemented]
        │        I_ring(Q, φ) = T(φ) × Σᵢ Aᵢ G(|Q|−qᵢ, σᵢ)
        │        Removes residual sample-holder ring.
        │
        │  (3) Backfill subtracted region                   [implemented]
        │        Radial interpolation from nearest uncontaminated neighbours.
        │
        ▼  FURTHER ANALYSIS
        │
        │  (4) Bragg peak removal  (punch-and-fill)         [implemented]
        │  (5) 3D-ΔPDF via Fourier transform                [implemented]
        │
        ▼
  [ 3D-ΔPDF in real space ]
```

---

## Phase 1 — Foundation  ✓ complete

| Task | Status |
|------|--------|
| `HKLVolume` — 3D ndarray + UB matrix + mask + σ | done |
| I/O — HDF5/NeXus read/write | done |
| I/O — ASCII legacy `.hkl`, tab-delimited | done |
| CI — pytest, ruff, mypy | skeleton done |

**Milestone:** load real Mantid output → display slices → save back.

---

## Phase 2 — Powder Ring Removal  ✓ implemented (slice-validated; full 3D pending)

### 2-1. Empty-scan subtraction  ✓

```
I_residual(Q) = I_sample(Q) − s × I_empty(Q)
```

Scale factor `s` estimated analytically from ring-dominated |Q| shells.
Removes environment ring; sample-holder ring remains.

### 2-2. Ring model  ✓

**Physical model:**

All rings from the same polycrystalline material share the same azimuthal
texture T(φ), because they all experience the same detector geometry.
Per-ring amplitudes Aᵢ scale with each ring's structure factor and absorption.

```
I_ring(Q, φ) = T(φ) × Σᵢ Aᵢ × G(|Q| − qᵢ, σᵢ)
```

**Coordinate system:**

φ = azimuthal angle in the hk0 plane.  `φ = atan2(k_Q, h_Q)` where
(h_Q, k_Q) are Cartesian Q-space coordinates.  Gaussians are evaluated
at the full 3D |Q| of each voxel — extends the 2D fit to the full 3D dataset.

**Fitting procedure:**

```
(a) Divide φ ∈ [0, 2π) into N overlapping patches
    (Hann weighting within overlap; typical N = 24–72, overlap 30 %)

(b) Per patch: NNLS fit of Gaussian amplitudes to weighted 1D profile I(|Q|)
    Ring positions qᵢ and widths σᵢ shared across patches (global fit).
    → Amplitude matrix A[n_rings × n_patches]

(c) Rank-1 SVD factorisation of A:
    A[i, P] ≈ Aᵢ × T[P]
    First singular vectors give per-ring amplitudes Aᵢ and per-patch texture T[P].

(d) Fourier series fit to {φ_P, T[P]}:
    T(φ) = c₀ + Σₖ (aₖ cos kφ + bₖ sin kφ)
    Smooth, periodic, C∞  → C¹ continuity automatic (typical n_fourier = 4–8)

(e) Subtract full model:
    I_residual(voxel) = I(voxel) − T(φ_voxel) × Σᵢ Aᵢ G(|Q_voxel|−qᵢ, σᵢ)
    Mask voxels where I_ring / σ_data > threshold.
```

**Diagnostics:**

- `rank1_variance` — fraction of amplitude-matrix variance explained by rank-1.
  Should be ≥ 0.90.  Lower values indicate per-ring texture differs between rings.
- `per_ring_texture_residual()` — per-ring RMS deviation from the shared T(φ).

**Open design questions (to resolve with real data):**

- How many patches and what overlap for a typical dataset?
- Is n_fourier = 6 sufficient, or do high-|Q| rings require more harmonics?
- What SNR threshold for the ring mask gives the best ΔPDF quality?
- How to handle overlapping Gaussian peaks (closely spaced rings)?

**2026-06-03 update:** the original Gaussian/SVD model remains available, but
real-data shakedown moved the preferred remover to the non-parametric
`PatchedRadialRingModel`: trimmed radial profiles, morphological-opening
baseline, low-order azimuthal Fourier texture, and sparse-azimuth masking.
The fast 0kl-slice harness now suppresses the main Al rings by roughly 85-90%
while preserving off-ring diffuse signal. The next risk is scaling this
slice-validated path to the full 3D volume efficiently.

### 2-3. Backfill  ✓

Masked voxels (ring-dominated) filled by:
1. KDTree on valid voxel indices; filter to uncontaminated (outside all ring |Q|).
2. Weighted interpolation across the thin gap → C¹ by construction.
3. TV inpainting (Chambolle-Pock) fallback for voxels with too few clean neighbours.

---

## Phase 3 — Bragg Peak Removal  ✓ implemented

- Ellipsoidal punch at each integer (h,k,l).
- Adaptive punch radius (scale with intensity).
- Sigmoid-tapered boundary.
- Backfill via `backfill_bragg` (default `symmetry+tv`).

**Pending refinements:**
- Profile subtraction before punch (reduces punch radius needed).
- Absolute monitor normalisation.
- Symmetry fill helps little for Bragg-adjacent diffuse: the Laue
  equivalents of a near-Bragg voxel sit next to other (also-punched)
  Bragg peaks, so TV inpainting does most of the work.  Consider making
  `method="tv"` the Bragg-fill default.

---

## Phase 4 — 3D-ΔPDF  ✓ implemented

```
Δρ(r) = FT[ I_diffuse(Q) ]
```

1. Hann / Gaussian / no apodization (selectable).
2. Zero-pad to next power-of-2.
3. `scipy.fft.fftn` → fftshift → real part.
4. Real-space axes in Å via UB matrix.

**Pending:** absolute units / monitor normalisation.

---

## Phase 5 — Validation & Release  (in progress)

| Task | Status |
|------|--------|
| Real-data trial: load and inspect Mantid NeXus volumes | done |
| Real-data trial: 0kl slice ring-removal validation | done |
| Real-data trial: full 3D ring remove, Bragg punch, ΔPDF | **next step** |
| Repository health check: syntax/imports/lightweight tests | done 2026-06-03; official pytest/ruff/mypy blocked by missing dev deps |
| Synthetic benchmark: injected ring + diffuse, evaluate ΔPDF quality | pending |
| Per-ring T_i(φ) if rank1_variance < 0.90 | pending |
| Mantid integration — export-compatible format | pending |
| PyPI v1.0 — changelog, DOI via Zenodo | pending |

---

## Module layout

```
src/ndiff/
├── core.py
├── io/
│   └── hkl_reader.py
├── preprocessing/
│   ├── empty_subtraction.py    (1) subtract empty-environment scan      ✓
│   ├── ring_model.py           (2) PatchedRingModel: fit + subtract     ✓
│   ├── powder_rings.py         utilities: detect_ring_shells, mask      ✓
│   ├── backfill.py             (3) radial interpolation backfill        ✓
│   └── residual_rings.py       experimental symmetry-sector alternative ⚠ not wired in
├── analysis/
│   ├── bragg.py                (4) Bragg punch                          ✓
│   ├── bragg_fill.py           (4b) Bragg backfill                      ✓
│   └── delta_pdf.py            (5) 3D-ΔPDF                             ✓
├── inpainting/
│   ├── tv_inpainting.py        Chambolle-Pock TV inpainting             ✓
│   ├── interpolation.py        RBF / biharmonic fill                    ✓
│   └── pipeline.py             orchestration: symmetry → TV → RBF      ✓
└── utils/
    └── reciprocal_space.py     ub_from_lattice, d_spacing, q_to_hkl    ✓
```
