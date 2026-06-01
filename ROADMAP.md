# Roadmap

## Pipeline overview

The package takes over immediately after instrument data reduction (binned 3D HKL volume).
It provides two sequential stages:

```
[ Raw 3D HKL volume ]
        │
        ▼  ── DATA PROCESSING ──────────────────────────────
        │  (1) Symmetrize
        │  (2) Remove Al signals
        │  (3) Backfill Al holes
        │
        ▼  ── FURTHER ANALYSIS ──────────────────────────────
        │  (4) Remove Bragg peaks  (punch-and-fill)
        │  (5) Backfill Bragg holes
        │  (6) 3D-ΔPDF via Fourier transform
        │
        ▼
  [ 3D-ΔPDF in real space ]
```

---

## Phase 1 — Foundation (months 1–2)

**Goal:** solid I/O, data structures, and visualization.

| Task | Details |
|------|---------|
| `HKLVolume` | 3D ndarray + UB matrix + Q-extent + validity mask + σ |
| I/O — HDF5/NeXus | Read SXD, CORELLI, TOPAZ, Meerkat outputs |
| I/O — ASCII | Legacy `.hkl`, tab/space-delimited grids |
| I/O — write | Round-trip HDF5 + ASCII; unit tests |
| Slice viewer | h=const, k=const, l=const + \|Q\| shell projections |
| CI | pytest, GitHub Actions, ruff, mypy |

**Milestone:** load real dataset → display slices → save back.

---

## Phase 2 — Data Processing (months 2–5)

### 2-1. Symmetrize

Apply the crystal Laue symmetry to the raw data **before** any removal step. This:
- Averages symmetry-equivalent voxels → improves statistics
- Makes subsequent Al detection more reliable (symmetry axis peaks become obvious)
- Flags voxels with high inter-equivalent variance (possible contamination)

Implementation:
- Accept a list of (3×3) point-group rotation matrices or a Laue class string
- Weighted average (inverse-variance) of all symmetry equivalents in the grid
- Output: symmetrized volume + per-voxel symmetry-averaged σ

### 2-2. Remove Al signals

Al (FCC, Fm-3m, a ≈ 4.046 Å) produces powder rings at fixed |Q| values.

- Enumerate Al reflection |Q| positions up to Q_max
- Map to HKL space via sample UB matrix: `|Q|_voxel = ||UB · hkl||`
- **Adaptive mask width**: σ-clipping on radial intensity profile per peak
- **Soft boundary**: sigmoid taper (width τ ≈ 0.01 Å⁻¹) to avoid Gibbs ringing
- Optional: fit and subtract Al peak profile before masking → smaller mask footprint

### 2-3. Backfill Al holes

Restore physically meaningful values in masked Al regions:

1. **Symmetry fill** (primary): use Laue equivalents from the symmetrized data
2. **TV inpainting** (secondary): Chambolle-Pock total-variation on remaining gaps
3. **RBF / biharmonic** (fallback): smooth interpolation for isolated residual gaps

Output: cleaned volume with filled-voxel flags and inflated σ.

**Phase 2 milestone:** on benchmark data, < 2 % diffuse signal lost; residual Al < 1 %.

---

## Phase 3 — Further Analysis (months 5–9)

### 3-1. Remove Bragg peaks (punch)

Bragg peaks at integer (h,k,l) are orders of magnitude stronger than diffuse signal
and must be excised before computing the 3D-ΔPDF.

- **Detection**: enumerate integer reflections within the HKL grid; optionally read a
  peak list from a `.peaks` / `.integrate` file (Mantid format)
- **Adaptive punch radius**: scale with |F|² or with observed peak intensity to avoid
  over- or under-punching
- **Anisotropic punch**: ellipsoidal mask aligned with instrumental resolution function
  (δh, δk, δl may differ)
- **Profile subtraction** (optional): fit 3D Gaussian + background, subtract fitted
  Bragg contribution → reduces punch size, preserves diffuse signal near Bragg peaks
- Output: volume with Bragg holes masked

### 3-2. Backfill Bragg holes

Bragg holes are typically larger and more numerous than Al holes.

- **Symmetry fill**: strongest tool when Laue multiplicity is high
- **TV inpainting**: preserves diffuse streaks that run through Bragg positions
- **Local polynomial fit**: fit smooth background in a shell around each Bragg position
  using only unmasked voxels, then evaluate at hole voxels
- **Uncertainty**: filled Bragg voxels carry larger σ; downstream ΔPDF analysis should
  weight accordingly

### 3-3. 3D-ΔPDF

The 3D difference pair distribution function is the real-space Fourier transform of the
diffuse scattering:

```
Δρ(r) = FT[ I_diffuse(Q) ]  =  FT[ I_total(Q) - I_Bragg(Q) ]
```

Implementation:
- **Normalization**: convert measured counts to absolute units or normalise by monitor
- **Apodization**: apply a 3D window function (Hann, Gaussian, or "punch-and-fill"
  specific weight) to suppress termination ripples
- **Zero-padding**: pad to next power-of-2 grid for efficient FFT
- **3D FFT** (`numpy.fft.fftn` / `scipy.fft.fftn`): output is Δρ(r) on a real-space grid
- **Symmetrisation**: apply point-group to Δρ(r) to improve signal-to-noise
- **Origin removal**: set r=0 peak to zero or model its contribution
- **Output**: real-space `HKLVolume`-like object with (Δx, Δy, Δz) axes in Å

Visualisation:
- 2D slices of Δρ(Δx, Δy, 0) etc.
- 1D projected pair distances
- Comparison with model ΔPDF from RMC / DFT structures

**Phase 3 milestone:** reproduce published 3D-ΔPDF from a reference dataset.

---

## Phase 4 — Validation, Integration & Release (months 9–12)

| Task | Details |
|------|---------|
| Synthetic tests | Simulate diffuse + Al rings + Bragg peaks; compare pipeline output to ground truth |
| Real benchmarks | ≥ 2 datasets (CORELLI + SXD) with manual expert reference |
| Mantid plugin | Workflow algorithm callable from Mantid scripts |
| DISCUS / mcs3d export | Compatible format for RMC refinement |
| Jupyter tutorials | Step-by-step notebooks for each pipeline stage |
| PyPI v1.0 | Changelog, DOI via Zenodo, semantic versioning |

---

## Module layout

```
src/ndiff/
├── core.py                        # HKLVolume data structure
├── io/                            # File I/O (HDF5, ASCII)
├── preprocessing/
│   ├── symmetrize.py              # (1) Symmetrize
│   ├── aluminum.py                # (2) Al removal
│   └── backfill.py                # (3) Al hole filling (calls inpainting/)
├── analysis/
│   ├── bragg.py                   # (4) Bragg removal (punch)
│   ├── bragg_fill.py              # (5) Bragg hole filling
│   └── delta_pdf.py               # (6) 3D-ΔPDF via FFT
├── inpainting/                    # Filling algorithms (shared)
│   ├── symmetry.py
│   ├── tv_inpainting.py
│   ├── interpolation.py
│   └── pipeline.py
└── utils/
    └── reciprocal_space.py
```

---

## Out of scope (v1)

- Incoherent background (treated as flat; trivial to subtract before this pipeline)
- Multi-crystal twins
- GPU acceleration (NumPy/SciPy first)
- Phonon / magnetic diffuse separation (future v2 feature)
