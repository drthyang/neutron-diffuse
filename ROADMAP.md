# Roadmap

## Pipeline overview

Input: symmetrized 3D HKL volume from Mantid (or equivalent data reduction).
Symmetrization can be done in Mantid before handing off to this package.

```
[ Symmetrized 3D HKL volume ]
        │
        ▼  DATA PROCESSING
        │  (1) Detect powder rings (isotropic |Q| peaks in radial profile)
        │      Fit Gaussian profiles per ring  →  I_ring(|Q|)
        │      Subtract I_ring from entire volume
        │      Mask voxels where SNR post-subtraction is poor
        │  (2) Backfill ring holes
        │      Interpolate I_diffuse from surrounding unmasked voxels
        │      (TV inpainting preserves anisotropic diffuse features)
        │
        ▼  FURTHER ANALYSIS
        │  (3) Punch Bragg peaks  (ellipsoidal mask at integer hkl)
        │  (4) Backfill Bragg holes  (same inpainting pipeline)
        │  (5) 3D-ΔPDF via Fourier transform
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
| I/O — write | Round-trip fidelity tests |
| Slice viewer | h/k/l=const slices + \|Q\| shell projections |
| CI | pytest, GitHub Actions, ruff, mypy |

**Milestone:** load real Mantid output → display slices → save back.

---

## Phase 2 — Powder Ring Removal (months 2–5)

### 2-1. Detection

The algorithm is **material-agnostic**: it detects any powder rings in the data
without needing to know the source material. The approach:

1. Compute radial intensity profile (mean per |Q| shell, sigma-clipped).
2. Fit a smooth spline as the diffuse background baseline.
3. Detect peaks in residuals using `scipy.signal.find_peaks` (threshold: 5σ_rms).
4. Fit a Gaussian per peak → (q₀, σ_ring, amplitude).

> **Aluminum note**: Al (FCC, a ≈ 4.046 Å) is the most common source. Its
> known peak positions (from crystallography) can optionally be used to seed or
> validate the detection, but the algorithm does not require this prior.

Robustness features:
- Works on any polycrystalline contaminant (Al, V, Cu, steel, MgO, …)
- User can also provide ring positions directly (e.g. from known material)
- Multiple overlapping rings are handled simultaneously

### 2-2. Profile subtraction

For each detected ring, fit a Gaussian I_ring(|Q|) and subtract from the
entire volume:

```
I_diffuse_est(Q) = I_measured(Q) − I_ring(|Q_voxel|)
```

This step recovers diffuse signal at ring positions (not just masking it away).
Uncertainty is propagated: σ_post = sqrt(σ_data² + σ_model²).

Mask is then applied only where the subtraction quality is poor (I_ring/σ_data > threshold).

### 2-3. Backfill ring holes

Powder ring holes are **thin spherical shells** in HKL space. Every point on a
shell is at the same |Q|, so:
- Symmetry equivalents of a masked voxel are ALSO on the same shell → cannot use symmetry to fill.
- Radial interpolation has no source data at the same |Q|.

What works: **smooth 3D interpolation from slightly-different-|Q| neighbours**.
The diffuse signal is smooth in HKL space; the shell is thin. TV inpainting
is the default (λ ≈ 0.08) — it preserves anisotropic streaks and diffuse features.

The filled values represent the estimated I_diffuse underneath the ring, NOT
a reconstruction of the ring itself.

**Phase 2 milestone:** on benchmark data, post-fill residual ring contribution < 1 % of diffuse signal; no visible TV artifact in ΔPDF.

---

## Phase 3 — Further Analysis (months 5–9)

### 3-1. Bragg peak removal (punch)

Bragg peaks at integer (h,k,l) are orders of magnitude stronger than diffuse signal.

- Enumerate integer reflections within the HKL grid.
- Ellipsoidal punch mask: radii (δh, δk, δl) aligned with instrument resolution.
- Adaptive radius option: scale with peak intensity.
- Soft (sigmoid-tapered) boundary to avoid truncation effects.

### 3-2. Backfill Bragg holes

Bragg holes are larger and more numerous than ring holes.

- TV inpainting (λ ≈ 0.2, higher for larger holes).
- Local polynomial fit in shell around each Bragg position as fallback.
- Inflated σ for filled voxels.

### 3-3. 3D-ΔPDF

```
Δρ(r) = FT[ I_diffuse(Q) ]
```

Steps:
1. Normalise to absolute units (or monitor-normalise).
2. Apodize: Hann or Gaussian window in 3D.
3. Zero-pad to next power-of-2 grid.
4. `scipy.fft.fftn` → shift origin → take real part.
5. Output real-space axes in Å via UB matrix.

**Phase 3 milestone:** reproduce published 3D-ΔPDF from a reference dataset.

---

## Phase 4 — Validation & Release (months 9–12)

| Task | Details |
|------|---------|
| Synthetic tests | Inject simulated rings + Bragg peaks into known diffuse signal; compare recovered ΔPDF |
| Real benchmarks | ≥ 2 datasets (CORELLI + SXD) |
| Mantid integration | Export-compatible format; optional Mantid workflow script |
| Jupyter tutorials | One notebook per pipeline stage |
| PyPI v1.0 | Changelog, DOI via Zenodo |

---

## Module layout

```
src/ndiff/
├── core.py                        # HKLVolume
├── io/                            # HDF5 + ASCII I/O
├── preprocessing/
│   ├── powder_rings.py            # (1) detect, fit, subtract powder rings
│   └── backfill.py                # (2) fill ring holes (TV / RBF / biharmonic)
├── analysis/
│   ├── bragg.py                   # (3) Bragg punch
│   ├── bragg_fill.py              # (4) Bragg backfill
│   └── delta_pdf.py               # (5) 3D-ΔPDF
├── inpainting/                    # Shared filling algorithms
│   ├── tv_inpainting.py
│   ├── interpolation.py
│   └── pipeline.py
└── utils/
    └── reciprocal_space.py
```
