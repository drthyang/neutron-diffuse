# Roadmap

## Vision

A reproducible, scientifically rigorous pipeline for cleaning 3D diffuse neutron scattering data:  
detect → mask → fill, with minimal artifacts and full uncertainty propagation.

---

## Phase 1 — Foundation (months 1–2)

**Goal:** working I/O, data structures, and visualization.

| Task | Details |
|------|---------|
| Data structures | `HKLVolume`: 3D ndarray + UB matrix + Q-extent + Ewald mask |
| I/O — NeXus/HDF5 | Read SXD, CORELLI, TOPAZ, BL-12 processed HDF5 outputs |
| I/O — ASCII | Read legacy `.hkl`, Meerkat `nxs`, and DIFFUSE-formatted text grids |
| I/O — write | Export HDF5 + ASCII; round-trip fidelity tests |
| Visualization | Slice viewer (h=const, k=const, l=const), |Q|-shell projections |
| CI / packaging | pytest, GitHub Actions, pyproject.toml, pre-commit (ruff + mypy) |

**Milestone:** load a real dataset, display three orthogonal slices, save back.

---

## Phase 2 — Aluminum Peak Detection (months 2–4)

**Goal:** reliably identify which voxels are Al-contaminated.

### 2a. Crystallographic model
- Al FCC structure (Fm-3m, a = 4.046 Å); enumerate all allowed hkl up to user-set Q_max
- Compute |Q|_Al for each allowed reflection
- Given sample UB matrix, express Al powder ring shells in fractional HKL coordinates

### 2b. Adaptive masking
- Per-shell mask radius: σ-clipping on radial intensity profile to tune width automatically
- Anisotropic option: elliptical masks when instrumental resolution varies with direction
- Boundary softening: sigmoid-tapered mask weights → avoids Gibbs-like ringing

### 2c. Validation
- Metrics: fraction of sample signal lost, mask volume, boundary sharpness
- Visual overlay: plot predicted Al rings on data slices

**Milestone:** on a benchmark dataset, mask > 98 % of Al signal while losing < 2 % of diffuse signal.

---

## Phase 3 — Robust Removal (months 4–6)

**Goal:** subtract residual Al background below the mask threshold.

### 3a. Profile-based subtraction
- Fit radial profiles of Al peaks (Voigt / pseudo-Voigt in |Q|)
- Subtract fitted profile from the raw data before masking (reduces mask width needed)

### 3b. Multi-material support
- Generalize to vanadium, copper, steel, MgO containers
- User-supplied CIF or lattice parameters

### 3c. Artifact reduction
- Soft-mask convolution to avoid step-function artifacts at mask boundary
- Q-dependent scale factor to handle absorption-varying background

### 3d. Error propagation
- Propagate raw counting statistics through subtraction step
- Output σ(I) volume alongside cleaned I volume

**Milestone:** residual at masked positions < 1 % of peak Al signal after subtraction.

---

## Phase 4 — Advanced Inpainting / Filling (months 6–9)

**Goal:** reconstruct physically plausible values in masked regions.

### 4a. Symmetry-based averaging (highest priority)
- Use sample crystal point group to map each masked voxel to symmetry-equivalent positions
- Average over unmasked equivalents; propagate statistical weights
- Works best when masks are small and symmetry multiplicity is high

### 4b. Interpolation methods
- **RBF / kriging**: smooth radial-basis-function interpolation in 3D from surrounding unmasked voxels
- **3D spline**: tensor-product B-spline for anisotropic but smooth regions
- **Nearest-neighbor baseline**: fast fallback, useful for diagnostics

### 4c. Variational inpainting
- **Total-variation (TV) minimization**: Chambolle–Pock primal-dual algorithm on 3D masked volume; preserves sharp diffuse features while being stable
- **Biharmonic / diffusion**: solve ∇⁴u = 0 inside mask (smoother, appropriate for broad diffuse features)

### 4d. Hybrid pipeline
- Default: symmetry → TV, falling back to RBF where symmetry coverage is insufficient
- Configurable via method string, e.g. `"symmetry+tv"`, `"rbf"`, `"diffusion"`

### 4e. Uncertainty maps
- Assign σ_fill based on inter-equivalent variance (symmetry method) or kriging variance
- Flag filled voxels in output mask channel

**Milestone:** on synthetic test data with known ground truth, relative RMS error < 5 % in filled region.

---

## Phase 5 — Validation, Integration & Release (months 9–12)

### 5a. Synthetic test suite
- Simulate diffuse scattering from known disorder models (e.g. correlated displacements)
- Inject synthetic Al rings; run pipeline; compare to ground truth
- Benchmark all filling methods against each other

### 5b. Real-data benchmarks
- At least two experimental datasets from different instruments (CORELLI, SXD)
- Manual expert annotation for comparison

### 5c. Software integration
- **Mantid**: plugin or workflow algorithm callable from Mantid scripts
- **mcs3d / DISCUS**: export in compatible format for RMC refinement
- **VESTA**: export as VESTA-readable reciprocal-space map

### 5d. Documentation & tutorials
- Algorithm reference (this repo's `docs/algorithms/`)
- Jupyter notebook tutorials for common use cases
- API reference (Sphinx + autodoc)

### 5e. PyPI release v1.0
- Semantic versioning, changelog, DOI via Zenodo

---

## Out of scope (v1)

- Background from incoherent scattering (treated as flat offset, trivial to subtract)
- Multi-crystal twins (complex case; flagged for v2)
- GPU acceleration (NumPy/SciPy first; CuPy wrapper deferred)
