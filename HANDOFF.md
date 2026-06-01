# Hand-off Notes — neutron-diffuse

**Date:** 2026-06-01  
**Repo:** `neutron-diffuse`  
**Status:** Design + skeleton implementation complete; docs/packaging cleaned;
ready for first real-data trial. I/O layer is next.

---

## Progress log

### 2026-06-01 — Docs/packaging cleanup
- Corrected the algorithm docs that wrongly described the powder ring as
  "isotropic in |Q|" — it is azimuthally anisotropic, captured by the
  factored T(φ) model (`powder_rings.md`, this file).
- Clarified `inpainting.md` scope: it is the general-purpose inpainter (mainly
  for Bragg holes); ring shells use radial interpolation, not symmetry averaging.
- Fixed README quickstart to the real API; fixed clone URL.
- Fixed `pyproject` build-backend and static version.
- Removed the dead `background/` (Al masking) module + test.
- Rewrote the integration test against the real pipeline API.
- Decision: repo/dist name stays `neutron-diffuse`, import stays `ndiff`.
- Commits authored by Tsung-Han Yang only (no co-author trailer).

### Next phase (planned)
User prepares a real input file, then we build, in order:
1. **Readers/loaders** matched to that real input format (do not assume the
   current skeleton `io/hkl_reader.py` matches — inspect the real file first).
2. **Data presentation / visualisation.**
3. **Writers.**
Design guidance: keep components separated, in small focused pieces, so each
stage (reader / processing / presentation / writer) is independently swappable.

---

## What this package does

Takes a **symmetrised 3D HKL volume** (output of Mantid or equivalent data reduction)
and produces a clean **3D diffuse scattering volume** ready for 3D-ΔPDF analysis.

```
[ Symmetrised HKL volume from Mantid ]
        │
        ▼  DATA PROCESSING
        │  (1) Empty-scan subtraction          → remove environment ring
        │  (2) Factored ring model fit         → remove residual sample-holder ring
        │  (3) Backfill ring holes             → interpolate diffuse signal
        │
        ▼  FURTHER ANALYSIS
        │  (4) Bragg peak removal (punch)
        │  (5) Backfill Bragg holes
        │  (6) 3D-ΔPDF via Fourier transform
        │
        ▼
  [ 3D-ΔPDF in real space ]
```

---

## Module map

```
src/ndiff/
├── core.py                        HKLVolume: main data container
│                                  (3D array + UB matrix + mask + σ)
│
├── io/hkl_reader.py               load() / save()  — HDF5 and ASCII
│
├── preprocessing/
│   ├── empty_subtraction.py       EmptySubtractor
│   │                              Step 1: I_residual = I_sample − s·I_empty
│   │                              Scale s estimated from ring-dominated |Q| shells.
│   │
│   ├── ring_model.py              PatchedRingModel   ← primary ring removal
│   │                              Model: I_ring(Q,φ) = T(φ) × Σᵢ Aᵢ G(|Q|−qᵢ,σᵢ)
│   │                              Fit: patches in φ → Gaussian NNLS per patch
│   │                                   → rank-1 SVD → Fourier T(φ)
│   │                              Diagnostics: rank1_variance, per_ring_texture_residual
│   │
│   ├── powder_rings.py            Supporting utilities:
│   │                              detect_ring_shells() — rolling-median 1D detection
│   │                              mask_ring_shells()   — sigmoid-tapered mask
│   │                              radial_profile()     — 1D |Q| binning
│   │                              al_ring_q_positions()— Al FCC peak positions (ref)
│   │
│   ├── backfill.py                backfill_ring_shells()
│   │                              Per masked voxel: nearest uncontaminated 3D
│   │                              neighbours (outside ring |Q|) → weighted interp.
│   │                              C¹ continuity from interpolation, not stitching.
│   │                              TV inpainting fallback for isolated voxels.
│   │
│   └── residual_rings.py          detect_and_fill_residual()  [superseded by ring_model]
│                                  Kept for comparison / alternative approach.
│
├── analysis/
│   ├── bragg.py                   BraggRemover / bragg_mask()
│   │                              Ellipsoidal punch at integer (h,k,l).
│   │                              Adaptive radius, sigmoid taper.
│   │
│   ├── bragg_fill.py              backfill_bragg()
│   │                              TV inpainting (λ=0.2) for Bragg holes.
│   │
│   └── delta_pdf.py               compute_delta_pdf() → DeltaPDF
│                                  Hann apodization → zero-pad → fftn → real part.
│                                  Real-space axes in Å via UB matrix.
│
├── inpainting/
│   ├── tv_inpainting.py           tv_inpaint()  Chambolle-Pock primal-dual
│   ├── interpolation.py           rbf_fill(), biharmonic_fill()
│   └── pipeline.py                fill()  — orchestrates symmetry→TV→RBF
│
└── utils/reciprocal_space.py      ub_from_lattice(), d_spacing(), q_to_hkl()
```

---

## Key design decisions and their rationale

### Why not use crystal symmetry for ring removal?
A powder ring is localised in |Q| (a thin shell) but its amplitude varies
with azimuthal direction — it is *not* isotropic. Regardless, all Laue
equivalents of a masked ring voxel sit on the same |Q| shell and are
equally contaminated, so symmetry averaging cannot separate ring from
diffuse signal. The azimuthal variation is instead captured by the
factored T(φ) model below.

### Why the factored model T(φ) × Σ Aᵢ G(|Q|)?
All rings from the same polycrystalline material share the same detector
geometry, so their azimuthal texture is the same function T(φ) scaled by
per-ring amplitudes Aᵢ. The SVD rank-1 factorisation extracts this optimally.

### Why Fourier series for T(φ)?
Periodic, smooth (C∞), no patch-boundary stitching needed. Continuity
is automatic. Typical n_fourier = 4–8 resolves detector-geometry variations.

### Why radial interpolation for backfill?
Ring holes are thin |Q| shells. The nearest uncontaminated neighbours in
3D HKL space are at the same angular position but just outside the shell.
Interpolating across this thin gap is C¹ by construction and imposes no
assumption on the diffuse signal shape.

### Concern: higher-|Q| rings may have more azimuthal texture
At larger scattering angles, detector solid-angle coverage and absorption
path length vary more strongly with direction, so T_i(φ) may differ
between inner and outer rings. Use `model.rank1_variance` and
`model.per_ring_texture_residual()` to diagnose this after the first run.
If rank-1 variance is below ~0.90, per-ring T_i(φ) fitting is needed.

---

## What is NOT yet done

| Item | Notes |
|------|-------|
| Real-data validation | Algorithm designed; needs first trial on actual dataset |
| Per-ring texture T_i(φ) | Extension for high-|Q| rings if rank1_variance < 0.90 |
| Patch size / overlap tuning | n_patches, overlap_frac are dataset-dependent |
| Detector-gap handling | Patches with few voxels currently skipped; needs robustness |
| Overlapping ring peaks | Closely spaced rings may alias in the NNLS fit |
| Bragg removal refinement | Adaptive punch radius; profile subtraction before punch |
| 3D-ΔPDF normalisation | Absolute units / monitor normalisation not yet wired |
| Mantid integration | Export format; Mantid workflow script |
| PyPI packaging | Once API is stable |

---

## Immediate next steps

1. **Load a real dataset** → `ndiff.load("your_file.h5")`
2. **Run empty subtraction** → inspect residual ring visually in a few |Q| slices
3. **Run PatchedRingModel.fit()** → check `rank1_variance` and `per_ring_texture_residual()`
4. **Inspect the ring model** → plot T(φ) and the per-patch amplitude matrix A
5. **Subtract and backfill** → compare before/after in hk0 and h0l slices
6. **Punch Bragg + compute ΔPDF** → compare to reference if available

---

## Open algorithmic questions (to resolve with real data)

- How many azimuthal patches are needed for typical datasets?
- Is n_fourier = 6 sufficient, or do high-|Q| rings require more harmonics?
- What SNR threshold for the ring mask gives the best ΔPDF quality?
- Sequential vs. simultaneous Gaussian fitting per patch — does it matter in practice?

---

## Dependencies

```
numpy >= 1.24
scipy >= 1.10      (SVD, NNLS, KDTree, FFT, spline)
h5py  >= 3.8       (HDF5 I/O)
matplotlib >= 3.7  (visualisation, not yet wired into the library)
```

Dev: `pip install -e ".[dev]"` (adds pytest, ruff, mypy, pre-commit).
