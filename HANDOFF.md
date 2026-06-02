# Hand-off Notes — neutron-diffuse

**Date:** 2026-06-02  
**Repo:** `neutron-diffuse`  
**Status:** Reader and visualization complete; real data loads and plots correctly
with consistent physics-convention |Q|. Next: writers.

---

## Progress log

### 2026-06-02 — UB convention fixes (real-data |Q| correctness)
Two latent correctness bugs surfaced while presenting the real files; both fixed
in `io/mantid_nxs.py`:
- **2π convention.** Mantid's stored `orientation_matrix` is *crystallographic*
  (|b*| = 1/d, no 2π), but ndiff uses the *physics* convention everywhere
  (`ub_from_lattice` returns 2π·B; `al_ring_q_positions` uses Q = 2π√…/a).
  `_read_ub_matrix` now multiplies the stored matrix by 2π. Verified on real
  data: background ring peaks now align exactly with the Al FCC peak positions
  (low-Q peak at 2.69 Å⁻¹ = Al(111)); data |Q|max went 1.76 → 11.08 Å⁻¹.
- **Missing background UB.** The `_bkg.nxs` file has no `experiment0` group, so
  the reader fell back to an identity UB (bogus |Q|max 32.3). `load_mantid_nxs`
  now takes an optional `ub_matrix=` override; pass the paired data volume's UB
  so empty-can scans share a consistent |Q| scale. New `_resolve_ub()` helper
  picks: explicit override > file value > identity fallback.
  Data and background share identical masks (same geometry), confirming this
  is the right UB to inherit.

Result: data and background radial profiles now ride on one |Q| axis; background
ring peaks are ~4× the data. The provided `_sub_bkg.nxs` (experiment's own
data−bkg) is over-subtracted (negative ring troughs) — expected without scaling;
ndiff's `EmptySubtractor` estimates the scale `s` automatically.

### 2026-06-01 — Mantid NeXus reader
Real data files confirmed: Mantid MDHistoWorkspace NeXus format (`.nxs`),
401×501×151 grid (K×L×H in file, permuted to H×K×L = 151×401×501).
Orthorhombic lattice a=5.48, b=10.32, c=24.83 Å. ~38.5% voxels valid
(rest outside detector coverage, stored as NaN in file).

New module `src/ndiff/io/mantid_nxs.py` — modular Mantid reader:
- `load_mantid_nxs(path)` — public entry point, returns HKLVolume
- `is_mantid_nxs(path)` — format probe for auto-dispatch
- 6 private single-purpose helpers (dim-axis parsing, UB reading, array
  assembly with permutation to canonical H,K,L order)
- `load()` in `hkl_reader.py` auto-detects Mantid format for `.nxs` files

Background file has no UB matrix in the file (no experiment0 group);
reader falls back to identity matrix. Background counts peak at ~11,740
vs data ~283 — scale factor `s` in `EmptySubtractor` will be well below 1
(estimated automatically from ring-dominated |Q| shells).

### 2026-06-01 — Visualization module
New package `src/ndiff/visualization/` — four modules, each single-purpose:
- `slices.py`: `extract_slice()` (returns `SliceData` NamedTuple),
  `plot_slice()` — 2D HKL plane views with percentile colour clipping,
  optional log scale, half-bin extent, grey masked regions.
  Accepts plane as `'kl'`/`'hl'`/`'hk'` or Mantid aliases `'0kl'`/`'h0l'`/`'hk0'`.
- `profiles.py`: `plot_radial_profile()` (wraps existing `radial_profile()`
  from `powder_rings.py`), `plot_azimuthal_map()` — φ vs I at a |Q| shell
  (useful for inspecting ring azimuthal texture before/after PatchedRingModel).
- `overview.py`: `plot_overview()` — 2×2 diagnostic figure: K-L, H-L, H-K
  slices + radial profile. Confirmed on real data: ring clearly visible in
  the K-L plane; multiple ring peaks visible in the radial profile.
- `__init__.py`: re-exports all six public names.

Next: writers.

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
1. ~~**Readers/loaders**~~ ✓ done (Mantid MDHistoWorkspace `.nxs`)
2. ~~**Data presentation / visualisation**~~ ✓ done (`ndiff.visualization`)
3. **Writers** — save processed volumes back to Mantid-compatible `.nxs`
   or ndiff HDF5.
Design guidance: keep components separated, in small focused pieces, so each
stage is independently swappable.

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
├── io/
│   ├── hkl_reader.py              load() / save()  — auto-dispatch by format
│   │                              .nxs → Mantid reader (auto-detected)
│   │                              .h5/.hdf5 → ndiff HDF5 schema
│   │                              .txt/.dat/.hkl → ASCII (h k l I sigma)
│   └── mantid_nxs.py              load_mantid_nxs() / is_mantid_nxs()
│                                  Reads MDHistoWorkspace: signal, σ, mask,
│                                  bin-edge axes, UB matrix.  Permutes file
│                                  (D2,D1,D0) order to canonical (H,K,L).
│                                  UB scaled ×2π (file is crystallographic,
│                                  ndiff is physics convention). Optional
│                                  ub_matrix= override for files lacking one
│                                  (e.g. background/empty-can scans).
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
├── visualization/
│   ├── slices.py                  extract_slice() → SliceData NamedTuple
│   │                              plot_slice() — 2D HKL plane view
│   │                              Planes: 'kl','hl','hk' (or '0kl','h0l','hk0')
│   │                              Percentile colour clip, log scale, grey mask.
│   ├── profiles.py                plot_radial_profile() — |Q| vs I
│   │                              plot_azimuthal_map()  — φ vs I at a |Q| shell
│   └── overview.py                plot_overview() — 2×2 diagnostic figure
│                                  (K-L, H-L, H-K slices + radial profile)
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
