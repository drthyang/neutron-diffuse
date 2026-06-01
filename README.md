# neutron-diffuse

**3D diffuse neutron scattering processing — powder ring removal and 3D-ΔPDF.**

Takes a symmetrised 3D HKL volume (output of Mantid or equivalent) and produces
a clean diffuse scattering volume ready for 3D-ΔPDF analysis.

```
[ Symmetrised 3D HKL volume from Mantid ]
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

## Why two steps for ring removal?

An **empty-scan subtraction** removes the environment ring (cryostat, furnace).
A residual ring from the sample holder remains and cannot be removed by subtraction
alone, because the sample holder scattering is present only with the sample mounted.
The **factored ring model** removes this residual ring by fitting

```
I_ring(Q, φ) = T(φ) × Σᵢ Aᵢ G(|Q| − qᵢ, σᵢ)
```

where T(φ) is a shared azimuthal texture function (Fourier series) extracted via
rank-1 SVD of the per-patch amplitude matrix.  This imposes no assumption on the
diffuse signal.

## Quickstart

```python
import ndiff
from ndiff.preprocessing import EmptySubtractor, PatchedRingModel, backfill_ring_shells

# Load symmetrised 3D HKL volume (from Mantid or equivalent)
vol = ndiff.load("experiment_sym.h5")
vol_empty = ndiff.load("empty_environment.h5")

# ── Step 1: empty-scan subtraction ──────────────────────────────────────────
sub = EmptySubtractor()
vol1, scale = sub.subtract(vol, vol_empty)
print(f"Empty scan scale factor: {scale:.4f}")

# ── Step 2: factored ring model (residual sample-holder ring) ────────────────
model_fitter = PatchedRingModel(n_patches=36, overlap_frac=0.3, n_fourier=6)
fitted = model_fitter.fit(vol1)
print(f"Rank-1 variance: {fitted.rank1_variance:.3f}")   # > 0.90 is good
print("Per-ring texture residual:", fitted.per_ring_texture_residual())

vol2, ring_snr = model_fitter.subtract(vol1, fitted)

# ── Step 3: backfill ring holes ──────────────────────────────────────────────
from ndiff.preprocessing import detect_ring_shells, mask_ring_shells
rings, *_ = detect_ring_shells(vol1)
keep = mask_ring_shells(vol1, rings)

import dataclasses
vol2_masked = dataclasses.replace(vol2, mask=vol2.mask & keep)
vol_clean = backfill_ring_shells(vol2_masked, rings, n_neighbors=16)

# ── Step 4–5: Bragg punch and fill ──────────────────────────────────────────
from ndiff.analysis import BraggRemover, backfill_bragg
bragg_keep = BraggRemover().mask(vol_clean)
import dataclasses
vol_punched = dataclasses.replace(vol_clean, mask=vol_clean.mask & bragg_keep)
vol_diffuse = backfill_bragg(vol_punched)

# ── Step 6: 3D-ΔPDF ─────────────────────────────────────────────────────────
from ndiff.analysis import compute_delta_pdf
dpdf = compute_delta_pdf(vol_diffuse, apodization="hann")

# Save
ndiff.save(vol_diffuse, "diffuse_only.h5")
```

## Installation

```bash
git clone https://github.com/thyang-phys/neutron-diffuse
cd neutron-diffuse && pip install -e ".[dev]"
```

## Module overview

```
src/ndiff/
├── core.py                     HKLVolume: 3D array + UB matrix + mask + σ
├── io/hkl_reader.py            load() / save() — HDF5 and ASCII
├── preprocessing/
│   ├── empty_subtraction.py    EmptySubtractor  (step 1)
│   ├── ring_model.py           PatchedRingModel  (step 2) ← primary ring removal
│   ├── powder_rings.py         detect_ring_shells, mask_ring_shells, radial_profile
│   └── backfill.py             backfill_ring_shells  (step 3)
├── analysis/
│   ├── bragg.py                BraggRemover / bragg_mask  (step 4)
│   ├── bragg_fill.py           backfill_bragg  (step 5)
│   └── delta_pdf.py            compute_delta_pdf → DeltaPDF  (step 6)
├── inpainting/
│   ├── tv_inpainting.py        tv_inpaint — Chambolle-Pock primal-dual
│   ├── interpolation.py        rbf_fill, biharmonic_fill
│   └── pipeline.py             fill — symmetry → TV → RBF orchestration
└── utils/reciprocal_space.py   ub_from_lattice, d_spacing, q_to_hkl
```

## Diagnostics

After fitting `PatchedRingModel`, inspect:

- `fitted.rank1_variance` — fraction of variance explained by the shared T(φ).
  Values ≥ 0.90 confirm the rank-1 factorisation is valid.  Values < 0.90
  indicate that higher-|Q| rings have stronger per-ring azimuthal texture and
  may need individual T_i(φ) fits.
- `fitted.per_ring_texture_residual()` — per-ring RMS deviation from the shared
  texture.  Identify which ring drives the rank-1 failure.

## Status

Design and skeleton implementation complete; ready for first real-data trial.
See [HANDOFF.md](HANDOFF.md) for hand-off notes and [ROADMAP.md](ROADMAP.md) for
the development plan.

## Dependencies

```
numpy >= 1.24
scipy >= 1.10      (SVD, NNLS, KDTree, FFT, spline)
h5py  >= 3.8       (HDF5 I/O)
matplotlib >= 3.7  (visualisation)
```

## License

MIT
