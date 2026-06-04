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
from ndiff.preprocessing import (
    EmptySubtractor, PatchedRingModel, backfill_ring_shells, detect_ring_shells,
)
from ndiff.analysis import bragg_mask, backfill_bragg, compute_delta_pdf

# Load symmetrised 3D HKL volume (from Mantid or equivalent)
vol = ndiff.load("experiment_sym.h5")
vol_empty = ndiff.load("empty_environment.h5")

# ── Step 1: empty-scan subtraction ──────────────────────────────────────────
sub = EmptySubtractor(vol_empty)            # empty scan is a constructor arg
vol1 = sub.subtract(vol)                    # scale estimated automatically
print(f"Empty scan scale factor: {sub.scale:.4f}")

# ── Step 2: factored ring model (residual sample-holder ring) ────────────────
model = PatchedRingModel(n_patches=36, overlap_frac=0.3, n_fourier=6)
fitted = model.fit(vol1)                     # or fit(vol1, ring_hints=[2.69, 3.10])
print(f"Rank-1 variance: {fitted.rank1_variance:.3f}")   # > 0.90 is good
print("Per-ring texture residual:", fitted.per_ring_texture_residual())

vol2, i_ring = model.subtract(vol1, fitted)  # ring-dominated voxels are masked

# ── Step 3: backfill the masked ring shells ──────────────────────────────────
rings, *_ = detect_ring_shells(vol1)         # |Q| ranges to treat as contaminated
vol_clean = backfill_ring_shells(vol2, rings, n_neighbors=16)

# ── Step 4–5: Bragg punch and fill ──────────────────────────────────────────
vol_clean.apply_mask(bragg_mask(vol_clean, punch_radius_hkl=0.3))
vol_diffuse = backfill_bragg(vol_clean)

# ── Step 6: 3D-ΔPDF ─────────────────────────────────────────────────────────
dpdf = compute_delta_pdf(vol_diffuse, apodization="hann")

# Save
ndiff.save(vol_diffuse, "diffuse_only.h5")
```

For the current real-data workflow, the example drivers are the most up-to-date
entry points:

```bash
PYTHONPATH=src python examples/remove_rings_3d.py
PYTHONPATH=src python examples/punch_bragg_3d.py
PYTHONPATH=src python examples/backfill_bragg_3d.py
```

Use `examples/explore_slice.py` for interactive QA.  It opens an H-slider viewer
with four panels: data, ring-removed, punched, and backfilled.

## Installation

```bash
git clone https://github.com/drthyang/neutron-diffuse
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

Real Mantid NeXus loading, full-3D ring removal, Bragg/satellite punching, local
Bragg backfill, and 3D-DeltaPDF computation are implemented.  The current Bragg
punch stage treats the `(0,0,0)` incident beam separately from Bragg peaks and
uses UB-aware phi-direction tail expansion for peaks smeared along powder rings.
The cleanup stack is ready for real-data 3D-DeltaPDF candidate generation and
inspection.  See [HANDOFF.md](HANDOFF.md) for current hand-off notes and
[ROADMAP.md](ROADMAP.md) for the development plan.

## Dependencies

```
numpy >= 1.24
scipy >= 1.10      (SVD, NNLS, KDTree, FFT, spline)
h5py  >= 3.8       (HDF5 I/O)
matplotlib >= 3.7  (visualisation)
```

## License

MIT
