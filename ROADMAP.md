# Roadmap

## Pipeline overview

Input: symmetrised 3D HKL volume from Mantid (or equivalent).

```
[ Symmetrised 3D HKL volume ]
        │
        ▼  DATA PROCESSING
        │
        │  (1) Empty-scan subtraction
        │        Subtract monitor-normalised empty-environment scan.
        │        Removes environment ring (cryostat, furnace, etc.).
        │        Residual: sample-holder Al ring remains.
        │
        │  (2) Residual ring removal  [active research]
        │        Ring amplitude is uneven → must work locally.
        │        Process patch by patch; enforce C¹ continuity across
        │        patch boundaries via overlapping Hann-window blending.
        │
        ▼  FURTHER ANALYSIS
        │
        │  (3) Bragg peak removal  (punch-and-fill)
        │  (4) 3D-ΔPDF via Fourier transform
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

Monitor-normalised subtraction:

```
I_residual(Q) = I_sample(Q) − s × I_empty(Q)
```

Scale factor `s` is estimated analytically by minimising the residual
in ring-dominated |Q| shells.  Removes the environment ring; sample-holder
Al ring remains.

### 2-2. Residual ring removal  *(active research — algorithm TBD)*

**The problem in full:**

- The residual ring (sample-holder Al) is not in the empty scan.
- Its amplitude varies across the |Q| shell: stronger where detector
  coverage is dense, weaker or absent in gaps.
- A global model (fit one Gaussian to the whole ring) fails because the
  amplitude is not constant.
- Crystal Laue symmetry is **not** applied here: the ring and the
  diffuse signal are both local quantities, and a global symmetry
  operation does not separate them cleanly.

**Approach: local patch-by-patch processing with C¹ continuity**

Work in small overlapping patches in HKL (or |Q| × solid-angle) space:

```
Within each patch:
  1. Estimate smooth local diffuse background
     (low-order polynomial or spline fit to uncontaminated voxels).
  2. Ring excess = data − background.
     Flag voxels where excess > threshold.
  3. Fill flagged voxels from the smooth background estimate
     (local interpolation, not symmetry equivalents).
  4. Weight the processed patch by a Hann window.

Combine patches:
  I_out(voxel) = Σ_patches  w_p(voxel) × I_p(voxel)
               / Σ_patches  w_p(voxel)
```

**Why Hann windows guarantee C¹ continuity:**

The Hann window is zero at both patch edges, with zero first derivative
at the edges.  Overlapping Hann windows form a partition of unity
(they sum to 1 everywhere in the interior).  Because the weight is C¹
and the blending is a weighted average, the output is automatically C¹
across all patch boundaries — no post-processing stitch step needed.

**Open design questions:**

| Question | Options |
|----------|---------|
| Patch geometry | Cartesian HKL boxes vs. |Q| shells × angular sectors |
| Patch size | Must be large enough to fit background; small enough that ring amplitude is ~constant |
| Background model | Low-order polynomial, local spline, median filter |
| Detection threshold | Per-patch adaptive (MAD-based) |
| Patch overlap fraction | 50 % is standard; may need tuning |

**What NOT to do:**
- Do not apply symmetry averaging within the ring removal step — the ring
  and diffuse signal are both local; symmetry is a global operation that
  conflates them.
- Do not process without overlap — hard patch edges produce C⁰
  discontinuities that appear as ringing in the 3D-ΔPDF.
- Do not fill with a flat value — the diffuse signal underneath the ring
  must be estimated from the local smooth background, not set to zero or
  the shell mean.

---

## Phase 3 — Bragg Peak Removal (months 6–8)

- Enumerate integer (h,k,l) within the grid.
- Ellipsoidal punch mask with soft (sigmoid-tapered) boundary.
- Adaptive punch radius: scale with peak intensity.
- Backfill with TV inpainting (C¹-continuous by construction).

---

## Phase 4 — 3D-ΔPDF (months 8–10)

```
Δρ(r) = FT[ I_diffuse(Q) ]
```

1. Apodize with 3D Hann window.
2. Zero-pad to next power-of-2.
3. `scipy.fft.fftn` → fftshift → real part.
4. Real-space axes in Å via UB matrix.

---

## Phase 5 — Validation & Release (months 10–12)

| Task | Details |
|------|---------|
| Synthetic tests | Inject ring + Bragg into known diffuse; evaluate ΔPDF quality |
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
│   ├── residual_rings.py       # (2) patch-by-patch ring removal [TBD]
│   └── backfill.py             # (2b) 3D inpainting for unfillable voxels
├── analysis/
│   ├── bragg.py                # (3) Bragg punch
│   ├── bragg_fill.py           # (3b) Bragg backfill
│   └── delta_pdf.py            # (4) 3D-ΔPDF
├── inpainting/
│   ├── tv_inpainting.py
│   ├── interpolation.py
│   └── pipeline.py
└── utils/
    └── reciprocal_space.py
```
