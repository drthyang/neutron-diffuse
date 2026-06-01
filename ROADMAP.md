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
        │        Removes: cryostat, furnace, pressure-cell walls, etc.
        │        Residual: sample-holder Al remains (not in empty scan).
        │
        │  (2) Residual ring detection & fill  [open research problem]
        │        Ring amplitude is UNEVEN around the shell (detector
        │        coverage, absorption, normalisation). Exploit this:
        │        within each Laue equivalent group, flag anomalously-high
        │        voxels and fill from the cleaner equivalents.
        │        Fallback: 3D inpainting for voxels with too few clean
        │        equivalents.
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

Straightforward monitor-normalised subtraction:

```
I_residual(Q) = I_sample(Q) − s × I_empty(Q)
```

where `s` is a scale factor fit from ring-dominated |Q| shells
(minimises the residual in those shells, solved analytically).

This removes the bulk of the ring from the sample environment.
The sample holder contribution is **not** removed here because the
sample holder is absent from the empty scan.

### 2-2. Residual ring removal  *(active research — no final algorithm yet)*

**The challenge:**

The residual ring (from sample holder Al) is:
- Not present in the empty scan.
- Non-uniform in amplitude around the |Q| shell — stronger where more
  frames contribute at that scattering geometry, weaker or absent in
  detector gaps.

**The key insight:**

Non-uniform amplitude is actually exploitable via crystal Laue symmetry.
Within a set of Laue-equivalent voxels (all at the same |Q|), voxels in
a high-ring-amplitude sector will be anomalously high compared to voxels
in a low-ring-amplitude sector. An asymmetric sigma-clip within each
equivalent group identifies the contaminated voxels; the clean equivalents
fill them.

This requires:
- Input symmetrised (from Mantid) so that genuine diffuse asymmetry is
  already removed.
- Enough Laue multiplicity that some equivalents escape the ring.

**Current status:** Initial implementation in `residual_rings.py`.
Algorithm needs validation on real data; threshold tuning is dataset-dependent.

**Open questions:**
- What fraction of equivalents are typically contaminated? (Depends on ring
  width and crystal symmetry.)
- What to do when all equivalents are contaminated (low-multiplicity or
  wide ring)? → 3D inpainting fallback.
- Does the method work for monoclinic / triclinic crystals?

**Alternative approaches (for future consideration):**
- Profile model per solid-angle sector (fit ring independently per patch).
- Iterative diffuse/ring separation (alternate between smooth diffuse estimate
  and ring residual).
- Instrument-geometry-based prediction of ring amplitude per HKL direction.

---

## Phase 3 — Bragg Peak Removal (months 6–8)

- Ellipsoidal punch at each integer (h,k,l).
- Adaptive radius: scale with peak intensity.
- Soft (sigmoid-tapered) boundary.
- Backfill with TV inpainting.

---

## Phase 4 — 3D-ΔPDF (months 8–10)

```
Δρ(r) = FT[ I_diffuse(Q) ]
```

1. Normalise / apodize (Hann window in 3D).
2. Zero-pad to next power-of-2.
3. `scipy.fft.fftn` → fftshift → take real part.
4. Output real-space axes in Å via UB matrix.

---

## Phase 5 — Validation & Release (months 10–12)

| Task | Details |
|------|---------|
| Synthetic tests | Inject ring + Bragg into known diffuse; evaluate pipeline |
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
│   ├── residual_rings.py       # (2) symmetry-based residual detection & fill
│   ├── backfill.py             # (2b) 3D inpainting fallback for unfillable voxels
│   └── powder_rings.py         # diagnostic / exploratory tools
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
