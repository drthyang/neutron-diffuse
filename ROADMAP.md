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

## Phase 2 — Powder Ring Removal  ✓ implemented (slice-validated and full 3D)

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

**2026-06-03 ring-removal continuation:** validation moved to the improved
`TbTi3Bi4_22K_mmm...401x401x301_mmm.nxs` dataset, especially the `H=0.3333`
slice where diffuse signal is visible.  Two over-subtraction fixes then landed
and became the new defaults:
- **SNIP baseline** (`baseline_method="snip"`): slope-aware peak-clipping
  replaces morphological opening, which was biased low on sloping backgrounds.
- **|Q|-pooled high-order azimuthal texture** (`n_fourier=8`,
  `texture_ridge=0.05`, `texture_q_smooth=0.06`): real rings have multi-lobed
  texture that low order can't follow; raising the order alone rang into sparse
  azimuths, so the texture *shape* is now pooled across each ring's |Q| width
  (amplitude-weighted, radial peak kept sharp).  Over-subtraction `neg_trough`
  fell ~51% on the H=0.3333 slice.
- **Per-ring adaptive baseline thickness** (`adaptive_ring_width=True`,
  `ring_width_scale=3.0`, `ring_width_cap_frac=0.9`): ring FWHM varies 2.6× in
  the data, so a single global window either under-captures broad rings or eats
  diffuse and bridges close ring pairs.  Each ring now gets a window of
  scale×FWHM (detected Bragg-robustly via the cross-patch median), capped to the
  neighbour spacing.  Close-pair valley over-subtraction fell ~31%.
The reference is now `q_step=0.02, texture_model="fourier", n_fourier=8,
texture_ridge=0.05, texture_q_smooth=0.06, baseline_method="snip",
adaptive_ring_width=True`.  Diagnostics
score signed residuals, negative troughs, radial roughness, and off-ring removal
(not positive ring leftover alone); `examples/_azimuthal_texture_cmp.py` scores
the azimuthal texture fit directly.  The experimental `texture_model="smooth"`
(L-BFGS-B per-patch with cyclic curvature penalty) remains available but is not
the default.

**2026-06-03 — residual leftover root-caused.**  The "uniform positive ring
leftover" was traced to two stacked causes (SNIP and the texture model in
aggregate are NOT the cause).  Three independent levers, all exposed as env knobs
in `explore_slice.py` (`TEXTURE_Q_SMOOTH`, `PROFILE_METHOD`, `Q_STEP`); defaults
unchanged pending 22K H=0.3333 validation:
- **`texture_q_smooth=0`** captures the ring's *azimuthally-varying width* —
  `q_smooth` pools the texture shape across |Q|, which homogenises the width when
  it varies with φ (under-subtracting broad arcs, over-subtracting narrow ones).
  Turning it off cut both under- and over-fill ~30% at H≠0 with no diffuse cost.
  The most principled lever; promote it (likely with `median`) once 22K confirms.
- **`profile_method="median"`** removes the asymmetric-trim bias of the default
  `trimmed_mean (10,80)` (−12% arc under-fill, no extra over-subtraction).
- **`q_step=0.015`** resolves the peak better (−15% leftover, no troughs).
The experimental mask-and-replace cleanup (`masked_rings.py`) was **removed** —
its excess-based mask could not separate ring from diffuse (27% of masked
intensity was real signal).  Ring removal is **subtractive only**.

### 2-3. Backfill  ✓

Masked voxels (ring-dominated) filled by:
1. KDTree on valid voxel indices; filter to uncontaminated (outside all ring |Q|).
2. Weighted interpolation across the thin gap → C¹ by construction.
3. TV inpainting (Chambolle-Pock) fallback for voxels with too few clean neighbours.

**2026-06-04 update:** ring removal is now promoted to the full 3D volume via
`examples/remove_rings_3d.py`, which fits the validated `PatchedRadialRingModel`
independently on each H plane.  Cross-H ring-shell confirmation and per-shell
amplitude ceilings suppress integer-H phantom troughs while keeping real powder
rings continuous across H.

---

## Phase 3 — Bragg Peak Removal  ✓ implemented

- Data-driven local-window punch at detected integer Bragg peaks.
- `mode="auto"` / `mode="search"` detects sharp off-integer satellites as
  robust per-|Q|-shell outliers.
- Adaptive anisotropic punch radius (scale with intensity).
- Forced origin punch removes the non-physical `(0,0,0)` direct-beam remnant.
- Optional `phi_tail_hkl` expands punches tangentially in the K-L plane to catch
  Bragg tails smeared along the powder-ring direction.
- Backfill via `backfill_bragg` (default `method="local"` shell-median fill).

**2026-06-04 auto-punch update:** `BraggRemover(mode="auto")` is now an alias
for search mode: robust per-|Q|-shell background (`median + n·MAD`) plus local
maxima, analogous to the ring-removal high-tail rejection logic.  The real-data
driver exposes `SEARCH_MIN_I` to catch weak Bragg/satellite peaks and
`SEARCH_PROM` to require local 3x3x3 sharpness so broad diffuse maxima are not
punched.  Current visual candidate:
`MODE=auto R_HKL=0.09,0.12,0.45 MAX_SCALE=2.0 MARGIN=0.02 SEARCH_NMAD=4
SEARCH_MIN_I=1.0 SEARCH_PROM=1.0`.  Inspect H=0.333/0.666 for diffuse
preservation and H=2 K=±8/±10 for weak-peak capture before promoting defaults.

**2026-06-04 punch/backfill update:** visual QA showed two remaining punch
issues: the strong `(0,0,0)` intensity is not real, and Bragg tails persisted
along φ.  `BraggRemover` now defaults to `force_origin=True` and supports
`phi_tail_hkl`; the example drivers default to `PHI_TAIL_HKL=0.12`.  The
interactive QA script `examples/explore_slice.py` now processes all H planes and
opens an H-slider viewer with four panels: `data`, `Removed ring`, `Punched`,
and `Backfilled`.  `backfill_bragg` now defaults to local shell-median fill,
with TV/symmetry retained for comparisons.

**Pending refinements:**
- Profile subtraction before punch (reduces punch radius needed).
- Absolute monitor normalisation.

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

> **✅ Ring off-centering — re-investigated 2026-06-03, NOT an issue.** On all
> testable data the ring centre is at the origin (H=0: |c|≈3×10⁻⁵; H=0.32: tiny
> H-projection artefact, correcting it changes residuals <0.003).  Verified the
> ring sits at constant 3D |Q| with φ, so the radial binning is correct.  No
> longer a blocker.  (Re-check on the 22K file with `_ring_center_fit.py` if its
> rings look off-centre there.)  See HANDOFF.md.

**Standard preview tool:** always investigate ring-removal results with the
interactive viewer `examples/explore_slice.py` (live 3-panel data | removed
rings | residual slice with vmin/vmax sliders, default slider travel 0.0–1.0).
Run via the `rmc-discord` env; defaults to the 22K mmm file, H=0.3333.

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
