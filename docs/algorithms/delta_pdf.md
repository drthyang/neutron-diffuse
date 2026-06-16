# 3D-ΔPDF Transform

## Purpose

The three-dimensional difference pair distribution function (3D-ΔPDF) is the
Fourier transform of the diffuse scattering intensity:

```text
Δρ(r) = FT[ I_diffuse(Q) ] = FT[ I_total(Q) − I_Bragg(Q) ]
```

It maps reciprocal-space diffuse scattering to real-space pair correlations.
Positive ΔPDF at a vector **r** means more interatomic pairs separated by **r**
than in the average structure; negative means fewer. References: Weber &
Simonov, *Z. Kristallogr.* **227**, 238 (2012); Simonov, Weber & Steurer,
*J. Appl. Cryst.* **47**, 2011 (2014).

API: `ndiff.analysis.compute_delta_pdf`. Drivers: `examples/delta_pdf.py`
(full 3D), `examples/delta_pdf_plane.py` (single reciprocal H-plane 2D),
`examples/explore_delta_pdf.py` (interactive y_K–z_L viewer with x_H slider).

## Correct transform recipe

The input volume stores `Q=0` at the **array centre** (index `s//2`), but
`fftn` treats index `[0,0,0]` as the origin. A correct, centred transform of a
real, centrosymmetric `I(Q)` must therefore be:

```python
Δρ = fftshift( fftn( ifftshift( I_windowed ) ) ).real
```

Step by step (as implemented in `src/ndiff/analysis/delta_pdf.py`):

1. **Fill** masked voxels with 0 (the backfilled volume should already be
   NaN-free).
2. **Apodize**: multiply by a separable window (`hann` default, or `gaussian` /
   `none`) to suppress termination ripples from the finite `|Q|` range.
3. **Remove DC**: subtract the mean *after* windowing so `Σ I = 0` exactly.
   This zeroes the `r=0` self-correlation spike. (Subtracting before windowing
   leaves a nonzero windowed sum → a large spurious peak at `r=0`.)
4. **Zero-pad symmetrically** to the next power of 2, keeping `Q=0` on the new
   centre. One-sided padding shifts the origin and breaks step 5.
5. **`ifftshift` → `fftn` → `fftshift`**: move `Q=0` to the corner, transform,
   then recentre `r=0`.
6. **Take the real part**: valid because the symmetrised (`mmm`) data is
   centrosymmetric, `I(Q)=I(−Q)`, so the transform is real. The imaginary part
   is numerical noise (a useful diagnostic — if it is not negligible, the input
   is not properly centred or symmetrised).

Real-space axes come from `fftshift(fftfreq(n, d=ΔHKL))`, converted to Å with
the direct-lattice vector lengths `2π·inv(UB)ᵀ`.

## Inverse transform & consistency check (`invert_delta_pdf`)

The recipe is exactly invertible, so the ΔPDF can be transformed **back** to the
reciprocal-space diffuse volume it came from — a round-trip consistency check.
`compute_delta_pdf` records the inverse metadata (pad width, the separable window
factors, the subtracted mean, the cropped axes) on its result, and
`invert_delta_pdf` undoes each step:

```python
I_recon = fftshift( ifftn( ifftshift( Δρ ) ) ).real    # inverse of the recipe
I_recon = unpad(I_recon)                                # strip symmetric padding
I_recon = (I_recon + mean) / window                     # restore DC, deapodize
```

The deapodization (divide by the window) is well-posed for the **gaussian**
window because it never reaches zero; for **hann** the edge planes (window → 0)
are clamped by `window_floor` and are unreliable. The recovered volume's `mask`
marks the reliably-recovered region.

`pdf_consistency_check` (pipeline stage `pdf_check`, default ON; standalone
`examples/delta_pdf_consistency.py`) runs this inverse and compares it to the
diffuse data the ΔPDF was built from (cropped to the transform window), writing a
metric JSON (Pearson r + normalised RMS residual) and a `data | back-FFT |
residual` figure. Because the `mmm` data is centrosymmetric and the gaussian
window is invertible, a faithful ΔPDF round-trips to **r ≈ 1** (22K: r = 0.9999,
RMS ≈ 1%). The check is therefore a regression/validation gate: a wrong axis,
sign flip, or normalisation bug, or an over-aggressive `crop_hkl` / apodization,
would surface here as a visible residual. (On an **even** grid the `Q=0` centre
leaves one index unpaired, so the real-part projection drops a small asymmetric
part — a known, tiny round-trip error; odd grids are exact.)

## The centring bug (fixed 2026-06-05)

Earlier code did `fftshift(fftn(data))` with **no `ifftshift`** and **one-sided**
zero-padding. With `Q=0` at the array centre, the missing `ifftshift` introduces
a linear phase ramp `e^{−iπk} = (−1)^k` across the output. Taking the real part
then **flips the sign of real-space features by pixel parity**, so each
correlation peak splits into mixed positive/negative lobes (a derivative-like
appearance), and slices such as `x_H=0` look scrambled.

Verification: for `I(Q)=1+cos(2π·3·(i−c)/N)` (a single positive correlation,
even about the centre) the buggy transform returned a peak of **−2048** where
the correct transform returns **+2048**. Regression guard:
`tests/test_bragg.py::test_delta_pdf_centring_positive_peak`.

This bug only affected sign/phase; the real-space axes and magnitudes were
already correct. Zero-padding is sinc-interpolation only — it gives a finer
display grid, not more intrinsic resolution (that is fixed by the `|Q|` range
and the apodization window).

## Near-origin spike (expected, not the bug)

A strong feature at `r < ~3 Å` remains after the fix. It comes from residual
high-`|Q|` Bragg leakage, the backfill discontinuities at punch boundaries, and
the direct-beam punch. Plot colour scales are set from the `p99` of `|ΔPDF|` at
`r > 3 Å` so this near-origin spike does not dominate the display.

## The axis cross is the residual diffuse background (diagnosed 2026-06-05)

A bright **cross** along the `y_K=0` and `z_L=0` axes appears in the real-space
map. It is **not** a Bragg/punch/masking artifact: it is present even on planes
with no Bragg peaks (e.g. `H=1/3`), the input has **0 % masked voxels** along
the axis lines, and replacing the exact `K=0`/`L=0` input lines with neighbour
averages changes nothing.

**Root cause.** Ring removal, Bragg punching, and backfill remove rings
(azimuthal), sharp local peaks, and holes — but **none of them removes the
broad, slowly-varying diffuse *envelope***: a smooth positive hump centred near
`K=L=0` that decays toward the edges, with ridges along the principal axes.
`subtract_mean` only removes the scalar DC term (killing the `r=0` spike); it
leaves the shape of the envelope untouched. That envelope is approximately
**separable** (`≈ f(K) + g(L)`), and the FT of a separable function concentrates
its energy *on the two axes* → a cross at `y_K=0` / `z_L=0`. The Hann window,
itself a centred separable hump, multiplies in and sharpens the cross.
Directionally: the horizontal arm (`z_L=0`) is the FT of the L-averaged
K-profile; the vertical arm (`y_K=0`) is the FT of the K-averaged L-profile.

Why `apodize="none"` *looks* like it has a smaller cross: hard truncation
sprays termination ripple everywhere, raising the off-axis floor, so the cross
ratio drops — the window does not *create* the cross, it cleans everything
*except* the cross (which is real low-frequency signal).

**Fix.** Subtract a smooth background **before** windowing so only the
oscillatory modulation transforms. A Gaussian blur (`σ ≈ 1.5 r.l.u.`) collapses
the cross while preserving the off-axis correlation lattice and the genuine
correlation peaks that happen to sit on the axes. Trade-off (standard for
ΔPDF): this also removes genuine very-long-period / low-`r` correlations — but
those live at the same scale as the un-subtractable background, so they cannot
be cleanly separated from it anyway. Subtracting the exact separable marginals
(`per-row + per-col means`) removes the artifact slightly more completely but is
cruder than a smooth blur.

### Methods compared (smooth-bg wins)

`examples/compare_delta_pdf_methods.py` puts three background-removal methods
side by side on a shared colour scale for H=0, 1/3, 2/3:

| method | what it does | effect on the cross |
| --- | --- | --- |
| **baseline** | subtract scalar mean only | cross present |
| **threshold-clip** | `I_new = max(I − c, 0)`, `c` = a percentile | **cross remains ≈ baseline** |
| **smooth-bg** | `I_new = I − GaussianBlur(I, σ≈1.5 rlu)` | **cross removed, lattice clean** |

Threshold-clip *sparsifies the input* (looks cleaner) but barely changes the
transform: on H=1/3 even keeping only 10 % of voxels drops the cross ratio from
~36 to ~11, versus ~6 for smooth-bg. It targets the wrong component — it removes
the dim background tails, but the cross is made by the **bright central
envelope**, which is the highest-intensity region and so survives any threshold.
It also adds hard-edge termination ripple and discards the negative excursions
of `I_diffuse` (regions with *fewer* pairs than average), which a ΔPDF needs.

Of the in-FFT options, **smooth-bg subtraction** (`subtract_smooth_bg` in
`compute_delta_pdf`; `SUBTRACT_BG=<σ rlu>` in `examples/delta_pdf_plane.py` /
`examples/delta_pdf.py`) is the one that removes the cross most completely.

### Background removal in the pipeline: the radial flatten (step 4)

The smooth-bg blur above removes the axis cross most completely, but the
per-H-plane form (`σ_H=0`, e.g. `0,1.5,1.5`) does so by subtracting each H
plane's integrated K–L intensity — which **is** the on-axis x_H Fourier
component. So it also **destroys the H-direction signal** (real lattice-`a`
peaks drop to ~1–3 %, for any σ; see the `radial_flatten` module and
`flatten-vs-subtractbg`).

The production pipeline (`examples/run_pipeline.py`) therefore removes the
background with an **explicit step 4**, the isotropic radial flatten
(`ndiff.preprocessing.flatten_radial_background`,
`examples/flatten_background_3d.py`), and leaves the in-FFT `SUBTRACT_BG` **off**
by default. The flatten subtracts a smooth `bg(|Q|)` per spherical shell without
touching per-plane DC, so it **preserves the on-axis H signal** while still
roughly halving the L=0 axis cross. The two are alternatives — never run both
(double subtraction, and the blur re-introduces the H-axis loss). Robustness of
the flatten is validated across 22/45/100 K by `examples/validate_flatten.py`.
Judge the effect on the L=0 (H–K) plane, where the methods diverge.
