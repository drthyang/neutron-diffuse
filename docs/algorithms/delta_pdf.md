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

## Near-origin artifact (expected, not the bug)

A strong feature at `r < ~3 Å` and a faint cross along the principal axes remain
after the fix. These come from residual high-`|Q|` Bragg leakage, the backfill
discontinuities at punch boundaries, and the direct-beam punch — not from the
transform itself. Plot colour scales are set from the `p99` of `|ΔPDF|` at
`r > 3 Å` so this near-origin spike does not dominate the display.
