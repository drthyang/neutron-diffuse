"""ΔPDF round-trip consistency check: inverse-FFT the 3D-ΔPDF, compare to data.

Takes the cleaned diffuse volume that feeds the ΔPDF stage (the *flattened*
volume by default), computes the 3D-ΔPDF with the pipeline's defaults, then
**inverse-transforms it back to reciprocal space** with
:func:`ndiff.analysis.invert_delta_pdf` and compares the reconstruction against
the input.

Why this is a consistency check, not a tautology
------------------------------------------------
The forward transform is invertible (centred FFT + a gaussian apodization that
never reaches zero), so a faithful ΔPDF *must* back-transform to the diffuse data
it came from.  Where the reconstruction departs from the data localises exactly
what the transform settings throw away:

* ``crop_hkl`` discards the high-|Q| shell (the reconstruction has no information
  there);
* on an even grid the real-part projection drops a small asymmetric part;
* a too-aggressive apodization / ``subtract_smooth_bg`` suppresses real features.

A wrong axis, sign flip, or normalisation bug in the transform pair would show up
here as a gross mismatch — which is the point.

Outputs
-------
* stdout: global Pearson correlation and normalised RMS residual between the
  back-FFT reconstruction and the data (over the reliably-recovered region), plus
  per-plane correlations.
* a PNG: rows = representative reciprocal H planes, columns =
  ``data | back-FFT reconstruction | residual`` (data − reconstruction).

Usage::

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
    python3 examples/delta_pdf_consistency.py

Env: ``DATA_FILE`` (input volume; default = newest 22K ``*_flattened.h5``),
``H_VALUES`` (comma list, default ``0,0.3333,1.0``), ``OUT_PNG``.
"""

from __future__ import annotations

import os
from pathlib import Path

import ndiff
from ndiff.analysis import compute_delta_pdf
from ndiff.pipeline import DeltaPdfParams, pdf_consistency_check


def _find_input() -> Path:
    df = os.environ.get("DATA_FILE")
    if df:
        return Path(df)
    proc = Path("data/processed")
    cands = (sorted(proc.glob("*22K*cc_sub_bkg*flattened.h5"))
             or sorted(proc.glob("*22K*flattened.h5"))
             or sorted(proc.glob("*flattened.h5"))
             or sorted(proc.glob("*backfilled.h5")))
    if not cands:
        raise FileNotFoundError(
            "No ΔPDF-stage input found in data/processed; set DATA_FILE=…")
    return cands[-1]


def main() -> None:
    in_path = _find_input()
    print(f"loading {in_path.name}", flush=True)
    vol = ndiff.load(in_path)

    p = DeltaPdfParams()
    print(f"ΔPDF params: apodize={p.apodization} sigma={p.gaussian_sigma} "
          f"crop_hkl={p.crop_hkl} subtract_bg={p.subtract_smooth_bg}", flush=True)

    dpdf = compute_delta_pdf(
        vol, apodization=p.apodization,  # type: ignore[arg-type]
        gaussian_sigma=p.gaussian_sigma, zero_pad=p.zero_pad,
        subtract_mean=p.subtract_mean, real_space_angstrom=True,
        crop_hkl=p.crop_hkl, subtract_smooth_bg=p.subtract_smooth_bg)
    print(f"ΔPDF shape {dpdf.data.shape}", flush=True)

    # Inverse-FFT back to reciprocal space and compare to the diffuse data — the
    # same library routine the pipeline's pdf_check stage runs.
    h_values = tuple(float(x) for x in
                     os.environ.get("H_VALUES", "0,0.3333,1.0").split(","))
    out_png = Path(os.environ.get(
        "OUT_PNG", "data/processed/delta_pdf_consistency.png"))
    metrics = pdf_consistency_check(
        vol, dpdf, p, h_values=h_values, figure_path=out_png)

    print(f"\nback-FFT vs data  (reliable region, {metrics['n_voxels']:,} voxels):")
    print(f"  Pearson r       = {metrics['pearson_r']:.5f}")
    print(f"  normalised RMS  = {metrics['normalized_rms']:.3e}  "
          f"(RMS {metrics['rms']:.3e})")
    for hk, rho in metrics["per_plane_r"].items():
        print(f"  H={hk}: r = {rho:.5f}", flush=True)
    print(f"\nwrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
