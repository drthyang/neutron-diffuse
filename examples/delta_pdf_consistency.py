"""ΔPDF round-trip consistency check: inverse-FFT the 3D-ΔPDF, compare to data.

Takes the cleaned diffuse volume that feeds the ΔPDF stage (the *flattened*
volume by default), computes the 3D-ΔPDF with the pipeline's defaults, then
**inverse-transforms it back to reciprocal space** with
:func:`nebula3d.analysis.invert_delta_pdf` and compares the reconstruction against
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

Env: ``DATA_FILE`` (input volume; default = newest ``*_flattened.h5``),
``H_VALUES`` (comma list, default ``0,0.3333,1.0``), ``OUT_PNG``, and the
standard ΔPDF knobs: ``CROP_H/K/L``, ``APODIZE``, ``GAUSSIAN_SIGMA``,
``ZERO_PAD``, ``SUBTRACT_MEAN``, ``SUBTRACT_BG``.
"""

from __future__ import annotations

import os
from pathlib import Path

import nebula3d
from nebula3d.analysis import compute_delta_pdf
from nebula3d.pipeline import DeltaPdfParams, pdf_consistency_check


def _find_input() -> Path:
    df = os.environ.get("DATA_FILE")
    if df:
        return Path(df)
    proc = Path("data/processed")
    cands = (sorted(proc.glob("*flattened.h5"))
             or sorted(proc.glob("*backfilled.h5")))
    if not cands:
        raise FileNotFoundError(
            "No ΔPDF-stage input found in data/processed; set DATA_FILE=…")
    return cands[-1]


def _optional_float(name: str) -> float | None:
    val = os.environ.get(name)
    return None if val is None or val == "" else float(val)


def _subtract_bg_from_env() -> float | tuple[float, float, float] | None:
    val = os.environ.get("SUBTRACT_BG")
    if val is None or val.strip() in {"", "0"}:
        return None
    if "," in val:
        parts = tuple(float(x) for x in val.split(","))
        if len(parts) != 3:
            raise ValueError("SUBTRACT_BG must be a scalar or three comma-separated values")
        return parts
    return float(val)


def _params_from_env() -> DeltaPdfParams:
    crop = tuple(_optional_float(k) for k in ("CROP_H", "CROP_K", "CROP_L"))
    crop_hkl = None if all(v is None for v in crop) else tuple(
        0.0 if v is None else v for v in crop)
    return DeltaPdfParams(
        apodization=os.environ.get("APODIZE", "gaussian"),
        gaussian_sigma=float(os.environ.get("GAUSSIAN_SIGMA", "0.4")),
        zero_pad=os.environ.get("ZERO_PAD", "1") != "0",
        subtract_mean=os.environ.get("SUBTRACT_MEAN", "1") != "0",
        crop_hkl=crop_hkl,  # type: ignore[arg-type]
        subtract_smooth_bg=_subtract_bg_from_env(),
    )


def main() -> None:
    in_path = _find_input()
    print(f"loading {in_path.name}", flush=True)
    vol = nebula3d.load(in_path)

    p = _params_from_env()
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
