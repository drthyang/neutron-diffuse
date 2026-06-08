"""End-to-end 3D-PDF workflow — KEEP the Bragg peaks (no punch, no backfill).

This is the **3D-PDF** counterpart to ``examples/run_pipeline.py`` (which is the
3D-ΔPDF workflow).  The ΔPDF removes the Bragg peaks (punch → backfill) to isolate
the diffuse; here we deliberately **skip** those two stages and Fourier-transform
the **total** scattering, Bragg included, giving a Patterson-like 3D-PDF of the
average-structure correlations plus the diffuse.

Stages::

    1. remove_rings_3d.py   raw .nxs          → *_ringremoved.h5   (optional)
    2. pdf_3d.py            *_ringremoved.h5  → *_3dpdf.h5         (FFT, Bragg KEPT)
    3. explore_delta_pdf_ortho.py             (interactive orthoslice viewer)

There is **no Bragg punch and no backfill** — that is the whole point of this
workflow.  Each stage is skipped if its output already exists (resume).  The
ring-removed files are the SAME ones ``run_pipeline.py`` writes, so if you have
already run the ΔPDF workflow, stage 1 here is instant.

Run::

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \\
      python3 examples/run_pipeline_pdf.py

Env:
    DATA_FILE     raw input .nxs (default: auto-detect 22K mmm cc_sub_bkg)
    RING_REMOVAL  1 (default) ring-remove first; 0 → FFT the raw data directly
    FORCE         1 → recompute every stage
    FORCE_FROM    rings | pdf — recompute from this stage on
    NO_VIEWER     1 → stop after the *_3dpdf.h5 is written (no GUI)
    RMAX          viewer half-window in Å (default 50)
    # 3D-PDF FFT knobs (note: SUBTRACT_BG is OFF — that is a ΔPDF-only trick):
    APODIZE (gaussian)  GAUSSIAN_SIGMA (0.4)  SUBTRACT_MEAN (1)
    CROP_H (4)  CROP_K (8)  CROP_L (15)  SUBTRACT_BG (0)
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PROC = REPO / "data" / "processed"
RAW = REPO / "data" / "raw"
PY = sys.executable

# Per-stage defaults.  The PDF stage mirrors the ΔPDF transform settings for an
# apples-to-apples comparison EXCEPT SUBTRACT_BG, which is off for the total PDF.
STAGE_DEFAULTS = {
    "rings": {"RING_PRESET": "cc_on"},
    "pdf": {
        "APODIZE": "gaussian", "GAUSSIAN_SIGMA": "0.4", "SUBTRACT_MEAN": "1",
        "CROP_H": "4", "CROP_K": "8", "CROP_L": "15", "SUBTRACT_BG": "0",
    },
    "viewer": {"RMAX": "50"},
}
CHAIN_KEYS = ("DATA_FILE", "OUT_FILE", "PROC_FILE", "PDF_FILE")

FORCE = os.environ.get("FORCE", "0") == "1"
FORCE_FROM = os.environ.get("FORCE_FROM", "").strip().lower()
RING_REMOVAL = os.environ.get("RING_REMOVAL", "1") != "0"
ORDER = ["rings", "pdf"]
if FORCE_FROM and FORCE_FROM not in ORDER:
    sys.exit(f"FORCE_FROM={FORCE_FROM!r}; choose one of {ORDER}")


def _forced(stage: str) -> bool:
    if FORCE:
        return True
    if FORCE_FROM:
        return ORDER.index(stage) >= ORDER.index(FORCE_FROM)
    return False


def _detect_raw() -> Path:
    env = os.environ.get("DATA_FILE")
    if env:
        return Path(env)

    def is_empty_bkg(p: Path) -> bool:
        return p.stem.endswith("_bkg") and not p.stem.endswith(("_sub_bkg", "_cc_sub_bkg"))

    cands = [p for p in sorted(RAW.glob("*.nxs")) if not is_empty_bkg(p)]
    if not cands:
        sys.exit("No input .nxs in data/raw; set DATA_FILE=/path/to/input.nxs.")
    return next(
        (p for p in cands if "22K_mmm" in p.stem and "cc_sub_bkg" in p.stem),
        next((p for p in cands if "22K_mmm" in p.stem), cands[0]),
    )


def _stage_env(stage: str, **explicit) -> dict:
    env = os.environ.copy()
    for k in CHAIN_KEYS:
        env.pop(k, None)
    env["PYTHONPATH"] = str(REPO / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    for k, v in STAGE_DEFAULTS.get(stage, {}).items():
        env.setdefault(k, v)
    env.update({k: str(v) for k, v in explicit.items()})
    return env


def _run(label: str, script: str, env: dict) -> None:
    print(f"\n{'='*70}\n▶ {label}\n{'='*70}", flush=True)
    r = subprocess.run([PY, str(HERE / script)], env=env, cwd=REPO)
    if r.returncode != 0:
        sys.exit(f"stage '{label}' failed (exit {r.returncode}).")


# ------------------------------------------------------------------
# resolve chained paths
# ------------------------------------------------------------------
raw = _detect_raw()
if not raw.exists():
    sys.exit(f"input not found: {raw}")
ring_out = PROC / f"{raw.stem}_ringremoved.h5"
fft_input = ring_out if RING_REMOVAL else raw
pdf_out = PROC / f"{fft_input.stem}_3dpdf.h5"
PROC.mkdir(parents=True, exist_ok=True)

print(f"input : {raw.name}")
print(f"chain : {'ringremoved → ' if RING_REMOVAL else ''}3D-PDF (Bragg KEPT — "
      "no punch, no backfill)")

# ------------------------------------------------------------------
# stage 1: ring removal (optional)
# ------------------------------------------------------------------
if RING_REMOVAL:
    if ring_out.exists() and not _forced("rings"):
        print(f"[skip] 1/2 ring removal: {ring_out.name} exists "
              "(FORCE=1 or FORCE_FROM=rings to redo)", flush=True)
    else:
        _run("1/2 ring removal", "remove_rings_3d.py",
             _stage_env("rings", DATA_FILE=raw, OUT_FILE=ring_out))
else:
    print("[skip] ring removal disabled (RING_REMOVAL=0) — FFT the raw data", flush=True)

# ------------------------------------------------------------------
# stage 2: 3D-PDF FFT (NO Bragg punch, NO backfill)
# ------------------------------------------------------------------
if pdf_out.exists() and not _forced("pdf"):
    print(f"[skip] 2/2 3D-PDF: {pdf_out.name} exists "
          "(FORCE=1 or FORCE_FROM=pdf to redo)", flush=True)
else:
    _run("2/2 3D-PDF (Bragg kept)", "pdf_3d.py",
         _stage_env("pdf", PROC_FILE=fft_input, OUT_FILE=pdf_out))

# ------------------------------------------------------------------
# stage 3: interactive viewer
# ------------------------------------------------------------------
if os.environ.get("NO_VIEWER", "0") == "1":
    print(f"\nNO_VIEWER=1 — done. 3D-PDF cached at {pdf_out}", flush=True)
    sys.exit(0)

print(f"\n{'='*70}\n▶ 3/3 3D-PDF orthoslice viewer\n{'='*70}", flush=True)
subprocess.run([PY, str(HERE / "explore_delta_pdf_ortho.py")],
               env=_stage_env("viewer", PDF_FILE=pdf_out), cwd=REPO)
print("workflow complete.", flush=True)
