"""End-to-end neutron-diffuse workflow: raw .nxs → 3D-ΔPDF → interactive viewer.

Runs the four processing stages in order, then opens the orthoslice viewer:

    1. remove_rings_3d.py     raw .nxs                  → *_ringremoved.h5
    2. punch_bragg_3d.py      *_ringremoved.h5          → *_braggpunched.h5
    3. backfill_bragg_3d.py   *_braggpunched.h5         → *_backfilled.h5
    4. delta_pdf.py           *_backfilled.h5           → examples/_delta_pdf.h5
    5. explore_slice.py       (4-panel KL QA: data | ring removed | punched |
                               backfilled — H + vmin/vmax sliders)
    6. explore_delta_pdf_ortho.py   (ΔPDF real-space orthoslices, sliders)

Each stage is **skipped if its output already exists** (resume), so re-running
only does the missing work.  Use FORCE / FORCE_FROM to recompute.

Defaults follow the validated `cc_on` presets in HANDOFF.md; the slice-wise
smooth-bg + crop + gaussian ΔPDF settings are the ones that gave the clean
maps.  Every individual stage's own env vars still pass through and override
these defaults.

Run::

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
      /Users/tt9/miniforge3/envs/rmc-discord/bin/python3 examples/run_pipeline.py

Env:
    DATA_FILE   raw input .nxs (default: auto-detect 22K mmm cc_sub_bkg in data/raw)
    FORCE       1 → recompute every stage even if its output exists
    FORCE_FROM  rings | punch | backfill | pdf — recompute from this stage on
    NO_VIEWER   1 → stop after the ΔPDF is written (no GUI)
    RMAX        viewer half-window in Å (default 50)
    # ΔPDF knobs (override the defaults below):
    SUBTRACT_BG (default 0,1.5,1.5)  CROP_K (8)  CROP_L (15)
    APODIZE (gaussian)  GAUSSIAN_SIGMA (0.4)
    # plus every stage's own env vars (RING_PRESET, MODE, METHOD, ...).
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

# Per-stage default presets (the validated cc_on / clean-ΔPDF settings).
# User-set env vars take precedence (these are only applied where unset).
STAGE_DEFAULTS = {
    "rings": {"RING_PRESET": "cc_on"},
    "punch": {
        "PUNCH_PRESET": "cc_on", "MODE": "both",
        "MIN_I": "0.8", "MIN_PROM": "0.8",
        "INTEGER_FIT_POSITION": "1", "INTEGER_FIT_SHAPE": "1",
        "INTEGER_H_GUARD": "0.12",
        "SEARCH_EXCLUDE_H": "-0.6667,-0.3333,0.3333,0.6667",
        "SEARCH_EXCLUDE_H_WIDTH": "0.08", "PREVIEW": "0",
    },
    "backfill": {"METHOD": "q_shell"},
    "pdf": {
        "SUBTRACT_BG": "0,1.5,1.5", "CROP_K": "8", "CROP_L": "15",
        "APODIZE": "gaussian", "GAUSSIAN_SIGMA": "0.4",
    },
    "qa": {"H_VALUE": "0.3333", "SLIDER_MIN": "0.0", "SLIDER_MAX": "1.0"},
    "viewer": {"RMAX": "50"},
}

# Keys this orchestrator sets explicitly per stage — must not leak between stages.
CHAIN_KEYS = ("DATA_FILE", "OUT_FILE", "PROC_FILE", "PDF_FILE")

FORCE = os.environ.get("FORCE", "0") == "1"
FORCE_FROM = os.environ.get("FORCE_FROM", "").strip().lower()
ORDER = ["rings", "punch", "backfill", "pdf"]
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
    """Build subprocess env: inherit, strip chain keys, apply defaults, set explicit."""
    env = os.environ.copy()
    for k in CHAIN_KEYS:
        env.pop(k, None)
    # ensure imports + headless matplotlib config work in the child
    src = str(REPO / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
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


def _stage(stage, label, script, out_path, in_key, in_path):
    if out_path.exists() and not _forced(stage):
        print(f"[skip] {label}: {out_path.name} exists "
              f"(FORCE=1 or FORCE_FROM={stage} to redo)", flush=True)
        return
    _run(label, script, _stage_env(stage, **{in_key: in_path, "OUT_FILE": out_path}))


# ------------------------------------------------------------------
# resolve the chained file paths
# ------------------------------------------------------------------
raw = _detect_raw()
if not raw.exists():
    sys.exit(f"input not found: {raw}")
ring_out = PROC / f"{raw.stem}_ringremoved.h5"
punch_out = PROC / f"{ring_out.stem}_braggpunched.h5"
fill_out = PROC / f"{punch_out.stem}_backfilled.h5"
pdf_out = HERE / "_delta_pdf.h5"
PROC.mkdir(parents=True, exist_ok=True)

print(f"input : {raw.name}")
print(f"chain : ringremoved → braggpunched → backfilled → _delta_pdf.h5")

# ------------------------------------------------------------------
# stages 1–3
# ------------------------------------------------------------------
_stage("rings", "1/5 ring removal", "remove_rings_3d.py", ring_out, "DATA_FILE", raw)
_stage("punch", "2/5 Bragg punch", "punch_bragg_3d.py", punch_out, "DATA_FILE", ring_out)
_stage("backfill", "3/5 Bragg backfill", "backfill_bragg_3d.py", fill_out, "DATA_FILE", punch_out)

# ------------------------------------------------------------------
# stage 4: 3D-ΔPDF  (delta_pdf.py uses PROC_FILE in, fixed output _delta_pdf.h5)
# The output name is fixed, so guard against a STALE cache from a different
# dataset: delta_pdf.py stamps source_file into the .h5; recompute unless it
# matches this run's backfilled input.
# ------------------------------------------------------------------
def _pdf_is_current(pdf_path: Path, expected_src: str) -> bool:
    if not pdf_path.exists():
        return False
    try:
        import h5py
        with h5py.File(pdf_path, "r") as fh:
            return fh.attrs.get("source_file", "") == expected_src
    except Exception:
        return False


if _pdf_is_current(pdf_out, fill_out.name) and not _forced("pdf"):
    print(f"[skip] 4/5 3D-ΔPDF: {pdf_out.name} is current for this dataset "
          "(FORCE=1 or FORCE_FROM=pdf to redo)", flush=True)
else:
    if pdf_out.exists():
        print(f"[stale] {pdf_out.name} is from a different dataset — recomputing",
              flush=True)
    _run("4/5 3D-ΔPDF", "delta_pdf.py", _stage_env("pdf", PROC_FILE=fill_out))

# ------------------------------------------------------------------
# stages 5–6: interactive viewers (close each window to advance)
# ------------------------------------------------------------------
if os.environ.get("NO_VIEWER", "0") == "1":
    print(f"\nNO_VIEWER=1 — done. ΔPDF cached at {pdf_out}", flush=True)
    sys.exit(0)

# 5/6 processed-data QA: 4-panel KL viewer (data | ring removed | punched |
# backfilled) with H + vmin/vmax sliders, loading the precomputed stages so
# nothing is recomputed.
print(f"\n{'='*70}\n▶ 5/6 processed-data QA viewer (KL plane; H + vmin/vmax sliders)"
      f"\n{'='*70}", flush=True)
qa_env = _stage_env("qa", DATA_FILE=raw, RING_FILE=ring_out,
                    PUNCH_FILE=punch_out, BACKFILL_FILE=fill_out)
subprocess.run([PY, str(HERE / "explore_slice.py")], env=qa_env, cwd=REPO)

# 6/6 ΔPDF real-space orthoslice viewer
print(f"\n{'='*70}\n▶ 6/6 3D-ΔPDF orthoslice viewer\n{'='*70}", flush=True)
viewer_env = _stage_env("viewer", PDF_FILE=pdf_out)
subprocess.run([PY, str(HERE / "explore_delta_pdf_ortho.py")], env=viewer_env, cwd=REPO)
print("workflow complete.", flush=True)
