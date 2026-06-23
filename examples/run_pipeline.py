"""End-to-end neutron-diffuse workflow: raw .nxs → 3D-ΔPDF → consistency QA.

Runs the processing stages in order, runs the back-FFT consistency check, then
opens the interactive viewers:

    1. remove_rings_3d.py     raw .nxs                  → *_ringremoved.h5
    2. punch_bragg_3d.py      *_ringremoved.h5          → *_braggpunched.h5
    3. backfill_bragg_3d.py   *_braggpunched.h5         → *_backfilled.h5
    4. flatten_background_3d.py  *_backfilled.h5        → *_flattened.h5
                              (isotropic radial-background flatten — the explicit
                               background-removal step; default ON, FLATTEN=0 to skip)
    5. delta_pdf.py           *_flattened.h5            → *_delta_pdf.h5
    6. delta_pdf_consistency.py  back-FFT ΔPDF check    → *_consistency.png
    7. explore_slice.py       (4-panel KL QA: data | ring removed | punched |
                               backfilled — H + vmin/vmax sliders)
    8. explore_delta_pdf_ortho.py   (ΔPDF real-space orthoslices, sliders)

Background removal is an explicit step (4 — the radial flatten), NOT a hidden
blur inside the FFT: the ΔPDF's own Gaussian `SUBTRACT_BG` is OFF by default
here.  The two are *alternative* background removers — running both removes the
background twice, and the per-H-plane blur (σ_H=0) destroys the on-axis
H-direction signal that the flatten preserves.

Each stage is **skipped if its output already exists** (resume), so re-running
only does the missing work.  Use FORCE / FORCE_FROM to recompute.

Defaults follow the validated `cc_on` presets in ROADMAP.md; the slice-wise
smooth-bg + crop + gaussian ΔPDF settings are the ones that gave the clean
maps.  Every individual stage's own env vars still pass through and override
these defaults.

Run::

    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
      python3 examples/run_pipeline.py

Env:
    DATA_FILE   raw input .nxs (default: auto-detect 22K mmm cc_sub_bkg in data/raw)
    FORCE       1 → recompute every stage even if its output exists
    FORCE_FROM  rings | punch | backfill | flatten | pdf | check — recompute from here on
    FLATTEN     radial-background flatten (step 4) — default ON.  FLATTEN=0 skips
                it (then NO background is removed unless you pass an explicit
                SUBTRACT_BG).  The flatten preserves the on-axis H-direction
                signal; judge its effect on the L=0 (H-K) plane.
    CONSISTENCY 1 → run the back-FFT consistency check after the ΔPDF (default).
                Set CONSISTENCY=0 to skip it.
    NO_VIEWER   1 → stop after the ΔPDF and consistency check are written (no GUI)
    RMAX        viewer half-window in Å (default 50)
    # ΔPDF knobs (override the defaults below):
    SUBTRACT_BG (default 0 = OFF; the step-4 flatten replaces it).  Set e.g.
                0,1.5,1.5 to use the legacy per-H-plane Gaussian blur instead —
                but do NOT combine it with the flatten (double subtraction; the
                σ_H=0 blur destroys the H-axis signal).
    CROP_H (4)  CROP_K (8)  CROP_L (15)  APODIZE (gaussian)  GAUSSIAN_SIGMA (0.4)
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
        # catch small-but-sharp weak Bragg at integer nodes (local-MAD prominence)
        "INTEGER_LOCAL_NMAD": "8",
        # protect the whole q=1/3 satellite family (integer±1/3) periodically
        "SEARCH_EXCLUDE_H_FRACTIONS": "0.3333,0.6667",
        "SEARCH_EXCLUDE_H_WIDTH": "0.08", "PREVIEW": "0",
    },
    "backfill": {"METHOD": "q_shell"},
    # isotropic radial-background flatten — the explicit step-4 background
    # remover (default ON).  floor (p25) keeps diffuse and is validated robust
    # across 22/45/100K (see examples/validate_flatten.py).
    "flatten": {
        "ESTIMATOR": "floor", "FLOOR_PCT": "25", "Q_STEP": "0.05",
        "SMOOTH": "0.10", "MIN_COUNT": "20",
    },
    # SUBTRACT_BG OFF by default: the step-4 flatten is the background remover.
    # The per-H-plane Gaussian blur is the legacy alternative — set it explicitly
    # to use it instead, but never together with the flatten (see the header).
    "pdf": {
        "SUBTRACT_BG": "0", "CROP_H": "4", "CROP_K": "8", "CROP_L": "15",
        "APODIZE": "gaussian", "GAUSSIAN_SIGMA": "0.4",
    },
    "qa": {"H_VALUE": "0.3333", "SLIDER_MIN": "0.0", "SLIDER_MAX": "1.0"},
    "viewer": {"RMAX": "50"},
}

# Keys this orchestrator sets explicitly per stage — must not leak between stages.
CHAIN_KEYS = ("DATA_FILE", "OUT_FILE", "PROC_FILE", "PDF_FILE")

FORCE = os.environ.get("FORCE", "0") == "1"
FORCE_FROM = os.environ.get("FORCE_FROM", "").strip().lower()
FLATTEN = os.environ.get("FLATTEN", "1") != "0"
CONSISTENCY = os.environ.get("CONSISTENCY", "1") != "0"
ORDER = ["rings", "punch", "backfill", "flatten", "pdf", "check"]
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
flatten_out = PROC / f"{fill_out.stem}_flattened.h5"
# The radial-flatten stage (step 4, default ON) feeds the ΔPDF; with FLATTEN=0
# the backfilled volume does.  pdf_out keeps a stable name either way —
# delta_pdf.py stamps the source file, so toggling FLATTEN recomputes a stale
# cache (see _pdf_is_current).
pdf_input = flatten_out if FLATTEN else fill_out
pdf_out = PROC / f"{fill_out.stem}_delta_pdf.h5"
consistency_png = PROC / f"{pdf_out.stem}_consistency.png"
PROC.mkdir(parents=True, exist_ok=True)

print(f"input : {raw.name}")
flatten_link = "→ flattened " if FLATTEN else ""
print(f"chain : ringremoved → braggpunched → backfilled {flatten_link}→ _delta_pdf.h5")

# ------------------------------------------------------------------
# stages 1–4 (rings, punch, backfill, background flatten)
# ------------------------------------------------------------------
_stage("rings", "1/8 ring removal", "remove_rings_3d.py", ring_out, "DATA_FILE", raw)
_stage("punch", "2/8 Bragg punch", "punch_bragg_3d.py", punch_out, "DATA_FILE", ring_out)
_stage("backfill", "3/8 Bragg backfill", "backfill_bragg_3d.py", fill_out, "DATA_FILE", punch_out)
if FLATTEN:
    _stage("flatten", "4/8 radial background flatten (background removal)",
           "flatten_background_3d.py", flatten_out, "DATA_FILE", fill_out)
else:
    print("[skip] 4/8 radial background flatten: FLATTEN=0 — background NOT "
          "removed unless an explicit SUBTRACT_BG is set", flush=True)

# ------------------------------------------------------------------
# stage 5: 3D-ΔPDF  (delta_pdf.py uses PROC_FILE in, fixed output _delta_pdf.h5)
# The output name is fixed, so guard against a STALE cache from a different
# dataset or transform configuration. delta_pdf.py stamps source_file and
# transform_config into the .h5; recompute unless both match this run.
# ------------------------------------------------------------------
def _param_string(value: str) -> str:
    return f"{float(value):.12g}" if value else ""


def _subtract_bg_config(value: str) -> str:
    if "," in value:
        return ",".join(_param_string(v) for v in value.split(","))
    return "" if not float(value or "0") else _param_string(value)


def _pdf_transform_config(env: dict) -> str:
    crop_hkl = ",".join(
        _param_string(env.get(k, ""))
        for k in ("CROP_H", "CROP_K", "CROP_L")
    )
    subtract_bg = _subtract_bg_config(env.get("SUBTRACT_BG", "0"))
    return ";".join(
        (
            f"apodize={env.get('APODIZE', 'hann')}",
            f"gaussian_sigma={_param_string(env.get('GAUSSIAN_SIGMA', '0.5'))}",
            f"zero_pad={int(env.get('ZERO_PAD', '1'))}",
            f"subtract_mean={int(env.get('SUBTRACT_MEAN', '1'))}",
            f"crop_hkl={crop_hkl}",
            f"subtract_bg={subtract_bg}",
        )
    )


def _pdf_is_current(pdf_path: Path, expected_src: str, expected_config: str) -> bool:
    if not pdf_path.exists():
        return False
    try:
        import h5py
        with h5py.File(pdf_path, "r") as fh:
            return (
                fh.attrs.get("source_file", "") == expected_src
                and fh.attrs.get("transform_config", "") == expected_config
            )
    except Exception:
        return False


pdf_env = _stage_env("pdf", PROC_FILE=pdf_input, OUT_FILE=pdf_out)
# Background removal is step 4 (the radial flatten).  The ΔPDF's own Gaussian
# SUBTRACT_BG is the *alternative* (legacy) remover and defaults OFF — running
# both subtracts the background twice, and the per-H-plane blur (σ_H=0) destroys
# the on-axis H signal the flatten preserves (validated: H-axis peaks → ~1-3%).
_sbg = pdf_env.get("SUBTRACT_BG", "0")
_sbg_on = any(float(v or 0) != 0 for v in _sbg.split(",")) if _sbg else False
if FLATTEN and _sbg_on:
    print(f"[warn] FLATTEN on AND SUBTRACT_BG={_sbg}: the background is being "
          "removed TWICE and the K-L blur will damage the on-axis H signal. Use "
          "one or the other (recommended: flatten only, SUBTRACT_BG=0).", flush=True)
print(f"[background] step-4 radial flatten: {'ON' if FLATTEN else 'OFF'}   |   "
      f"ΔPDF SUBTRACT_BG (legacy blur): {_sbg if _sbg_on else 'off'}", flush=True)
pdf_config = _pdf_transform_config(pdf_env)
ran_pdf = False
if _pdf_is_current(pdf_out, pdf_input.name, pdf_config) and not _forced("pdf"):
    print(f"[skip] 5/8 3D-ΔPDF: {pdf_out.name} is current for this dataset "
          "and transform config (FORCE=1 or FORCE_FROM=pdf to redo)", flush=True)
else:
    if pdf_out.exists():
        print(f"[stale] {pdf_out.name} is from a different dataset/config — recomputing",
              flush=True)
    _run("5/8 3D-ΔPDF", "delta_pdf.py", pdf_env)
    ran_pdf = True

# ------------------------------------------------------------------
# stage 6: back-FFT round-trip consistency check
# ------------------------------------------------------------------
if CONSISTENCY:
    if consistency_png.exists() and not ran_pdf and not _forced("check"):
        print(f"[skip] 6/8 consistency check: {consistency_png.name} exists "
              "(FORCE=1 or FORCE_FROM=check to redo)", flush=True)
    else:
        _run(
            "6/8 consistency check (back-FFT ΔPDF → reciprocal space)",
            "delta_pdf_consistency.py",
            _stage_env("pdf", DATA_FILE=pdf_input, OUT_PNG=consistency_png),
        )
else:
    print("[skip] 6/8 consistency check: CONSISTENCY=0", flush=True)

# ------------------------------------------------------------------
# stages 7–8: interactive viewers (close each window to advance)
# ------------------------------------------------------------------
if os.environ.get("NO_VIEWER", "0") == "1":
    print(f"\nNO_VIEWER=1 — done. ΔPDF cached at {pdf_out}", flush=True)
    sys.exit(0)

# 7/8 processed-data QA: 4-panel KL viewer (data | ring removed | punched |
# backfilled) with H + vmin/vmax sliders, loading the precomputed stages so
# nothing is recomputed.
print(f"\n{'='*70}\n▶ 7/8 processed-data QA viewer (KL plane; H + vmin/vmax sliders)"
      f"\n{'='*70}", flush=True)
qa_env = _stage_env("qa", DATA_FILE=raw, RING_FILE=ring_out,
                    PUNCH_FILE=punch_out, BACKFILL_FILE=fill_out)
subprocess.run([PY, str(HERE / "explore_slice.py")], env=qa_env, cwd=REPO)

# 8/8 ΔPDF real-space orthoslice viewer
print(f"\n{'='*70}\n▶ 8/8 3D-ΔPDF orthoslice viewer\n{'='*70}", flush=True)
viewer_env = _stage_env("viewer", PDF_FILE=pdf_out)
subprocess.run([PY, str(HERE / "explore_delta_pdf_ortho.py")], env=viewer_env, cwd=REPO)
print("workflow complete.", flush=True)
