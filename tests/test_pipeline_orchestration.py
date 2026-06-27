"""Tests for nebula3d.pipeline orchestration: chaining, resume, force, progress.

The heavy per-stage maths is covered elsewhere (test_pipeline.py,
test_radial_flatten.py, test_bragg.py, ...).  These tests pin the *orchestration*
contract of :func:`nebula3d.pipeline.run_pipeline`:

* the chained output file names match the original ``run_pipeline.py``;
* stages are skipped when their output already exists (resume);
* ``force_from`` recomputes the tail only;
* ``flatten_enabled=False`` routes the backfilled volume into the FFT;
* the progress callback fires per stage;
* the ΔPDF ``.h5`` is written in the schema the viewers read, with a working
  stale-cache guard.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import nebula3d
from nebula3d import pipeline
from nebula3d.core import HKLVolume

UB = 2 * np.pi * np.eye(3) / 4.0


def _vol(shape=(8, 8, 8), seed=0, fill=1.0):
    rng = np.random.default_rng(seed)
    data = fill + rng.normal(0.0, 0.01, shape)
    return HKLVolume.from_arrays(data, (-3, 3), (-3, 3), (-3, 3), ub_matrix=UB)


def _fake_dpdf(shape=(8, 8, 8)):
    """Minimal stand-in for a DeltaPDF (what write_delta_pdf_h5 reads)."""
    n = shape[0]
    ax = np.linspace(-10, 10, n)
    return SimpleNamespace(
        data=np.random.default_rng(1).normal(0, 1, shape),
        x_axis=ax, y_axis=ax, z_axis=ax, q_max=3.0, apodization="gaussian",
    )


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------
def test_pipeline_paths_match_run_pipeline_naming(tmp_path):
    paths = pipeline.pipeline_paths(tmp_path / "sample.nxs", proc_dir=tmp_path,
                                    flatten_enabled=True)
    assert paths.ringremoved.name == "sample_ringremoved.h5"
    assert paths.braggpunched.name == "sample_ringremoved_braggpunched.h5"
    assert paths.backfilled.name == "sample_ringremoved_braggpunched_backfilled.h5"
    assert paths.flattened.name == (
        "sample_ringremoved_braggpunched_backfilled_flattened.h5"
    )
    # ΔPDF name is derived from the *backfilled* stem regardless of flatten.
    assert paths.delta_pdf.name == (
        "sample_ringremoved_braggpunched_backfilled_delta_pdf.h5"
    )
    assert paths.pdf_input == paths.flattened


def test_pdf_input_is_backfilled_when_flatten_disabled(tmp_path):
    paths = pipeline.pipeline_paths(tmp_path / "s.nxs", proc_dir=tmp_path,
                                    flatten_enabled=False)
    assert paths.pdf_input == paths.backfilled


def _ring_vol_3d(shape=(5, 41, 41), ring_q=3.0, ring_fwhm=0.14, seed=0):
    """3-D volume with a textured powder ring in each 0kl plane (slice_axis='H')."""
    from nebula3d.preprocessing.parametric_ring import _pseudo_voigt

    rng = np.random.default_rng(seed)
    vol = HKLVolume.from_arrays(
        np.ones(shape), (-2, 2), (-4, 4), (-4, 4), ub_matrix=UB)
    q = vol.q_magnitude()
    _, K, L = vol.hkl_grid()
    Q = np.stack([np.zeros_like(K), K, L], axis=-1) @ UB.T
    phi = np.arctan2(Q[..., 2], Q[..., 1])
    diffuse = 1.0 + 0.2 * np.cos(np.pi * K) * np.cos(np.pi * L)
    ring = (1.0 + 0.4 * np.cos(2 * phi)) * 3.0 * _pseudo_voigt(q, ring_q, ring_fwhm, 0.5)
    data = diffuse + ring + rng.normal(0, 0.02, shape)
    return HKLVolume.from_arrays(
        data, (-2, 2), (-4, 4), (-4, 4), ub_matrix=UB), q


def test_remove_rings_parametric_runs_and_suppresses_ring():
    """The ring_model='parametric' branch runs through the per-slice 3-D driver
    and subtracts the planted ring while preserving the off-ring diffuse."""
    vol, q = _ring_vol_3d()
    params = pipeline.RingParams(
        ring_model="parametric", slice_axis="H", q_min=1.0, q_max=8.0,
        confirm_rings=False)
    out = pipeline.remove_rings(vol, params)

    assert out.data.shape == vol.data.shape
    assert np.isfinite(out.data).all()

    on_lo, on_hi = 2.9, 3.1
    base = float(np.median(vol.data[(q > 1.3) & (q < 1.9)]))
    before = float(np.median(vol.data[(q > on_lo) & (q < on_hi)])) - base
    after = float(np.median(out.data[(q > on_lo) & (q < on_hi)])) - base
    assert before > 0.3
    assert after < 0.4 * before


def test_transform_config_string_format():
    cfg = pipeline.delta_pdf_transform_config(
        pipeline.DeltaPdfParams(apodization="gaussian", gaussian_sigma=0.4,
                                zero_pad=True, subtract_mean=True,
                                crop_hkl=(4.0, 8.0, 15.0), subtract_smooth_bg=None)
    )
    assert cfg == (
        "apodize=gaussian;gaussian_sigma=0.4;zero_pad=1;subtract_mean=1;"
        "crop_hkl=4,8,15;q_band=;subtract_bg="
    )


# ---------------------------------------------------------------------------
# orchestration with stubbed stage compute (fast, deterministic)
# ---------------------------------------------------------------------------
@pytest.fixture
def stubbed(monkeypatch):
    """Replace the five stage compute fns with fast recording stubs."""
    calls = []
    vol = _vol()

    def mk(name):
        def stub(v, params=None, *, progress=None):
            calls.append(name)
            return vol
        return stub

    monkeypatch.setattr(pipeline, "remove_rings", mk("rings"))
    monkeypatch.setattr(pipeline, "punch_bragg", mk("punch"))
    monkeypatch.setattr(pipeline, "backfill", mk("backfill"))
    monkeypatch.setattr(pipeline, "flatten", mk("flatten"))

    def pdf_stub(v, params=None, *, progress=None):
        calls.append("pdf")
        return _fake_dpdf()

    monkeypatch.setattr(pipeline, "delta_pdf", pdf_stub)

    # Stage 6 (pdf_check) — stub the back-FFT comparison so orchestration tests
    # stay fast and don't need a real invertible DeltaPDF; it touches the figure
    # so the resume/skip guard (json+png both exist) behaves.
    def pdf_check_stub(vol, dpdf, params, *, h_values=(0.0,), figure_path=None):
        calls.append("pdf_check")
        if figure_path is not None:
            Path(figure_path).write_bytes(b"")
        return {"pearson_r": 1.0, "normalized_rms": 0.0}

    monkeypatch.setattr(pipeline, "pdf_consistency_check", pdf_check_stub)
    return SimpleNamespace(calls=calls, vol=vol)


def _seed_input(tmp_path):
    inp = tmp_path / "sample.nxs"
    nebula3d.save(_vol(), tmp_path / "sample.nxs")
    return inp


def test_run_pipeline_runs_all_stages_in_order(tmp_path, stubbed):
    inp = _seed_input(tmp_path)
    events = []
    paths = pipeline.run_pipeline(
        inp, proc_dir=tmp_path,
        progress=lambda *a: events.append(a),
    )
    assert stubbed.calls == [
        "rings", "punch", "backfill", "flatten", "pdf", "pdf_check"]
    for p in (paths.ringremoved, paths.braggpunched, paths.backfilled,
              paths.flattened, paths.delta_pdf,
              paths.pdf_check_json, paths.pdf_check_png):
        assert p.exists(), p
    # nothing skipped on a fresh run
    assert not any(ev[1] == "skip" for ev in events)


def test_run_pipeline_resumes_and_skips_existing(tmp_path, stubbed):
    inp = _seed_input(tmp_path)
    pipeline.run_pipeline(inp, proc_dir=tmp_path)        # first run writes outputs
    stubbed.calls.clear()

    events = []
    pipeline.run_pipeline(inp, proc_dir=tmp_path,
                          progress=lambda *a: events.append(a))
    assert stubbed.calls == []                           # everything skipped
    skipped = {ev[0] for ev in events if ev[1] == "skip"}
    assert skipped == {
        "rings", "punch", "backfill", "flatten", "pdf", "pdf_check"}


def test_force_from_recomputes_tail_only(tmp_path, stubbed):
    inp = _seed_input(tmp_path)
    pipeline.run_pipeline(inp, proc_dir=tmp_path)
    stubbed.calls.clear()

    pipeline.run_pipeline(inp, proc_dir=tmp_path, force_from="backfill")
    assert stubbed.calls == ["backfill", "flatten", "pdf", "pdf_check"]


def test_force_recomputes_everything(tmp_path, stubbed):
    inp = _seed_input(tmp_path)
    pipeline.run_pipeline(inp, proc_dir=tmp_path)
    stubbed.calls.clear()

    pipeline.run_pipeline(inp, proc_dir=tmp_path, force=True)
    assert stubbed.calls == [
        "rings", "punch", "backfill", "flatten", "pdf", "pdf_check"]


def test_stage_subset_runs_only_requested(tmp_path, stubbed):
    inp = _seed_input(tmp_path)
    # seed the punched file so backfill has an input to load
    nebula3d.save(stubbed.vol, pipeline.pipeline_paths(inp, proc_dir=tmp_path).braggpunched)
    pipeline.run_pipeline(inp, proc_dir=tmp_path, stages=("backfill",))
    assert stubbed.calls == ["backfill"]


def test_disabled_stage_passes_input_through(tmp_path, stubbed):
    """A disabled cleanup stage is skipped and its input flows to the next
    enabled stage: with rings off, punch runs on the raw input and the
    ring-removed output is never written."""
    inp = _seed_input(tmp_path)
    paths = pipeline.pipeline_paths(inp, proc_dir=tmp_path)
    pipeline.run_pipeline(
        inp, proc_dir=tmp_path,
        stages=("punch", "backfill", "flatten", "pdf", "pdf_check"),
    )
    assert "rings" not in stubbed.calls          # disabled stage skipped
    assert stubbed.calls[0] == "punch"           # punch ran first (on the raw input)
    assert not paths.ringremoved.exists()        # disabled → no output written
    assert paths.braggpunched.exists()           # downstream still produced
    assert paths.delta_pdf.exists()


def test_flatten_disabled_routes_backfilled_into_pdf(tmp_path, stubbed):
    import h5py

    inp = _seed_input(tmp_path)
    params = pipeline.PipelineParams(flatten_enabled=False)
    paths = pipeline.run_pipeline(inp, params, proc_dir=tmp_path)

    assert not paths.flattened.exists()
    assert "flatten" not in stubbed.calls
    with h5py.File(paths.delta_pdf, "r") as fh:
        assert fh.attrs["source_file"] == paths.backfilled.name


def test_invalid_force_from_raises(tmp_path):
    with pytest.raises(ValueError, match="force_from"):
        pipeline.run_pipeline(tmp_path / "x.nxs", proc_dir=tmp_path,
                              force_from="nonsense")


# ---------------------------------------------------------------------------
# real stage calls (backfill → flatten → pdf) on a small synthetic volume
# ---------------------------------------------------------------------------
def _punched_vol(shape=(20, 20, 20)):
    rng = np.random.default_rng(0)
    vol = HKLVolume.from_arrays(
        np.zeros(shape), (-3, 3), (-3, 3), (-3, 3), ub_matrix=UB)
    q = vol.q_magnitude()
    vol.data[...] = 5.0 * np.exp(-q / 3.0) + 0.5 + rng.normal(0, 0.02, shape)
    # punch a few interior holes (so q_shell backfill has shell neighbours)
    for idx in [(9, 9, 9), (10, 9, 8), (8, 11, 10), (11, 10, 9)]:
        vol.data[idx] = -99.0
        vol.mask[idx] = False
    return vol


def test_real_backfill_flatten_pdf_writes_viewer_schema(tmp_path):
    import h5py

    inp = tmp_path / "s.nxs"
    paths = pipeline.pipeline_paths(inp, proc_dir=tmp_path)
    nebula3d.save(_punched_vol(), paths.braggpunched)

    params = pipeline.PipelineParams(
        delta_pdf=pipeline.DeltaPdfParams(apodization="hann", zero_pad=False,
                                          crop_hkl=None),
    )
    events = []
    pipeline.run_pipeline(inp, params, proc_dir=tmp_path,
                          stages=("backfill", "flatten", "pdf"),
                          progress=lambda *a: events.append(a))

    # backfill produced an all-finite volume
    filled = nebula3d.load(paths.backfilled)
    assert np.isfinite(filled.data).all()
    assert paths.flattened.exists()

    # ΔPDF written in the schema the viewers read
    with h5py.File(paths.delta_pdf, "r") as fh:
        for key in ("data", "x_axis", "y_axis", "z_axis"):
            assert key in fh
        assert fh.attrs["source_file"] == paths.flattened.name
        for latk in ("lat_a", "lat_b", "lat_c"):
            assert latk in fh.attrs
            assert np.isclose(float(fh.attrs[latk]), 4.0, atol=1e-6)  # 2π/(2π/4)

    # per-stage start/done events fired
    pairs = {(ev[0], ev[1]) for ev in events}
    for stage in ("backfill", "flatten", "pdf"):
        assert (stage, "start") in pairs
        assert (stage, "done") in pairs


def test_real_pdf_stale_guard_skips_on_rerun(tmp_path):
    inp = tmp_path / "s.nxs"
    paths = pipeline.pipeline_paths(inp, proc_dir=tmp_path)
    nebula3d.save(_punched_vol(), paths.braggpunched)
    params = pipeline.PipelineParams(
        delta_pdf=pipeline.DeltaPdfParams(apodization="hann", zero_pad=False,
                                          crop_hkl=None),
    )
    kw = dict(proc_dir=tmp_path, stages=("backfill", "flatten", "pdf"))
    pipeline.run_pipeline(inp, params, **kw)

    events = []
    pipeline.run_pipeline(inp, params, progress=lambda *a: events.append(a), **kw)
    skipped = {ev[0] for ev in events if ev[1] == "skip"}
    assert {"backfill", "flatten", "pdf"} <= skipped


def _diffuse_vol(n=15):
    """A clean centrosymmetric diffuse volume on an odd grid (exact round trip)."""
    vol = HKLVolume.from_arrays(
        np.zeros((n, n, n)), (-3, 3), (-3, 3), (-3, 3), ub_matrix=UB)
    vol.data[...] = 5.0 * np.exp(-vol.q_magnitude() / 2.0) + 1.0
    return vol


def test_real_pdf_check_roundtrip_metrics(tmp_path):
    """The real pdf_check stage inverts the ΔPDF and recovers the diffuse data."""
    import json

    inp = tmp_path / "s.nxs"
    params = pipeline.PipelineParams(
        flatten_enabled=False,
        delta_pdf=pipeline.DeltaPdfParams(apodization="gaussian", zero_pad=False,
                                          crop_hkl=None),
    )
    paths = pipeline.pipeline_paths(inp, proc_dir=tmp_path, flatten_enabled=False)
    nebula3d.save(_diffuse_vol(15), paths.backfilled)   # = pdf_input when flatten off

    events = []
    pipeline.run_pipeline(inp, params, proc_dir=tmp_path,
                          stages=("pdf", "pdf_check"),
                          progress=lambda *a: events.append(a))

    assert paths.pdf_check_json.exists()
    assert paths.pdf_check_png.exists()
    assert ("pdf_check", "done") in {(ev[0], ev[1]) for ev in events}
    metrics = json.loads(paths.pdf_check_json.read_text())
    assert metrics["pearson_r"] > 0.999        # gaussian window → near-exact
    assert metrics["normalized_rms"] < 1e-2
