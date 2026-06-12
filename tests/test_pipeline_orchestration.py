"""Tests for ndiff.pipeline orchestration: chaining, resume, force, progress.

The heavy per-stage maths is covered elsewhere (test_pipeline.py,
test_radial_flatten.py, test_bragg.py, ...).  These tests pin the *orchestration*
contract of :func:`ndiff.pipeline.run_pipeline`:

* the chained output file names match the original ``run_pipeline.py``;
* stages are skipped when their output already exists (resume);
* ``force_from`` recomputes the tail only;
* ``flatten_enabled=False`` routes the backfilled volume into the FFT;
* the progress callback fires per stage;
* the ΔPDF ``.h5`` is written in the schema the viewers read, with a working
  stale-cache guard.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import ndiff
from ndiff import pipeline
from ndiff.core import HKLVolume

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


def test_transform_config_string_format():
    cfg = pipeline.delta_pdf_transform_config(
        pipeline.DeltaPdfParams(apodization="gaussian", gaussian_sigma=0.4,
                                zero_pad=True, subtract_mean=True,
                                crop_hkl=(4.0, 8.0, 15.0), subtract_smooth_bg=None)
    )
    assert cfg == (
        "apodize=gaussian;gaussian_sigma=0.4;zero_pad=1;subtract_mean=1;"
        "crop_hkl=4,8,15;subtract_bg="
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
    return SimpleNamespace(calls=calls, vol=vol)


def _seed_input(tmp_path):
    inp = tmp_path / "sample.nxs"
    ndiff.save(_vol(), tmp_path / "sample.nxs")
    return inp


def test_run_pipeline_runs_all_stages_in_order(tmp_path, stubbed):
    inp = _seed_input(tmp_path)
    events = []
    paths = pipeline.run_pipeline(
        inp, proc_dir=tmp_path,
        progress=lambda *a: events.append(a),
    )
    assert stubbed.calls == ["rings", "punch", "backfill", "flatten", "pdf"]
    for p in (paths.ringremoved, paths.braggpunched, paths.backfilled,
              paths.flattened, paths.delta_pdf):
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
    assert skipped == {"rings", "punch", "backfill", "flatten", "pdf"}


def test_force_from_recomputes_tail_only(tmp_path, stubbed):
    inp = _seed_input(tmp_path)
    pipeline.run_pipeline(inp, proc_dir=tmp_path)
    stubbed.calls.clear()

    pipeline.run_pipeline(inp, proc_dir=tmp_path, force_from="backfill")
    assert stubbed.calls == ["backfill", "flatten", "pdf"]


def test_force_recomputes_everything(tmp_path, stubbed):
    inp = _seed_input(tmp_path)
    pipeline.run_pipeline(inp, proc_dir=tmp_path)
    stubbed.calls.clear()

    pipeline.run_pipeline(inp, proc_dir=tmp_path, force=True)
    assert stubbed.calls == ["rings", "punch", "backfill", "flatten", "pdf"]


def test_stage_subset_runs_only_requested(tmp_path, stubbed):
    inp = _seed_input(tmp_path)
    # seed the punched file so backfill has an input to load
    ndiff.save(stubbed.vol, pipeline.pipeline_paths(inp, proc_dir=tmp_path).braggpunched)
    pipeline.run_pipeline(inp, proc_dir=tmp_path, stages=("backfill",))
    assert stubbed.calls == ["backfill"]


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
    ndiff.save(_punched_vol(), paths.braggpunched)

    params = pipeline.PipelineParams(
        delta_pdf=pipeline.DeltaPdfParams(apodization="hann", zero_pad=False,
                                          crop_hkl=None),
    )
    events = []
    pipeline.run_pipeline(inp, params, proc_dir=tmp_path,
                          stages=("backfill", "flatten", "pdf"),
                          progress=lambda *a: events.append(a))

    # backfill produced an all-finite volume
    filled = ndiff.load(paths.backfilled)
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
    ndiff.save(_punched_vol(), paths.braggpunched)
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
