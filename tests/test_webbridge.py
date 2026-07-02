"""End-to-end test of the in-browser bridge (:mod:`nebula3d.webbridge`).

Exercises the same code path the Pyodide build uses — minus Pyodide itself —
on a tiny synthetic volume: set up the virtual workspace, run the full pipeline,
then pull the dataset listing, HKL/ΔPDF/consistency metadata, and binary slice
envelopes, asserting they have the shapes the React viewers decode.

The bridge must stay importable without FastAPI (it runs under Pyodide), so the
import itself is part of the contract under test.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys

import numpy as np
import pytest

from nebula3d import webbridge


def _decode_envelope(buf: bytes) -> tuple[dict, np.ndarray]:
    """Decode [uint32 header_len][JSON header][float32 data] (the wire format)."""
    (header_len,) = struct.unpack_from("<I", buf, 0)
    header = json.loads(buf[4 : 4 + header_len].decode("utf-8"))
    data = np.frombuffer(buf[4 + header_len :], dtype="<f4")
    return header, data


def test_webbridge_imports_without_fastapi():
    """The bridge (and the server helpers it reuses) must not import FastAPI.

    Run in a fresh interpreter: FastAPI is unavailable under Pyodide, so dragging
    it in transitively would break the in-browser build.  A subprocess isolates
    this from other tests in the suite (e.g. test_server) that do import FastAPI.
    """
    code = (
        "import sys; import nebula3d.webbridge; "
        "assert 'fastapi' not in sys.modules, sorted(sys.modules)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.fixture
def ran_pipeline(tmp_path):
    """Run the full pipeline once on a synthetic volume in a temp workspace."""
    webbridge.setup(workdir=str(tmp_path / "work"))
    dataset_id = webbridge.make_demo_input(n=24)

    events: list[tuple] = []

    def progress(stage, status, fraction, message):
        events.append((stage, status, fraction, message))

    datasets_json = webbridge.run(
        "", "{}", flatten_enabled=True, force=True, progress=progress)
    return dataset_id, json.loads(datasets_json), events


def test_run_streams_progress_for_every_stage(ran_pipeline):
    _id, _datasets, events = ran_pipeline
    stages_seen = {ev[0] for ev in events}
    # Every pipeline stage should have emitted at least one progress event.
    assert set(webbridge.run.__globals__["STAGES"]).issubset(stages_seen)
    assert any(status == "done" for _s, status, _f, _m in events)


def test_datasets_listing_has_all_stages(ran_pipeline):
    dataset_id, datasets, _events = ran_pipeline
    assert len(datasets) == 1
    ds = datasets[0]
    assert ds["id"] == dataset_id
    assert ds["temperature"] is None
    by_name = {s["name"]: s for s in ds["stages"]}
    for stage in ("raw", "ringremoved", "braggpunched", "backfilled",
                  "flattened", "delta_pdf"):
        assert stage in by_name, stage
        assert by_name[stage]["volume_id"] == f"{dataset_id}.{stage}"
    # The cleanup stages and the ΔPDF should all exist on disk after the run.
    assert by_name["ringremoved"]["exists"]
    assert by_name["delta_pdf"]["exists"]
    assert by_name["delta_pdf"]["kind"] == "delta_pdf"


def test_hkl_meta_and_slice(ran_pipeline):
    dataset_id, _datasets, _events = ran_pipeline
    vid = f"{dataset_id}.ringremoved"
    meta = json.loads(webbridge.volume_meta_json(vid))
    assert meta["id"] == vid
    assert meta["kind"] == "hkl"
    assert meta["shape"] == [24, 24, 24]
    assert "hk" in meta["planes"]

    env = webbridge.volume_slice(vid, "hk", 0.0, False)
    header, data = _decode_envelope(bytes(env))
    assert header["nx"] * header["ny"] == data.size
    assert header["ny"] == 24 and header["nx"] == 24


def test_dpdf_meta_and_slice(ran_pipeline):
    dataset_id, _datasets, _events = ran_pipeline
    vid = f"{dataset_id}.delta_pdf"
    meta = json.loads(webbridge.dpdf_meta_json(vid))
    assert meta["id"] == vid
    assert len(meta["shape"]) == 3
    assert set(meta["planes"]) == {"xy", "xz", "yz"}

    env = webbridge.dpdf_slice(vid, "xy", 0.0)
    header, data = _decode_envelope(bytes(env))
    assert header["nx"] * header["ny"] == data.size
    assert header["robust_max"] > 0


def test_consistency_meta_and_slice(ran_pipeline):
    dataset_id, _datasets, _events = ran_pipeline
    meta = json.loads(webbridge.consistency_meta_json(dataset_id))
    assert "metrics" in meta
    assert "pearson_r" in meta["metrics"]
    assert len(meta["shape"]) == 3

    # The reciprocal panels slice on HKL planes; the ΔPDF panel on real-space ones.
    for panel, plane in (("data", "hk"), ("recon", "hk"), ("residual", "hk"),
                         ("dpdf", "xy")):
        env = webbridge.consistency_slice(dataset_id, panel, plane, 0.0)
        header, data = _decode_envelope(bytes(env))
        assert header["nx"] * header["ny"] == data.size


def test_unknown_volume_raises(ran_pipeline):
    with pytest.raises(KeyError):
        webbridge.volume_meta_json("nope.ringremoved")


def test_bragg_profile_json(ran_pipeline):
    dataset_id, _datasets, _events = ran_pipeline
    prof = json.loads(webbridge.bragg_profile_json(dataset_id))
    assert prof["dataset_id"] == dataset_id
    assert prof["has_profile"] is True
    assert prof["n_peaks"] == len(prof["peaks"]) > 0
    assert prof["width_labels"]


def test_bragg_profile_json_missing_file(ran_pipeline):
    from pathlib import Path

    dataset_id, _datasets, _events = ran_pipeline
    prof = json.loads(webbridge.bragg_profile_json(dataset_id))
    Path(prof["profile_path"]).unlink()
    prof2 = json.loads(webbridge.bragg_profile_json(dataset_id))
    assert prof2["has_profile"] is False
    assert prof2["peaks"] == []


def test_bragg_profile_json_unknown_dataset(ran_pipeline):
    with pytest.raises(KeyError):
        webbridge.bragg_profile_json("nope")


def test_save_dpdf_envelope(ran_pipeline):
    dataset_id, _datasets, _events = ran_pipeline
    env = bytes(webbridge.save_dpdf(dataset_id, q_min=0.5, q_max=3.0))
    (header_len,) = struct.unpack_from("<I", env, 0)
    header = json.loads(env[4 : 4 + header_len].decode("utf-8"))
    assert header["filename"].endswith(".h5")
    assert "delta_pdf" in header["filename"]
    payload = env[4 + header_len:]
    assert payload[:8] == b"\x89HDF\r\n\x1a\n"  # HDF5 magic signature
    # The temp file must not linger in the workspace (unlinked after read).
    cfg = webbridge._S.cfg
    assert cfg is not None
    assert header["filename"] not in {p.name for p in cfg.processed_dir.iterdir()}


def test_inspect_input_admits_full_resolution_volume(tmp_path):
    """A real full-resolution grid (301×401×401 = 48.4 M voxels, float64) must
    pass the browser memory gate — the 4 GB Pyodide heap plus the memory-lean
    float64 pipeline admit it (no precision reduction, see webbridge budget).

    Metadata-only file: the gate reads just the HDF5 dataset shape, so the
    dataset is never written (h5py allocates contiguous storage lazily).
    """
    import h5py

    webbridge.setup(workdir=str(tmp_path / "work"))
    path = tmp_path / "full_res.h5"
    with h5py.File(path, "w") as f:
        f.create_group("entry").create_dataset(
            "data", shape=(301, 401, 401), dtype="f8")

    report = json.loads(webbridge.inspect_input("full_res.h5", str(path)))
    assert report["shape"] == [301, 401, 401]
    assert report["n_voxels"] == 301 * 401 * 401
    assert report["ok"] is True, report["message"]


def test_inspect_input_still_rejects_oversized_volume(tmp_path):
    """Beyond the physical 4 GB WASM ceiling the gate must still refuse."""
    import h5py

    webbridge.setup(workdir=str(tmp_path / "work"))
    path = tmp_path / "huge.h5"
    with h5py.File(path, "w") as f:
        f.create_group("entry").create_dataset(
            "data", shape=(501, 501, 501), dtype="f8")  # 125 M voxels

    report = json.loads(webbridge.inspect_input("huge.h5", str(path)))
    assert report["ok"] is False
    assert "native build" in report["message"]
