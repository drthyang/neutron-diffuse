"""Tests for the FastAPI backend (datasets + slice endpoints).

Uses a synthetic data root so it is fast and independent of the real volumes.
The key correctness check is that the binary slice endpoint returns exactly what
``extract_slice`` produces directly.
"""

import json
import struct

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

import ndiff
from ndiff.core import HKLVolume
from ndiff.pipeline import pipeline_paths
from ndiff.server import deltapdf as dpdf_mod
from ndiff.server import volumes as vol_mod
from ndiff.server.app import create_app
from ndiff.server.config import ServerConfig
from ndiff.server.datasets import discover_datasets
from ndiff.server.routers import datasets as datasets_router
from ndiff.visualization import extract_slice

UB = 2 * np.pi * np.eye(3) / 4.0
STEM = "TbTi3Bi4_22K_test_cc_sub_bkg"
SLUG = "TbTi3Bi4-22K-test-cc-sub-bkg"


def _vol(shape=(7, 9, 11), seed=0):
    rng = np.random.default_rng(seed)
    data = 1.0 + rng.normal(0.0, 0.3, shape)
    return HKLVolume.from_arrays(data, (-3, 3), (-4, 4), (-5, 5), ub_matrix=UB)


@pytest.fixture
def env(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "processed").mkdir()
    vol = _vol()
    paths = pipeline_paths(tmp_path / "raw" / f"{STEM}.nxs",
                           proc_dir=tmp_path / "processed")
    ndiff.save(vol, paths.ringremoved)
    ndiff.save(vol, paths.backfilled)
    vol_mod.clear_cache()
    app = create_app(ServerConfig(data_root=tmp_path))
    return TestClient(app), vol


def _parse_envelope(body: bytes):
    (hlen,) = struct.unpack("<I", body[:4])
    header = json.loads(body[4:4 + hlen].decode("utf-8"))
    data = np.frombuffer(body[4 + hlen:], dtype="<f4").reshape(
        header["ny"], header["nx"])
    return header, data


def test_health(env):
    client, _ = env
    assert client.get("/api/health").json() == {"status": "ok"}


def test_list_datasets_reflects_disk(env):
    client, _ = env
    data = client.get("/api/datasets").json()
    assert len(data) == 1
    ds = data[0]
    assert ds["id"] == SLUG
    assert ds["temperature"] == "22K"
    status = {s["name"]: s["exists"] for s in ds["stages"]}
    assert status["ringremoved"] is True
    assert status["backfilled"] is True
    assert status["raw"] is False          # no .nxs synthesised
    assert status["braggpunched"] is False
    assert status["flattened"] is False
    assert status["delta_pdf"] is False
    # volume ids are addressable
    vid = {s["name"]: s["volume_id"] for s in ds["stages"]}
    assert vid["ringremoved"] == f"{SLUG}.ringremoved"


def test_data_root_can_be_switched(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "processed").mkdir()
    paths = pipeline_paths(tmp_path / "raw" / f"{STEM}.nxs",
                           proc_dir=tmp_path / "processed")
    ndiff.save(_vol(), paths.ringremoved)

    alt = tmp_path / "alt_data"
    (alt / "raw").mkdir(parents=True)
    (alt / "processed").mkdir()
    alt_stem = "TbTi3Bi4_45K_alt_cc_sub_bkg"
    alt_slug = "TbTi3Bi4-45K-alt-cc-sub-bkg"
    alt_paths = pipeline_paths(alt / "raw" / f"{alt_stem}.nxs",
                               proc_dir=alt / "processed")
    ndiff.save(_vol(seed=1), alt_paths.ringremoved)

    app = create_app(ServerConfig(data_root=tmp_path))
    client = TestClient(app)

    assert client.get("/api/data-root").json()["data_root"] == str(tmp_path)
    assert client.get("/api/datasets").json()[0]["id"] == SLUG

    switched = client.put("/api/data-root", json={"data_root": str(alt)})
    assert switched.status_code == 200
    body = switched.json()
    assert body["data_root"] == str(alt)
    assert body["raw_exists"] is True
    assert body["processed_exists"] is True
    assert body["n_datasets"] == 1
    assert client.get("/api/datasets").json()[0]["id"] == alt_slug

    missing = client.put("/api/data-root", json={"data_root": str(tmp_path / "nope")})
    assert missing.status_code == 400


def test_data_root_browse_switches_selected_folder(tmp_path, monkeypatch):
    (tmp_path / "raw").mkdir()
    (tmp_path / "processed").mkdir()

    alt = tmp_path / "picked"
    (alt / "raw").mkdir(parents=True)
    (alt / "processed").mkdir()
    alt_stem = "TbTi3Bi4_100K_picked_cc_sub_bkg"
    alt_slug = "TbTi3Bi4-100K-picked-cc-sub-bkg"
    alt_paths = pipeline_paths(alt / "raw" / f"{alt_stem}.nxs",
                               proc_dir=alt / "processed")
    ndiff.save(_vol(seed=2), alt_paths.ringremoved)

    monkeypatch.setattr(datasets_router, "_choose_directory", lambda _initial: alt)

    app = create_app(ServerConfig(data_root=tmp_path))
    client = TestClient(app)

    blocked = client.post("/api/data-root/browse")
    assert blocked.status_code == 403

    picked = client.post("/api/data-root/browse", headers={"X-Ndiff-Local": "1"})
    assert picked.status_code == 200
    body = picked.json()
    assert body["data_root"] == str(alt)
    assert body["n_datasets"] == 1
    assert client.get("/api/datasets").json()[0]["id"] == alt_slug


def test_data_root_browse_cancel_leaves_root(tmp_path, monkeypatch):
    (tmp_path / "raw").mkdir()
    (tmp_path / "processed").mkdir()
    monkeypatch.setattr(datasets_router, "_choose_directory", lambda _initial: None)

    app = create_app(ServerConfig(data_root=tmp_path))
    client = TestClient(app)

    canceled = client.post("/api/data-root/browse", headers={"X-Ndiff-Local": "1"})
    assert canceled.status_code == 409
    assert client.get("/api/data-root").json()["data_root"] == str(tmp_path)


@pytest.mark.parametrize("dpdf_name", ["full_chain", "short_legacy", "both"])
def test_discovery_no_phantom_from_delta_pdf_names(tmp_path, dpdf_name):
    """``*_delta_pdf.h5`` files attach to their dataset instead of spawning a
    phantom one — whether named with the full pipeline chain
    (``…_backfilled_delta_pdf.h5``) or the short legacy stem
    (``{stem}_delta_pdf.h5``).  Regression: both names previously slipped past
    the seeding skip-check (in opposite directions) and created a duplicate
    dataset carrying only the ΔPDF stage."""
    (tmp_path / "raw").mkdir()
    proc = tmp_path / "processed"
    proc.mkdir()
    paths = pipeline_paths(tmp_path / "raw" / f"{STEM}.nxs", proc_dir=proc)
    ndiff.save(_vol(), paths.ringremoved)
    ndiff.save(_vol(), paths.backfilled)
    if dpdf_name in ("full_chain", "both"):
        _write_dpdf(paths.delta_pdf)               # …_backfilled_delta_pdf.h5
    if dpdf_name in ("short_legacy", "both"):
        _write_dpdf(proc / f"{STEM}_delta_pdf.h5")  # short legacy name

    datasets = discover_datasets(ServerConfig(data_root=tmp_path))

    assert len(datasets) == 1
    ds = datasets[0]
    assert ds.id == SLUG
    assert next(s for s in ds.stages if s.name == "delta_pdf").exists is True


def test_volume_meta(env):
    client, vol = env
    m = client.get(f"/api/volumes/{SLUG}.ringremoved/meta").json()
    assert m["shape"] == list(vol.data.shape)
    assert m["h_range"] == [-3.0, 3.0]
    assert m["l_range"] == [-5.0, 5.0]
    assert m["lattice"]["a"] == pytest.approx(4.0)
    assert "hk" in m["planes"] and "0kl" in m["planes"]


@pytest.mark.parametrize("plane,value", [("hk", 0.0), ("0kl", 0.0), ("h0l", 1.0)])
def test_slice_matches_extract_slice(env, plane, value):
    client, vol = env
    r = client.get(f"/api/volumes/{SLUG}.ringremoved/slice",
                   params={"plane": plane, "value": value})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"
    header, data = _parse_envelope(r.content)

    ref = extract_slice(vol, plane=plane, value=value)
    assert data.shape == ref.data.shape
    np.testing.assert_allclose(data, ref.data.astype("float32"), rtol=1e-5,
                               atol=1e-5, equal_nan=True)
    assert header["x_label"] == ref.x_label
    assert header["y_label"] == ref.y_label
    np.testing.assert_allclose(header["x_axis"], ref.x_axis, rtol=1e-6)


def test_slice_unknown_volume_404(env):
    client, _ = env
    assert client.get("/api/volumes/nope.ringremoved/slice").status_code == 404


def test_slice_missing_stage_404(env):
    client, _ = env
    # flattened was never written
    assert client.get(f"/api/volumes/{SLUG}.flattened/slice").status_code == 404


def test_slice_bad_plane_400(env):
    client, _ = env
    r = client.get(f"/api/volumes/{SLUG}.ringremoved/slice",
                   params={"plane": "zz"})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# ΔPDF endpoints
# ---------------------------------------------------------------------------
def _write_dpdf(path, nx=6, ny=8, nz=10):
    rng = np.random.default_rng(2)
    data = rng.normal(0.0, 1.0, (nx, ny, nz)).astype(np.float64)
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-12, 12, ny)
    z = np.linspace(-15, 15, nz)
    with h5py.File(path, "w") as fh:
        fh.create_dataset("data", data=data)
        fh.create_dataset("x_axis", data=x)
        fh.create_dataset("y_axis", data=y)
        fh.create_dataset("z_axis", data=z)
        fh.attrs["lat_a"] = 5.8
        fh.attrs["lat_b"] = 10.4
        fh.attrs["lat_c"] = 24.7
        fh.attrs["q_max"] = 11.9
    return data, x, y, z


@pytest.fixture
def dpdf_env(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "processed").mkdir()
    paths = pipeline_paths(tmp_path / "raw" / f"{STEM}.nxs",
                           proc_dir=tmp_path / "processed")
    data, x, y, z = _write_dpdf(paths.delta_pdf)
    dpdf_mod.clear_cache()
    app = create_app(ServerConfig(data_root=tmp_path))
    return TestClient(app), data, x, y, z


def test_dpdf_meta(dpdf_env):
    client, data, _, _, _ = dpdf_env
    m = client.get(f"/api/deltapdf/{SLUG}.delta_pdf/meta").json()
    assert m["shape"] == list(data.shape)
    assert m["x_range"] == [-10.0, 10.0]
    assert m["z_range"] == [-15.0, 15.0]
    assert m["lattice"]["a"] == pytest.approx(5.8)
    assert m["q_max"] == pytest.approx(11.9)
    assert m["planes"] == ["xy", "xz", "yz"]


@pytest.mark.parametrize(
    "plane,fixed_axis,xl,yl",
    [("xy", "z", "x_H (Å)", "y_K (Å)"),
     ("xz", "y", "x_H (Å)", "z_L (Å)"),
     ("yz", "x", "y_K (Å)", "z_L (Å)")],
)
def test_dpdf_slice_matches_transpose(dpdf_env, plane, fixed_axis, xl, yl):
    client, data, x, y, z = dpdf_env
    r = client.get(f"/api/deltapdf/{SLUG}.delta_pdf/slice",
                   params={"plane": plane, "value": 0.0})
    assert r.status_code == 200
    header, arr = _parse_envelope(r.content)

    axis = {"x": x, "y": y, "z": z}[fixed_axis]
    i = int(np.argmin(np.abs(axis - 0.0)))
    ref = {
        "xy": data[:, :, i].T,
        "xz": data[:, i, :].T,
        "yz": data[i, :, :].T,
    }[plane].astype("float32")
    assert arr.shape == ref.shape
    np.testing.assert_allclose(arr, ref, rtol=1e-5, atol=1e-5)
    assert header["x_label"] == xl
    assert header["y_label"] == yl


def test_dpdf_bad_plane_400(dpdf_env):
    client, *_ = dpdf_env
    r = client.get(f"/api/deltapdf/{SLUG}.delta_pdf/slice", params={"plane": "hk"})
    assert r.status_code == 400


def test_dpdf_endpoint_rejects_hkl_volume(env):
    client, _ = env
    assert client.get(f"/api/deltapdf/{SLUG}.ringremoved/meta").status_code == 400


def test_hkl_endpoint_rejects_dpdf_volume(dpdf_env):
    client, *_ = dpdf_env
    assert client.get(f"/api/volumes/{SLUG}.delta_pdf/slice").status_code == 400


# ---------------------------------------------------------------------------
# back-FFT consistency endpoints
# ---------------------------------------------------------------------------
def test_consistency_meta_and_band(env):
    """Meta returns metrics + |Q| span; a sub-band limits the metric region."""
    client, _ = env
    full = client.get(f"/api/consistency/{SLUG}/meta")
    assert full.status_code == 200
    m = full.json()
    assert m["q_data_max"] > 0
    assert "kl" in m["planes"]
    assert m["metrics"]["q_band"] is None
    full_vox = m["metrics"]["n_voxels"]

    qmax = m["q_data_max"]
    banded = client.get(f"/api/consistency/{SLUG}/meta",
                        params={"q_min": qmax * 0.3, "q_max": qmax * 0.7})
    assert banded.status_code == 200
    mb = banded.json()
    assert mb["metrics"]["q_band"][0] == pytest.approx(qmax * 0.3)
    assert 0 < mb["metrics"]["n_voxels"] < full_vox   # the shell is a subset


def test_consistency_slice_envelope(env):
    client, _ = env
    r = client.get(f"/api/consistency/{SLUG}/slice",
                   params={"panel": "recon", "plane": "kl", "value": 0.0})
    assert r.status_code == 200
    header, data = _parse_envelope(r.content)
    assert data.shape == (header["ny"], header["nx"])
    assert header["x_label"] and header["y_label"]


def test_consistency_bad_panel_400(env):
    client, _ = env
    r = client.get(f"/api/consistency/{SLUG}/slice",
                   params={"panel": "nope", "plane": "kl"})
    assert r.status_code == 400


def test_consistency_unknown_dataset_404(env):
    client, _ = env
    assert client.get("/api/consistency/does-not-exist/meta").status_code == 404


# ---------------------------------------------------------------------------
# request → PipelineParams mapping
# ---------------------------------------------------------------------------
def test_build_params_ring_overrides():
    """Ring-removal overrides reach RingParams; omitted ones keep the defaults."""
    from ndiff.pipeline import RingParams
    from ndiff.server.routers.pipeline import build_params
    from ndiff.server.schemas import PipelineRunRequest, StageParamsIn

    defaults = build_params(PipelineRunRequest(dataset_id="x"))
    assert defaults.rings.n_patches == RingParams().n_patches
    assert defaults.rings.n_fourier == RingParams().n_fourier
    assert defaults.rings.slice_axis == RingParams().slice_axis

    overridden = build_params(PipelineRunRequest(
        dataset_id="x",
        params=StageParamsIn(rings_n_patches=48, rings_n_fourier=10,
                             rings_slice_axis="L"),
    ))
    assert overridden.rings.n_patches == 48
    assert overridden.rings.n_fourier == 10
    assert overridden.rings.slice_axis == "L"
    # ring model defaults to the patched estimator; the parametric one is opt-in
    assert defaults.rings.ring_model == "patched"
    parametric = build_params(PipelineRunRequest(
        dataset_id="x",
        params=StageParamsIn(rings_model="parametric", rings_ring_width=0.3,
                             rings_radial_mode="peaks"),
    ))
    assert parametric.rings.ring_model == "parametric"
    assert parametric.rings.ring_width == 0.3
    assert parametric.rings.ring_radial_mode == "peaks"
    # the parametric radial model defaults to the continuous rolling sweep
    assert defaults.rings.ring_radial_mode == "rolling"
    # an unrelated stage keeps its default
    from ndiff.pipeline import PunchParams
    assert overridden.punch.min_intensity == PunchParams().min_intensity


def test_build_params_punch_overrides():
    """Punch overrides reach PunchParams; per-axis radii fall back to defaults."""
    from ndiff.pipeline import PunchParams
    from ndiff.server.routers.pipeline import build_params
    from ndiff.server.schemas import PipelineRunRequest, StageParamsIn

    base = PunchParams()
    overridden = build_params(PipelineRunRequest(
        dataset_id="x",
        params=StageParamsIn(
            punch_mode="search",
            punch_radius_h=0.2,
            punch_radius_l=0.6,
            punch_margin=0.05,
            punch_phi_tail_hkl=0.3,
        ),
    )).punch

    assert overridden.mode == "search"
    # h and l overridden; k untouched keeps its default
    assert overridden.punch_radii == (0.2, base.punch_radii[1], 0.6)
    assert overridden.margin == 0.05
    assert overridden.phi_tail_hkl == 0.3

    # omitting every punch field leaves PunchParams untouched
    defaults = build_params(PipelineRunRequest(dataset_id="x")).punch
    assert defaults.mode == base.mode
    assert defaults.punch_radii == base.punch_radii
    assert defaults.phi_tail_hkl == base.phi_tail_hkl
    # the Q-space frame is the default (since Phase 4)
    assert defaults.punch_frame == "q"


def test_build_params_qspace_punch_overrides():
    """Q-space punch overrides (frame + isotropic / per-axis radii) reach PunchParams."""
    from ndiff.pipeline import PunchParams
    from ndiff.server.routers.pipeline import build_params
    from ndiff.server.schemas import PipelineRunRequest, StageParamsIn

    base = PunchParams()
    iso = build_params(PipelineRunRequest(
        dataset_id="x",
        params=StageParamsIn(punch_frame="q", punch_q_radius=0.1),
    )).punch
    assert iso.punch_frame == "q"
    assert iso.punch_q_radius == 0.1

    # per-axis: a and c overridden, b falls back to the default punch_q_radii
    anis = build_params(PipelineRunRequest(
        dataset_id="x",
        params=StageParamsIn(punch_frame="q",
                             punch_q_radius_a=0.08, punch_q_radius_c=0.2),
    )).punch
    assert anis.punch_q_radii == (0.08, base.punch_q_radii[1], 0.2)

    # the Q frame is the default since Phase 4
    assert base.punch_frame == "q"
    assert base.punch_q_radii == (0.097, 0.072, 0.115)

    # Phase 3 covariance-fit toggle
    cov = build_params(PipelineRunRequest(
        dataset_id="x", params=StageParamsIn(punch_fit_covariance=True),
    )).punch
    assert cov.integer_fit_covariance is True
    assert base.integer_fit_covariance is False


# ---------------------------------------------------------------------------
# pipeline job execution (real worker process)
# ---------------------------------------------------------------------------
def test_job_manager_runs_pipeline_in_process(tmp_path):
    import time

    from ndiff.pipeline import DeltaPdfParams, PipelineParams, pipeline_paths
    from ndiff.server.jobs import JobManager

    proc = tmp_path / "processed"
    proc.mkdir()
    inp = tmp_path / "s.nxs"
    paths = pipeline_paths(inp, proc_dir=proc)

    # seed a punched volume so backfill → flatten → pdf can run without the slow
    # ring/punch stages (and without a real Mantid .nxs).
    rng = np.random.default_rng(0)
    shape = (16, 16, 16)
    vol = HKLVolume.from_arrays(
        np.abs(rng.normal(1.0, 0.2, shape)), (-3, 3), (-3, 3), (-3, 3), ub_matrix=UB)
    for idx in [(8, 8, 8), (7, 8, 9)]:
        vol.data[idx] = -9.0
        vol.mask[idx] = False
    ndiff.save(vol, paths.braggpunched)

    params = PipelineParams(
        delta_pdf=DeltaPdfParams(apodization="hann", zero_pad=False, crop_hkl=None))
    job = JobManager().start(inp, params, proc_dir=proc,
                             stages=("backfill", "flatten", "pdf"))

    deadline = time.time() + 120
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.2)

    assert job.status == "done", (job.status, job.error)
    assert paths.delta_pdf.exists()
    stages_done = {e["stage"] for e in job.events if e.get("status") == "done"}
    assert {"backfill", "flatten", "pdf"} <= stages_done
