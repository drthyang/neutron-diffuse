# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tsung-han Yang

"""Dataset listing endpoints."""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from nebula3d.server import consistency as consistency_mod
from nebula3d.server import datasets as ds_mod
from nebula3d.server import deltapdf as dpdf_mod
from nebula3d.server import volumes as vol_mod
from nebula3d.server.config import ServerConfig
from nebula3d.server.deps import get_config
from nebula3d.server.schemas import DataRootIn, DataRootOut, DatasetOut, StageStatusOut

router = APIRouter(prefix="/api", tags=["datasets"])


def _data_root_out(cfg: ServerConfig) -> DataRootOut:
    return DataRootOut(
        data_root=str(cfg.data_root),
        raw_exists=cfg.raw_dir.is_dir(),
        processed_exists=cfg.processed_dir.is_dir(),
        n_datasets=len(ds_mod.discover_datasets(cfg)),
    )


def _switch_data_root(root: Path, request: Request) -> DataRootOut:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise HTTPException(status_code=400, detail=f"Folder does not exist: {root}")

    cfg = ServerConfig(data_root=root)
    request.app.state.config = cfg
    vol_mod.clear_cache()
    dpdf_mod.clear_cache()
    consistency_mod.clear_cache()
    return _data_root_out(cfg)


def _choose_directory(initial: Path) -> Path | None:
    """Ask the server's desktop session for a folder path."""
    initial = initial.expanduser().resolve()
    system = platform.system()

    if system == "Darwin":
        escaped = str(initial).replace("\\", "\\\\").replace('"', '\\"')
        script = (
            f'set initialFolder to POSIX file "{escaped}"\n'
            "try\n"
            '  set chosenFolder to choose folder with prompt "Choose nebula3d data folder" '
            "default location initialFolder\n"
            "  POSIX path of chosenFolder\n"
            "on error number -128\n"
            '  return ""\n'
            "end try\n"
        )
        proc = subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            msg = proc.stderr.strip() or "macOS folder picker failed"
            raise HTTPException(status_code=500, detail=msg)
        selected = proc.stdout.strip()
        return Path(selected) if selected else None

    if system == "Linux" and shutil.which("zenity"):
        proc = subprocess.run(
            [
                "zenity",
                "--file-selection",
                "--directory",
                f"--filename={initial}/",
                "--title=Choose nebula3d data folder",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 1:
            return None
        if proc.returncode != 0:
            msg = proc.stderr.strip() or "Linux folder picker failed"
            raise HTTPException(status_code=500, detail=msg)
        return Path(proc.stdout.strip())

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - platform fallback
        raise HTTPException(
            status_code=501,
            detail="No local folder picker is available; paste the full folder path.",
        ) from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(
            initialdir=str(initial),
            title="Choose nebula3d data folder",
            mustexist=True,
        )
    finally:
        root.destroy()
    return Path(selected) if selected else None


@router.get("/data-root", response_model=DataRootOut)
def get_data_root(cfg: ServerConfig = Depends(get_config)) -> DataRootOut:
    """Return the active data root scanned by the web server."""
    return _data_root_out(cfg)


@router.put("/data-root", response_model=DataRootOut)
def set_data_root(req: DataRootIn, request: Request) -> DataRootOut:
    """Switch the active data root for this running server process."""
    return _switch_data_root(Path(req.data_root), request)


@router.post("/data-root/browse", response_model=DataRootOut)
def browse_data_root(
    request: Request,
    cfg: ServerConfig = Depends(get_config),
) -> DataRootOut:
    """Open a trusted local folder picker and switch to the selected root."""
    if request.headers.get("x-nebula3d-local") != "1":
        raise HTTPException(status_code=403, detail="Missing local request header")

    selected = _choose_directory(cfg.data_root)
    if selected is None:
        raise HTTPException(status_code=409, detail="Folder selection canceled")
    return _switch_data_root(selected, request)


@router.get("/datasets", response_model=list[DatasetOut])
def list_datasets(cfg: ServerConfig = Depends(get_config)) -> list[DatasetOut]:
    """List datasets grouped by raw input, with per-stage output status."""
    out: list[DatasetOut] = []
    for ds in ds_mod.discover_datasets(cfg):
        stages = [
            StageStatusOut(name=s.name, exists=s.exists, kind=s.kind,
                           volume_id=f"{ds.id}.{s.name}")
            for s in ds.stages
        ]
        out.append(DatasetOut(
            id=ds.id, temperature=ds.temperature, raw_name=ds.raw_name,
            stem=ds.stem, stages=stages,
        ))
    return out
