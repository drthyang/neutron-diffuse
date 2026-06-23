"""Server configuration (data locations)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServerConfig:
    """Where the server reads/writes data.

    ``data_root`` holds ``raw/`` (input ``.nxs``) and ``processed/`` (pipeline
    ``.h5`` outputs), matching the repository layout used by the example scripts.
    """

    data_root: Path

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "processed"


def config_from_env() -> ServerConfig:
    """Build a config from ``NEBULA3D_DATA_ROOT`` (default: ``./data``)."""
    root = os.environ.get("NEBULA3D_DATA_ROOT")
    data_root = Path(root).expanduser().resolve() if root else (Path.cwd() / "data")
    return ServerConfig(data_root=data_root)
