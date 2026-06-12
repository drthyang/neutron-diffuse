"""Dataset discovery: group raw inputs with their pipeline-stage outputs.

A *dataset* is anchored by a raw input stem (e.g. the 22 K ``.nxs``).  Its stage
outputs are the chained ``.h5`` files named by :func:`ndiff.pipeline.pipeline_paths`
— so discovery reuses that single source of truth for the naming convention.
Datasets are also discovered from processed files alone (when the raw ``.nxs`` is
absent), so the UI still works on a checkout that only has the processed outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ndiff.pipeline import pipeline_paths
from ndiff.server.config import ServerConfig

# Stage name -> PipelinePaths attribute.  "raw" is handled separately.
_STAGE_ATTRS = {
    "ringremoved": "ringremoved",
    "braggpunched": "braggpunched",
    "backfilled": "backfilled",
    "flattened": "flattened",
    "delta_pdf": "delta_pdf",
}

# delta_pdf is a real-space ΔPDF (different .h5 schema); the rest are HKLVolumes.
_DELTA_PDF_STAGES = {"delta_pdf"}

#: HKLVolume-valued stages, in pipeline order (for the reciprocal-space viewer).
HKL_STAGES: tuple[str, ...] = (
    "raw", "ringremoved", "braggpunched", "backfilled", "flattened",
)

_TEMP_RE = re.compile(r"(\d+)\s*K", re.IGNORECASE)


@dataclass(frozen=True)
class StageStatus:
    name: str
    exists: bool
    path: Path
    kind: str  # "hkl" | "delta_pdf"


@dataclass(frozen=True)
class Dataset:
    id: str
    stem: str
    temperature: str | None
    raw_name: str
    raw_path: Path
    stages: list[StageStatus]


def _slug(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-")


def _detect_temp(stem: str) -> str | None:
    m = _TEMP_RE.search(stem)
    return f"{m.group(1)}K" if m else None


def discover_datasets(cfg: ServerConfig) -> list[Dataset]:
    """Enumerate datasets and the on-disk status of each pipeline stage."""
    stems: dict[str, Path] = {}
    if cfg.raw_dir.is_dir():
        for p in sorted(cfg.raw_dir.glob("*.nxs")):
            stems.setdefault(p.stem, p)
    if cfg.processed_dir.is_dir():
        for p in sorted(cfg.processed_dir.glob("*_ringremoved*.h5")):
            base = p.name.split("_ringremoved")[0]
            stems.setdefault(base, cfg.raw_dir / f"{base}.nxs")

    datasets: list[Dataset] = []
    for stem in sorted(stems):
        raw = stems[stem]
        paths = pipeline_paths(raw, proc_dir=cfg.processed_dir)
        stages = [StageStatus("raw", raw.exists(), raw, "hkl")]
        for name, attr in _STAGE_ATTRS.items():
            path: Path = getattr(paths, attr)
            kind = "delta_pdf" if name in _DELTA_PDF_STAGES else "hkl"
            stages.append(StageStatus(name, path.exists(), path, kind))
        datasets.append(Dataset(
            id=_slug(stem), stem=stem, temperature=_detect_temp(stem),
            raw_name=raw.name, raw_path=raw, stages=stages,
        ))
    return datasets


def find_dataset(cfg: ServerConfig, dataset_id: str) -> Dataset | None:
    return next((d for d in discover_datasets(cfg) if d.id == dataset_id), None)


def resolve_volume(cfg: ServerConfig, volume_id: str) -> StageStatus | None:
    """Resolve ``"<dataset_id>.<stage>"`` to its :class:`StageStatus`."""
    dataset_id, _, stage = volume_id.rpartition(".")
    if not dataset_id or not stage:
        return None
    ds = find_dataset(cfg, dataset_id)
    if ds is None:
        return None
    return next((s for s in ds.stages if s.name == stage), None)
