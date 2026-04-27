"""Checkpoint saving and loading utilities for Unsloth training runs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CheckpointMeta:
    """Metadata persisted alongside a model checkpoint."""

    step: int
    epoch: float
    best_loss: float
    model_name: str
    extra: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError(f"step must be >= 0, got {self.step}")
        if self.epoch < 0.0:
            raise ValueError(f"epoch must be >= 0.0, got {self.epoch}")
        if self.extra is None:
            self.extra = {}

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def save(self, directory: str | os.PathLike) -> Path:
        """Write metadata as JSON inside *directory*. Returns the file path."""
        path = Path(directory) / "checkpoint_meta.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)
        return path

    @classmethod
    def load(cls, directory: str | os.PathLike) -> "CheckpointMeta":
        """Load metadata from *directory/checkpoint_meta.json*."""
        path = Path(directory) / "checkpoint_meta.json"
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint metadata found at {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(**data)


def latest_checkpoint(base_dir: str | os.PathLike) -> Optional[Path]:
    """Return the sub-directory with the highest *step* among all checkpoints.

    Scans immediate children of *base_dir* for ``checkpoint_meta.json`` files
    and returns the directory whose metadata has the largest ``step`` value.
    Returns ``None`` when no checkpoints are found.
    """
    base = Path(base_dir)
    best_path: Optional[Path] = None
    best_step = -1

    if not base.is_dir():
        return None

    for child in base.iterdir():
        meta_file = child / "checkpoint_meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = CheckpointMeta.load(child)
        except Exception:  # noqa: BLE001
            continue
        if meta.step > best_step:
            best_step = meta.step
            best_path = child

    return best_path
