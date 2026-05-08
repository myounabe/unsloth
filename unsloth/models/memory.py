"""GPU/CPU memory tracking utilities for Unsloth training runs."""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional


@dataclasses.dataclass
class MemorySnapshot:
    """A single point-in-time memory measurement (values in MiB)."""

    step: int
    allocated_mib: float
    reserved_mib: float
    peak_allocated_mib: float

    def to_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class MemoryTracker:
    """Tracks GPU memory usage across training steps."""

    device: int = 0
    _history: List[MemorySnapshot] = dataclasses.field(
        default_factory=list, init=False, repr=False
    )

    def snapshot(self, step: int) -> MemorySnapshot:
        """Record current memory state and return the snapshot."""
        try:
            import torch

            props = torch.cuda.memory_stats(self.device)
            allocated = props.get("allocated_bytes.all.current", 0) / 1024 ** 2
            reserved = props.get("reserved_bytes.all.current", 0) / 1024 ** 2
            peak = props.get("allocated_bytes.all.peak", 0) / 1024 ** 2
        except Exception:
            allocated = reserved = peak = 0.0

        snap = MemorySnapshot(
            step=step,
            allocated_mib=round(allocated, 2),
            reserved_mib=round(reserved, 2),
            peak_allocated_mib=round(peak, 2),
        )
        self._history.append(snap)
        return snap

    def peak(self) -> Optional[float]:
        """Return the highest *allocated* MiB seen across all snapshots."""
        if not self._history:
            return None
        return max(s.allocated_mib for s in self._history)

    def history(self) -> List[Dict[str, float]]:
        """Return all snapshots as plain dicts."""
        return [s.to_dict() for s in self._history]

    def reset(self) -> None:
        """Clear recorded history."""
        self._history.clear()
