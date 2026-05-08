# Memory Tracker

The `memory` module provides lightweight GPU memory tracking that can be
embedded in training loops or callbacks without requiring a full profiler.

## Classes

### `MemorySnapshot`

Immutable record of memory at a single training step.

| Field | Type | Description |
|---|---|---|
| `step` | `int` | Training step index |
| `allocated_mib` | `float` | Currently allocated memory (MiB) |
| `reserved_mib` | `float` | Memory reserved by the CUDA allocator (MiB) |
| `peak_allocated_mib` | `float` | Peak allocated since last reset (MiB) |

### `MemoryTracker`

Stateful tracker that accumulates `MemorySnapshot` objects.

```python
from unsloth.models.memory import MemoryTracker

tracker = MemoryTracker(device=0)

for step in range(100):
    # ... training step ...
    snap = tracker.snapshot(step)
    print(snap.allocated_mib)

print("Peak allocated:", tracker.peak(), "MiB")
print("Full history:", tracker.history())
```

## Notes

- When CUDA is unavailable all memory values are reported as `0.0`; no
  exception is raised, so the same code runs on CPU-only machines.
- Call `tracker.reset()` between epochs if you want per-epoch peaks.
