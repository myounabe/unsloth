# Scheduler Module

The `scheduler` module provides `SchedulerConfig` — a validated dataclass for
configuring learning-rate schedulers used during fine-tuning.

## Supported scheduler types

| Type | Description |
|------|-------------|
| `linear` | Linear decay to 0 after warmup |
| `cosine` | Cosine annealing after warmup |
| `cosine_with_restarts` | Cosine with hard restarts |
| `polynomial` | Polynomial decay |
| `constant` | No decay |
| `constant_with_warmup` | Constant LR with linear warmup |

## Quick start

```python
from unsloth.models.scheduler import SchedulerConfig

cfg = SchedulerConfig(
    scheduler_type="cosine",
    warmup_ratio=0.05,
    num_training_steps=1000,
)

kwargs = cfg.build_kwargs(total_steps=1000)
# Pass **kwargs to transformers.get_scheduler(optimizer=optimizer, **kwargs)
```

## Loading from a dict

```python
cfg = SchedulerConfig.from_dict({
    "scheduler_type": "linear",
    "warmup_steps": 50,
    "unknown_key": "ignored",
})
```

## Validation rules

- `scheduler_type` must be one of the supported types.
- `warmup_steps` and `warmup_ratio` are mutually exclusive.
- `warmup_ratio` must be in `[0.0, 1.0]`.
- `num_training_steps`, when provided, must be a positive integer.
- `num_cycles` must be ≥ 1; `power` must be > 0.
