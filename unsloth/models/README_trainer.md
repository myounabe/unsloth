# `unsloth.models.trainer` — TrainingArguments

A lightweight, dependency-free dataclass that holds supervised fine-tuning
hyper-parameters and validates them before a training run starts.

## Quick start

```python
from unsloth.models.trainer import TrainingArguments

args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    bf16=True,
)

# or build from a plain dict (e.g. loaded from YAML / JSON)
args = TrainingArguments.from_dict(config_dict)

print("Effective batch size:", args.effective_batch_size)
```

## Fields

| Field | Default | Description |
|---|---|---|
| `output_dir` | `"./outputs"` | Directory for checkpoints |
| `num_train_epochs` | `1` | Total training epochs |
| `per_device_train_batch_size` | `2` | Micro-batch size per device |
| `gradient_accumulation_steps` | `4` | Steps before an optimiser update |
| `learning_rate` | `2e-4` | Peak LR |
| `warmup_steps` | `10` | Linear warm-up steps |
| `fp16` / `bf16` | `False` | Mixed-precision flags (mutually exclusive) |
| `lr_scheduler_type` | `"linear"` | One of `linear`, `cosine`, `constant` |
| `max_steps` | `-1` | Override epoch-based stopping when > 0 |

## Validation rules

- `num_train_epochs >= 1`
- `per_device_train_batch_size >= 1`
- `gradient_accumulation_steps >= 1`
- `learning_rate > 0`
- `fp16` and `bf16` are mutually exclusive
- `lr_scheduler_type` must be one of the supported values

Unknown keys passed to `from_dict` are silently stored in `extra_kwargs` and
never cause an error, making the class forward-compatible with future options.
