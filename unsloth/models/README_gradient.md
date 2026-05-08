# Gradient Module

The `gradient` module provides lightweight helpers for gradient clipping and
accumulation bookkeeping used during fine-tuning.

## GradientConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `max_grad_norm` | `float` | `1.0` | Maximum L2 norm for gradient clipping. |
| `accumulation_steps` | `int` | `1` | Number of micro-steps before an optimizer step. |
| `clip_value` | `float \| None` | `None` | Hard clip value; overrides `max_grad_norm` when set. |
| `log_grad_norm` | `bool` | `False` | Whether to log the pre-clip gradient norm. |

### Validation rules

- `max_grad_norm` must be **positive**.
- `accumulation_steps` must be **≥ 1**.
- `clip_value`, when provided, must be **positive**.

## clip_gradients

```python
from unsloth.models.gradient import GradientConfig, clip_gradients

cfg = GradientConfig(max_grad_norm=1.0, accumulation_steps=4)
norm = clip_gradients(model.parameters(), cfg)
print(f"Pre-clip norm: {norm:.4f}")
```

Returns the global gradient norm *before* clipping so callers can log it.
