# Optimizer Module

This module provides a lightweight wrapper around PyTorch (and HuggingFace
Transformers) optimizers so that optimizer hyper-parameters can be stored,
validated, and serialised in the same way as every other Unsloth config.

## Classes

### `OptimizerConfig`

A `dataclass` that holds all hyper-parameters for a single optimizer.

| Field | Type | Default | Description |
|---|---|---|---|
| `optimizer_type` | `str` | `"adamw"` | One of `adamw`, `adam`, `sgd`, `adafactor` |
| `learning_rate` | `float` | `2e-4` | Must be > 0 |
| `weight_decay` | `float` | `0.01` | Must be ≥ 0 |
| `beta1` | `float` | `0.9` | Adam β₁ — must be in `[0, 1)` |
| `beta2` | `float` | `0.999` | Adam β₂ — must be in `[0, 1)` |
| `epsilon` | `float` | `1e-8` | Adam ε — must be > 0 |
| `momentum` | `float` | `0.0` | SGD momentum |
| `extra` | `dict` | `{}` | Forwarded verbatim to the optimizer |

Validation runs automatically on construction via `__post_init__`.

#### `from_dict(data)`

Convenience constructor — unknown keys are silently ignored, making it safe
to pass a raw configuration dictionary that may contain unrelated fields.

## Functions

### `build_optimizer(params, config)`

Instantiates and returns a `torch.optim.Optimizer` (or a Transformers
`Adafactor` instance) from the supplied `OptimizerConfig`.

Only parameters that have `requires_grad=True` are passed to the optimizer.

## Example

```python
from unsloth.models.optimizer import OptimizerConfig, build_optimizer

cfg = OptimizerConfig(optimizer_type="adamw", learning_rate=3e-4)
optimizer = build_optimizer(model.named_parameters(), cfg)
```
