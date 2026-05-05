# Kernel Patching

The `patching` module provides a **registry-based** mechanism for swapping
PyTorch / Transformers methods with faster Unsloth kernels at runtime.

## Core types

| Symbol | Description |
|---|---|
| `PatchRecord` | Dataclass holding the target module path, attribute (`ClassName.method`), the original callable, and the replacement. |
| `PatchRegistry` | Tracks applied patches and allows bulk reversion. |
| `default_registry` | Module-level singleton used throughout Unsloth. |

## Quick start

```python
from unsloth.models.patching import PatchRecord, default_registry
import transformers.models.llama.modeling_llama as llama_mod
from unsloth.kernels.fast_attention import fast_forward

record = PatchRecord(
    module_path="transformers.models.llama.modeling_llama",
    attribute="LlamaAttention.forward",
    original=llama_mod.LlamaAttention.forward,
    replacement=fast_forward,
    description="Fused RoPE + attention kernel",
)
default_registry.apply(record)
```

## Reverting patches

Call `default_registry.revert_all()` to restore every patched method to its
original implementation.  This is useful in test teardown or when unloading
the model:

```python
reverted = default_registry.revert_all()
print(f"Reverted {reverted} patch(es).")
```

## Inspecting applied patches

```python
for patch in default_registry.applied_patches():
    print(patch["module"], patch["attribute"], "-", patch["description"])
```

## Notes

- `attribute` must follow the `"ClassName.method"` format exactly.
- Patches are reverted in **LIFO** order to handle dependency chains safely.
- The registry does **not** deduplicate; applying the same patch twice will
  register two revert steps.
