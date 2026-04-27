# Quantization Support

Unsloth supports post-training quantization via [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes).

## Quick start

```python
from unsloth.models.quantization import QuantizationConfig, make_bnb_config

# 4-bit NF4 quantization (default)
qcfg = QuantizationConfig(bits=4, quant_type="nf4", compute_dtype="bfloat16")
bnb_config = make_bnb_config(qcfg)

# Pass to the loader
from unsloth.models.loader import load_model_and_tokenizer
from unsloth.models.config import UnslothConfig

cfg = UnslothConfig(model_name="unsloth/llama-3-8b", load_in_4bit=True)
model, tokenizer = load_model_and_tokenizer(cfg)
```

## QuantizationConfig fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `bits` | `4` \| `8` | `4` | Number of quantization bits |
| `double_quant` | `bool` | `True` | Enable double quantization |
| `quant_type` | `"nf4"` \| `"fp4"` | `"nf4"` | Quantization data type |
| `compute_dtype` | `str` | `"float16"` | Dtype used for compute (`"float16"`, `"bfloat16"`, `"float32"`) |
| `threshold` | `float` | `6.0` | Outlier threshold for 8-bit quantization |
| `extra` | `dict` | `{}` | Pass-through kwargs reserved for future use |

## Building from a dictionary

```python
qcfg = QuantizationConfig.from_dict({
    "bits": 8,
    "compute_dtype": "bfloat16",
    "unknown_future_key": "ignored",
})
```

Unknown keys are silently ignored, making forward-compatible configuration files easy to maintain.

## Disabling quantization

Pass `None` to `make_bnb_config` and no quantization config will be created:

```python
bnb_config = make_bnb_config(None)  # returns None
```
