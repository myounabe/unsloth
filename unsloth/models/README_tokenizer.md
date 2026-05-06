# Tokenizer Utilities

This module provides helpers for loading and configuring tokenizers used during
fine-tuning with Unsloth.

## `TokenizerConfig`

A dataclass that holds all tokenizer-related settings.

| Field | Type | Default | Description |
|---|---|---|---|
| `model_name_or_path` | `str` | `""` | HuggingFace model id or local path |
| `max_seq_length` | `int` | `2048` | Maximum token sequence length |
| `padding_side` | `str` | `"right"` | `"left"` or `"right"` |
| `truncation_side` | `str` | `"right"` | `"left"` or `"right"` |
| `add_special_tokens` | `bool` | `True` | Whether to add special tokens on encode |
| `extra_special_tokens` | `list[str]` | `[]` | Additional tokens to register |

### Validation rules

- `max_seq_length` must be a positive integer.
- `padding_side` and `truncation_side` must each be `"left"` or `"right"`.

## `prepare_tokenizer(tokenizer, config)`

Applies the settings from a `TokenizerConfig` to an existing tokenizer object.
If the tokenizer has no `pad_token`, it is set to `eos_token` (or `[PAD]` is
added as a fallback).

## `get_tokenizer_vocab_size(tokenizer)`

Convenience wrapper that returns `len(tokenizer)`.

## Example

```python
from unsloth.models.tokenizer import TokenizerConfig, prepare_tokenizer

cfg = TokenizerConfig(model_name_or_path="meta-llama/Llama-2-7b-hf", max_seq_length=4096)
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
tokenizer = prepare_tokenizer(tokenizer, cfg)
```
