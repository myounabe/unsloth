# Model loader for unsloth
# Wraps model + tokenizer loading with sensible defaults

import torch
from typing import Tuple, Optional
from unsloth.models.config import UnslothConfig


def _resolve_dtype(dtype_str: Optional[str]) -> torch.dtype:
    """Resolve dtype string to torch.dtype."""
    if dtype_str is None:
        # Auto-detect based on hardware capability
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unknown dtype '{dtype_str}'. Choose from {list(mapping)}.")
    return mapping[dtype_str]


def load_model_and_tokenizer(config: UnslothConfig) -> Tuple[object, object]:
    """
    Load a model and tokenizer using the given UnslothConfig.

    Returns:
        (model, tokenizer) tuple
    """
    config.validate()

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        raise ImportError("transformers is required. Install with: pip install transformers") from e

    dtype = _resolve_dtype(config.dtype)

    quantization_config = None
    if config.load_in_4bit or config.load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=config.load_in_4bit,
                load_in_8bit=config.load_in_8bit,
                bnb_4bit_compute_dtype=dtype,
            )
        except ImportError as e:
            raise ImportError("bitsandbytes is required for quantization.") from e

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quantization_config,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    return model, tokenizer
