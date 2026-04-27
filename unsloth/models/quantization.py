"""Quantization utilities for Unsloth models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


SUPPORTED_BITS = (4, 8)
QuantBits = Literal[4, 8]


@dataclass
class QuantizationConfig:
    """Configuration for BitsAndBytes quantization."""

    bits: QuantBits = 4
    double_quant: bool = True
    quant_type: str = "nf4"
    compute_dtype: str = "float16"
    threshold: float = 6.0
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bits not in SUPPORTED_BITS:
            raise ValueError(
                f"bits must be one of {SUPPORTED_BITS}, got {self.bits}"
            )
        if self.quant_type not in ("nf4", "fp4"):
            raise ValueError(
                f"quant_type must be 'nf4' or 'fp4', got {self.quant_type!r}"
            )
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")

    @classmethod
    def from_dict(cls, data: dict) -> "QuantizationConfig":
        """Build a QuantizationConfig from a dictionary, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def to_bnb_kwargs(self) -> dict:
        """Return keyword arguments suitable for BitsAndBytesConfig."""
        import torch  # local import to keep module importable without torch

        _dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        compute_dtype = _dtype_map.get(self.compute_dtype, torch.float16)
        kwargs: dict = {
            "load_in_4bit": self.bits == 4,
            "load_in_8bit": self.bits == 8,
            "bnb_4bit_use_double_quant": self.double_quant,
            "bnb_4bit_quant_type": self.quant_type,
            "bnb_4bit_compute_dtype": compute_dtype,
        }
        return kwargs


def make_bnb_config(cfg: Optional[QuantizationConfig]):
    """Return a BitsAndBytesConfig or None if cfg is None."""
    if cfg is None:
        return None
    try:
        from transformers import BitsAndBytesConfig  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("transformers is required for quantization") from exc
    return BitsAndBytesConfig(**cfg.to_bnb_kwargs())
