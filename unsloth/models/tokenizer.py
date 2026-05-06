"""Tokenizer utilities for Unsloth."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer loading and preparation."""

    model_name_or_path: str = ""
    max_seq_length: int = 2048
    padding_side: str = "right"
    truncation_side: str = "right"
    add_special_tokens: bool = True
    extra_special_tokens: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {self.max_seq_length}")
        if self.padding_side not in ("left", "right"):
            raise ValueError(f"padding_side must be 'left' or 'right', got {self.padding_side!r}")
        if self.truncation_side not in ("left", "right"):
            raise ValueError(
                f"truncation_side must be 'left' or 'right', got {self.truncation_side!r}"
            )

    @classmethod
    def from_dict(cls, data: dict) -> "TokenizerConfig":
        """Create a TokenizerConfig from a dictionary, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def prepare_tokenizer(tokenizer, config: TokenizerConfig):
    """Apply config settings to a tokenizer instance and ensure a pad token exists."""
    tokenizer.padding_side = config.padding_side
    tokenizer.truncation_side = config.truncation_side
    tokenizer.model_max_length = config.max_seq_length

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if config.extra_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": config.extra_special_tokens})

    return tokenizer


def get_tokenizer_vocab_size(tokenizer) -> int:
    """Return the vocabulary size of a tokenizer."""
    return len(tokenizer)
