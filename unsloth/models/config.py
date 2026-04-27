# Model configuration utilities for unsloth
# Handles loading and validating model configs

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class UnslothConfig:
    """Configuration for Unsloth model loading and training."""

    model_name: str = "unsloth/llama-3-8b-bnb-4bit"
    max_seq_length: int = 2048
    dtype: Optional[str] = None  # None for auto-detection
    load_in_4bit: bool = True
    load_in_8bit: bool = False

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    bias: str = "none"
    use_gradient_checkpointing: bool = True
    random_state: int = 3407

    def validate(self):
        """Validate configuration values."""
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot enable both load_in_4bit and load_in_8bit.")
        if self.lora_r <= 0:
            raise ValueError(f"lora_r must be positive, got {self.lora_r}.")
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}.")
        if not (0.0 <= self.lora_dropout < 1.0):
            raise ValueError(f"lora_dropout must be in [0, 1), got {self.lora_dropout}.")
        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {self.max_seq_length}.")
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "UnslothConfig":
        """Create an UnslothConfig from a dictionary."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
