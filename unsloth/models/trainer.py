"""Lightweight training utilities for Unsloth fine-tuning workflows."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingArguments:
    """Minimal training arguments for supervised fine-tuning."""

    output_dir: str = "./outputs"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    logging_steps: int = 10
    save_steps: int = 100
    fp16: bool = False
    bf16: bool = False
    seed: int = 42
    max_steps: int = -1
    lr_scheduler_type: str = "linear"
    weight_decay: float = 0.01
    extra_kwargs: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.num_train_epochs < 1:
            raise ValueError("num_train_epochs must be >= 1")
        if self.per_device_train_batch_size < 1:
            raise ValueError("per_device_train_batch_size must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.fp16 and self.bf16:
            raise ValueError("fp16 and bf16 cannot both be True")
        if self.lr_scheduler_type not in ("linear", "cosine", "constant"):
            raise ValueError(
                f"lr_scheduler_type '{self.lr_scheduler_type}' is not supported; "
                "choose from 'linear', 'cosine', 'constant'"
            )

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingArguments":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        obj = cls(**filtered)
        obj.extra_kwargs = extra
        obj.validate()
        return obj

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps
