"""Learning rate scheduler configuration and builder for Unsloth training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


VALID_SCHEDULER_TYPES = (
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
)


@dataclass
class SchedulerConfig:
    """Configuration for a learning rate scheduler."""

    scheduler_type: str = "linear"
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    num_training_steps: Optional[int] = None
    num_cycles: int = 1
    power: float = 1.0
    last_epoch: int = -1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.scheduler_type not in VALID_SCHEDULER_TYPES:
            raise ValueError(
                f"scheduler_type must be one of {VALID_SCHEDULER_TYPES}, "
                f"got '{self.scheduler_type}'."
            )
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0.")
        if not (0.0 <= self.warmup_ratio <= 1.0):
            raise ValueError("warmup_ratio must be in [0.0, 1.0].")
        if self.warmup_steps > 0 and self.warmup_ratio > 0.0:
            raise ValueError(
                "Specify either warmup_steps or warmup_ratio, not both."
            )
        if self.num_training_steps is not None and self.num_training_steps <= 0:
            raise ValueError("num_training_steps must be a positive integer.")
        if self.num_cycles < 1:
            raise ValueError("num_cycles must be >= 1.")
        if self.power <= 0.0:
            raise ValueError("power must be > 0.")

    @classmethod
    def from_dict(cls, data: dict) -> "SchedulerConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def effective_warmup_steps(self, total_steps: int) -> int:
        """Return concrete warmup steps, resolving ratio if needed."""
        if self.warmup_steps > 0:
            return self.warmup_steps
        return int(self.warmup_ratio * total_steps)

    def build_kwargs(self, total_steps: int) -> dict:
        """Return keyword arguments suitable for a HuggingFace get_scheduler call."""
        return {
            "name": self.scheduler_type,
            "num_warmup_steps": self.effective_warmup_steps(total_steps),
            "num_training_steps": total_steps,
        }
