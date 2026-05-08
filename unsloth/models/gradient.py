"""Gradient clipping and accumulation utilities for Unsloth training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GradientConfig:
    """Configuration for gradient clipping and accumulation."""

    max_grad_norm: float = 1.0
    accumulation_steps: int = 1
    clip_value: Optional[float] = None
    log_grad_norm: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {self.max_grad_norm}")
        if self.accumulation_steps < 1:
            raise ValueError(
                f"accumulation_steps must be >= 1, got {self.accumulation_steps}"
            )
        if self.clip_value is not None and self.clip_value <= 0:
            raise ValueError(f"clip_value must be positive, got {self.clip_value}")

    @classmethod
    def from_dict(cls, data: dict) -> "GradientConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @property
    def effective_clip(self) -> float:
        """Return clip_value if set, otherwise max_grad_norm."""
        return self.clip_value if self.clip_value is not None else self.max_grad_norm


def clip_gradients(parameters, config: GradientConfig) -> float:
    """Clip gradients and return the pre-clip global norm.

    Parameters
    ----------
    parameters:
        Iterable of ``torch.nn.Parameter`` objects.
    config:
        A :class:`GradientConfig` instance controlling clipping behaviour.

    Returns
    -------
    float
        The global gradient norm *before* clipping.
    """
    import torch

    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0

    total_norm = torch.nn.utils.clip_grad_norm_(params, config.effective_clip)
    return float(total_norm)
