"""Optimizer configuration and factory utilities for Unsloth training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional, Tuple

_SUPPORTED_OPTIMIZERS = {"adamw", "adam", "sgd", "adafactor"}


@dataclass
class OptimizerConfig:
    """Configuration for the training optimizer."""

    optimizer_type: str = "adamw"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    momentum: float = 0.0  # only used by SGD
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.optimizer_type not in _SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"optimizer_type must be one of {_SUPPORTED_OPTIMIZERS}, "
                f"got '{self.optimizer_type}'"
            )
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if not (0.0 <= self.weight_decay):
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if not (0.0 <= self.beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0, 1), got {self.beta1}")
        if not (0.0 <= self.beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0, 1), got {self.beta2}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizerConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def build_optimizer(
    params: Iterator[Tuple[str, Any]],
    config: OptimizerConfig,
) -> Any:
    """Instantiate a PyTorch optimizer from *config*.

    Parameters
    ----------
    params:
        An iterable of ``(name, parameter)`` pairs, as returned by
        ``model.named_parameters()``.  Only parameters that require grad
        are included automatically.
    config:
        An :class:`OptimizerConfig` instance.

    Returns
    -------
    torch.optim.Optimizer
    """
    import torch.optim as optim

    trainable = [p for _, p in params if p.requires_grad]

    otype = config.optimizer_type
    if otype == "adamw":
        return optim.AdamW(
            trainable,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay,
            **config.extra,
        )
    if otype == "adam":
        return optim.Adam(
            trainable,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay,
            **config.extra,
        )
    if otype == "sgd":
        return optim.SGD(
            trainable,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            **config.extra,
        )
    if otype == "adafactor":
        try:
            from transformers.optimization import Adafactor
        except ImportError as exc:
            raise ImportError(
                "transformers must be installed to use adafactor"
            ) from exc
        return Adafactor(
            trainable,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            **config.extra,
        )
    raise ValueError(f"Unsupported optimizer_type: '{otype}'")
