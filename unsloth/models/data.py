"""Dataset configuration and preparation utilities for Unsloth training."""

from dataclasses import dataclass, field
from typing import Optional, List


VALID_DATASET_FORMATS = ("alpaca", "sharegpt", "raw", "instruction")


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing."""

    dataset_name: str = ""
    dataset_format: str = "alpaca"
    split: str = "train"
    max_samples: Optional[int] = None
    seed: int = 42
    text_column: str = "text"
    instruction_column: str = "instruction"
    input_column: str = "input"
    output_column: str = "output"
    extra_columns: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ValueError if the configuration is invalid."""
        if self.dataset_format not in VALID_DATASET_FORMATS:
            raise ValueError(
                f"dataset_format must be one of {VALID_DATASET_FORMATS}, "
                f"got {self.dataset_format!r}"
            )
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError(
                f"max_samples must be a positive integer, got {self.max_samples}"
            )
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if not self.split:
            raise ValueError("split must be a non-empty string")

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetConfig":
        """Construct a DatasetConfig from a dictionary, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def column_mapping(self) -> dict:
        """Return a mapping of logical roles to actual column names."""
        return {
            "text": self.text_column,
            "instruction": self.instruction_column,
            "input": self.input_column,
            "output": self.output_column,
        }
