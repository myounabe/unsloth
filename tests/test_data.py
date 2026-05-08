"""Tests for unsloth.models.data.DatasetConfig."""

import pytest
from unsloth.models.data import DatasetConfig, VALID_DATASET_FORMATS


def test_default_config_is_valid():
    cfg = DatasetConfig()
    assert cfg.dataset_format == "alpaca"
    assert cfg.split == "train"
    assert cfg.seed == 42
    assert cfg.max_samples is None
    assert cfg.extra_columns == []


def test_from_dict_basic():
    cfg = DatasetConfig.from_dict({"dataset_name": "my_data", "split": "test", "seed": 7})
    assert cfg.dataset_name == "my_data"
    assert cfg.split == "test"
    assert cfg.seed == 7


def test_from_dict_ignores_unknown_keys():
    cfg = DatasetConfig.from_dict({"dataset_name": "x", "nonexistent_key": 999})
    assert cfg.dataset_name == "x"
    assert not hasattr(cfg, "nonexistent_key")


def test_invalid_dataset_format_raises():
    with pytest.raises(ValueError, match="dataset_format"):
        DatasetConfig(dataset_format="csv")


def test_invalid_max_samples_raises():
    with pytest.raises(ValueError, match="max_samples"):
        DatasetConfig(max_samples=0)


def test_negative_max_samples_raises():
    with pytest.raises(ValueError, match="max_samples"):
        DatasetConfig(max_samples=-10)


def test_negative_seed_raises():
    with pytest.raises(ValueError, match="seed"):
        DatasetConfig(seed=-1)


def test_empty_split_raises():
    with pytest.raises(ValueError, match="split"):
        DatasetConfig(split="")


def test_all_valid_formats_accepted():
    for fmt in VALID_DATASET_FORMATS:
        cfg = DatasetConfig(dataset_format=fmt)
        assert cfg.dataset_format == fmt


def test_column_mapping_returns_correct_keys():
    cfg = DatasetConfig()
    mapping = cfg.column_mapping()
    assert set(mapping.keys()) == {"text", "instruction", "input", "output"}
    assert mapping["text"] == cfg.text_column
    assert mapping["instruction"] == cfg.instruction_column


def test_column_mapping_reflects_custom_columns():
    cfg = DatasetConfig(text_column="content", output_column="response")
    mapping = cfg.column_mapping()
    assert mapping["text"] == "content"
    assert mapping["output"] == "response"
