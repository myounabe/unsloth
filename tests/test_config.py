# Tests for unsloth/models/config.py

import pytest
from unsloth.models.config import UnslothConfig


def test_default_config_is_valid():
    cfg = UnslothConfig()
    assert cfg.validate() is cfg


def test_from_dict_basic():
    cfg = UnslothConfig.from_dict({"model_name": "my/model", "lora_r": 8})
    assert cfg.model_name == "my/model"
    assert cfg.lora_r == 8
    # defaults preserved
    assert cfg.max_seq_length == 2048


def test_from_dict_ignores_unknown_keys():
    cfg = UnslothConfig.from_dict({"lora_r": 4, "unknown_key": "value"})
    assert cfg.lora_r == 4
    assert not hasattr(cfg, "unknown_key")


def test_both_quantization_raises():
    cfg = UnslothConfig(load_in_4bit=True, load_in_8bit=True)
    with pytest.raises(ValueError, match="Cannot enable both"):
        cfg.validate()


def test_invalid_lora_r_raises():
    cfg = UnslothConfig(lora_r=0)
    with pytest.raises(ValueError, match="lora_r must be positive"):
        cfg.validate()


def test_invalid_lora_alpha_raises():
    cfg = UnslothConfig(lora_alpha=-1)
    with pytest.raises(ValueError, match="lora_alpha must be positive"):
        cfg.validate()


def test_invalid_lora_dropout_raises():
    cfg = UnslothConfig(lora_dropout=1.5)
    with pytest.raises(ValueError, match="lora_dropout must be in"):
        cfg.validate()


def test_invalid_max_seq_length_raises():
    cfg = UnslothConfig(max_seq_length=-512)
    with pytest.raises(ValueError, match="max_seq_length must be positive"):
        cfg.validate()


def test_no_quantization_is_valid():
    cfg = UnslothConfig(load_in_4bit=False, load_in_8bit=False)
    assert cfg.validate() is cfg
