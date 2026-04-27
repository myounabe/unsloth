"""Tests for unsloth.models.quantization."""
import pytest
from unittest.mock import MagicMock, patch

from unsloth.models.quantization import QuantizationConfig, make_bnb_config


# ---------------------------------------------------------------------------
# QuantizationConfig construction
# ---------------------------------------------------------------------------

def test_default_config_is_valid():
    cfg = QuantizationConfig()
    assert cfg.bits == 4
    assert cfg.quant_type == "nf4"
    assert cfg.double_quant is True


def test_from_dict_basic():
    cfg = QuantizationConfig.from_dict({"bits": 8, "quant_type": "fp4"})
    assert cfg.bits == 8
    assert cfg.quant_type == "fp4"


def test_from_dict_ignores_unknown_keys():
    cfg = QuantizationConfig.from_dict({"bits": 4, "nonexistent_key": "value"})
    assert cfg.bits == 4
    assert not hasattr(cfg, "nonexistent_key")


def test_invalid_bits_raises():
    with pytest.raises(ValueError, match="bits must be one of"):
        QuantizationConfig(bits=3)  # type: ignore[arg-type]


def test_invalid_quant_type_raises():
    with pytest.raises(ValueError, match="quant_type must be"):
        QuantizationConfig(quant_type="int8")


def test_invalid_threshold_raises():
    with pytest.raises(ValueError, match="threshold must be positive"):
        QuantizationConfig(threshold=-1.0)


def test_extra_defaults_to_empty_dict():
    cfg = QuantizationConfig()
    assert cfg.extra == {}


# ---------------------------------------------------------------------------
# to_bnb_kwargs
# ---------------------------------------------------------------------------

def test_to_bnb_kwargs_4bit():
    import torch
    cfg = QuantizationConfig(bits=4, compute_dtype="bfloat16")
    kwargs = cfg.to_bnb_kwargs()
    assert kwargs["load_in_4bit"] is True
    assert kwargs["load_in_8bit"] is False
    assert kwargs["bnb_4bit_compute_dtype"] == torch.bfloat16


def test_to_bnb_kwargs_8bit():
    cfg = QuantizationConfig(bits=8)
    kwargs = cfg.to_bnb_kwargs()
    assert kwargs["load_in_8bit"] is True
    assert kwargs["load_in_4bit"] is False


# ---------------------------------------------------------------------------
# make_bnb_config
# ---------------------------------------------------------------------------

def test_make_bnb_config_none_returns_none():
    assert make_bnb_config(None) is None


def test_make_bnb_config_calls_transformers():
    fake_bnb_cls = MagicMock(return_value=MagicMock())
    with patch.dict("sys.modules", {"transformers": MagicMock(BitsAndBytesConfig=fake_bnb_cls)}):
        cfg = QuantizationConfig(bits=4)
        result = make_bnb_config(cfg)
    fake_bnb_cls.assert_called_once()
    assert result is not None
