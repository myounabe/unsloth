# Tests for unsloth/models/loader.py

import pytest
import torch
from unittest.mock import patch, MagicMock
from unsloth.models.loader import _resolve_dtype, load_model_and_tokenizer
from unsloth.models.config import UnslothConfig


def test_resolve_dtype_float16():
    assert _resolve_dtype("float16") == torch.float16


def test_resolve_dtype_bfloat16():
    assert _resolve_dtype("bfloat16") == torch.bfloat16


def test_resolve_dtype_float32():
    assert _resolve_dtype("float32") == torch.float32


def test_resolve_dtype_unknown_raises():
    with pytest.raises(ValueError, match="Unknown dtype"):
        _resolve_dtype("int8")


def test_resolve_dtype_auto_returns_torch_dtype():
    dtype = _resolve_dtype(None)
    assert dtype in (torch.float16, torch.bfloat16)


@patch("unsloth.models.loader.AutoModelForCausalLM")
@patch("unsloth.models.loader.AutoTokenizer")
def test_load_model_and_tokenizer_called(mock_tokenizer_cls, mock_model_cls):
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
    mock_model_cls.from_pretrained.return_value = mock_model

    cfg = UnslothConfig(
        model_name="test/model",
        load_in_4bit=False,
        load_in_8bit=False,
        dtype="float16",
    )

    with patch("unsloth.models.loader.BitsAndBytesConfig", create=True):
        model, tokenizer = load_model_and_tokenizer(cfg)

    assert model is mock_model
    assert tokenizer is mock_tokenizer
    mock_tokenizer_cls.from_pretrained.assert_called_once_with(
        "test/model", trust_remote_code=True
    )


def test_load_model_invalid_config_raises():
    cfg = UnslothConfig(load_in_4bit=True, load_in_8bit=True)
    with pytest.raises(ValueError, match="Cannot enable both"):
        load_model_and_tokenizer(cfg)
