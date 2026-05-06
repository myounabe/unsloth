"""Tests for unsloth.models.tokenizer."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from unsloth.models.tokenizer import (
    TokenizerConfig,
    get_tokenizer_vocab_size,
    prepare_tokenizer,
)


# ---------------------------------------------------------------------------
# TokenizerConfig tests
# ---------------------------------------------------------------------------

def test_default_config_is_valid():
    cfg = TokenizerConfig(model_name_or_path="dummy")
    assert cfg.max_seq_length == 2048
    assert cfg.padding_side == "right"
    assert cfg.truncation_side == "right"
    assert cfg.extra_special_tokens == []


def test_from_dict_basic():
    cfg = TokenizerConfig.from_dict({"model_name_or_path": "x", "max_seq_length": 512})
    assert cfg.model_name_or_path == "x"
    assert cfg.max_seq_length == 512


def test_from_dict_ignores_unknown_keys():
    cfg = TokenizerConfig.from_dict({"model_name_or_path": "y", "unknown_key": 99})
    assert cfg.model_name_or_path == "y"
    assert not hasattr(cfg, "unknown_key")


def test_invalid_max_seq_length_raises():
    with pytest.raises(ValueError, match="max_seq_length"):
        TokenizerConfig(model_name_or_path="m", max_seq_length=0)


def test_invalid_padding_side_raises():
    with pytest.raises(ValueError, match="padding_side"):
        TokenizerConfig(model_name_or_path="m", padding_side="center")


def test_invalid_truncation_side_raises():
    with pytest.raises(ValueError, match="truncation_side"):
        TokenizerConfig(model_name_or_path="m", truncation_side="middle")


# ---------------------------------------------------------------------------
# prepare_tokenizer tests
# ---------------------------------------------------------------------------

def _make_tokenizer(pad_token=None, eos_token="</s>"):
    tok = MagicMock()
    tok.pad_token = pad_token
    tok.eos_token = eos_token
    tok.__len__ = MagicMock(return_value=32000)
    return tok


def test_prepare_tokenizer_sets_padding_side():
    cfg = TokenizerConfig(model_name_or_path="m", padding_side="left")
    tok = _make_tokenizer(pad_token="<pad>")
    prepare_tokenizer(tok, cfg)
    assert tok.padding_side == "left"


def test_prepare_tokenizer_sets_pad_token_from_eos_when_missing():
    cfg = TokenizerConfig(model_name_or_path="m")
    tok = _make_tokenizer(pad_token=None, eos_token="</s>")
    prepare_tokenizer(tok, cfg)
    assert tok.pad_token == "</s>"


def test_prepare_tokenizer_adds_extra_special_tokens():
    cfg = TokenizerConfig(model_name_or_path="m", extra_special_tokens=["<custom>"])
    tok = _make_tokenizer(pad_token="<pad>")
    prepare_tokenizer(tok, cfg)
    tok.add_special_tokens.assert_called_once_with(
        {"additional_special_tokens": ["<custom>"]}
    )


# ---------------------------------------------------------------------------
# get_tokenizer_vocab_size tests
# ---------------------------------------------------------------------------

def test_get_tokenizer_vocab_size():
    tok = _make_tokenizer(pad_token="<pad>")
    assert get_tokenizer_vocab_size(tok) == 32000
