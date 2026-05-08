"""Tests for unsloth.models.gradient."""

import pytest
from unsloth.models.gradient import GradientConfig, clip_gradients


# ---------------------------------------------------------------------------
# GradientConfig – construction & validation
# ---------------------------------------------------------------------------

def test_default_config_is_valid():
    cfg = GradientConfig()
    assert cfg.max_grad_norm == 1.0
    assert cfg.accumulation_steps == 1
    assert cfg.clip_value is None
    assert cfg.log_grad_norm is False


def test_from_dict_basic():
    cfg = GradientConfig.from_dict({"max_grad_norm": 0.5, "accumulation_steps": 8})
    assert cfg.max_grad_norm == 0.5
    assert cfg.accumulation_steps == 8


def test_from_dict_ignores_unknown_keys():
    cfg = GradientConfig.from_dict({"max_grad_norm": 1.0, "unknown_key": 99})
    assert cfg.max_grad_norm == 1.0


def test_invalid_max_grad_norm_raises():
    with pytest.raises(ValueError, match="max_grad_norm"):
        GradientConfig(max_grad_norm=0.0)


def test_negative_max_grad_norm_raises():
    with pytest.raises(ValueError, match="max_grad_norm"):
        GradientConfig(max_grad_norm=-1.0)


def test_invalid_accumulation_steps_raises():
    with pytest.raises(ValueError, match="accumulation_steps"):
        GradientConfig(accumulation_steps=0)


def test_invalid_clip_value_raises():
    with pytest.raises(ValueError, match="clip_value"):
        GradientConfig(clip_value=-0.5)


def test_effective_clip_uses_clip_value_when_set():
    cfg = GradientConfig(max_grad_norm=1.0, clip_value=0.25)
    assert cfg.effective_clip == 0.25


def test_effective_clip_falls_back_to_max_grad_norm():
    cfg = GradientConfig(max_grad_norm=2.0)
    assert cfg.effective_clip == 2.0


# ---------------------------------------------------------------------------
# clip_gradients helper
# ---------------------------------------------------------------------------

def test_clip_gradients_returns_float():
    torch = pytest.importorskip("torch")

    linear = torch.nn.Linear(4, 4)
    loss = linear(torch.randn(2, 4)).sum()
    loss.backward()

    cfg = GradientConfig(max_grad_norm=1.0)
    norm = clip_gradients(linear.parameters(), cfg)
    assert isinstance(norm, float)
    assert norm >= 0.0


def test_clip_gradients_no_grad_returns_zero():
    torch = pytest.importorskip("torch")

    linear = torch.nn.Linear(4, 4)  # no backward call → no .grad
    cfg = GradientConfig(max_grad_norm=1.0)
    norm = clip_gradients(linear.parameters(), cfg)
    assert norm == 0.0
