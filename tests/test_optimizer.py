"""Tests for unsloth.models.optimizer."""

from __future__ import annotations

import pytest

from unsloth.models.optimizer import OptimizerConfig, build_optimizer


# ---------------------------------------------------------------------------
# OptimizerConfig construction & validation
# ---------------------------------------------------------------------------

def test_default_config_is_valid():
    cfg = OptimizerConfig()
    assert cfg.optimizer_type == "adamw"
    assert cfg.learning_rate == pytest.approx(2e-4)


def test_from_dict_basic():
    cfg = OptimizerConfig.from_dict({"optimizer_type": "sgd", "learning_rate": 1e-3})
    assert cfg.optimizer_type == "sgd"
    assert cfg.learning_rate == pytest.approx(1e-3)


def test_from_dict_ignores_unknown_keys():
    cfg = OptimizerConfig.from_dict({"optimizer_type": "adam", "unknown_key": 99})
    assert cfg.optimizer_type == "adam"


def test_invalid_optimizer_type_raises():
    with pytest.raises(ValueError, match="optimizer_type"):
        OptimizerConfig(optimizer_type="rmsprop")


def test_invalid_learning_rate_raises():
    with pytest.raises(ValueError, match="learning_rate"):
        OptimizerConfig(learning_rate=-1e-4)


def test_invalid_beta1_raises():
    with pytest.raises(ValueError, match="beta1"):
        OptimizerConfig(beta1=1.0)


def test_invalid_beta2_raises():
    with pytest.raises(ValueError, match="beta2"):
        OptimizerConfig(beta2=-0.1)


def test_invalid_epsilon_raises():
    with pytest.raises(ValueError, match="epsilon"):
        OptimizerConfig(epsilon=0.0)


# ---------------------------------------------------------------------------
# build_optimizer
# ---------------------------------------------------------------------------

def _dummy_params(requires_grad: bool = True):
    """Return a minimal list of (name, param) pairs backed by real tensors."""
    import torch

    p = torch.nn.Parameter(torch.randn(4, 4), requires_grad=requires_grad)
    return [("w", p)]


def test_build_adamw_optimizer():
    torch = pytest.importorskip("torch")
    cfg = OptimizerConfig(optimizer_type="adamw", learning_rate=1e-3)
    opt = build_optimizer(iter(_dummy_params()), cfg)
    assert type(opt).__name__ == "AdamW"


def test_build_adam_optimizer():
    torch = pytest.importorskip("torch")
    cfg = OptimizerConfig(optimizer_type="adam")
    opt = build_optimizer(iter(_dummy_params()), cfg)
    assert type(opt).__name__ == "Adam"


def test_build_sgd_optimizer():
    torch = pytest.importorskip("torch")
    cfg = OptimizerConfig(optimizer_type="sgd", momentum=0.9)
    opt = build_optimizer(iter(_dummy_params()), cfg)
    assert type(opt).__name__ == "SGD"


def test_no_grad_params_excluded():
    torch = pytest.importorskip("torch")
    cfg = OptimizerConfig()
    opt = build_optimizer(iter(_dummy_params(requires_grad=False)), cfg)
    # param_groups should have an empty params list
    assert opt.param_groups[0]["params"] == []
