"""Tests for unsloth.models.scheduler."""

import pytest

from unsloth.models.scheduler import SchedulerConfig, VALID_SCHEDULER_TYPES


# ---------------------------------------------------------------------------
# Construction / defaults
# ---------------------------------------------------------------------------

def test_default_config_is_valid():
    cfg = SchedulerConfig()
    assert cfg.scheduler_type == "linear"
    assert cfg.warmup_steps == 0
    assert cfg.warmup_ratio == 0.0
    assert cfg.num_training_steps is None


def test_from_dict_basic():
    cfg = SchedulerConfig.from_dict({"scheduler_type": "cosine", "warmup_steps": 10})
    assert cfg.scheduler_type == "cosine"
    assert cfg.warmup_steps == 10


def test_from_dict_ignores_unknown_keys():
    cfg = SchedulerConfig.from_dict({"scheduler_type": "constant", "foo": "bar"})
    assert cfg.scheduler_type == "constant"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_invalid_scheduler_type_raises():
    with pytest.raises(ValueError, match="scheduler_type"):
        SchedulerConfig(scheduler_type="magic_scheduler")


def test_negative_warmup_steps_raises():
    with pytest.raises(ValueError, match="warmup_steps"):
        SchedulerConfig(warmup_steps=-1)


def test_warmup_ratio_out_of_range_raises():
    with pytest.raises(ValueError, match="warmup_ratio"):
        SchedulerConfig(warmup_ratio=1.5)


def test_both_warmup_params_raises():
    with pytest.raises(ValueError, match="mutually exclusive"):
        SchedulerConfig(warmup_steps=10, warmup_ratio=0.1)


def test_zero_training_steps_raises():
    with pytest.raises(ValueError, match="num_training_steps"):
        SchedulerConfig(num_training_steps=0)


def test_invalid_num_cycles_raises():
    with pytest.raises(ValueError, match="num_cycles"):
        SchedulerConfig(num_cycles=0)


def test_invalid_power_raises():
    with pytest.raises(ValueError, match="power"):
        SchedulerConfig(power=0.0)


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------

def test_effective_warmup_steps_from_steps():
    cfg = SchedulerConfig(warmup_steps=50)
    assert cfg.effective_warmup_steps(total_steps=1000) == 50


def test_effective_warmup_steps_from_ratio():
    cfg = SchedulerConfig(warmup_ratio=0.1)
    assert cfg.effective_warmup_steps(total_steps=200) == 20


def test_build_kwargs_contains_required_keys():
    cfg = SchedulerConfig(scheduler_type="cosine", warmup_steps=5)
    kwargs = cfg.build_kwargs(total_steps=100)
    assert kwargs["name"] == "cosine"
    assert kwargs["num_warmup_steps"] == 5
    assert kwargs["num_training_steps"] == 100


def test_all_valid_scheduler_types_accepted():
    for stype in VALID_SCHEDULER_TYPES:
        cfg = SchedulerConfig(scheduler_type=stype)
        assert cfg.scheduler_type == stype
