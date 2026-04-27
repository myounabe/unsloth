"""Tests for unsloth.models.trainer.TrainingArguments."""

import pytest
from unsloth.models.trainer import TrainingArguments


def test_default_training_arguments_are_valid():
    args = TrainingArguments()
    args.validate()  # should not raise


def test_from_dict_basic():
    args = TrainingArguments.from_dict(
        {"num_train_epochs": 3, "learning_rate": 1e-4, "output_dir": "/tmp/out"}
    )
    assert args.num_train_epochs == 3
    assert args.learning_rate == 1e-4
    assert args.output_dir == "/tmp/out"


def test_from_dict_ignores_unknown_keys():
    args = TrainingArguments.from_dict({"unknown_param": "value", "num_train_epochs": 2})
    assert args.num_train_epochs == 2
    assert args.extra_kwargs == {"unknown_param": "value"}


def test_invalid_epochs_raises():
    with pytest.raises(ValueError, match="num_train_epochs"):
        TrainingArguments(num_train_epochs=0).validate()


def test_invalid_learning_rate_raises():
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingArguments(learning_rate=-1e-4).validate()


def test_fp16_and_bf16_together_raises():
    with pytest.raises(ValueError, match="fp16 and bf16"):
        TrainingArguments(fp16=True, bf16=True).validate()


def test_invalid_lr_scheduler_raises():
    with pytest.raises(ValueError, match="lr_scheduler_type"):
        TrainingArguments(lr_scheduler_type="polynomial").validate()


def test_effective_batch_size():
    args = TrainingArguments(per_device_train_batch_size=4, gradient_accumulation_steps=8)
    assert args.effective_batch_size == 32


def test_cosine_scheduler_is_valid():
    args = TrainingArguments(lr_scheduler_type="cosine")
    args.validate()  # should not raise


def test_constant_scheduler_is_valid():
    args = TrainingArguments(lr_scheduler_type="constant")
    args.validate()  # should not raise
