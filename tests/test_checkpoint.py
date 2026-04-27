"""Tests for unsloth.models.checkpoint."""

import json
import pytest
from pathlib import Path

from unsloth.models.checkpoint import CheckpointMeta, latest_checkpoint


# ---------------------------------------------------------------------------
# CheckpointMeta construction
# ---------------------------------------------------------------------------

def test_default_extra_is_empty_dict():
    meta = CheckpointMeta(step=0, epoch=0.0, best_loss=1.0, model_name="m")
    assert meta.extra == {}


def test_negative_step_raises():
    with pytest.raises(ValueError, match="step"):
        CheckpointMeta(step=-1, epoch=0.0, best_loss=1.0, model_name="m")


def test_negative_epoch_raises():
    with pytest.raises(ValueError, match="epoch"):
        CheckpointMeta(step=0, epoch=-0.1, best_loss=1.0, model_name="m")


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

def test_save_creates_json_file(tmp_path):
    meta = CheckpointMeta(step=10, epoch=1.5, best_loss=0.42, model_name="llama")
    saved = meta.save(tmp_path)
    assert saved.exists()
    assert saved.name == "checkpoint_meta.json"


def test_round_trip(tmp_path):
    meta = CheckpointMeta(step=5, epoch=2.0, best_loss=0.3, model_name="mistral", extra={"lr": 1e-4})
    meta.save(tmp_path)
    loaded = CheckpointMeta.load(tmp_path)
    assert loaded.step == 5
    assert loaded.epoch == 2.0
    assert abs(loaded.best_loss - 0.3) < 1e-9
    assert loaded.model_name == "mistral"
    assert loaded.extra == {"lr": 1e-4}


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CheckpointMeta.load(tmp_path)


# ---------------------------------------------------------------------------
# latest_checkpoint
# ---------------------------------------------------------------------------

def test_latest_checkpoint_returns_none_for_empty_dir(tmp_path):
    assert latest_checkpoint(tmp_path) is None


def test_latest_checkpoint_returns_none_for_missing_dir(tmp_path):
    assert latest_checkpoint(tmp_path / "nonexistent") is None


def test_latest_checkpoint_picks_highest_step(tmp_path):
    for step in (1, 5, 3):
        ckpt_dir = tmp_path / f"checkpoint-{step}"
        ckpt_dir.mkdir()
        CheckpointMeta(step=step, epoch=float(step), best_loss=1.0 / step, model_name="m").save(ckpt_dir)

    result = latest_checkpoint(tmp_path)
    assert result is not None
    assert result.name == "checkpoint-5"


def test_latest_checkpoint_skips_invalid_dirs(tmp_path):
    # One valid checkpoint
    valid = tmp_path / "checkpoint-2"
    valid.mkdir()
    CheckpointMeta(step=2, epoch=1.0, best_loss=0.5, model_name="m").save(valid)

    # One directory with a broken JSON file
    broken = tmp_path / "checkpoint-99"
    broken.mkdir()
    (broken / "checkpoint_meta.json").write_text("not-json", encoding="utf-8")

    result = latest_checkpoint(tmp_path)
    assert result == valid
