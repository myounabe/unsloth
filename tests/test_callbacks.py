"""Tests for unsloth/models/callbacks.py"""

import pytest
from unsloth.models.callbacks import TrainingMetrics, UnslothProgressCallback


# --- TrainingMetrics ---

def test_initial_metrics_have_no_history():
    m = TrainingMetrics()
    assert m.history == []
    assert m.best_loss() is None


def test_record_appends_to_history():
    m = TrainingMetrics()
    m.record(step=1, epoch=0.1, loss=2.5, lr=1e-4, elapsed=10.0)
    assert len(m.history) == 1
    assert m.history[0]["loss"] == 2.5


def test_best_loss_returns_minimum():
    m = TrainingMetrics()
    m.record(step=1, epoch=0.1, loss=2.5, lr=1e-4, elapsed=5.0)
    m.record(step=2, epoch=0.2, loss=1.8, lr=1e-4, elapsed=10.0)
    m.record(step=3, epoch=0.3, loss=2.1, lr=1e-4, elapsed=15.0)
    assert m.best_loss() == pytest.approx(1.8)


def test_summary_keys_present():
    m = TrainingMetrics()
    m.record(step=5, epoch=1.0, loss=0.9, lr=5e-5, elapsed=30.0)
    summary = m.summary()
    for key in ("total_steps", "final_epoch", "final_loss", "best_loss", "elapsed_seconds"):
        assert key in summary


def test_summary_reflects_last_record():
    m = TrainingMetrics()
    m.record(step=10, epoch=2.0, loss=0.5, lr=1e-5, elapsed=60.0)
    s = m.summary()
    assert s["total_steps"] == 10
    assert s["final_loss"] == pytest.approx(0.5)


# --- UnslothProgressCallback ---

def test_callback_records_on_log():
    cb = UnslothProgressCallback(verbose=False)
    cb.on_train_begin()
    cb.on_log(step=1, epoch=0.5, logs={"loss": 1.2, "learning_rate": 2e-4})
    assert cb.metrics.loss == pytest.approx(1.2)
    assert cb.metrics.step == 1


def test_callback_on_train_end_returns_summary():
    cb = UnslothProgressCallback(verbose=False)
    cb.on_train_begin()
    cb.on_log(step=2, epoch=1.0, logs={"loss": 0.8, "learning_rate": 1e-4})
    summary = cb.on_train_end()
    assert summary["total_steps"] == 2
    assert summary["best_loss"] == pytest.approx(0.8)


def test_callback_multiple_logs_track_best():
    cb = UnslothProgressCallback(verbose=False)
    cb.on_train_begin()
    for step, loss in enumerate([3.0, 2.0, 1.5, 1.8], start=1):
        cb.on_log(step=step, epoch=step * 0.25, logs={"loss": loss, "learning_rate": 1e-4})
    assert cb.metrics.best_loss() == pytest.approx(1.5)


def test_callback_handles_missing_log_keys():
    cb = UnslothProgressCallback(verbose=False)
    cb.on_train_begin()
    # logs without 'learning_rate' or 'samples_per_second'
    cb.on_log(step=1, epoch=0.1, logs={"loss": 0.7})
    assert cb.metrics.learning_rate is None
    assert cb.metrics.samples_per_second is None
