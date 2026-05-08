"""Tests for unsloth.models.memory."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from unsloth.models.memory import MemorySnapshot, MemoryTracker


# ---------------------------------------------------------------------------
# MemorySnapshot tests
# ---------------------------------------------------------------------------

def test_snapshot_to_dict_has_required_keys():
    snap = MemorySnapshot(step=1, allocated_mib=100.0, reserved_mib=200.0, peak_allocated_mib=150.0)
    d = snap.to_dict()
    assert set(d.keys()) == {"step", "allocated_mib", "reserved_mib", "peak_allocated_mib"}


def test_snapshot_values_preserved():
    snap = MemorySnapshot(step=5, allocated_mib=42.5, reserved_mib=80.0, peak_allocated_mib=55.0)
    assert snap.step == 5
    assert snap.allocated_mib == 42.5


# ---------------------------------------------------------------------------
# MemoryTracker – no CUDA path
# ---------------------------------------------------------------------------

def _tracker_no_cuda() -> MemoryTracker:
    """Return a tracker that always falls back to zeros (no CUDA)."""
    return MemoryTracker(device=0)


def test_initial_history_is_empty():
    tracker = _tracker_no_cuda()
    assert tracker.history() == []


def test_peak_with_no_history_returns_none():
    tracker = _tracker_no_cuda()
    assert tracker.peak() is None


def test_snapshot_appends_to_history():
    tracker = _tracker_no_cuda()
    tracker.snapshot(0)
    tracker.snapshot(1)
    assert len(tracker.history()) == 2


def test_snapshot_step_recorded_correctly():
    tracker = _tracker_no_cuda()
    snap = tracker.snapshot(7)
    assert snap.step == 7


def test_reset_clears_history():
    tracker = _tracker_no_cuda()
    tracker.snapshot(0)
    tracker.snapshot(1)
    tracker.reset()
    assert tracker.history() == []
    assert tracker.peak() is None


def test_peak_returns_maximum_allocated():
    tracker = _tracker_no_cuda()
    # Manually inject snapshots to control values
    tracker._history.append(MemorySnapshot(0, 10.0, 20.0, 10.0))
    tracker._history.append(MemorySnapshot(1, 50.0, 60.0, 50.0))
    tracker._history.append(MemorySnapshot(2, 30.0, 40.0, 50.0))
    assert tracker.peak() == 50.0


def test_history_returns_list_of_dicts():
    tracker = _tracker_no_cuda()
    tracker.snapshot(0)
    h = tracker.history()
    assert isinstance(h, list)
    assert isinstance(h[0], dict)


# ---------------------------------------------------------------------------
# MemoryTracker – mocked CUDA path
# ---------------------------------------------------------------------------

def test_snapshot_uses_cuda_stats_when_available():
    fake_stats = {
        "allocated_bytes.all.current": 512 * 1024 ** 2,
        "reserved_bytes.all.current": 1024 * 1024 ** 2,
        "allocated_bytes.all.peak": 768 * 1024 ** 2,
    }
    mock_torch = MagicMock()
    mock_torch.cuda.memory_stats.return_value = fake_stats

    with patch.dict(sys.modules, {"torch": mock_torch}):
        tracker = MemoryTracker(device=0)
        snap = tracker.snapshot(3)

    assert snap.allocated_mib == 512.0
    assert snap.reserved_mib == 1024.0
    assert snap.peak_allocated_mib == 768.0
