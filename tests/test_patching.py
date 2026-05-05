"""Tests for unsloth.models.patching."""

from __future__ import annotations

import types
import sys
import pytest

from unsloth.models.patching import PatchRecord, PatchRegistry


# ---------------------------------------------------------------------------
# Helpers – build a tiny fake module so we don't need transformers installed.
# ---------------------------------------------------------------------------

def _make_fake_module(module_path: str, class_name: str, method_name: str):
    """Inject a disposable module into sys.modules for testing."""

    def original_method(self):
        return "original"

    cls = type(class_name, (), {method_name: original_method})
    mod = types.ModuleType(module_path)
    setattr(mod, class_name, cls)
    sys.modules[module_path] = mod
    return mod, cls, original_method


def _cleanup(module_path: str):
    sys.modules.pop(module_path, None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_apply_patches_method():
    mod_path = "_fake_mod_apply"
    mod, cls, orig = _make_fake_module(mod_path, "FakeCls", "forward")
    try:
        registry = PatchRegistry()
        replacement = lambda self: "patched"
        record = PatchRecord(
            module_path=mod_path,
            attribute="FakeCls.forward",
            original=orig,
            replacement=replacement,
            description="test patch",
        )
        registry.apply(record)
        assert cls().forward() == "patched"
    finally:
        _cleanup(mod_path)


def test_revert_all_restores_original():
    mod_path = "_fake_mod_revert"
    mod, cls, orig = _make_fake_module(mod_path, "FakeCls", "forward")
    try:
        registry = PatchRegistry()
        record = PatchRecord(
            module_path=mod_path,
            attribute="FakeCls.forward",
            original=orig,
            replacement=lambda self: "patched",
        )
        registry.apply(record)
        count = registry.revert_all()
        assert count == 1
        assert cls().forward() == "original"
        assert registry.applied_patches() == []
    finally:
        _cleanup(mod_path)


def test_is_patched_true_and_false():
    mod_path = "_fake_mod_is_patched"
    mod, cls, orig = _make_fake_module(mod_path, "FakeCls", "forward")
    try:
        registry = PatchRegistry()
        assert not registry.is_patched(mod_path, "FakeCls.forward")
        record = PatchRecord(
            module_path=mod_path,
            attribute="FakeCls.forward",
            original=orig,
            replacement=lambda self: "patched",
        )
        registry.apply(record)
        assert registry.is_patched(mod_path, "FakeCls.forward")
    finally:
        _cleanup(mod_path)


def test_invalid_attribute_raises():
    registry = PatchRegistry()
    with pytest.raises(ValueError, match="ClassName.method"):
        registry.apply(
            PatchRecord(
                module_path="some.module",
                attribute="no_dot_here",
                original=None,
                replacement=lambda: None,
            )
        )


def test_applied_patches_summary_contains_description():
    mod_path = "_fake_mod_summary"
    mod, cls, orig = _make_fake_module(mod_path, "FakeCls", "forward")
    try:
        registry = PatchRegistry()
        record = PatchRecord(
            module_path=mod_path,
            attribute="FakeCls.forward",
            original=orig,
            replacement=lambda self: None,
            description="my description",
        )
        registry.apply(record)
        summary = registry.applied_patches()
        assert len(summary) == 1
        assert summary[0]["description"] == "my description"
    finally:
        _cleanup(mod_path)
