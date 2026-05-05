"""Kernel patching utilities for Unsloth speed optimisations.

This module provides a lightweight registry that tracks which model
modules have been patched with faster kernels, and helpers to apply
or revert those patches at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class PatchRecord:
    """Metadata about a single applied patch."""

    module_path: str  # e.g. "transformers.models.llama.modeling_llama"
    attribute: str   # e.g. "LlamaAttention.forward"
    original: object
    replacement: Callable
    description: str = ""


class PatchRegistry:
    """Central registry for all Unsloth kernel patches."""

    def __init__(self) -> None:
        self._records: List[PatchRecord] = []

    # ------------------------------------------------------------------
    def apply(self, record: PatchRecord) -> None:
        """Apply *record* and remember it for later reversal."""
        parts = record.attribute.split(".")
        if len(parts) != 2:
            raise ValueError(
                f"attribute must be 'ClassName.method', got {record.attribute!r}"
            )
        class_name, method_name = parts

        import importlib
        mod = importlib.import_module(record.module_path)
        cls = getattr(mod, class_name)
        setattr(cls, method_name, record.replacement)
        self._records.append(record)

    def revert_all(self) -> int:
        """Restore all patched methods to their originals.

        Returns the number of patches reverted.
        """
        import importlib
        count = 0
        for rec in reversed(self._records):
            parts = rec.attribute.split(".")
            class_name, method_name = parts
            mod = importlib.import_module(rec.module_path)
            cls = getattr(mod, class_name)
            setattr(cls, method_name, rec.original)
            count += 1
        self._records.clear()
        return count

    def applied_patches(self) -> List[Dict[str, str]]:
        """Return a summary list of currently applied patches."""
        return [
            {
                "module": r.module_path,
                "attribute": r.attribute,
                "description": r.description,
            }
            for r in self._records
        ]

    def is_patched(self, module_path: str, attribute: str) -> bool:
        """Return True if the given attribute is currently patched."""
        return any(
            r.module_path == module_path and r.attribute == attribute
            for r in self._records
        )


# Module-level singleton used by the rest of Unsloth.
default_registry = PatchRegistry()
